"""Main RAG evaluator for orchestrating the evaluation pipeline."""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..logging import get_logger
from ..search import EnhancedSearchEngine
from ..vector_store import VectorStore
from .judge import LLMJudge
from .metrics import EvaluationMetrics
from .test_suite import TestCase, TestResult, TestSuite


class RAGEvaluator:
    """Main evaluator for RAG system quality assessment."""

    def __init__(
        self,
        vector_store: VectorStore,
        search_engine: EnhancedSearchEngine,
        llm_judge: LLMJudge,
        test_suite: TestSuite,
    ):
        self.vector_store = vector_store
        self.search_engine = search_engine
        self.llm_judge = llm_judge
        self.test_suite = test_suite
        self.logger = get_logger(__name__)

        # Evaluation state
        self.current_evaluation_id: Optional[str] = None
        self.test_results: List[TestResult] = []
        self.metrics: Optional[EvaluationMetrics] = None

        # Callbacks for progress tracking
        self.progress_callback: Optional[Callable[[str, int, int], None]] = None
        self.result_callback: Optional[Callable[[TestResult], None]] = None

    def set_progress_callback(self, callback: Callable[[str, int, int], None]) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback

    def set_result_callback(self, callback: Callable[[TestResult], None]) -> None:
        """Set callback for individual result updates."""
        self.result_callback = callback

    async def run_evaluation(
        self,
        evaluation_id: Optional[str] = None,
        max_concurrent: int = 5,
        enable_judging: bool = True,
    ) -> EvaluationMetrics:
        """
        Run the complete RAG evaluation pipeline.

        Args:
            evaluation_id: Optional evaluation ID, auto-generated if not provided
            max_concurrent: Maximum concurrent test executions
            enable_judging: Whether to enable LLM judging

        Returns:
            EvaluationMetrics with complete results
        """
        # Initialize evaluation
        self.current_evaluation_id = evaluation_id or f"eval_{uuid.uuid4().hex[:8]}"
        self.test_results = []
        self.metrics = EvaluationMetrics()

        self.logger.info(f"Starting RAG evaluation: {self.current_evaluation_id}")
        self.logger.info(
            f"Test suite: {self.test_suite.name} ({len(self.test_suite.test_cases)} test cases)"
        )

        # Start timing
        start_time = time.time()
        self.metrics.evaluation_started = datetime.utcnow()

        try:
            # Phase 1: Execute all test cases
            await self._execute_test_cases(max_concurrent)

            # Phase 2: LLM judging (if enabled)
            if enable_judging:
                await self._judge_responses()

            # Phase 3: Calculate metrics
            self._calculate_metrics()

            # Complete timing
            end_time = time.time()
            self.metrics.evaluation_completed = datetime.utcnow()
            self.metrics.evaluation_duration_seconds = end_time - start_time

            self.logger.info(
                f"Evaluation completed: {self.current_evaluation_id} "
                f"({self.metrics.evaluation_duration_seconds:.2f}s)"
            )

            return self.metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

    async def _execute_test_cases(self, max_concurrent: int) -> None:
        """Execute all test cases with controlled concurrency."""
        self.logger.info(f"Executing {len(self.test_suite.test_cases)} test cases")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create tasks for all test cases
        tasks = []
        for i, test_case in enumerate(self.test_suite.test_cases):
            task = asyncio.create_task(
                self._execute_single_test_case(
                    test_case, i + 1, len(self.test_suite.test_cases), semaphore
                )
            )
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info(f"Test execution completed: {len(self.test_results)} results")

    async def _execute_single_test_case(
        self,
        test_case: TestCase,
        current: int,
        total: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a single test case."""
        async with semaphore:
            try:
                # Update progress
                if self.progress_callback:
                    self.progress_callback(
                        f"Executing test case {test_case.id}", current, total
                    )

                self.logger.info(
                    f"Executing test case {current}/{total}: {test_case.id}"
                )

                # Execute the test case
                test_result = await self._execute_test_case(test_case)

                # Store result
                self.test_results.append(test_result)

                # Call result callback
                if self.result_callback:
                    self.result_callback(test_result)

                self.logger.info(
                    f"Test case {test_case.id} completed: "
                    f"response_time={test_result.response_time_ms:.2f}ms"
                )

            except Exception as e:
                self.logger.error(f"Test case {test_case.id} failed: {e}")

                # Create failed result
                failed_result = TestResult(
                    test_case=test_case,
                    execution_id=f"{self.current_evaluation_id}_{test_case.id}",
                    rag_response="",
                    errors=[str(e)],
                )

                self.test_results.append(failed_result)

    async def _execute_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case and return the result."""
        execution_id = f"{self.current_evaluation_id}_{test_case.id}"
        start_time = time.time()

        try:
            # Execute search query
            search_results = await self.search_engine.search(
                query=test_case.query,
                n_results=5,
                library=test_case.expected_libraries[0]
                if test_case.expected_libraries
                else None,
                content_type=test_case.expected_content_types[0]
                if test_case.expected_content_types
                else None,
            )

            # Extract response and metadata
            rag_response = self._format_search_results(search_results)
            retrieved_chunks = [
                {
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                }
                for result in search_results
            ]

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Create test result
            test_result = TestResult(
                test_case=test_case,
                execution_id=execution_id,
                rag_response=rag_response,
                retrieved_chunks=retrieved_chunks,
                search_metadata={
                    "n_results": len(search_results),
                    "query": test_case.query,
                    "filters": {
                        "library": test_case.expected_libraries,
                        "content_type": test_case.expected_content_types,
                    },
                },
                response_time_ms=response_time_ms,
            )

            return test_result

        except Exception as e:
            # Handle execution errors
            response_time_ms = (time.time() - start_time) * 1000

            return TestResult(
                test_case=test_case,
                execution_id=execution_id,
                rag_response="",
                errors=[f"Execution failed: {e}"],
                response_time_ms=response_time_ms,
            )

    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable response."""
        if not search_results:
            return "No relevant results found."

        formatted_response = []

        for i, result in enumerate(search_results[:3]):  # Limit to top 3 results
            content = result.get("content", "")
            metadata = result.get("metadata", {})

            # Add result header
            title = metadata.get("title", f"Result {i+1}")
            url = metadata.get("url", "")

            formatted_response.append(f"**{title}**")
            if url:
                formatted_response.append(f"Source: {url}")

            # Add content (truncated)
            content_preview = content[:300] + "..." if len(content) > 300 else content
            formatted_response.append(f"{content_preview}\n")

        return "\n".join(formatted_response)

    async def _judge_responses(self) -> None:
        """Use LLM judge to evaluate all responses."""
        if not self.test_results:
            self.logger.warning("No test results to judge")
            return

        self.logger.info(f"Starting LLM judging of {len(self.test_results)} responses")

        # Filter successful results for judging
        successful_results = [r for r in self.test_results if r.is_successful()]

        if not successful_results:
            self.logger.warning("No successful test results to judge")
            return

        # Update progress
        if self.progress_callback:
            self.progress_callback("LLM judging responses", 0, len(successful_results))

        # Judge responses in batches
        judged_results = await self.llm_judge.evaluate_batch(successful_results)

        # Update test results with judged scores
        for judged_result in judged_results:
            for i, original_result in enumerate(self.test_results):
                if original_result.execution_id == judged_result.execution_id:
                    self.test_results[i] = judged_result
                    break

        self.logger.info(
            f"LLM judging completed: {len(judged_results)} responses evaluated"
        )

    def _calculate_metrics(self) -> None:
        """Calculate comprehensive evaluation metrics."""
        if not self.metrics:
            return

        self.logger.info("Calculating evaluation metrics")
        self.metrics.calculate_from_results(self.test_results)

        # Log summary
        summary = self.metrics.get_summary()
        self.logger.info(
            f"Evaluation summary: {summary['quality_scores']['overall']}/5 overall score"
        )
        self.logger.info(f"Success rate: {summary['test_execution']['success_rate']}")
        self.logger.info(f"Grade: {self.metrics.get_grade()}")

        # Log improvement areas
        improvement_areas = self.metrics.get_improvement_areas()
        if improvement_areas:
            self.logger.info("Areas for improvement:")
            for area in improvement_areas:
                self.logger.info(f"  - {area}")

    def get_test_result(self, test_case_id: str) -> Optional[TestResult]:
        """Get test result by test case ID."""
        for result in self.test_results:
            if result.test_case.id == test_case_id:
                return result
        return None

    def get_results_by_category(self, category: str) -> List[TestResult]:
        """Get test results filtered by category."""
        return [r for r in self.test_results if r.test_case.category.value == category]

    def get_results_by_difficulty(self, difficulty: str) -> List[TestResult]:
        """Get test results filtered by difficulty."""
        return [
            r for r in self.test_results if r.test_case.difficulty.value == difficulty
        ]

    def export_results(self, format: str = "json") -> str:
        """Export evaluation results in specified format."""
        if format.lower() == "json":
            import json

            return json.dumps(self.metrics.get_summary(), indent=2)
        elif format.lower() == "csv":
            return self._export_csv()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_csv(self) -> str:
        """Export results as CSV."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(
            [
                "Test ID",
                "Query",
                "Category",
                "Difficulty",
                "Relevance",
                "Accuracy",
                "Completeness",
                "Helpfulness",
                "Overall",
                "Response Time (ms)",
                "Errors",
            ]
        )

        # Write data
        for result in self.test_results:
            writer.writerow(
                [
                    result.test_case.id,
                    result.test_case.query[:50] + "..."
                    if len(result.test_case.query) > 50
                    else result.test_case.query,
                    result.test_case.category.value,
                    result.test_case.difficulty.value,
                    result.relevance_score or "N/A",
                    result.accuracy_score or "N/A",
                    result.completeness_score or "N/A",
                    result.helpfulness_score or "N/A",
                    result.overall_score or "N/A",
                    result.response_time_ms,
                    "; ".join(result.errors) if result.errors else "",
                ]
            )

        return output.getvalue()

    def save_results(self, filepath: str, format: str = "json") -> None:
        """Save evaluation results to file."""
        content = self.export_results(format)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        self.logger.info(f"Results saved to: {filepath}")

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a comprehensive evaluation summary."""
        if not self.metrics:
            return {}

        summary = self.metrics.get_summary()
        summary["grade"] = self.metrics.get_grade()
        summary["improvement_areas"] = self.metrics.get_improvement_areas()

        return summary
