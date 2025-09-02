"""LLM judge system for evaluating RAG response quality."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..llm.client import LLMClient
from ..logging import get_logger
from .test_suite import TestCase, TestResult


@dataclass
class JudgeCriteria:
    """Criteria for LLM judge evaluation."""

    relevance_weight: float = 0.25
    accuracy_weight: float = 0.25
    completeness_weight: float = 0.25
    helpfulness_weight: float = 0.25

    # Scoring guidelines
    relevance_guidelines: str = """
    Score 1-5 based on how well the response addresses the user's query:
    1: Completely irrelevant, off-topic
    2: Mostly irrelevant, barely related
    3: Somewhat relevant, partially addresses query
    4: Relevant, addresses most of the query
    5: Highly relevant, directly answers the query
    """

    accuracy_guidelines: str = """
    Score 1-5 based on factual correctness:
    1: Completely incorrect, misleading information
    2: Mostly incorrect, some accuracy issues
    3: Partially correct, some inaccuracies
    4: Mostly correct, minor inaccuracies
    5: Completely accurate, factually correct
    """

    completeness_guidelines: str = """
    Score 1-5 based on how complete the answer is:
    1: Very incomplete, missing key information
    2: Incomplete, missing important details
    3: Partially complete, covers main points
    4: Mostly complete, covers most aspects
    5: Complete, comprehensive coverage
    """

    helpfulness_guidelines: str = """
    Score 1-5 based on how helpful the response is:
    1: Not helpful, confusing or uninformative
    2: Slightly helpful, minimal value
    3: Somewhat helpful, provides some value
    4: Helpful, provides good value
    5: Very helpful, highly valuable response
    """


class LLMJudge:
    """LLM-based judge for evaluating RAG response quality."""

    def __init__(self, llm_client: LLMClient, criteria: Optional[JudgeCriteria] = None):
        self.llm_client = llm_client
        self.criteria = criteria or JudgeCriteria()
        self.logger = get_logger(__name__)

        # Evaluation prompt template
        self.evaluation_prompt = self._build_evaluation_prompt()

    def _build_evaluation_prompt(self) -> str:
        """Build the evaluation prompt template."""
        return f"""
You are an expert evaluator of RAG (Retrieval-Augmented Generation) system responses. Your task is to evaluate the quality of a response to a user query.

## EVALUATION CRITERIA

### 1. RELEVANCE (Weight: {self.criteria.relevance_weight})
{self.criteria.relevance_guidelines}

### 2. ACCURACY (Weight: {self.criteria.accuracy_weight})
{self.criteria.accuracy_guidelines}

### 3. COMPLETENESS (Weight: {self.criteria.completeness_weight})
{self.criteria.completeness_guidelines}

### 4. HELPFULNESS (Weight: {self.criteria.helpfulness_weight})
{self.criteria.helpfulness_guidelines}

## EVALUATION PROCESS

1. Read the user query carefully
2. Review the expected answer (ground truth)
3. Evaluate the RAG system's response
4. Score each criterion from 1-5
5. Calculate weighted overall score
6. Provide detailed feedback for each criterion

## OUTPUT FORMAT

Respond with a JSON object in this exact format:
{{
    "relevance_score": <1-5>,
    "accuracy_score": <1-5>,
    "completeness_score": <1-5>,
    "helpfulness_score": <1-5>,
    "overall_score": <weighted_average>,
    "relevance_feedback": "<detailed feedback>",
    "accuracy_feedback": "<detailed feedback>",
    "completeness_feedback": "<detailed feedback>",
    "helpfulness_feedback": "<detailed feedback>",
    "overall_feedback": "<summary of evaluation>"
}}

## IMPORTANT NOTES

- Scores must be integers from 1-5 only
- Overall score should be the weighted average
- Provide specific, constructive feedback
- Consider the user's context and needs
- Be objective and consistent in scoring
"""

    async def evaluate_response(
        self,
        test_case: TestCase,
        rag_response: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG response using the LLM judge.

        Args:
            test_case: The test case with query and expected answer
            rag_response: The response from the RAG system
            retrieved_chunks: Optional retrieved chunks for context

        Returns:
            Dictionary with evaluation scores and feedback
        """
        try:
            # Build the evaluation context
            evaluation_context = self._build_evaluation_context(
                test_case, rag_response, retrieved_chunks
            )

            # Get LLM evaluation
            evaluation_result = await self._get_llm_evaluation(evaluation_context)

            # Parse and validate the result
            parsed_result = self._parse_evaluation_result(evaluation_result)

            # Validate scores
            self._validate_scores(parsed_result)

            self.logger.info(
                f"Evaluation completed for test case {test_case.id}: "
                f"overall_score={parsed_result['overall_score']}"
            )

            return parsed_result

        except Exception as e:
            self.logger.error(f"Evaluation failed for test case {test_case.id}: {e}")
            return self._get_fallback_evaluation()

    def _build_evaluation_context(
        self,
        test_case: TestCase,
        rag_response: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build the context for LLM evaluation."""
        context = f"""
## USER QUERY
{test_case.query}

## EXPECTED ANSWER (Ground Truth)
{test_case.expected_answer}

## RAG SYSTEM RESPONSE
{rag_response}

## ADDITIONAL CONTEXT
"""

        if test_case.user_context:
            context += f"\nUser Context: {test_case.user_context}"

        if test_case.code_context:
            context += f"\nCode Context: {test_case.code_context}"

        if test_case.version_constraints:
            context += f"\nVersion Constraints: {test_case.version_constraints}"

        if retrieved_chunks:
            context += f"\n\n## RETRIEVED CHUNKS ({len(retrieved_chunks)} chunks)"
            for i, chunk in enumerate(retrieved_chunks[:3]):  # Show first 3 chunks
                context += f"\n\nChunk {i+1}:"
                context += f"\nContent: {chunk.get('content', '')[:200]}..."
                context += f"\nMetadata: {chunk.get('metadata', {})}"

        context += "\n\n## EVALUATION"
        context += "\nPlease evaluate the RAG response based on the criteria above."

        return context

    async def _get_llm_evaluation(self, evaluation_context: str) -> str:
        """Get evaluation from the LLM."""
        try:
            # Combine prompt and context
            full_prompt = self.evaluation_prompt + "\n\n" + evaluation_context

            # Get LLM response
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent evaluation
                response_format={"type": "json_object"},
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"LLM evaluation failed: {e}")
            raise

    def _parse_evaluation_result(self, evaluation_result: str) -> Dict[str, Any]:
        """Parse the LLM evaluation result."""
        try:
            # Clean the response and parse JSON
            cleaned_response = evaluation_result.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            parsed = json.loads(cleaned_response.strip())

            # Ensure all required fields are present
            required_fields = [
                "relevance_score",
                "accuracy_score",
                "completeness_score",
                "helpfulness_score",
                "overall_score",
            ]

            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")

            return parsed

        except Exception as e:
            self.logger.error(f"Failed to parse evaluation result: {e}")
            raise

    def _validate_scores(self, parsed_result: Dict[str, Any]) -> None:
        """Validate that scores are within expected ranges."""
        score_fields = [
            "relevance_score",
            "accuracy_score",
            "completeness_score",
            "helpfulness_score",
            "overall_score",
        ]

        for field in score_fields:
            score = parsed_result[field]
            if not isinstance(score, (int, float)) or score < 1 or score > 5:
                raise ValueError(f"Invalid score for {field}: {score}. Must be 1-5.")

    def _get_fallback_evaluation(self) -> Dict[str, Any]:
        """Get fallback evaluation when LLM evaluation fails."""
        return {
            "relevance_score": 1.0,
            "accuracy_score": 1.0,
            "completeness_score": 1.0,
            "helpfulness_score": 1.0,
            "overall_score": 1.0,
            "relevance_feedback": "Evaluation failed - fallback score",
            "accuracy_feedback": "Evaluation failed - fallback score",
            "completeness_feedback": "Evaluation failed - fallback score",
            "helpfulness_feedback": "Evaluation failed - fallback score",
            "overall_feedback": "LLM evaluation failed, using fallback scores",
            "evaluation_error": True,
        }

    async def evaluate_batch(self, test_results: List[TestResult]) -> List[TestResult]:
        """
        Evaluate a batch of test results.

        Args:
            test_results: List of test results to evaluate

        Returns:
            List of test results with evaluation scores added
        """
        self.logger.info(
            f"Starting batch evaluation of {len(test_results)} test results"
        )

        evaluated_results = []

        for i, test_result in enumerate(test_results):
            try:
                self.logger.info(
                    f"Evaluating test result {i+1}/{len(test_results)}: {test_result.test_case.id}"
                )

                # Evaluate the response
                evaluation = await self.evaluate_response(
                    test_result.test_case,
                    test_result.rag_response,
                    test_result.retrieved_chunks,
                )

                # Update the test result with evaluation scores
                test_result.relevance_score = evaluation["relevance_score"]
                test_result.accuracy_score = evaluation["accuracy_score"]
                test_result.completeness_score = evaluation["completeness_score"]
                test_result.helpfulness_score = evaluation["helpfulness_score"]
                test_result.overall_score = evaluation["overall_score"]
                test_result.judge_feedback = evaluation.get("overall_feedback", "")
                test_result.score_breakdown = evaluation

                evaluated_results.append(test_result)

                # Small delay to avoid overwhelming the LLM API
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate test result {test_result.test_case.id}: {e}"
                )
                test_result.errors.append(f"Evaluation failed: {e}")
                evaluated_results.append(test_result)

        self.logger.info(
            f"Batch evaluation completed: {len(evaluated_results)} results processed"
        )
        return evaluated_results
