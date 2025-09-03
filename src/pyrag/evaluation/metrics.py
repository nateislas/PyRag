"""Evaluation metrics and quality scoring for RAG system evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .test_suite import DifficultyLevel, QueryCategory, TestResult


@dataclass
class QualityScore:
    """Quality score for a specific metric."""

    metric: str
    score: float
    max_score: float = 5.0
    weight: float = 1.0

    @property
    def normalized_score(self) -> float:
        """Get normalized score (0-1)."""
        return self.score / self.max_score

    @property
    def weighted_score(self) -> float:
        """Get weighted score."""
        return self.score * self.weight


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for RAG system performance."""

    # Test execution metrics
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    success_rate: float = 0.0

    # Quality scores (averages)
    average_relevance: float = 0.0
    average_accuracy: float = 0.0
    average_completeness: float = 0.0
    average_helpfulness: float = 0.0
    average_overall: float = 0.0

    # Performance metrics
    average_response_time_ms: float = 0.0
    total_token_usage: Optional[int] = None
    average_tokens_per_query: float = 0.0

    # Category breakdown
    category_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    difficulty_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Score distribution
    score_distribution: Dict[str, Dict[int, int]] = field(default_factory=dict)

    # Timestamps
    evaluation_started: Optional[datetime] = None
    evaluation_completed: Optional[datetime] = None
    evaluation_duration_seconds: float = 0.0

    def calculate_from_results(self, test_results: List[TestResult]) -> None:
        """Calculate metrics from test results."""
        if not test_results:
            return

        self.total_tests = len(test_results)
        self.successful_tests = sum(1 for r in test_results if r.is_successful())
        self.failed_tests = self.total_tests - self.successful_tests
        self.success_rate = (
            self.successful_tests / self.total_tests if self.total_tests > 0 else 0.0
        )

        # Calculate average scores
        self._calculate_average_scores(test_results)

        # Calculate performance metrics
        self._calculate_performance_metrics(test_results)

        # Calculate category and difficulty breakdowns
        self._calculate_breakdowns(test_results)

        # Calculate score distributions
        self._calculate_score_distributions(test_results)

    def _calculate_average_scores(self, test_results: List[TestResult]) -> None:
        """Calculate average quality scores."""
        successful_results = [r for r in test_results if r.is_successful()]

        if not successful_results:
            return

        # Calculate averages for successful tests
        self.average_relevance = sum(
            r.relevance_score or 0 for r in successful_results
        ) / len(successful_results)
        self.average_accuracy = sum(
            r.accuracy_score or 0 for r in successful_results
        ) / len(successful_results)
        self.average_completeness = sum(
            r.completeness_score or 0 for r in successful_results
        ) / len(successful_results)
        self.average_helpfulness = sum(
            r.helpfulness_score or 0 for r in successful_results
        ) / len(successful_results)
        self.average_overall = sum(
            r.overall_score or 0 for r in successful_results
        ) / len(successful_results)

    def _calculate_performance_metrics(self, test_results: List[TestResult]) -> None:
        """Calculate performance metrics."""
        successful_results = [r for r in test_results if r.is_successful()]

        if not successful_results:
            return

        # Response time
        response_times = [
            r.response_time_ms for r in successful_results if r.response_time_ms > 0
        ]
        if response_times:
            self.average_response_time_ms = sum(response_times) / len(response_times)

        # Token usage
        token_usage = []
        for r in successful_results:
            if r.token_usage:
                if isinstance(r.token_usage, dict):
                    total_tokens = sum(r.token_usage.values())
                    token_usage.append(total_tokens)
                elif isinstance(r.token_usage, (int, float)):
                    token_usage.append(int(r.token_usage))

        if token_usage:
            self.total_token_usage = sum(token_usage)
            self.average_tokens_per_query = self.total_token_usage / len(token_usage)

    def _calculate_breakdowns(self, test_results: List[TestResult]) -> None:
        """Calculate category and difficulty breakdowns."""
        # Category breakdown
        for category in QueryCategory:
            category_results = [
                r
                for r in test_results
                if r.test_case.category == category and r.is_successful()
            ]
            if category_results:
                self.category_scores[category.value] = {
                    "count": len(category_results),
                    "average_relevance": sum(
                        r.relevance_score or 0 for r in category_results
                    )
                    / len(category_results),
                    "average_accuracy": sum(
                        r.accuracy_score or 0 for r in category_results
                    )
                    / len(category_results),
                    "average_completeness": sum(
                        r.completeness_score or 0 for r in category_results
                    )
                    / len(category_results),
                    "average_helpfulness": sum(
                        r.helpfulness_score or 0 for r in category_results
                    )
                    / len(category_results),
                    "average_overall": sum(
                        r.overall_score or 0 for r in category_results
                    )
                    / len(category_results),
                }

        # Difficulty breakdown
        for difficulty in DifficultyLevel:
            difficulty_results = [
                r
                for r in test_results
                if r.test_case.difficulty == difficulty and r.is_successful()
            ]
            if difficulty_results:
                self.difficulty_scores[difficulty.value] = {
                    "count": len(difficulty_results),
                    "average_relevance": sum(
                        r.relevance_score or 0 for r in difficulty_results
                    )
                    / len(difficulty_results),
                    "average_accuracy": sum(
                        r.accuracy_score or 0 for r in difficulty_results
                    )
                    / len(difficulty_results),
                    "average_completeness": sum(
                        r.completeness_score or 0 for r in difficulty_results
                    )
                    / len(difficulty_results),
                    "average_helpfulness": sum(
                        r.helpfulness_score or 0 for r in difficulty_results
                    )
                    / len(difficulty_results),
                    "average_overall": sum(
                        r.overall_score or 0 for r in difficulty_results
                    )
                    / len(difficulty_results),
                }

    def _calculate_score_distributions(self, test_results: List[TestResult]) -> None:
        """Calculate score distributions for each metric."""
        metrics = ["relevance", "accuracy", "completeness", "helpfulness", "overall"]

        for metric in metrics:
            self.score_distribution[metric] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

            for result in test_results:
                if not result.is_successful():
                    continue

                score = getattr(result, f"{metric}_score")
                if score is not None:
                    score_int = int(round(score))
                    if 1 <= score_int <= 5:
                        self.score_distribution[metric][score_int] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "test_execution": {
                "total_tests": self.total_tests,
                "successful_tests": self.successful_tests,
                "failed_tests": self.failed_tests,
                "success_rate": f"{self.success_rate:.2%}",
            },
            "quality_scores": {
                "relevance": round(self.average_relevance, 2),
                "accuracy": round(self.average_accuracy, 2),
                "completeness": round(self.average_completeness, 2),
                "helpfulness": round(self.average_helpfulness, 2),
                "overall": round(self.average_overall, 2),
            },
            "performance": {
                "average_response_time_ms": round(self.average_response_time_ms, 2),
                "total_token_usage": self.total_token_usage,
                "average_tokens_per_query": round(self.average_tokens_per_query, 2),
            },
            "category_breakdown": self.category_scores,
            "difficulty_breakdown": self.difficulty_scores,
            "score_distributions": self.score_distribution,
            "evaluation_info": {
                "started": (
                    self.evaluation_started.isoformat()
                    if self.evaluation_started
                    else None
                ),
                "completed": (
                    self.evaluation_completed.isoformat()
                    if self.evaluation_completed
                    else None
                ),
                "duration_seconds": round(self.evaluation_duration_seconds, 2),
            },
        }

    def get_grade(self) -> str:
        """Get a letter grade based on overall performance."""
        if self.average_overall >= 4.5:
            return "A+"
        elif self.average_overall >= 4.0:
            return "A"
        elif self.average_overall >= 3.5:
            return "B+"
        elif self.average_overall >= 3.0:
            return "B"
        elif self.average_overall >= 2.5:
            return "C+"
        elif self.average_overall >= 2.0:
            return "C"
        elif self.average_overall >= 1.5:
            return "D+"
        elif self.average_overall >= 1.0:
            return "D"
        else:
            return "F"

    def get_improvement_areas(self) -> List[str]:
        """Get areas that need improvement based on scores."""
        improvement_areas = []

        if self.average_relevance < 3.0:
            improvement_areas.append(
                "Query relevance - responses often don't address the user's question"
            )

        if self.average_accuracy < 3.0:
            improvement_areas.append(
                "Factual accuracy - responses contain incorrect information"
            )

        if self.average_completeness < 3.0:
            improvement_areas.append(
                "Answer completeness - responses are missing key information"
            )

        if self.average_helpfulness < 3.0:
            improvement_areas.append("Helpfulness - responses are not useful to users")

        if self.success_rate < 0.8:
            improvement_areas.append("System reliability - many queries are failing")

        if self.average_response_time_ms > 1000:
            improvement_areas.append(
                "Response time - queries are taking too long to process"
            )

        return improvement_areas

    def compare_with_baseline(
        self, baseline_metrics: "EvaluationMetrics"
    ) -> Dict[str, Any]:
        """Compare current metrics with a baseline."""
        comparison = {}

        # Quality score comparisons
        for metric in [
            "relevance",
            "accuracy",
            "completeness",
            "helpfulness",
            "overall",
        ]:
            current = getattr(self, f"average_{metric}")
            baseline = getattr(baseline_metrics, f"average_{metric}")

            if baseline > 0:
                change = current - baseline
                change_percent = (change / baseline) * 100
                comparison[f"{metric}_change"] = {
                    "absolute": round(change, 2),
                    "percentage": round(change_percent, 1),
                }

        # Performance comparisons
        if baseline_metrics.average_response_time_ms > 0:
            response_time_change = (
                self.average_response_time_ms
                - baseline_metrics.average_response_time_ms
            )
            response_time_change_percent = (
                response_time_change / baseline_metrics.average_response_time_ms
            ) * 100
            comparison["response_time_change"] = {
                "absolute_ms": round(response_time_change, 2),
                "percentage": round(response_time_change_percent, 1),
            }

        # Success rate comparison
        if baseline_metrics.success_rate > 0:
            success_rate_change = self.success_rate - baseline_metrics.success_rate
            success_rate_change_percent = (
                success_rate_change / baseline_metrics.success_rate
            ) * 100
            comparison["success_rate_change"] = {
                "absolute": round(success_rate_change, 3),
                "percentage": round(success_rate_change_percent, 1),
            }

        return comparison
