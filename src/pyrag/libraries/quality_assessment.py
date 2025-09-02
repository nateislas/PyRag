"""Quality assessment for library documentation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for library documentation."""

    documentation_completeness: float
    code_example_quality: float
    api_coverage: float
    update_frequency: float
    community_activity: float
    overall_score: float


class QualityAssessment:
    """Assess documentation quality for libraries."""

    def __init__(self):
        """Initialize the quality assessment system."""
        self.logger = get_logger(__name__)

        # Quality thresholds
        self.excellent_threshold = 0.8
        self.good_threshold = 0.6
        self.poor_threshold = 0.4

    async def assess_library_quality(
        self, library_info: Dict[str, Any]
    ) -> QualityMetrics:
        """Assess the quality of a library's documentation."""
        try:
            self.logger.info(
                f"Assessing quality for library: {library_info.get('name', 'unknown')}"
            )

            # Calculate individual metrics
            doc_completeness = self._assess_documentation_completeness(library_info)
            code_quality = self._assess_code_example_quality(library_info)
            api_coverage = self._assess_api_coverage(library_info)
            update_freq = self._assess_update_frequency(library_info)
            community_activity = self._assess_community_activity(library_info)

            # Calculate overall score (weighted average)
            overall_score = (
                doc_completeness * 0.3
                + code_quality * 0.25
                + api_coverage * 0.2
                + update_freq * 0.15
                + community_activity * 0.1
            )

            metrics = QualityMetrics(
                documentation_completeness=doc_completeness,
                code_example_quality=code_quality,
                api_coverage=api_coverage,
                update_frequency=update_freq,
                community_activity=community_activity,
                overall_score=overall_score,
            )

            self.logger.info(f"Quality assessment complete: {overall_score:.2f}")
            return metrics

        except Exception as e:
            self.logger.error(f"Failed to assess library quality: {e}")
            # Return default metrics
            return QualityMetrics(
                documentation_completeness=0.0,
                code_example_quality=0.0,
                api_coverage=0.0,
                update_frequency=0.0,
                community_activity=0.0,
                overall_score=0.0,
            )

    def _assess_documentation_completeness(self, library_info: Dict[str, Any]) -> float:
        """Assess documentation completeness."""
        score = 0.0

        # Check for documentation URL
        if library_info.get("documentation_url"):
            score += 0.3

        # Check for repository URL
        if library_info.get("repository_url"):
            score += 0.2

        # Check description quality
        description = library_info.get("description", "")
        if len(description) > 100:
            score += 0.3
        elif len(description) > 50:
            score += 0.2
        elif len(description) > 20:
            score += 0.1

        # Check for license information
        if library_info.get("license"):
            score += 0.2

        return min(score, 1.0)

    def _assess_code_example_quality(self, library_info: Dict[str, Any]) -> float:
        """Assess code example quality."""
        # Mock implementation - in production this would analyze actual documentation
        # For now, return a score based on library popularity

        download_count = library_info.get("download_count", 0)
        github_stars = library_info.get("github_stars", 0)

        # Popular libraries tend to have better examples
        if download_count > 1000000 or github_stars > 10000:
            return 0.9
        elif download_count > 100000 or github_stars > 1000:
            return 0.7
        elif download_count > 10000 or github_stars > 100:
            return 0.5
        else:
            return 0.3

    def _assess_api_coverage(self, library_info: Dict[str, Any]) -> float:
        """Assess API coverage in documentation."""
        # Mock implementation - in production this would analyze API documentation
        # For now, return a score based on library maturity

        version = library_info.get("version", "0.0.0")

        try:
            major_version = int(version.split(".")[0])
            if major_version >= 1:
                return 0.8
            elif major_version >= 0:
                return 0.6
            else:
                return 0.4
        except (ValueError, IndexError):
            return 0.5

    def _assess_update_frequency(self, library_info: Dict[str, Any]) -> float:
        """Assess update frequency."""
        # Mock implementation - in production this would analyze release history
        # For now, return a score based on library activity

        github_forks = library_info.get("github_forks", 0)

        if github_forks > 1000:
            return 0.9
        elif github_forks > 100:
            return 0.7
        elif github_forks > 10:
            return 0.5
        else:
            return 0.3

    def _assess_community_activity(self, library_info: Dict[str, Any]) -> float:
        """Assess community activity."""
        # Mock implementation - in production this would analyze GitHub activity
        # For now, return a score based on GitHub stars

        github_stars = library_info.get("github_stars", 0)

        if github_stars > 10000:
            return 0.9
        elif github_stars > 1000:
            return 0.7
        elif github_stars > 100:
            return 0.5
        else:
            return 0.3

    def get_quality_level(self, metrics: QualityMetrics) -> str:
        """Get quality level based on overall score."""
        if metrics.overall_score >= self.excellent_threshold:
            return "excellent"
        elif metrics.overall_score >= self.good_threshold:
            return "good"
        elif metrics.overall_score >= self.poor_threshold:
            return "fair"
        else:
            return "poor"

    def get_quality_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Get recommendations for improving quality."""
        recommendations = []

        if metrics.documentation_completeness < 0.7:
            recommendations.append("Improve documentation completeness")

        if metrics.code_example_quality < 0.6:
            recommendations.append("Add more code examples")

        if metrics.api_coverage < 0.7:
            recommendations.append("Improve API documentation coverage")

        if metrics.update_frequency < 0.5:
            recommendations.append("Increase update frequency")

        if metrics.community_activity < 0.5:
            recommendations.append("Increase community engagement")

        return recommendations

    async def compare_libraries_quality(
        self, libraries: List[Dict[str, Any]]
    ) -> Dict[str, QualityMetrics]:
        """Compare quality across multiple libraries."""
        try:
            self.logger.info(f"Comparing quality for {len(libraries)} libraries")

            quality_results = {}

            for library in libraries:
                metrics = await self.assess_library_quality(library)
                quality_results[library.get("name", "unknown")] = metrics

            return quality_results

        except Exception as e:
            self.logger.error(f"Failed to compare library quality: {e}")
            return {}

    async def get_quality_report(self, library_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        try:
            metrics = await self.assess_library_quality(library_info)
            quality_level = self.get_quality_level(metrics)
            recommendations = self.get_quality_recommendations(metrics)

            return {
                "library_name": library_info.get("name"),
                "quality_level": quality_level,
                "metrics": {
                    "documentation_completeness": metrics.documentation_completeness,
                    "code_example_quality": metrics.code_example_quality,
                    "api_coverage": metrics.api_coverage,
                    "update_frequency": metrics.update_frequency,
                    "community_activity": metrics.community_activity,
                    "overall_score": metrics.overall_score,
                },
                "recommendations": recommendations,
                "assessment_date": "2024-01-01",  # Mock date
            }

        except Exception as e:
            self.logger.error(f"Failed to generate quality report: {e}")
            return {
                "library_name": library_info.get("name"),
                "quality_level": "unknown",
                "metrics": {},
                "recommendations": ["Unable to assess quality"],
                "assessment_date": "2024-01-01",
            }
