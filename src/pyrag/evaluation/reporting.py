"""Evaluation reporting and visualization for RAG system assessment."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import EvaluationMetrics
from .test_suite import TestResult, TestSuite


class EvaluationReporter:
    """Generate comprehensive evaluation reports."""

    def __init__(
        self,
        test_suite: TestSuite,
        metrics: EvaluationMetrics,
        test_results: List[TestResult],
    ):
        self.test_suite = test_suite
        self.metrics = metrics
        self.test_results = test_results
        self.report_timestamp = datetime.utcnow()

    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown report."""
        report = []

        # Header
        report.append("# RAG System Evaluation Report")
        report.append(
            f"**Generated**: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        report.append(f"**Test Suite**: {self.test_suite.name}")
        report.append(f"**Version**: {self.test_suite.version}")
        report.append("")

        # Executive Summary
        report.append("## 📊 Executive Summary")
        report.append("")

        summary = self.metrics.get_summary()
        grade = self.metrics.get_grade()

        report.append(f"**Overall Grade**: {grade}")
        report.append(f"**Overall Score**: {summary['quality_scores']['overall']}/5.0")
        report.append(f"**Success Rate**: {summary['test_execution']['success_rate']}")
        report.append(f"**Total Tests**: {summary['test_execution']['total_tests']}")
        report.append(
            f"**Evaluation Duration**: {summary['evaluation_info']['duration_seconds']:.2f} seconds"
        )
        report.append("")

        # Quality Scores Overview
        report.append("### Quality Scores")
        report.append("")
        report.append("| Metric | Score | Grade |")
        report.append("|--------|-------|-------|")

        for metric, score in summary["quality_scores"].items():
            grade_emoji = self._get_grade_emoji(score)
            report.append(f"| {metric.title()} | {score}/5.0 | {grade_emoji} |")

        report.append("")

        # Performance Metrics
        report.append("### Performance Metrics")
        report.append("")
        report.append(
            f"**Average Response Time**: {summary['performance']['average_response_time_ms']:.2f} ms"
        )
        if summary["performance"]["total_token_usage"]:
            report.append(
                f"**Total Token Usage**: {summary['performance']['total_token_usage']:,}"
            )
            report.append(
                f"**Average Tokens per Query**: {summary['performance']['average_tokens_per_query']:.1f}"
            )
        report.append("")

        # Test Execution Summary
        report.append("## 🧪 Test Execution Summary")
        report.append("")

        report.append(
            f"**Successful Tests**: {summary['test_execution']['successful_tests']}"
        )
        report.append(f"**Failed Tests**: {summary['test_execution']['failed_tests']}")
        report.append(f"**Success Rate**: {summary['test_execution']['success_rate']}")
        report.append("")

        # Category Breakdown
        if summary["category_breakdown"]:
            report.append("### Performance by Category")
            report.append("")
            report.append(
                "| Category | Count | Avg Relevance | Avg Accuracy | Avg Completeness | Avg Helpfulness | Avg Overall |"
            )
            report.append(
                "|----------|-------|---------------|--------------|------------------|-----------------|-------------|"
            )

            for category, scores in summary["category_breakdown"].items():
                report.append(
                    f"| {category.replace('_', ' ').title()} | {scores['count']} | "
                    f"{scores['average_relevance']:.2f} | {scores['average_accuracy']:.2f} | "
                    f"{scores['average_completeness']:.2f} | {scores['average_helpfulness']:.2f} | "
                    f"{scores['average_overall']:.2f} |"
                )
            report.append("")

        # Difficulty Breakdown
        if summary["difficulty_breakdown"]:
            report.append("### Performance by Difficulty")
            report.append("")
            report.append(
                "| Difficulty | Count | Avg Relevance | Avg Accuracy | Avg Completeness | Avg Helpfulness | Avg Overall |"
            )
            report.append(
                "|------------|-------|---------------|--------------|------------------|-----------------|-------------|"
            )

            for difficulty, scores in summary["difficulty_breakdown"].items():
                report.append(
                    f"| {difficulty.title()} | {scores['count']} | "
                    f"{scores['average_relevance']:.2f} | {scores['average_accuracy']:.2f} | "
                    f"{scores['average_completeness']:.2f} | {scores['average_helpfulness']:.2f} | "
                    f"{scores['average_overall']:.2f} |"
                )
            report.append("")

        # Score Distributions
        report.append("## 📈 Score Distributions")
        report.append("")

        for metric, distribution in summary["score_distributions"].items():
            report.append(f"### {metric.title()} Score Distribution")
            report.append("")

            # Create bar chart representation
            total = sum(distribution.values())
            if total > 0:
                for score in range(1, 6):
                    count = distribution.get(score, 0)
                    percentage = (count / total) * 100
                    bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
                    report.append(f"**{score}/5**: {bar} {count} ({percentage:.1f}%)")
            report.append("")

        # Improvement Areas
        improvement_areas = self.metrics.get_improvement_areas()
        if improvement_areas:
            report.append("## 🚨 Areas for Improvement")
            report.append("")
            for area in improvement_areas:
                report.append(f"- {area}")
            report.append("")

        # Detailed Test Results
        report.append("## 📋 Detailed Test Results")
        report.append("")

        # Group by category
        categories = {}
        for result in self.test_results:
            category = result.test_case.category.value
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        for category, results in categories.items():
            report.append(f"### {category.replace('_', ' ').title()} Tests")
            report.append("")

            report.append(
                "| Test ID | Query | Difficulty | Overall | Response Time | Status |"
            )
            report.append(
                "|---------|-------|------------|---------|---------------|--------|"
            )

            for result in results:
                query_preview = (
                    result.test_case.query[:40] + "..."
                    if len(result.test_case.query) > 40
                    else result.test_case.query
                )
                difficulty = result.test_case.difficulty.value
                overall = result.overall_score or "N/A"
                response_time = f"{result.response_time_ms:.0f}ms"
                status = "✅" if result.is_successful() else "❌"

                report.append(
                    f"| {result.test_case.id} | {query_preview} | {difficulty} | {overall} | {response_time} | {status} |"
                )

            report.append("")

        # Failed Tests Analysis
        failed_results = [r for r in self.test_results if not r.is_successful()]
        if failed_results:
            report.append("## ❌ Failed Tests Analysis")
            report.append("")

            for result in failed_results:
                report.append(f"### {result.test_case.id}")
                report.append(f"**Query**: {result.test_case.query}")
                report.append(f"**Errors**: {'; '.join(result.errors)}")
                report.append("")

        # Recommendations
        report.append("## 💡 Recommendations")
        report.append("")

        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"- {rec}")

        report.append("")

        # Footer
        report.append("---")
        report.append(
            f"*Report generated by PyRAG Evaluation System on {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}*"
        )

        return "\n".join(report)

    def _get_grade_emoji(self, score: float) -> str:
        """Get emoji representation of a score."""
        if score >= 4.5:
            return "🟢 A+"
        elif score >= 4.0:
            return "🟢 A"
        elif score >= 3.5:
            return "🟡 B+"
        elif score >= 3.0:
            return "🟡 B"
        elif score >= 2.5:
            return "🟠 C+"
        elif score >= 2.0:
            return "🟠 C"
        elif score >= 1.5:
            return "🔴 D+"
        elif score >= 1.0:
            return "🔴 D"
        else:
            return "🔴 F"

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        summary = self.metrics.get_summary()

        # Quality-based recommendations
        if summary["quality_scores"]["relevance"] < 3.0:
            recommendations.append(
                "Improve query understanding and retrieval relevance"
            )

        if summary["quality_scores"]["accuracy"] < 3.0:
            recommendations.append("Enhance fact-checking and information validation")

        if summary["quality_scores"]["completeness"] < 3.0:
            recommendations.append(
                "Expand retrieval to include more comprehensive information"
            )

        if summary["quality_scores"]["helpfulness"] < 3.0:
            recommendations.append("Improve response formatting and user experience")

        # Performance-based recommendations
        if summary["performance"]["average_response_time_ms"] > 1000:
            recommendations.append(
                "Optimize search algorithms and reduce response time"
            )

        if summary["test_execution"]["success_rate"] < 0.8:
            recommendations.append("Investigate and fix system failures and errors")

        # Category-based recommendations
        for category, scores in summary["category_breakdown"].items():
            if scores["average_overall"] < 3.0:
                recommendations.append(
                    f"Focus on improving {category.replace('_', ' ')} performance"
                )

        # Difficulty-based recommendations
        for difficulty, scores in summary["difficulty_breakdown"].items():
            if scores["average_overall"] < 3.0:
                recommendations.append(
                    f"Enhance handling of {difficulty} complexity queries"
                )

        if not recommendations:
            recommendations.append("System is performing well across all metrics")

        return recommendations

    def generate_json_report(self) -> str:
        """Generate a JSON report with all evaluation data."""
        report_data = {
            "metadata": {
                "generated_at": self.report_timestamp.isoformat(),
                "test_suite": {
                    "name": self.test_suite.name,
                    "version": self.test_suite.version,
                    "description": self.test_suite.description,
                    "total_test_cases": len(self.test_suite.test_cases),
                },
            },
            "summary": self.metrics.get_summary(),
            "grade": self.metrics.get_grade(),
            "improvement_areas": self.metrics.get_improvement_areas(),
            "recommendations": self._generate_recommendations(),
            "detailed_results": [
                {
                    "test_id": result.test_case.id,
                    "query": result.test_case.query,
                    "category": result.test_case.category.value,
                    "difficulty": result.test_case.difficulty.value,
                    "expected_answer": result.test_case.expected_answer,
                    "rag_response": result.rag_response,
                    "scores": {
                        "relevance": result.relevance_score,
                        "accuracy": result.accuracy_score,
                        "completeness": result.completeness_score,
                        "helpfulness": result.helpfulness_score,
                        "overall": result.overall_score,
                    },
                    "performance": {
                        "response_time_ms": result.response_time_ms,
                        "token_usage": result.token_usage,
                    },
                    "status": "success" if result.is_successful() else "failed",
                    "errors": result.errors,
                    "judge_feedback": result.judge_feedback,
                }
                for result in self.test_results
            ],
        }

        return json.dumps(report_data, indent=2)

    def generate_html_report(self) -> str:
        """Generate an HTML report for web viewing."""
        # This would generate a full HTML report with charts and styling
        # For now, return a simple HTML version of the markdown
        markdown_content = self.generate_markdown_report()

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .grade-a {{ color: #28a745; }}
        .grade-b {{ color: #ffc107; }}
        .grade-c {{ color: #fd7e14; }}
        .grade-d {{ color: #dc3545; }}
        .grade-f {{ color: #dc3545; }}
        .metric-score {{ font-weight: bold; }}
        .recommendation {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }}
    </style>
</head>
<body>
    <div class="container">
        {markdown_content.replace('**', '<strong>').replace('**', '</strong>')}
    </div>
</body>
</html>
        """

        return html

    def save_report(self, filepath: str, format: str = "markdown") -> None:
        """Save the report to a file."""
        if format.lower() == "markdown":
            content = self.generate_markdown_report()
            extension = ".md"
        elif format.lower() == "json":
            content = self.generate_json_report()
            extension = ".json"
        elif format.lower() == "html":
            content = self.generate_html_report()
            extension = ".html"
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Ensure filepath has correct extension
        if not filepath.endswith(extension):
            filepath += extension

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Report saved to: {filepath}")

    def print_summary(self) -> None:
        """Print a summary to the console."""
        summary = self.metrics.get_summary()
        grade = self.metrics.get_grade()

        print("\n" + "=" * 60)
        print("🎯 RAG SYSTEM EVALUATION SUMMARY")
        print("=" * 60)
        print(f"📊 Overall Grade: {grade}")
        print(f"📈 Overall Score: {summary['quality_scores']['overall']}/5.0")
        print(f"✅ Success Rate: {summary['test_execution']['success_rate']}")
        print(
            f"⏱️  Response Time: {summary['performance']['average_response_time_ms']:.0f}ms"
        )
        print(f"🧪 Total Tests: {summary['test_execution']['total_tests']}")
        print("=" * 60)

        print("\n📋 Quality Scores:")
        for metric, score in summary["quality_scores"].items():
            grade_emoji = self._get_grade_emoji(score)
            print(f"   {metric.title()}: {score}/5.0 {grade_emoji}")

        print("\n🚨 Areas for Improvement:")
        improvement_areas = self.metrics.get_improvement_areas()
        if improvement_areas:
            for area in improvement_areas:
                print(f"   • {area}")
        else:
            print("   • System performing well across all metrics")

        print("\n" + "=" * 60)
