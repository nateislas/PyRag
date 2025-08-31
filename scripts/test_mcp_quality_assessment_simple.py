#!/usr/bin/env python3
"""
PyRAG Quality Assessment System (Simplified)

This script tests PyRAG MCP tools with various queries and provides
quantitative metrics without requiring external LLM APIs.
"""

import asyncio
import sys
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.core import PyRAG
from pyrag.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TestQuery:
    """Represents a test query with expected outcomes."""
    query: str
    library: str
    content_type: Optional[str] = None
    description: str = ""
    expected_keywords: List[str] = None
    expected_concepts: List[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"


@dataclass
class QualityAssessment:
    """Automated assessment of response quality."""
    relevance_score: float  # 0-10
    accuracy_score: float   # 0-10
    completeness_score: float  # 0-10
    helpfulness_score: float   # 0-10
    overall_score: float    # 0-10
    reasoning: str
    suggestions: List[str]
    keywords_found: List[str]
    concepts_addressed: List[str]


@dataclass
class TestResult:
    """Result of a single test query."""
    query: TestQuery
    response: str
    response_length: int
    response_time: float
    success: bool
    error: Optional[str] = None
    assessment: Optional[QualityAssessment] = None


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    test_results: List[TestResult]
    overall_metrics: Dict[str, float]
    tool_performance: Dict[str, Dict[str, float]]
    recommendations: List[str]
    timestamp: datetime


class PyRAGQualityAssessor:
    """Assesses PyRAG response quality using automated metrics."""
    
    def __init__(self, pyrag: PyRAG):
        self.pyrag = pyrag
        self.test_results: List[TestResult] = []
    
    def assess_response_quality(
        self, 
        query: TestQuery, 
        response: str
    ) -> QualityAssessment:
        """Automatically assess the quality of a PyRAG response."""
        
        # Initialize scores
        relevance_score = 0.0
        accuracy_score = 0.0
        completeness_score = 0.0
        helpfulness_score = 0.0
        
        # Extract keywords and concepts
        keywords_found = []
        concepts_addressed = []
        suggestions = []
        reasoning_parts = []
        
        # Check if response is empty
        if not response or response.strip() == "No results found.":
            reasoning_parts.append("No results returned")
            suggestions.append("Improve search algorithms or add more documentation")
            return QualityAssessment(
                relevance_score=0.0,
                accuracy_score=0.0,
                completeness_score=0.0,
                helpfulness_score=0.0,
                overall_score=0.0,
                reasoning="No results found",
                suggestions=suggestions,
                keywords_found=[],
                concepts_addressed=[]
            )
        
        # Check response length
        if len(response) < 50:
            completeness_score += 2.0
            reasoning_parts.append("Very short response")
        elif len(response) < 200:
            completeness_score += 5.0
            reasoning_parts.append("Short response")
        elif len(response) < 500:
            completeness_score += 7.0
            reasoning_parts.append("Moderate response length")
        else:
            completeness_score += 9.0
            reasoning_parts.append("Good response length")
        
        # Check for expected keywords
        if query.expected_keywords:
            found_keywords = []
            for keyword in query.expected_keywords:
                if keyword.lower() in response.lower():
                    found_keywords.append(keyword)
                    relevance_score += 2.0
            
            keywords_found = found_keywords
            keyword_coverage = len(found_keywords) / len(query.expected_keywords)
            relevance_score += keyword_coverage * 5.0
            
            if keyword_coverage > 0.8:
                reasoning_parts.append(f"Excellent keyword coverage ({len(found_keywords)}/{len(query.expected_keywords)})")
            elif keyword_coverage > 0.5:
                reasoning_parts.append(f"Good keyword coverage ({len(found_keywords)}/{len(query.expected_keywords)})")
            else:
                reasoning_parts.append(f"Poor keyword coverage ({len(found_keywords)}/{len(query.expected_keywords)})")
                suggestions.append("Improve keyword matching in search")
        
        # Check for expected concepts
        if query.expected_concepts:
            found_concepts = []
            for concept in query.expected_concepts:
                if concept.lower() in response.lower():
                    found_concepts.append(concept)
                    helpfulness_score += 1.5
            
            concepts_addressed = found_concepts
            concept_coverage = len(found_concepts) / len(query.expected_concepts)
            helpfulness_score += concept_coverage * 3.0
            
            if concept_coverage > 0.8:
                reasoning_parts.append(f"Excellent concept coverage ({len(found_concepts)}/{len(query.expected_concepts)})")
            elif concept_coverage > 0.5:
                reasoning_parts.append(f"Good concept coverage ({len(found_concepts)}/{len(query.expected_concepts)})")
            else:
                reasoning_parts.append(f"Poor concept coverage ({len(found_concepts)}/{len(query.expected_concepts)})")
        
        # Check for library-specific content
        if query.library.lower() in response.lower():
            relevance_score += 2.0
            accuracy_score += 1.0
            reasoning_parts.append(f"Contains {query.library} content")
        else:
            reasoning_parts.append(f"Missing {query.library} specific content")
            suggestions.append(f"Ensure {query.library} documentation is properly indexed")
        
        # Check for code examples
        if re.search(r'```python|def |class |import |from ', response, re.IGNORECASE):
            helpfulness_score += 2.0
            reasoning_parts.append("Contains code examples")
        else:
            reasoning_parts.append("No code examples found")
            suggestions.append("Include more code examples in responses")
        
        # Check for API references
        if re.search(r'\.|\(|\)|\[|\]', response):
            accuracy_score += 1.0
            reasoning_parts.append("Contains API references")
        
        # Check for structured content
        if re.search(r'Parameters:|Returns:|Examples:|Note:', response, re.IGNORECASE):
            completeness_score += 1.0
            reasoning_parts.append("Contains structured documentation")
        
        # Cap scores at 10
        relevance_score = min(relevance_score, 10.0)
        accuracy_score = min(accuracy_score, 10.0)
        completeness_score = min(completeness_score, 10.0)
        helpfulness_score = min(helpfulness_score, 10.0)
        
        # Calculate overall score
        overall_score = (relevance_score + accuracy_score + completeness_score + helpfulness_score) / 4.0
        
        # Generate reasoning
        reasoning = "; ".join(reasoning_parts)
        
        # Add general suggestions
        if overall_score < 5.0:
            suggestions.append("Significant improvements needed in search relevance")
        elif overall_score < 7.0:
            suggestions.append("Moderate improvements needed")
        else:
            suggestions.append("Good performance, minor optimizations possible")
        
        return QualityAssessment(
            relevance_score=relevance_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            helpfulness_score=helpfulness_score,
            overall_score=overall_score,
            reasoning=reasoning,
            suggestions=suggestions,
            keywords_found=keywords_found,
            concepts_addressed=concepts_addressed
        )
    
    async def test_search_functionality(self, test_queries: List[TestQuery]) -> List[TestResult]:
        """Test search_python_docs functionality with quality assessment."""
        
        print(f"\n🔍 Testing search_python_docs with {len(test_queries)} queries...")
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}/{len(test_queries)}: {query.description}")
            print(f"   Query: '{query.query}'")
            
            start_time = time.time()
            
            try:
                # Map content_type to internal format
                mapped_content_type = None
                if query.content_type:
                    if query.content_type in ["examples", "tutorials"]:
                        mapped_content_type = "examples"
                    elif query.content_type in ["reference", "api_reference"]:
                        mapped_content_type = "api_reference"
                    elif query.content_type == "overview":
                        mapped_content_type = "overview"
                
                # Get PyRAG response
                search_results = await self.pyrag.search_documentation(
                    query=query.query,
                    library=query.library,
                    content_type=mapped_content_type,
                    max_results=5,
                )
                
                response_time = time.time() - start_time
                
                # Format response
                if search_results:
                    response = "\n\n".join([
                        f"Result {j+1}:\n{result['content'][:500]}..."
                        for j, result in enumerate(search_results[:3])
                    ])
                else:
                    response = "No results found."
                
                # Assess quality
                assessment = self.assess_response_quality(query, response)
                
                result = TestResult(
                    query=query,
                    response=response,
                    response_length=len(response),
                    response_time=response_time,
                    success=True,
                    assessment=assessment
                )
                
                print(f"   ✅ Success in {response_time:.2f}s")
                print(f"   📊 Quality Score: {assessment.overall_score:.1f}/10")
                print(f"   🎯 Relevance: {assessment.relevance_score:.1f}/10")
                print(f"   📚 Keywords: {', '.join(assessment.keywords_found)}")
                
            except Exception as e:
                response_time = time.time() - start_time
                result = TestResult(
                    query=query,
                    response="",
                    response_length=0,
                    response_time=response_time,
                    success=False,
                    error=str(e)
                )
                print(f"   ❌ Failed: {e}")
            
            results.append(result)
        
        return results
    
    async def test_api_reference_functionality(self, test_apis: List[Dict]) -> List[TestResult]:
        """Test get_api_reference functionality with quality assessment."""
        
        print(f"\n📚 Testing get_api_reference with {len(test_apis)} API queries...")
        results = []
        
        for i, api_test in enumerate(test_apis, 1):
            query = TestQuery(
                query=f"API reference for {api_test['api_path']}",
                library=api_test['library'],
                description=f"API reference: {api_test['api_path']}",
                expected_keywords=[api_test['api_path'].split('.')[-1], api_test['library']]
            )
            
            print(f"\n📝 Test {i}/{len(test_apis)}: {query.description}")
            
            start_time = time.time()
            
            try:
                # Get PyRAG response
                response = await self.pyrag.get_api_reference(
                    library=api_test['library'],
                    api_path=api_test['api_path'],
                    include_examples=api_test.get('include_examples', True)
                )
                
                response_time = time.time() - start_time
                
                if not response:
                    response = "No API reference found."
                
                # Assess quality
                assessment = self.assess_response_quality(query, response)
                
                result = TestResult(
                    query=query,
                    response=response,
                    response_length=len(response),
                    response_time=response_time,
                    success=True,
                    assessment=assessment
                )
                
                print(f"   ✅ Success in {response_time:.2f}s")
                print(f"   📊 Quality Score: {assessment.overall_score:.1f}/10")
                print(f"   📚 Keywords: {', '.join(assessment.keywords_found)}")
                
            except Exception as e:
                response_time = time.time() - start_time
                result = TestResult(
                    query=query,
                    response="",
                    response_length=0,
                    response_time=response_time,
                    success=False,
                    error=str(e)
                )
                print(f"   ❌ Failed: {e}")
            
            results.append(result)
        
        return results
    
    def calculate_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall quality metrics."""
        
        successful_results = [r for r in self.test_results if r.success and r.assessment]
        
        if not successful_results:
            return {
                "success_rate": 0.0,
                "average_quality": 0.0,
                "average_response_time": 0.0,
                "total_tests": len(self.test_results)
            }
        
        quality_scores = [r.assessment.overall_score for r in successful_results]
        response_times = [r.response_time for r in successful_results]
        
        return {
            "success_rate": len(successful_results) / len(self.test_results),
            "average_quality": sum(quality_scores) / len(quality_scores),
            "average_response_time": sum(response_times) / len(response_times),
            "total_tests": len(self.test_results),
            "successful_tests": len(successful_results),
            "max_quality": max(quality_scores),
            "min_quality": min(quality_scores),
            "quality_std": (sum((x - sum(quality_scores)/len(quality_scores))**2 for x in quality_scores) / len(quality_scores))**0.5
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        
        recommendations = []
        successful_results = [r for r in self.test_results if r.success and r.assessment]
        
        if not successful_results:
            recommendations.append("❌ No successful tests - investigate system issues")
            return recommendations
        
        # Analyze quality scores
        quality_scores = [r.assessment.overall_score for r in successful_results]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        if avg_quality >= 8.0:
            recommendations.append("✅ Excellent quality - system is performing very well")
        elif avg_quality >= 6.0:
            recommendations.append("✅ Good quality - system is performing well")
        elif avg_quality >= 4.0:
            recommendations.append("⚠️  Moderate quality - consider improvements")
        else:
            recommendations.append("❌ Poor quality - significant improvements needed")
        
        # Analyze response times
        response_times = [r.response_time for r in successful_results]
        avg_response_time = sum(response_times) / len(response_times)
        
        if avg_response_time < 0.5:
            recommendations.append("🚀 Excellent response time")
        elif avg_response_time < 1.0:
            recommendations.append("⚡ Good response time")
        elif avg_response_time < 2.0:
            recommendations.append("⚠️  Moderate response time - consider optimization")
        else:
            recommendations.append("🐌 Slow response time - optimization needed")
        
        # Analyze specific aspects
        relevance_scores = [r.assessment.relevance_score for r in successful_results]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        if avg_relevance < 6.0:
            recommendations.append("🎯 Improve query relevance - better search algorithms needed")
        
        completeness_scores = [r.assessment.completeness_score for r in successful_results]
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        
        if avg_completeness < 6.0:
            recommendations.append("📚 Improve content completeness - more documentation needed")
        
        # Check for empty responses
        empty_responses = [r for r in successful_results if r.response.strip() == "No results found."]
        if empty_responses:
            recommendations.append(f"📝 {len(empty_responses)} queries returned no results - improve documentation coverage")
        
        return recommendations


# Predefined test queries for pandas
PANDAS_TEST_QUERIES = [
    TestQuery(
        query="how to create a pandas DataFrame",
        library="pandas",
        content_type="examples",
        description="DataFrame creation",
        expected_keywords=["DataFrame", "create", "pandas"],
        expected_concepts=["data structure", "initialization"],
        difficulty="easy",
        category="basics"
    ),
    TestQuery(
        query="pandas DataFrame operations and methods",
        library="pandas",
        content_type="reference",
        description="DataFrame operations",
        expected_keywords=["DataFrame", "operations", "methods"],
        expected_concepts=["data manipulation", "API reference"],
        difficulty="medium",
        category="operations"
    ),
    TestQuery(
        query="how to read CSV files with pandas",
        library="pandas",
        content_type="tutorials",
        description="CSV file reading",
        expected_keywords=["read_csv", "CSV", "file"],
        expected_concepts=["file I/O", "data import"],
        difficulty="easy",
        category="file_io"
    ),
    TestQuery(
        query="pandas data manipulation and filtering",
        library="pandas",
        content_type="all",
        description="Data manipulation",
        expected_keywords=["filter", "manipulation", "data"],
        expected_concepts=["data processing", "filtering"],
        difficulty="medium",
        category="manipulation"
    ),
    TestQuery(
        query="pandas groupby operations and aggregations",
        library="pandas",
        content_type="examples",
        description="Groupby operations",
        expected_keywords=["groupby", "aggregation", "group"],
        expected_concepts=["data grouping", "statistics"],
        difficulty="medium",
        category="aggregation"
    ),
    TestQuery(
        query="how to handle missing values in pandas DataFrame",
        library="pandas",
        content_type="tutorials",
        description="Missing data handling",
        expected_keywords=["missing", "NaN", "dropna", "fillna"],
        expected_concepts=["data cleaning", "missing data"],
        difficulty="medium",
        category="data_cleaning"
    ),
    TestQuery(
        query="pandas time series operations and datetime handling",
        library="pandas",
        content_type="reference",
        description="Time series operations",
        expected_keywords=["time", "datetime", "series"],
        expected_concepts=["time series", "date handling"],
        difficulty="hard",
        category="time_series"
    ),
    TestQuery(
        query="pandas performance optimization and memory usage",
        library="pandas",
        content_type="all",
        description="Performance optimization",
        expected_keywords=["performance", "memory", "optimization"],
        expected_concepts=["performance", "memory management"],
        difficulty="hard",
        category="performance"
    ),
    TestQuery(
        query="pandas merge and join operations",
        library="pandas",
        content_type="reference",
        description="Data merging",
        expected_keywords=["merge", "join", "combine"],
        expected_concepts=["data combination", "relational operations"],
        difficulty="medium",
        category="merging"
    ),
    TestQuery(
        query="pandas plotting and visualization with matplotlib",
        library="pandas",
        content_type="examples",
        description="Data visualization",
        expected_keywords=["plot", "visualization", "matplotlib"],
        expected_concepts=["data visualization", "plotting"],
        difficulty="medium",
        category="visualization"
    )
]

# API reference tests
PANDAS_API_TESTS = [
    {
        "library": "pandas",
        "api_path": "pandas.DataFrame",
        "include_examples": True,
        "description": "DataFrame class"
    },
    {
        "library": "pandas",
        "api_path": "pandas.read_csv",
        "include_examples": True,
        "description": "read_csv function"
    },
    {
        "library": "pandas",
        "api_path": "pandas.DataFrame.groupby",
        "include_examples": False,
        "description": "DataFrame groupby method"
    },
    {
        "library": "pandas",
        "api_path": "pandas.DataFrame.merge",
        "include_examples": True,
        "description": "DataFrame merge method"
    },
    {
        "library": "pandas",
        "api_path": "pandas.Series",
        "include_examples": True,
        "description": "Series class"
    }
]


async def main():
    """Main quality assessment function."""
    
    print("🚀 PyRAG Quality Assessment System (Simplified)")
    print("=" * 60)
    print("Testing PyRAG with automated quality evaluation")
    print("=" * 60)
    
    # Initialize PyRAG
    try:
        pyrag = PyRAG()
        print("✅ PyRAG initialized successfully")
    except Exception as e:
        print(f"❌ PyRAG initialization error: {e}")
        return
    
    # Create quality assessor
    assessor = PyRAGQualityAssessor(pyrag)
    
    # Run search functionality tests
    print(f"\n🔍 Running search functionality tests...")
    search_results = await assessor.test_search_functionality(PANDAS_TEST_QUERIES)
    assessor.test_results.extend(search_results)
    
    # Run API reference tests
    print(f"\n📚 Running API reference tests...")
    api_results = await assessor.test_api_reference_functionality(PANDAS_API_TESTS)
    assessor.test_results.extend(api_results)
    
    # Calculate metrics
    print(f"\n📊 Calculating quality metrics...")
    overall_metrics = assessor.calculate_overall_metrics()
    
    # Generate recommendations
    recommendations = assessor.generate_recommendations()
    
    # Create quality report
    report = QualityReport(
        test_results=assessor.test_results,
        overall_metrics=overall_metrics,
        tool_performance={},  # Could be expanded
        recommendations=recommendations,
        timestamp=datetime.now()
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("🎯 QUALITY ASSESSMENT RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"   Total Tests: {overall_metrics['total_tests']}")
    print(f"   Successful Tests: {overall_metrics['successful_tests']}")
    print(f"   Success Rate: {overall_metrics['success_rate']:.1%}")
    print(f"   Average Quality Score: {overall_metrics['average_quality']:.1f}/10")
    print(f"   Average Response Time: {overall_metrics['average_response_time']:.2f}s")
    print(f"   Quality Range: {overall_metrics['min_quality']:.1f} - {overall_metrics['max_quality']:.1f}")
    print(f"   Quality Standard Deviation: {overall_metrics['quality_std']:.2f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"   • {rec}")
    
    # Show detailed results for top and bottom performers
    successful_results = [r for r in assessor.test_results if r.success and r.assessment]
    if successful_results:
        # Sort by quality score
        successful_results.sort(key=lambda x: x.assessment.overall_score, reverse=True)
        
        print(f"\n🏆 TOP 3 PERFORMERS:")
        for i, result in enumerate(successful_results[:3], 1):
            print(f"   {i}. {result.query.description}")
            print(f"      Query: {result.query.query}")
            print(f"      Quality: {result.assessment.overall_score:.1f}/10")
            print(f"      Response Time: {result.response_time:.2f}s")
            print(f"      Keywords: {', '.join(result.assessment.keywords_found)}")
        
        print(f"\n📉 BOTTOM 3 PERFORMERS:")
        for i, result in enumerate(successful_results[-3:], 1):
            print(f"   {i}. {result.query.description}")
            print(f"      Query: {result.query.query}")
            print(f"      Quality: {result.assessment.overall_score:.1f}/10")
            print(f"      Response Time: {result.response_time:.2f}s")
            print(f"      Keywords: {', '.join(result.assessment.keywords_found)}")
    
    # Save detailed report
    report_path = Path("reports") / f"quality_assessment_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    print(f"\n🎉 Quality assessment completed!")
    print(f"📝 PyRAG quality score: {overall_metrics['average_quality']:.1f}/10")


if __name__ == "__main__":
    asyncio.run(main())
