#!/usr/bin/env python3
"""Test script for the RAG evaluation pipeline."""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.config import get_config
from pyrag.vector_store import VectorStore
from pyrag.embeddings import EmbeddingService
from pyrag.search import EnhancedSearchEngine
from pyrag.llm.client import LLMClient
from pyrag.evaluation import (
    RAGEvaluator, LLMJudge, TestSuite, TestCase, 
    QueryCategory, DifficultyLevel, EvaluationReporter
)


def create_sample_test_suite() -> TestSuite:
    """Create a sample test suite with various query types and difficulties."""
    
    test_suite = TestSuite(
        name="PyRAG Quality Evaluation Suite",
        description="Comprehensive test suite for evaluating RAG system quality across different query types and difficulties",
        version="1.0.0",
        tags=["python", "documentation", "rag", "quality"]
    )
    
    # API Reference Tests
    test_suite.add_test_case(TestCase(
        id="api_ref_001",
        query="How do I make an HTTP GET request with the requests library?",
        category=QueryCategory.API_REFERENCE,
        difficulty=DifficultyLevel.SIMPLE,
        expected_answer="Use requests.get(url) to make an HTTP GET request. The basic syntax is: response = requests.get('https://api.example.com/data')",
        expected_libraries=["requests"],
        expected_content_types=["api_reference", "examples"],
        tags=["http", "requests", "basic"]
    ))
    
    test_suite.add_test_case(TestCase(
        id="api_ref_002",
        query="What are the parameters for pandas.DataFrame.merge?",
        category=QueryCategory.API_REFERENCE,
        difficulty=DifficultyLevel.MODERATE,
        expected_answer="pandas.DataFrame.merge() takes parameters like 'right', 'how', 'on', 'left_on', 'right_on', 'left_index', 'right_index', 'sort', 'suffixes', 'copy', 'indicator', 'validate'",
        expected_libraries=["pandas"],
        expected_content_types=["api_reference"],
        tags=["pandas", "dataframe", "merge", "parameters"]
    ))
    
    test_suite.add_test_case(TestCase(
        id="api_ref_003",
        query="How do I implement a custom FastAPI dependency with error handling?",
        category=QueryCategory.API_REFERENCE,
        difficulty=DifficultyLevel.COMPLEX,
        expected_answer="Create a function that returns the dependency value, use Depends() decorator, and implement proper error handling with HTTPException or custom exception classes",
        expected_libraries=["fastapi"],
        expected_content_types=["api_reference", "examples", "tutorials"],
        tags=["fastapi", "dependencies", "error_handling", "advanced"]
    ))
    
    # Examples Tests
    test_suite.add_test_case(TestCase(
        id="examples_001",
        query="Show me an example of reading a CSV file with pandas",
        category=QueryCategory.EXAMPLES,
        difficulty=DifficultyLevel.SIMPLE,
        expected_answer="Use pd.read_csv('filename.csv') to read a CSV file. Example: df = pd.read_csv('data.csv')",
        expected_libraries=["pandas"],
        expected_content_types=["examples"],
        tags=["pandas", "csv", "reading", "basic"]
    ))
    
    test_suite.add_test_case(TestCase(
        id="examples_002",
        query="Give me a complete example of a FastAPI endpoint with database integration",
        category=QueryCategory.EXAMPLES,
        difficulty=DifficultyLevel.MODERATE,
        expected_answer="Create an endpoint with proper database session management, error handling, and response models. Include imports, database connection, and proper HTTP status codes",
        expected_libraries=["fastapi", "sqlalchemy"],
        expected_content_types=["examples", "tutorials"],
        tags=["fastapi", "database", "endpoint", "complete"]
    ))
    
    # Tutorial Tests
    test_suite.add_test_case(TestCase(
        id="tutorial_001",
        query="How do I set up a Python virtual environment and install packages?",
        category=QueryCategory.TUTORIALS,
        difficulty=DifficultyLevel.SIMPLE,
        expected_answer="Use 'python -m venv env_name' to create, 'source env_name/bin/activate' (Unix) or 'env_name\\Scripts\\activate' (Windows) to activate, and 'pip install package_name' to install packages",
        expected_libraries=[],
        expected_content_types=["tutorials"],
        tags=["python", "virtual_environment", "packages", "setup"]
    ))
    
    test_suite.add_test_case(TestCase(
        id="tutorial_002",
        query="Walk me through building a REST API with FastAPI from scratch",
        category=QueryCategory.TUTORIALS,
        difficulty=DifficultyLevel.COMPLEX,
        expected_answer="Start with project setup, create FastAPI app, define models with Pydantic, create endpoints, add database integration, implement authentication, add validation, and deploy",
        expected_libraries=["fastapi", "pydantic", "sqlalchemy"],
        expected_content_types=["tutorials", "examples"],
        tags=["fastapi", "rest_api", "tutorial", "complete"]
    ))
    
    # Troubleshooting Tests
    test_suite.add_test_case(TestCase(
        id="troubleshooting_001",
        query="I'm getting 'ModuleNotFoundError: No module named X' - how do I fix this?",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=DifficultyLevel.SIMPLE,
        expected_answer="Check if the module is installed with 'pip list', install it with 'pip install module_name', or check your Python path and virtual environment activation",
        expected_libraries=[],
        expected_content_types=["tutorials", "examples"],
        tags=["python", "modules", "installation", "error_fixing"]
    ))
    
    test_suite.add_test_case(TestCase(
        id="troubleshooting_002",
        query="My FastAPI app is running but I can't access the docs - what's wrong?",
        category=QueryCategory.TROUBLESHOOTING,
        difficulty=DifficultyLevel.MODERATE,
        expected_answer="Check if you're accessing the correct URL (/docs), ensure the app is running, check for CORS issues, verify the docs_url parameter isn't disabled, and check server logs for errors",
        expected_libraries=["fastapi"],
        expected_content_types=["tutorials", "troubleshooting"],
        tags=["fastapi", "documentation", "cors", "debugging"]
    ))
    
    # Comparison Tests
    test_suite.add_test_case(TestCase(
        id="comparison_001",
        query="What's the difference between requests and httpx libraries?",
        category=QueryCategory.COMPARISON,
        difficulty=DifficultyLevel.MODERATE,
        expected_answer="Requests is synchronous, httpx supports both sync and async. Httpx has better HTTP/2 support, modern Python features, and async capabilities. Requests is more mature and widely used",
        expected_libraries=["requests", "httpx"],
        expected_content_types=["tutorials", "examples"],
        tags=["http", "requests", "httpx", "comparison", "async"]
    ))
    
    # Best Practices Tests
    test_suite.add_test_case(TestCase(
        id="best_practices_001",
        query="What are the best practices for handling environment variables in Python?",
        category=QueryCategory.BEST_PRACTICES,
        difficulty=DifficultyLevel.MODERATE,
        expected_answer="Use python-dotenv for .env files, never commit secrets to version control, use environment-specific configs, validate required variables, and use Pydantic for type safety",
        expected_libraries=["python-dotenv", "pydantic"],
        expected_content_types=["tutorials", "best_practices"],
        tags=["python", "environment", "security", "configuration"]
    ))
    
    return test_suite


def progress_callback(message: str, current: int, total: int):
    """Progress callback for evaluation updates."""
    percentage = (current / total) * 100
    print(f"🔄 {message}: {current}/{total} ({percentage:.1f}%)")


def result_callback(test_result):
    """Result callback for individual test results."""
    status = "✅" if test_result.is_successful() else "❌"
    score = test_result.overall_score or "N/A"
    print(f"{status} {test_result.test_case.id}: {score}/5.0 ({test_result.response_time_ms:.0f}ms)")


async def main():
    """Main evaluation function."""
    print("🚀 Starting RAG System Quality Evaluation")
    print("=" * 60)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config = get_config()
        
        # Initialize components
        print("🔧 Initializing components...")
        vector_store = VectorStore()
        embedding_service = EmbeddingService()
        search_engine = EnhancedSearchEngine(vector_store, embedding_service)
        llm_client = LLMClient(config.llm)
        
        # Create test suite
        print("📚 Creating test suite...")
        test_suite = create_sample_test_suite()
        print(f"   Created {len(test_suite.test_cases)} test cases")
        
        # Print test suite statistics
        stats = test_suite.get_statistics()
        print(f"   Categories: {stats['category_distribution']}")
        print(f"   Difficulties: {stats['difficulty_distribution']}")
        
        # Initialize LLM judge
        print("⚖️  Initializing LLM judge...")
        llm_judge = LLMJudge(llm_client)
        
        # Initialize evaluator
        print("🎯 Initializing RAG evaluator...")
        evaluator = RAGEvaluator(
            vector_store=vector_store,
            search_engine=search_engine,
            llm_judge=llm_judge,
            test_suite=test_suite
        )
        
        # Set callbacks
        evaluator.set_progress_callback(progress_callback)
        evaluator.set_result_callback(result_callback)
        
        print("\n" + "=" * 60)
        print("🧪 RUNNING EVALUATION")
        print("=" * 60)
        
        # Run evaluation
        start_time = time.time()
        metrics = await evaluator.run_evaluation(
            evaluation_id="demo_eval_001",
            max_concurrent=3,
            enable_judging=True
        )
        end_time = time.time()
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION COMPLETED")
        print("=" * 60)
        
        # Print summary
        print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        print(f"📈 Overall score: {metrics.average_overall:.2f}/5.0")
        print(f"✅ Success rate: {metrics.success_rate:.1%}")
        print(f"🎯 Grade: {metrics.get_grade()}")
        
        # Create reporter and generate reports
        print("\n📝 Generating reports...")
        reporter = EvaluationReporter(test_suite, metrics, evaluator.test_results)
        
        # Print console summary
        reporter.print_summary()
        
        # Save reports
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"rag_evaluation_report_{timestamp}"
        
        reporter.save_report(f"reports/{base_filename}.md", "markdown")
        reporter.save_report(f"reports/{base_filename}.json", "json")
        reporter.save_report(f"reports/{base_filename}.html", "html")
        
        print(f"\n📁 Reports saved to reports/ directory")
        
        # Export results
        evaluator.save_results(f"reports/{base_filename}_results.csv", "csv")
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Create reports directory if it doesn't exist
    Path("reports").mkdir(exist_ok=True)
    
    # Run evaluation
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
