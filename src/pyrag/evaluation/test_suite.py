"""Test suite models for RAG evaluation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class QueryCategory(Enum):
    """Categories of test queries."""
    API_REFERENCE = "api_reference"
    EXAMPLES = "examples"
    TUTORIALS = "tutorials"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    BEST_PRACTICES = "best_practices"
    MIGRATION = "migration"


class DifficultyLevel(Enum):
    """Difficulty levels for test queries."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class TestCase:
    """A single test case for RAG evaluation."""
    
    # Basic identification
    id: str
    query: str
    category: QueryCategory
    difficulty: DifficultyLevel
    
    # Ground truth and context
    expected_answer: str
    expected_libraries: List[str] = field(default_factory=list)
    expected_content_types: List[str] = field(default_factory=list)
    
    # Additional context
    user_context: Optional[str] = None
    code_context: Optional[str] = None
    version_constraints: Optional[str] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate test case data."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        if not self.expected_answer.strip():
            raise ValueError("Expected answer cannot be empty")


@dataclass
class TestResult:
    """Result of a single test case execution."""
    
    # Test identification
    test_case: TestCase
    execution_id: str
    
    # RAG system response
    rag_response: str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    search_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    response_time_ms: float = 0.0
    token_usage: Optional[Dict[str, int]] = None
    
    # Quality scores (filled by LLM judge)
    relevance_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    completeness_score: Optional[float] = None
    helpfulness_score: Optional[float] = None
    overall_score: Optional[float] = None
    
    # Judge feedback
    judge_feedback: Optional[str] = None
    score_breakdown: Optional[Dict[str, Any]] = None
    
    # Execution metadata
    executed_at: datetime = field(default_factory=datetime.utcnow)
    errors: List[str] = field(default_factory=list)
    
    def is_successful(self) -> bool:
        """Check if the test execution was successful."""
        return len(self.errors) == 0 and self.overall_score is not None
    
    def get_score_summary(self) -> Dict[str, float]:
        """Get summary of all quality scores."""
        return {
            "relevance": self.relevance_score or 0.0,
            "accuracy": self.accuracy_score or 0.0,
            "completeness": self.completeness_score or 0.0,
            "helpfulness": self.helpfulness_score or 0.0,
            "overall": self.overall_score or 0.0,
        }


@dataclass
class TestSuite:
    """A collection of test cases for systematic evaluation."""
    
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    
    # Suite metadata
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    
    # Configuration
    default_library: Optional[str] = None
    default_content_type: Optional[str] = None
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
    
    def get_test_cases_by_category(self, category: QueryCategory) -> List[TestCase]:
        """Get test cases filtered by category."""
        return [tc for tc in self.test_cases if tc.category == category]
    
    def get_test_cases_by_difficulty(self, difficulty: DifficultyLevel) -> List[TestCase]:
        """Get test cases filtered by difficulty."""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]
    
    def get_test_case_by_id(self, test_id: str) -> Optional[TestCase]:
        """Get a test case by its ID."""
        for tc in self.test_cases:
            if tc.id == test_id:
                return tc
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the test suite."""
        total_cases = len(self.test_cases)
        
        category_counts = {}
        difficulty_counts = {}
        
        for tc in self.test_cases:
            category_counts[tc.category.value] = category_counts.get(tc.category.value, 0) + 1
            difficulty_counts[tc.difficulty.value] = difficulty_counts.get(tc.difficulty.value, 0) + 1
        
        return {
            "total_test_cases": total_cases,
            "category_distribution": category_counts,
            "difficulty_distribution": difficulty_counts,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }
