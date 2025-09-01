# RAG Quality Evaluation Pipeline Guide

## Overview

The PyRAG evaluation pipeline provides a systematic way to assess the quality of your RAG system using LLM judges. This system goes beyond simple test scripts to provide comprehensive, objective scoring of RAG responses across multiple dimensions.

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Test Dataset  │───▶│  RAG System      │───▶│  LLM Judge      │
│   (Queries +    │    │  (Search +       │    │  (Quality       │
│   Ground Truth) │    │   Retrieval)     │    │   Scoring)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Evaluation    │    │   Performance    │    │   Quality       │
│   Metrics       │    │   Monitoring     │    │   Reports       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Features

- **Systematic Testing**: Structured test cases with ground truth
- **LLM-Powered Judging**: Objective quality assessment using AI
- **Multi-Dimensional Scoring**: Relevance, accuracy, completeness, helpfulness
- **Comprehensive Metrics**: Performance, category breakdowns, score distributions
- **Multiple Report Formats**: Markdown, JSON, HTML, CSV
- **Progress Tracking**: Real-time updates and callbacks
- **Batch Processing**: Efficient evaluation of large test suites

## 🚀 Quick Start

### 1. Basic Usage

```python
from pyrag.evaluation import RAGEvaluator, LLMJudge, TestSuite, TestCase
from pyrag.evaluation import QueryCategory, DifficultyLevel

# Create test suite
test_suite = TestSuite(
    name="My RAG Tests",
    description="Testing my RAG system quality"
)

# Add test cases
test_suite.add_test_case(TestCase(
    id="test_001",
    query="How do I make HTTP requests?",
    category=QueryCategory.API_REFERENCE,
    difficulty=DifficultyLevel.SIMPLE,
    expected_answer="Use requests.get(url) for GET requests",
    expected_libraries=["requests"]
))

# Initialize evaluator
evaluator = RAGEvaluator(
    vector_store=vector_store,
    search_engine=search_engine,
    llm_judge=llm_judge,
    test_suite=test_suite
)

# Run evaluation
metrics = await evaluator.run_evaluation()
```

### 2. Run the Demo

```bash
# Activate virtual environment
source venv/bin/activate

# Run the evaluation
python scripts/test_rag_evaluation.py
```

## 📋 Test Case Design

### Test Case Structure

```python
TestCase(
    id="unique_id",                    # Unique identifier
    query="User's question",           # The query to test
    category=QueryCategory.API_REFERENCE,  # Query category
    difficulty=DifficultyLevel.SIMPLE,     # Complexity level
    expected_answer="Expected response",   # Ground truth
    expected_libraries=["lib1", "lib2"],  # Expected libraries
    expected_content_types=["api_ref"],    # Expected content types
    user_context="Optional context",       # User context
    code_context="Optional code context",  # Code context
    version_constraints=">=3.8",          # Version requirements
    tags=["tag1", "tag2"],               # Searchable tags
    description="Optional description"     # Detailed description
)
```

### Query Categories

- **`API_REFERENCE`**: Function/class documentation queries
- **`EXAMPLES`**: Code example requests
- **`TUTORIALS`**: Step-by-step instruction requests
- **`TROUBLESHOOTING`**: Error fixing and debugging queries
- **`COMPARISON`**: Library/approach comparison queries
- **`BEST_PRACTICES`**: Best practice and recommendation queries
- **`MIGRATION`**: Upgrade and migration queries

### Difficulty Levels

- **`SIMPLE`**: Basic, straightforward queries
- **`MODERATE`**: Intermediate complexity queries
- **`COMPLEX`**: Advanced, multi-step queries

## ⚖️ LLM Judge System

### Evaluation Criteria

The LLM judge evaluates responses across 4 dimensions:

1. **Relevance** (25% weight): How well the response addresses the query
2. **Accuracy** (25% weight): Factual correctness of the information
3. **Completeness** (25% weight): How complete the answer is
4. **Helpfulness** (25% weight): How useful the response is to users

### Scoring Scale

- **5/5**: Excellent - Directly answers the question perfectly
- **4/5**: Good - Addresses most aspects of the question
- **3/5**: Fair - Partially answers the question
- **2/5**: Poor - Barely relevant or helpful
- **1/5**: Very Poor - Off-topic or incorrect

### Customizing Judge Criteria

```python
from pyrag.evaluation import JudgeCriteria

# Custom criteria with different weights
custom_criteria = JudgeCriteria(
    relevance_weight=0.3,      # Emphasize relevance
    accuracy_weight=0.3,       # Emphasize accuracy
    completeness_weight=0.2,   # Reduce completeness weight
    helpfulness_weight=0.2     # Reduce helpfulness weight
)

# Initialize judge with custom criteria
llm_judge = LLMJudge(llm_client, criteria=custom_criteria)
```

## 📊 Evaluation Metrics

### Quality Scores

- **Overall Score**: Weighted average of all dimensions
- **Category Breakdown**: Performance by query category
- **Difficulty Breakdown**: Performance by complexity level
- **Score Distributions**: Histogram of scores for each metric

### Performance Metrics

- **Response Time**: Average query processing time
- **Token Usage**: LLM API consumption tracking
- **Success Rate**: Percentage of successful queries
- **Error Analysis**: Detailed failure investigation

### Grade System

- **A+** (4.5-5.0): Exceptional performance
- **A** (4.0-4.4): Excellent performance
- **B+** (3.5-3.9): Good performance
- **B** (3.0-3.4): Above average
- **C+** (2.5-2.9): Average performance
- **C** (2.0-2.4): Below average
- **D+** (1.5-1.9): Poor performance
- **D** (1.0-1.4): Very poor
- **F** (0.0-0.9): Failing

## 📝 Report Generation

### Available Formats

1. **Markdown**: Human-readable reports with tables and formatting
2. **JSON**: Machine-readable data for further analysis
3. **HTML**: Web-friendly reports with styling
4. **CSV**: Spreadsheet-compatible data export

### Report Contents

- **Executive Summary**: High-level performance overview
- **Quality Scores**: Detailed scoring breakdown
- **Performance Metrics**: System performance analysis
- **Category Analysis**: Performance by query type
- **Difficulty Analysis**: Performance by complexity
- **Score Distributions**: Visual score representations
- **Improvement Areas**: Actionable recommendations
- **Detailed Results**: Individual test case results
- **Failed Tests**: Error analysis and debugging

### Generating Reports

```python
from pyrag.evaluation import EvaluationReporter

# Create reporter
reporter = EvaluationReporter(test_suite, metrics, test_results)

# Generate different formats
markdown_report = reporter.generate_markdown_report()
json_report = reporter.generate_json_report()
html_report = reporter.generate_html_report()

# Save reports
reporter.save_report("evaluation_report.md", "markdown")
reporter.save_report("evaluation_report.json", "json")
reporter.save_report("evaluation_report.html", "html")
```

## 🔄 Progress Tracking

### Callbacks

```python
def progress_callback(message: str, current: int, total: int):
    """Track evaluation progress."""
    percentage = (current / total) * 100
    print(f"🔄 {message}: {current}/{total} ({percentage:.1f}%)")

def result_callback(test_result):
    """Handle individual test results."""
    status = "✅" if test_result.is_successful() else "❌"
    score = test_result.overall_score or "N/A"
    print(f"{status} {test_result.test_case.id}: {score}/5.0")

# Set callbacks
evaluator.set_progress_callback(progress_callback)
evaluator.set_result_callback(result_callback)
```

### Real-time Updates

The evaluation pipeline provides real-time updates on:
- Test case execution progress
- Individual test results
- LLM judging progress
- Overall completion status

## 🧪 Testing Strategies

### 1. Comprehensive Coverage

Create test cases that cover:
- All query categories
- All difficulty levels
- Edge cases and error conditions
- Different library combinations
- Various content types

### 2. Realistic Queries

Design test cases that mirror real user behavior:
- Natural language questions
- Specific technical queries
- Ambiguous or complex requests
- Context-dependent questions

### 3. Ground Truth Quality

Ensure your expected answers are:
- Accurate and up-to-date
- Comprehensive and detailed
- Realistic for the query complexity
- Consistent across similar queries

### 4. Systematic Variation

Vary test cases systematically:
- Different query formulations
- Various complexity levels
- Multiple library combinations
- Different content type preferences

## 📈 Continuous Improvement

### Baseline Comparison

```python
# Compare with previous evaluation
comparison = current_metrics.compare_with_baseline(baseline_metrics)

print("Improvement Analysis:")
for metric, change in comparison.items():
    print(f"{metric}: {change['absolute']} ({change['percentage']}%)")
```

### Trend Analysis

Track metrics over time to identify:
- Performance improvements
- Regression detection
- Seasonal variations
- Impact of system changes

### A/B Testing

Use the evaluation pipeline to compare:
- Different RAG configurations
- Various embedding models
- Alternative search algorithms
- Different chunking strategies

## 🚨 Troubleshooting

### Common Issues

1. **LLM API Failures**
   - Check API key configuration
   - Verify rate limits and quotas
   - Implement retry mechanisms

2. **Test Case Failures**
   - Validate test case data
   - Check RAG system connectivity
   - Verify expected answer format

3. **Performance Issues**
   - Reduce concurrent test execution
   - Implement caching strategies
   - Optimize search algorithms

4. **Memory Issues**
   - Process test cases in batches
   - Clear intermediate results
   - Monitor resource usage

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with reduced concurrency
metrics = await evaluator.run_evaluation(
    max_concurrent=1,  # Single thread for debugging
    enable_judging=True
)
```

## 🔧 Advanced Configuration

### Custom Evaluation Criteria

```python
class CustomJudgeCriteria(JudgeCriteria):
    """Custom evaluation criteria for specific use cases."""
    
    def __init__(self):
        super().__init__()
        self.code_quality_weight = 0.2
        self.documentation_quality_weight = 0.2
        # Adjust other weights accordingly
```

### Batch Processing

```python
# Process large test suites efficiently
async def run_large_evaluation():
    # Split test suite into batches
    batch_size = 100
    for i in range(0, len(test_suite.test_cases), batch_size):
        batch = test_suite.test_cases[i:i+batch_size]
        # Process batch
        batch_results = await process_batch(batch)
        # Save intermediate results
        save_batch_results(batch_results)
```

### Custom Report Formats

```python
class CustomReporter(EvaluationReporter):
    """Custom reporter with additional analysis."""
    
    def generate_custom_report(self):
        """Generate custom report format."""
        # Custom report logic
        pass
```

## 📚 Best Practices

### 1. Test Suite Design
- Start with a small, focused test suite
- Gradually expand coverage
- Include edge cases and error conditions
- Maintain realistic query complexity distribution

### 2. Ground Truth Quality
- Use authoritative sources for expected answers
- Keep answers up-to-date with library changes
- Ensure consistency across similar queries
- Include examples and code snippets

### 3. Evaluation Frequency
- Run evaluations after significant changes
- Establish baseline performance metrics
- Track improvements over time
- Use automated evaluation in CI/CD

### 4. Result Analysis
- Focus on actionable insights
- Identify systematic issues
- Track performance trends
- Share results with stakeholders

## 🎯 Success Metrics

### Quality Targets

- **Overall Score**: >4.0/5.0 (Grade A)
- **Success Rate**: >95%
- **Response Time**: <500ms average
- **Category Performance**: >3.5/5.0 across all categories

### Improvement Goals

- **Monthly**: 5% improvement in overall score
- **Quarterly**: 15% improvement in overall score
- **Annually**: 50% improvement in overall score

## 🔮 Future Enhancements

### Planned Features

1. **Multi-Judge Consensus**: Multiple LLM judges for reliability
2. **Automated Test Generation**: AI-generated test cases
3. **Performance Benchmarking**: Industry standard comparisons
4. **Real-time Monitoring**: Live quality assessment
5. **Custom Scoring Models**: Domain-specific evaluation criteria

### Integration Opportunities

- **CI/CD Pipelines**: Automated quality gates
- **Monitoring Systems**: Real-time quality alerts
- **Analytics Platforms**: Performance trend analysis
- **User Feedback**: Human validation of LLM scores

---

## 🚀 Getting Started Checklist

- [ ] Set up PyRAG evaluation environment
- [ ] Create initial test suite with 10-20 test cases
- [ ] Configure LLM judge with appropriate criteria
- [ ] Run baseline evaluation
- [ ] Analyze results and identify improvement areas
- [ ] Implement improvements to RAG system
- [ ] Re-run evaluation to measure improvements
- [ ] Establish regular evaluation schedule
- [ ] Track performance trends over time
- [ ] Share results with stakeholders

The PyRAG evaluation pipeline provides a robust, systematic approach to measuring and improving RAG system quality. By following this guide, you can establish a comprehensive quality assessment framework that drives continuous improvement in your RAG system.
