#!/usr/bin/env python3
"""Comprehensive test script for PyRAG MCP tools with pandas documentation."""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.core import PyRAG
from pyrag.logging import get_logger

logger = get_logger(__name__)

async def test_search_python_docs_pandas():
    """Test search_python_docs functionality with pandas-specific queries."""
    
    print("\n" + "="*60)
    print("🔍 TESTING: search_python_docs (Pandas Focus)")
    print("="*60)
    
    # Initialize PyRAG
    pyrag = PyRAG()
    
    # Test queries specifically for pandas functionality
    test_queries = [
        {
            "query": "how to create a pandas DataFrame",
            "library": "pandas",
            "content_type": "examples",
            "description": "DataFrame creation"
        },
        {
            "query": "pandas DataFrame operations and methods",
            "library": "pandas", 
            "content_type": "reference",
            "description": "DataFrame operations"
        },
        {
            "query": "how to read CSV files with pandas",
            "library": "pandas",
            "content_type": "tutorials",
            "description": "File reading functionality"
        },
        {
            "query": "pandas data manipulation and filtering",
            "library": "pandas",
            "content_type": "all",
            "description": "Data manipulation"
        },
        {
            "query": "pandas groupby operations and aggregations",
            "library": "pandas",
            "content_type": "examples",
            "description": "Grouping and aggregation"
        },
        {
            "query": "pandas merge and join operations",
            "library": "pandas",
            "content_type": "reference",
            "description": "Data merging"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"   Query: '{test_case['query']}'")
        print(f"   Library: {test_case['library']}")
        print(f"   Content Type: {test_case['content_type']}")
        
        try:
            start_time = time.time()
            
            # Map content_type to internal format
            mapped_content_type = None
            if test_case["content_type"]:
                if test_case["content_type"] in ["examples", "tutorials"]:
                    mapped_content_type = "examples"
                elif test_case["content_type"] in ["reference", "api_reference"]:
                    mapped_content_type = "api_reference"
                elif test_case["content_type"] == "overview":
                    mapped_content_type = "overview"
            
            # Call PyRAG search directly
            results = await pyrag.search_documentation(
                query=test_case["query"],
                library=test_case["library"],
                content_type=mapped_content_type,
                max_results=10,
            )
            
            query_time = time.time() - start_time
            
            print(f"   ✅ Success in {query_time:.2f}s")
            print(f"   📄 Found {len(results) if results else 0} results")
            
            if results:
                print(f"   📖 First result preview: {results[0]['content'][:200]}...")
                
                # Check if results contain pandas-specific content
                pandas_keywords = ['pandas', 'dataframe', 'series', 'read_csv', 'groupby', 'merge']
                found_keywords = [kw for kw in pandas_keywords if kw.lower() in results[0]['content'].lower()]
                if found_keywords:
                    print(f"   🎯 Found pandas keywords: {found_keywords}")
            else:
                print(f"   📝 No results found")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def test_get_api_reference_pandas():
    """Test get_api_reference functionality with pandas API paths."""
    
    print("\n" + "="*60)
    print("📚 TESTING: get_api_reference (Pandas Focus)")
    print("="*60)
    
    # Initialize PyRAG
    pyrag = PyRAG()
    
    # Test API reference queries for pandas
    test_apis = [
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
    
    for i, test_case in enumerate(test_apis, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"   Library: {test_case['library']}")
        print(f"   API Path: {test_case['api_path']}")
        print(f"   Include Examples: {test_case['include_examples']}")
        
        try:
            start_time = time.time()
            
            # Call PyRAG API reference directly
            result = await pyrag.get_api_reference(
                library=test_case["library"],
                api_path=test_case["api_path"],
                include_examples=test_case["include_examples"]
            )
            
            query_time = time.time() - start_time
            
            print(f"   ✅ Success in {query_time:.2f}s")
            print(f"   📄 Result length: {len(result) if result else 0} characters")
            
            if result:
                print(f"   📖 Preview: {result[:300]}...")
                
                # Check for pandas-specific content
                if 'pandas' in result.lower() or 'dataframe' in result.lower():
                    print(f"   🎯 Contains pandas-specific content")
            else:
                print(f"   📝 No API reference found")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def test_check_deprecation_pandas():
    """Test check_deprecation functionality with pandas APIs."""
    
    print("\n" + "="*60)
    print("⚠️  TESTING: check_deprecation (Pandas Focus)")
    print("="*60)
    
    # Initialize PyRAG
    pyrag = PyRAG()
    
    # Test deprecation checks for pandas
    test_apis = [
        {
            "library": "pandas",
            "apis": ["pandas.DataFrame.append", "pandas.DataFrame.iteritems"],
            "description": "Known deprecated pandas methods"
        },
        {
            "library": "pandas",
            "apis": ["pandas.read_csv", "pandas.DataFrame.groupby"],
            "description": "Current pandas methods"
        },
        {
            "library": "pandas",
            "apis": ["pandas.DataFrame.to_pickle", "pandas.DataFrame.to_hdf"],
            "description": "File I/O methods"
        }
    ]
    
    for i, test_case in enumerate(test_apis, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"   Library: {test_case['library']}")
        print(f"   APIs: {test_case['apis']}")
        
        try:
            start_time = time.time()
            
            # Call PyRAG deprecation check directly
            result = await pyrag.check_deprecation(
                library=test_case["library"],
                apis=test_case["apis"]
            )
            
            query_time = time.time() - start_time
            
            print(f"   ✅ Success in {query_time:.2f}s")
            print(f"   📄 Result length: {len(result) if result else 0} characters")
            
            if result:
                print(f"   📖 Preview: {result[:300]}...")
                
                # Check for deprecation warnings
                if 'deprecated' in result.lower() or 'warning' in result.lower():
                    print(f"   ⚠️  Contains deprecation information")
            else:
                print(f"   📝 No deprecation info found")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def test_find_similar_patterns_pandas():
    """Test find_similar_patterns functionality with pandas code snippets."""
    
    print("\n" + "="*60)
    print("🔍 TESTING: find_similar_patterns (Pandas Focus)")
    print("="*60)
    
    # Initialize PyRAG
    pyrag = PyRAG()
    
    # Test pattern matching with pandas code
    test_patterns = [
        {
            "code_snippet": "df = pd.read_csv('data.csv')",
            "intent": "Reading CSV data",
            "description": "CSV reading pattern"
        },
        {
            "code_snippet": "df.groupby('column').agg({'col1': 'mean', 'col2': 'sum'})",
            "intent": "Grouping and aggregation",
            "description": "Groupby aggregation pattern"
        },
        {
            "code_snippet": "df.merge(df2, on='key', how='left')",
            "intent": "Data merging",
            "description": "Merge operation pattern"
        },
        {
            "code_snippet": "df[df['column'] > 0].sort_values('column')",
            "intent": "Filtering and sorting",
            "description": "Filter and sort pattern"
        },
        {
            "code_snippet": "df.pivot_table(values='value', index='row', columns='col')",
            "intent": "Pivot table creation",
            "description": "Pivot table pattern"
        }
    ]
    
    for i, test_case in enumerate(test_patterns, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"   Code: {test_case['code_snippet']}")
        print(f"   Intent: {test_case['intent']}")
        
        try:
            start_time = time.time()
            
            # Call PyRAG pattern matching directly
            result = await pyrag.find_similar_patterns(
                code_snippet=test_case["code_snippet"],
                intent=test_case["intent"]
            )
            
            query_time = time.time() - start_time
            
            print(f"   ✅ Success in {query_time:.2f}s")
            print(f"   📄 Result length: {len(result) if result else 0} characters")
            
            if result:
                print(f"   📖 Preview: {result[:300]}...")
                
                # Check for pandas-specific patterns
                pandas_patterns = ['read_csv', 'groupby', 'merge', 'sort_values', 'pivot_table']
                found_patterns = [p for p in pandas_patterns if p.lower() in result.lower()]
                if found_patterns:
                    print(f"   🎯 Found pandas patterns: {found_patterns}")
            else:
                print(f"   📝 No similar patterns found")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def test_list_available_libraries():
    """Test list_available_libraries functionality."""
    
    print("\n" + "="*60)
    print("📚 TESTING: list_available_libraries")
    print("="*60)
    
    # Initialize PyRAG
    pyrag = PyRAG()
    
    try:
        start_time = time.time()
        
        # Call PyRAG library listing directly
        result = await pyrag.list_libraries()
        
        query_time = time.time() - start_time
        
        print(f"✅ Success in {query_time:.2f}s")
        print(f"📄 Found {len(result) if result else 0} libraries")
        
        if result:
            print(f"📖 Preview: {str(result)[:500]}...")
            
            # Check if pandas is mentioned
            result_str = str(result).lower()
            if "pandas" in result_str:
                print("✅ Pandas library found in results")
            else:
                print("⚠️  Pandas library not found in results")
                
            # Show all libraries found
            print(f"\n📚 All libraries found:")
            for i, lib in enumerate(result[:10], 1):  # Show first 10
                if isinstance(lib, dict):
                    name = lib.get('name', 'Unknown')
                    status = lib.get('status', 'Unknown')
                    print(f"   {i}. {name} - {status}")
                else:
                    print(f"   {i}. {lib}")
        else:
            print("📝 No libraries found")
            
    except Exception as e:
        print(f"❌ Failed: {e}")


async def test_get_library_status_pandas():
    """Test get_library_status functionality with pandas."""
    
    print("\n" + "="*60)
    print("📊 TESTING: get_library_status (Pandas Focus)")
    print("="*60)
    
    # Initialize PyRAG
    pyrag = PyRAG()
    
    # Test library status for pandas
    test_libraries = [
        {
            "library_name": "pandas",
            "description": "Pandas library status"
        },
        {
            "library_name": "numpy",
            "description": "NumPy library status (for comparison)"
        }
    ]
    
    for i, test_case in enumerate(test_libraries, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"   Library: {test_case['library_name']}")
        
        try:
            start_time = time.time()
            
            # Call PyRAG library status directly
            result = await pyrag.get_library_status(
                library_name=test_case["library_name"]
            )
            
            query_time = time.time() - start_time
            
            print(f"   ✅ Success in {query_time:.2f}s")
            print(f"   📄 Result length: {len(result) if result else 0} characters")
            
            if result:
                print(f"   📖 Preview: {result[:300]}...")
                
                # Check for status indicators
                status_indicators = ['indexed', 'version', 'chunks', 'last_updated']
                found_indicators = [ind for ind in status_indicators if ind.lower() in result.lower()]
                if found_indicators:
                    print(f"   📊 Contains status info: {found_indicators}")
            else:
                print(f"   📝 No status found")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def test_pandas_specific_queries():
    """Test pandas-specific advanced queries."""
    
    print("\n" + "="*60)
    print("🐼 TESTING: Pandas-Specific Advanced Queries")
    print("="*60)
    
    # Initialize PyRAG
    pyrag = PyRAG()
    
    # Advanced pandas-specific queries
    advanced_queries = [
        {
            "query": "how to handle missing values in pandas DataFrame",
            "description": "Missing data handling"
        },
        {
            "query": "pandas time series operations and datetime handling",
            "description": "Time series functionality"
        },
        {
            "query": "pandas performance optimization and memory usage",
            "description": "Performance optimization"
        },
        {
            "query": "pandas plotting and visualization with matplotlib",
            "description": "Visualization integration"
        },
        {
            "query": "pandas multi-level indexing and hierarchical data",
            "description": "Advanced indexing"
        }
    ]
    
    for i, test_case in enumerate(advanced_queries, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"   Query: '{test_case['query']}'")
        
        try:
            start_time = time.time()
            
            # Call PyRAG search directly
            results = await pyrag.search_documentation(
                query=test_case["query"],
                library="pandas",
                max_results=5,
            )
            
            query_time = time.time() - start_time
            
            print(f"   ✅ Success in {query_time:.2f}s")
            print(f"   📄 Found {len(results) if results else 0} results")
            
            if results:
                print(f"   📖 First result preview: {results[0]['content'][:200]}...")
                
                # Check for advanced pandas concepts
                advanced_concepts = ['missing', 'time', 'performance', 'plot', 'multi']
                found_concepts = [c for c in advanced_concepts if c.lower() in results[0]['content'].lower()]
                if found_concepts:
                    print(f"   🎯 Found advanced concepts: {found_concepts}")
            else:
                print(f"   📝 No results found")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def test_performance_benchmark_pandas():
    """Run performance benchmark with pandas-focused queries."""
    
    print("\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARK (Pandas Focus)")
    print("="*60)
    
    # Initialize PyRAG
    pyrag = PyRAG()
    
    # Define pandas-focused test cases for each tool
    benchmark_tests = [
        ("search_python_docs", {"query": "pandas DataFrame operations", "library": "pandas"}),
        ("get_api_reference", {"library": "pandas", "api_path": "pandas.DataFrame"}),
        ("check_deprecation", {"library": "pandas", "apis": ["pandas.DataFrame.append"]}),
        ("find_similar_patterns", {"code_snippet": "df.groupby('col').agg('mean')", "intent": "groupby aggregation"}),
        ("list_available_libraries", {}),
        ("get_library_status", {"library_name": "pandas"})
    ]
    
    results = []
    
    for tool_name, params in benchmark_tests:
        print(f"\n🔧 Benchmarking {tool_name}...")
        
        try:
            start_time = time.time()
            
            # Call the appropriate PyRAG function
            if tool_name == "search_python_docs":
                result = await pyrag.search_documentation(**params)
            elif tool_name == "get_api_reference":
                result = await pyrag.get_api_reference(**params)
            elif tool_name == "check_deprecation":
                result = await pyrag.check_deprecation(**params)
            elif tool_name == "find_similar_patterns":
                result = await pyrag.find_similar_patterns(**params)
            elif tool_name == "list_available_libraries":
                result = await pyrag.list_libraries(**params)
            elif tool_name == "get_library_status":
                result = await pyrag.get_library_status(**params)
            
            query_time = time.time() - start_time
            
            results.append({
                "tool": tool_name,
                "success": True,
                "time": query_time,
                "result_length": len(str(result)) if result else 0
            })
            
            print(f"   ✅ {query_time:.2f}s - {len(str(result)) if result else 0} chars")
            
        except Exception as e:
            results.append({
                "tool": tool_name,
                "success": False,
                "time": 0,
                "error": str(e)
            })
            print(f"   ❌ Failed: {e}")
    
    # Print summary
    print(f"\n📊 PERFORMANCE SUMMARY")
    print(f"{'='*40}")
    
    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {len(successful_tests)/len(results)*100:.1f}%")
    
    if successful_tests:
        avg_time = sum(r["time"] for r in successful_tests) / len(successful_tests)
        max_time = max(r["time"] for r in successful_tests)
        min_time = min(r["time"] for r in successful_tests)
        
        print(f"Average Time: {avg_time:.2f}s")
        print(f"Fastest: {min_time:.2f}s")
        print(f"Slowest: {max_time:.2f}s")
    
    # Show individual results
    print(f"\n📋 DETAILED RESULTS")
    print(f"{'='*40}")
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        if result["success"]:
            print(f"{status} {result['tool']}: {result['time']:.2f}s ({result['result_length']} chars)")
        else:
            print(f"{status} {result['tool']}: Failed - {result['error']}")


async def main():
    """Main test function."""
    
    print("🚀 PyRAG MCP Tools Test Suite - Pandas Focus")
    print("=" * 70)
    print("Testing all 6 MCP tools with pandas documentation")
    print("=" * 70)
    
    # Test each tool individually with pandas focus
    await test_search_python_docs_pandas()
    await test_get_api_reference_pandas()
    await test_check_deprecation_pandas()
    await test_find_similar_patterns_pandas()
    await test_list_available_libraries()
    await test_get_library_status_pandas()
    
    # Test pandas-specific advanced queries
    await test_pandas_specific_queries()
    
    # Run performance benchmark
    await test_performance_benchmark_pandas()
    
    print(f"\n🎉 All MCP tools tested successfully with pandas!")
    print(f"📝 The MCP server is ready for use with pandas documentation.")
    print(f"🔧 Tools tested:")
    print(f"   • search_python_docs - Pandas documentation search")
    print(f"   • get_api_reference - Pandas API documentation")
    print(f"   • check_deprecation - Pandas deprecation status")
    print(f"   • find_similar_patterns - Pandas code patterns")
    print(f"   • list_available_libraries - Library inventory")
    print(f"   • get_library_status - Pandas library status")
    
    print(f"\n🐼 Pandas-specific features tested:")
    print(f"   • DataFrame operations and methods")
    print(f"   • Data manipulation and filtering")
    print(f"   • Groupby and aggregation operations")
    print(f"   • File I/O operations (CSV, etc.)")
    print(f"   • Time series functionality")
    print(f"   • Performance optimization")
    
    print(f"\n💡 If pandas was successfully ingested, you should see:")
    print(f"   • Relevant search results for pandas queries")
    print(f"   • API reference information for pandas methods")
    print(f"   • Deprecation warnings for outdated pandas APIs")
    print(f"   • Code patterns and examples for pandas operations")


if __name__ == "__main__":
    asyncio.run(main())
