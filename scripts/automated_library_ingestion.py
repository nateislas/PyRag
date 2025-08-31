#!/usr/bin/env python3
"""
Automated PyRAG Library Ingestion System

This script automates the ingestion of multiple Python libraries into PyRAG,
providing batch processing, progress tracking, and comprehensive reporting.
Libraries are configured in config/libraries.json for easy management.
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.config import get_config, validate_config
from pyrag.vector_store import VectorStore
from pyrag.embeddings import EmbeddingService
from pyrag.llm.client import LLMClient
from pyrag.ingestion.documentation_manager import DocumentationManager, DocumentationJob


@dataclass
class LibraryConfig:
    """Configuration for a library to be ingested."""
    name: str
    url: str
    description: str
    category: str
    expected_complexity: str  # low, medium, high
    max_crawl_pages: int
    max_content_pages: int
    priority: int  # 1=high, 2=medium, 3=low
    enabled: bool = True
    notes: Optional[str] = None


@dataclass
class IngestionResult:
    """Result of a library ingestion attempt."""
    library_name: str
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    crawl_stats: Optional[Dict] = None
    extraction_stats: Optional[Dict] = None
    processing_stats: Optional[Dict] = None
    storage_stats: Optional[Dict] = None
    errors: Optional[List[str]] = None
    search_test_results: Optional[Dict] = None


def load_library_config(config_path: str = "config/libraries.json") -> List[LibraryConfig]:
    """Load library configuration from JSON file."""
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Library configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    libraries = []
    for lib_data in data.get("libraries", []):
        library = LibraryConfig(
            name=lib_data["name"],
            url=lib_data["url"],
            description=lib_data["description"],
            category=lib_data["category"],
            expected_complexity=lib_data["expected_complexity"],
            max_crawl_pages=lib_data["max_crawl_pages"],
            max_content_pages=lib_data["max_content_pages"],
            priority=lib_data["priority"],
            enabled=lib_data.get("enabled", True),
            notes=lib_data.get("notes")
        )
        libraries.append(library)
    
    return libraries


class AutomatedIngestionManager:
    """Manages automated ingestion of multiple libraries."""
    
    def __init__(self, doc_manager: DocumentationManager, vector_store: VectorStore):
        self.doc_manager = doc_manager
        self.vector_store = vector_store
        self.results: List[IngestionResult] = []
        self.start_time = datetime.now()
    
    async def ingest_library(self, library: LibraryConfig) -> IngestionResult:
        """Ingest a single library with comprehensive coverage."""
        
        print(f"\n{'='*80}")
        print(f"🔄 INGESTING: {library.name.upper()}")
        print(f"📚 Category: {library.category}")
        print(f"🔗 URL: {library.url}")
        print(f"📊 Expected Complexity: {library.expected_complexity}")
        print(f"📄 Max Pages: {library.max_crawl_pages} crawl, {library.max_content_pages} content")
        if library.notes:
            print(f"📝 Notes: {library.notes}")
        print(f"{'='*80}")
        
        result = IngestionResult(
            library_name=library.name,
            success=False,
            start_time=datetime.now()
        )
        
        try:
            # Create ingestion job with comprehensive settings
            job = DocumentationJob(
                library_name=library.name,
                version="latest",
                base_url=library.url,
                output_dir=f"./cache/{library.name}",
                max_crawl_depth=4,  # Deep crawling for comprehensive coverage
                max_crawl_pages=library.max_crawl_pages,
                max_content_pages=library.max_content_pages,
                use_llm_filtering=True
            )
            
            # Execute ingestion
            ingestion_result = await self.doc_manager.ingest_documentation(job)
            
            # Update result
            result.success = ingestion_result.success
            result.crawl_stats = ingestion_result.crawl_result.crawl_stats if ingestion_result.crawl_result else None
            result.extraction_stats = ingestion_result.extraction_stats
            result.processing_stats = ingestion_result.processing_stats
            result.storage_stats = ingestion_result.storage_stats
            result.errors = ingestion_result.errors if ingestion_result.errors else None
            
            # Test search functionality
            if result.success:
                search_results = await self.vector_store.search(
                    f"how to use {library.name}",
                    n_results=5
                )
                result.search_test_results = {
                    "query": f"how to use {library.name}",
                    "results_count": len(search_results),
                    "success": len(search_results) > 0
                }
            
            # Print results
            if result.success:
                print(f"✅ SUCCESS: {library.name}")
                print(f"📊 Crawl Stats: {result.crawl_stats}")
                print(f"📄 Processing Stats: {result.processing_stats}")
                if result.search_test_results:
                    print(f"🔍 Search Test: {result.search_test_results['results_count']} results")
            else:
                print(f"❌ FAILED: {library.name}")
                if result.errors:
                    print(f"⚠️  Errors: {result.errors}")
            
        except Exception as e:
            print(f"❌ ERROR: {library.name} - {e}")
            result.errors = [str(e)]
        
        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            print(f"⏱️  Duration: {result.duration_seconds:.2f}s")
        
        return result
    
    async def ingest_libraries(self, libraries: List[LibraryConfig], max_concurrent: int = 2) -> List[IngestionResult]:
        """Ingest multiple libraries with controlled concurrency."""
        
        print(f"\n🚀 STARTING COMPREHENSIVE INGESTION")
        print(f"📚 Total Libraries: {len(libraries)}")
        print(f"⚡ Max Concurrent: {max_concurrent}")
        print(f"🕐 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Goal: Comprehensive documentation coverage for robust AI pipeline")
        
        # Sort by priority (lower number = higher priority)
        sorted_libraries = sorted(libraries, key=lambda x: x.priority)
        
        # Process libraries with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def ingest_with_semaphore(library: LibraryConfig) -> IngestionResult:
            async with semaphore:
                return await self.ingest_library(library)
        
        # Create tasks
        tasks = [ingest_with_semaphore(lib) for lib in sorted_libraries if lib.enabled]
        
        # Execute with progress tracking
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            result = await task
            results.append(result)
            
            # Progress update
            success_count = sum(1 for r in results if r.success)
            print(f"\n📈 PROGRESS: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%)")
            print(f"✅ Success: {success_count}/{i} ({success_count/i*100:.1f}%)")
        
        self.results = results
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive ingestion report."""
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        # Calculate statistics
        total_chunks = sum(
            r.processing_stats.get('total_chunks', 0) 
            for r in successful 
            if r.processing_stats
        )
        
        avg_ingestion_time = sum(r.duration_seconds for r in successful) / len(successful) if successful else 0
        
        # Categorize results
        by_category = {}
        for result in self.results:
            # Get library config for category
            library_configs = load_library_config()
            library = next((lib for lib in library_configs if lib.name == result.library_name), None)
            if library:
                category = library.category
                if category not in by_category:
                    by_category[category] = {"success": 0, "failed": 0, "total": 0}
                by_category[category]["total"] += 1
                if result.success:
                    by_category[category]["success"] += 1
                else:
                    by_category[category]["failed"] += 1
        
        # Generate report
        report = {
            "summary": {
                "total_libraries": len(self.results),
                "successful_ingestions": len(successful),
                "failed_ingestions": len(failed),
                "success_rate": len(successful) / len(self.results) * 100 if self.results else 0,
                "total_duration_seconds": total_duration,
                "average_ingestion_time": avg_ingestion_time,
                "total_chunks_created": total_chunks,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "by_category": by_category,
            "detailed_results": [asdict(result) for result in self.results],
            "recommendations": self._generate_recommendations(successful, failed)
        }
        
        return report
    
    def _generate_recommendations(self, successful: List[IngestionResult], failed: List[IngestionResult]) -> List[str]:
        """Generate recommendations based on results."""
        
        recommendations = []
        
        if len(successful) >= 4:
            recommendations.append("✅ Excellent success rate! Comprehensive GenAI library coverage achieved.")
        elif len(successful) >= 3:
            recommendations.append("✅ Good success rate. Core GenAI libraries ingested successfully.")
        else:
            recommendations.append("⚠️  Low success rate. Investigate failures and improve system.")
        
        if failed:
            failed_libraries = [r.library_name for r in failed]
            recommendations.append(f"🔧 Investigate failures: {', '.join(failed_libraries)}")
        
        if len(successful) > 0:
            avg_time = sum(r.duration_seconds for r in successful) / len(successful)
            if avg_time > 600:  # 10 minutes
                recommendations.append("⚡ Consider optimizing ingestion performance for faster processing.")
            elif avg_time < 120:
                recommendations.append("🚀 Excellent performance! System is very efficient.")
        
        total_chunks = sum(r.processing_stats.get('total_chunks', 0) for r in successful if r.processing_stats)
        if total_chunks > 2000:
            recommendations.append("📚 Large amount of content ingested. Consider implementing content deduplication.")
        
        recommendations.append("🎯 Ready for comprehensive GenAI application development queries!")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save report to JSON file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_ingestion_report_{timestamp}.json"
        
        filepath = Path("reports") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(filepath)


async def main():
    """Main function for automated library ingestion."""
    
    print("🚀 PyRAG Comprehensive Library Ingestion System")
    print("=" * 60)
    
    # Load configuration
    try:
        config = get_config()
        if not validate_config(config):
            print("❌ Configuration validation failed")
            return
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Load library configuration
    try:
        libraries = load_library_config()
        enabled_libraries = [lib for lib in libraries if lib.enabled]
        print(f"✅ Loaded {len(enabled_libraries)} libraries from configuration")
    except Exception as e:
        print(f"❌ Library configuration error: {e}")
        return
    
    # Initialize components
    try:
        vector_store = VectorStore()
        embedding_service = EmbeddingService()
        llm_client = LLMClient(config.llm)
        
        doc_manager = DocumentationManager(
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_client=llm_client,
            firecrawl_api_key=config.firecrawl.api_key
        )
        
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Component initialization error: {e}")
        return
    
    # Create ingestion manager
    manager = AutomatedIngestionManager(doc_manager, vector_store)
    
    # Display library information
    print(f"\n🤖 TOP 5 GENAI LIBRARIES - COMPREHENSIVE INGESTION")
    print(f"📚 Total Libraries: {len(enabled_libraries)}")
    print(f"⚡ Parallel Workers: 2")
    print(f"🎯 Focus: Maximum coverage for robust GenAI application development")
    print(f"\n📚 Libraries to be ingested:")
    
    total_crawl_pages = sum(lib.max_crawl_pages for lib in enabled_libraries)
    total_content_pages = sum(lib.max_content_pages for lib in enabled_libraries)
    
    for lib in enabled_libraries:
        print(f"  • {lib.name.upper()} ({lib.category})")
        print(f"    📄 {lib.max_crawl_pages} crawl pages, {lib.max_content_pages} content pages")
        print(f"    📝 {lib.description}")
        if lib.notes:
            print(f"    💡 {lib.notes}")
        print()
    
    print(f"📊 TOTAL COVERAGE: {total_crawl_pages} crawl pages, {total_content_pages} content pages")
    print(f"🎯 This will provide comprehensive documentation for GenAI development!")
    
    # Ingest libraries with 2 parallel workers
    results = await manager.ingest_libraries(enabled_libraries, max_concurrent=2)
    
    # Generate and save report
    report = manager.generate_report()
    report_path = manager.save_report(report)
    
    # Print summary
    print(f"\n{'='*80}")
    print("🎉 COMPREHENSIVE INGESTION COMPLETED")
    print(f"{'='*80}")
    
    summary = report["summary"]
    print(f"📊 SUMMARY:")
    print(f"  • Total Libraries: {summary['total_libraries']}")
    print(f"  • Successful: {summary['successful_ingestions']}")
    print(f"  • Failed: {summary['failed_ingestions']}")
    print(f"  • Success Rate: {summary['success_rate']:.1f}%")
    print(f"  • Total Duration: {summary['total_duration_seconds']:.1f}s")
    print(f"  • Average Time: {summary['average_ingestion_time']:.1f}s")
    print(f"  • Total Chunks: {summary['total_chunks_created']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    print(f"\n📄 Report saved to: {report_path}")
    
    # GenAI-specific summary
    print(f"\n🤖 GENAI LIBRARY INGESTION COMPLETE!")
    print(f"🎯 These 5 libraries provide comprehensive coverage for GenAI development:")
    for lib in enabled_libraries:
        print(f"   • {lib.name} - {lib.description}")
    
    print(f"\n💡 Next steps for users:")
    print(f"   • Search for specific GenAI patterns and integrations")
    print(f"   • Explore cross-library usage examples")
    print(f"   • Find best practices for AI application development")
    print(f"   • Discover advanced features and optimization techniques")


if __name__ == "__main__":
    asyncio.run(main())
