"""Streaming support for FastMCP PyRAG server."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass

from fastmcp import Context
from ..logging import (
    MultiDimensionalSearchLogger,
    log_mcp_streaming_start, 
    log_mcp_streaming_step,
    CorrelationContext
)


@dataclass
class StreamingStep:
    """Represents a step in the streaming response."""
    step: int
    total_steps: int
    step_name: str
    data: Dict[str, Any]
    is_final: bool = False


@dataclass
class StreamingResult:
    """Complete streaming result."""
    steps: List[StreamingStep]
    final_result: Dict[str, Any]
    correlation_id: str
    total_time: float


class PyRAGStreamer:
    """Handles streaming responses for PyRAG operations."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize the PyRAG streamer."""
        self.correlation_id = correlation_id or str(uuid.uuid4())
        
    async def stream_comprehensive_search(
        self, 
        pyrag, 
        query: str, 
        library: Optional[str] = None, 
        max_results: int = 20,
        ctx: Optional[Context] = None
    ) -> AsyncGenerator[StreamingStep, None]:
        """Stream a comprehensive search operation with progress updates."""
        
        # Initialize logging
        search_logger = MultiDimensionalSearchLogger(query, self.correlation_id)
        
        # Step 1: Start search
        await self._send_progress(ctx, 1, 6, "Analyzing query intent", {"query": query})
        search_logger.log_query_start(library, max_results)
        yield StreamingStep(1, 6, "query_analysis", {"query": query, "library": library})
        
        # Get comprehensive search result (this will do the actual work)
        comprehensive_result = await pyrag.search_comprehensive(
            query=query, 
            library=library, 
            max_results=max_results
        )
        
        results = comprehensive_result["results"]
        search_strategy = comprehensive_result["search_strategy"]
        multi_dim_metadata = comprehensive_result.get("multi_dimensional_metadata")
        
        # Step 2: Intent analysis complete
        await self._send_progress(ctx, 2, 6, "Intent analysis complete", {
            "strategy": search_strategy,
            "is_multi_dimensional": search_strategy == "multi_dimensional"
        })
        yield StreamingStep(2, 6, "intent_complete", {
            "strategy": search_strategy,
            "multi_dimensional": search_strategy == "multi_dimensional"
        })
        
        if search_strategy == "multi_dimensional" and multi_dim_metadata:
            # Step 3: Multi-dimensional search progress
            await self._send_progress(ctx, 3, 6, "Multi-dimensional search", {
                "dimensions": multi_dim_metadata.get("dimensions_searched", 0),
                "coverage_score": multi_dim_metadata.get("coverage_score", 0)
            })
            yield StreamingStep(3, 6, "multidim_search", {
                "dimensions_searched": multi_dim_metadata.get("dimensions_searched", 0),
                "coverage_score": multi_dim_metadata.get("coverage_score", 0),
                "result_distribution": multi_dim_metadata.get("result_distribution", {})
            })
            
            # Step 4: Result synthesis
            await self._send_progress(ctx, 4, 6, "Synthesizing results", {
                "total_results": len(results)
            })
            yield StreamingStep(4, 6, "result_synthesis", {
                "total_results": len(results),
                "search_time": multi_dim_metadata.get("total_search_time", 0)
            })
        else:
            # Standard search - combine steps 3&4
            await self._send_progress(ctx, 3, 6, "Standard search complete", {
                "total_results": len(results)
            })
            yield StreamingStep(3, 6, "standard_search", {
                "total_results": len(results)
            })
            
            await self._send_progress(ctx, 4, 6, "Processing results", {
                "result_count": len(results)
            })
            yield StreamingStep(4, 6, "processing_results", {
                "result_count": len(results)
            })
        
        # Step 5: Building response
        await self._send_progress(ctx, 5, 6, "Building structured response", {
            "sections": "processing"
        })
        yield StreamingStep(5, 6, "building_response", {
            "building_sections": True
        })
        
        # Step 6: Complete with final result
        final_result = {
            "query": query,
            "library": library,
            "results": results,
            "search_strategy": search_strategy,
            "multi_dimensional_metadata": multi_dim_metadata,
            "correlation_id": self.correlation_id
        }
        
        await self._send_progress(ctx, 6, 6, "Search complete", {
            "final_result_count": len(results),
            "strategy": search_strategy
        })
        
        yield StreamingStep(6, 6, "search_complete", final_result, is_final=True)
    
    async def _send_progress(self, ctx: Optional[Context], step: int, total: int, 
                           message: str, data: Dict[str, Any]):
        """Send progress notification through FastMCP context."""
        if ctx:
            progress_data = {
                "step": step,
                "total_steps": total,
                "message": message,
                "correlation_id": self.correlation_id,
                **data
            }
            await ctx.info(f"[{step}/{total}] {message}")
            
            # Log the streaming step
            log_mcp_streaming_step(
                "search_comprehensive_streaming", 
                step, 
                message,
                data,
                self.correlation_id
            )

    async def stream_dimension_searches(
        self,
        dimension_results,
        ctx: Optional[Context] = None
    ) -> AsyncGenerator[StreamingStep, None]:
        """Stream individual dimension search results as they complete."""
        
        total_dimensions = len(dimension_results.dimensions_searched)
        
        for i, dim_result in enumerate(dimension_results.dimension_results, 1):
            await self._send_progress(ctx, i, total_dimensions, 
                f"Dimension '{dim_result.dimension.name}' complete", {
                    "dimension": dim_result.dimension.name,
                    "result_count": dim_result.result_count,
                    "avg_score": dim_result.avg_score,
                    "search_time": dim_result.search_time
                })
            
            yield StreamingStep(i, total_dimensions, f"dimension_{dim_result.dimension.name}", {
                "dimension": dim_result.dimension.name,
                "category": dim_result.dimension.category,
                "importance": dim_result.dimension.importance,
                "result_count": dim_result.result_count,
                "avg_score": dim_result.avg_score,
                "search_time": dim_result.search_time,
                "results": dim_result.results[:3]  # Sample of results
            })


class StreamingResponseBuilder:
    """Builds streaming responses in different formats."""
    
    @staticmethod
    async def build_streaming_comprehensive_response(
        streaming_steps: List[StreamingStep],
        query: str,
        library: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build comprehensive response from streaming steps."""
        
        # Get the final step
        final_step = None
        for step in reversed(streaming_steps):
            if step.is_final:
                final_step = step
                break
        
        if not final_step:
            return {"error": "No final result found in streaming steps"}
        
        # Extract results from final step
        results = final_step.data.get("results", [])
        search_strategy = final_step.data.get("search_strategy", "unknown")
        multi_dim_metadata = final_step.data.get("multi_dimensional_metadata")
        
        # Build sections from results
        sections = {}
        total_content_length = 0
        coverage_areas = set()
        
        for result in results:
            content_type = result.get("metadata", {}).get("content_type", "general")
            main_topic = result.get("metadata", {}).get("main_topic", "Documentation")
            
            section_key = f"{main_topic} ({content_type.title()})"
            
            if section_key not in sections:
                sections[section_key] = {
                    "content_items": [],
                    "total_length": 0,
                    "source_count": 0
                }
            
            content = result["content"]
            sections[section_key]["content_items"].append({
                "content": content,
                "source_url": result.get("metadata", {}).get("source_url", ""),
                "title": result.get("metadata", {}).get("title", ""),
                "score": result.get("score", 0.0),
                "library": result.get("metadata", {}).get("library_name", library or "unknown"),
                "version": result.get("metadata", {}).get("version", "latest")
            })
            
            sections[section_key]["total_length"] += len(content)
            sections[section_key]["source_count"] += 1
            total_content_length += len(content)
            
            # Track coverage areas
            if "key_concepts" in result.get("metadata", {}):
                key_concepts = result["metadata"]["key_concepts"]
                if isinstance(key_concepts, list):
                    coverage_areas.update(key_concepts)
        
        # Build comprehensive answer
        comprehensive_parts = [f"# {StreamingResponseBuilder._generate_title(query)}"]
        
        for section_name, section_data in sections.items():
            comprehensive_parts.append(f"\n## {section_name}")
            
            for item in section_data["content_items"]:
                if item["title"]:
                    comprehensive_parts.append(f"\n### {item['title']}")
                comprehensive_parts.append(f"\n{item['content']}")
                
                if item["source_url"]:
                    comprehensive_parts.append(f"\n*Source: {item['source_url']}*")
            
            comprehensive_parts.append("")
        
        comprehensive_answer = "\n".join(comprehensive_parts)
        
        # Calculate completeness score
        completeness_score = min(1.0, len(sections) * 0.2 + min(total_content_length / 10000, 0.6))
        
        return {
            "query": query,
            "library": library,
            "response_format": "comprehensive_streaming",
            "comprehensive_answer": comprehensive_answer,
            "sections": sections,
            "streaming_metadata": {
                "total_steps": len(streaming_steps),
                "streaming_correlation_id": streaming_steps[0].data.get("correlation_id") if streaming_steps else None,
                "steps_summary": [{"step": s.step, "name": s.step_name} for s in streaming_steps]
            },
            "metadata": {
                "total_results": len(results),
                "total_sections": len(sections),
                "total_content_length": total_content_length,
                "completeness_score": round(completeness_score, 2),
                "coverage_areas": list(coverage_areas),
                "response_type": "comprehensive_streaming",
                "search_strategy": search_strategy,
                "multi_dimensional_search": multi_dim_metadata
            }
        }
    
    @staticmethod
    def _generate_title(query: str) -> str:
        """Generate a title from query."""
        words = query.split()
        if len(words) > 8:
            return " ".join(words[:8]).title() + "..."
        return query.title()


async def create_streaming_response(
    pyrag,
    query: str, 
    library: Optional[str] = None,
    max_results: int = 20,
    ctx: Optional[Context] = None,
    correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a streaming comprehensive response."""
    
    # Initialize streamer
    streamer = PyRAGStreamer(correlation_id)
    
    # Log streaming start
    log_mcp_streaming_start("search_comprehensive_streaming", 6, streamer.correlation_id)
    
    # Collect all streaming steps
    streaming_steps = []
    
    # Stream the search process
    async for step in streamer.stream_comprehensive_search(
        pyrag, query, library, max_results, ctx
    ):
        streaming_steps.append(step)
        
        # If this is the final step, we have our result
        if step.is_final:
            break
    
    # Build the comprehensive response from streaming steps
    response = await StreamingResponseBuilder.build_streaming_comprehensive_response(
        streaming_steps, query, library
    )
    
    return response