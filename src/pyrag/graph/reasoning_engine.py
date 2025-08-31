"""Multi-hop reasoning engine for complex queries."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReasoningStep:
    """Represents a step in multi-hop reasoning."""
    step_id: int
    query: str
    reasoning_type: str
    context: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    confidence: float


@dataclass
class ReasoningResult:
    """Result of multi-hop reasoning."""
    query: str
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float
    supporting_evidence: List[Dict[str, Any]]
    execution_time: float


class ReasoningEngine:
    """Multi-hop reasoning for complex queries."""
    
    def __init__(self):
        """Initialize the reasoning engine."""
        self.logger = get_logger(__name__)
        
        # Reasoning strategies
        self.reasoning_strategies = {
            "comparison": self._compare_entities,
            "composition": self._compose_solution,
            "inference": self._infer_relationships,
            "synthesis": self._synthesize_information,
        }
    
    async def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """Execute multi-step reasoning."""
        try:
            self.logger.info(f"Starting multi-hop reasoning for query: {query}")
            
            # Parse reasoning steps
            steps = await self._parse_reasoning_steps(query, context)
            
            # Execute each step
            executed_steps = []
            for step in steps:
                step_result = await self._execute_reasoning_step(step, context)
                executed_steps.append(step_result)
            
            # Validate intermediate results
            valid_steps = [step for step in executed_steps if step.confidence > 0.5]
            
            # Combine final answer
            final_answer = await self._combine_reasoning_results(valid_steps)
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(valid_steps)
            
            return ReasoningResult(
                query=query,
                steps=executed_steps,
                final_answer=final_answer,
                confidence=confidence,
                supporting_evidence=[step.result for step in valid_steps if step.result],
                execution_time=0.0  # TODO: Add timing
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute reasoning: {e}")
            return ReasoningResult(
                query=query,
                steps=[],
                final_answer="Unable to process reasoning query",
                confidence=0.0,
                supporting_evidence=[],
                execution_time=0.0
            )
    
    async def _parse_reasoning_steps(self, query: str, context: Dict[str, Any]) -> List[ReasoningStep]:
        """Parse query into reasoning steps."""
        steps = []
        step_id = 1
        
        # Analyze query to determine reasoning type
        reasoning_type = self._determine_reasoning_type(query)
        
        if reasoning_type == "comparison":
            # Comparison reasoning: compare two or more entities
            entities = self._extract_comparison_entities(query)
            if len(entities) >= 2:
                steps.append(ReasoningStep(
                    step_id=step_id,
                    query=f"Compare {entities[0]} and {entities[1]}",
                    reasoning_type="comparison",
                    context={"entities": entities},
                    result=None,
                    confidence=0.0
                ))
                step_id += 1
        
        elif reasoning_type == "composition":
            # Composition reasoning: build solution from parts
            components = self._extract_composition_components(query)
            if components:
                steps.append(ReasoningStep(
                    step_id=step_id,
                    query=f"Compose solution using {', '.join(components)}",
                    reasoning_type="composition",
                    context={"components": components},
                    result=None,
                    confidence=0.0
                ))
                step_id += 1
        
        elif reasoning_type == "inference":
            # Inference reasoning: infer relationships
            entities = self._extract_inference_entities(query)
            if entities:
                steps.append(ReasoningStep(
                    step_id=step_id,
                    query=f"Infer relationships for {', '.join(entities)}",
                    reasoning_type="inference",
                    context={"entities": entities},
                    result=None,
                    confidence=0.0
                ))
                step_id += 1
        
        else:
            # Default to synthesis
            steps.append(ReasoningStep(
                step_id=step_id,
                query=query,
                reasoning_type="synthesis",
                context=context,
                result=None,
                confidence=0.0
            ))
        
        return steps
    
    async def _execute_reasoning_step(self, step: ReasoningStep, context: Dict[str, Any]) -> ReasoningStep:
        """Execute a single reasoning step."""
        try:
            self.logger.info(f"Executing reasoning step {step.step_id}: {step.reasoning_type}")
            
            # Get the appropriate reasoning strategy
            strategy = self.reasoning_strategies.get(step.reasoning_type)
            if strategy:
                result = await strategy(step.query, step.context, context)
                step.result = result
                step.confidence = result.get("confidence", 0.0)
            else:
                step.result = {"error": f"Unknown reasoning type: {step.reasoning_type}"}
                step.confidence = 0.0
            
            return step
            
        except Exception as e:
            self.logger.error(f"Failed to execute reasoning step {step.step_id}: {e}")
            step.result = {"error": str(e)}
            step.confidence = 0.0
            return step
    
    async def _compare_entities(self, query: str, step_context: Dict[str, Any], global_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two or more entities."""
        entities = step_context.get("entities", [])
        
        if len(entities) < 2:
            return {
                "error": "Need at least 2 entities for comparison",
                "confidence": 0.0
            }
        
        # Mock comparison logic
        comparison_result = {
            "type": "comparison",
            "entities": entities,
            "similarities": [
                "Both are HTTP libraries",
                "Both support async operations",
                "Both have similar API design"
            ],
            "differences": [
                f"{entities[0]} is more feature-rich",
                f"{entities[1]} has better performance",
                f"{entities[0]} has more documentation"
            ],
            "recommendation": f"Use {entities[0]} for complex projects, {entities[1]} for simple cases",
            "confidence": 0.8
        }
        
        return comparison_result
    
    async def _compose_solution(self, query: str, step_context: Dict[str, Any], global_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compose solution from multiple components."""
        components = step_context.get("components", [])
        
        if not components:
            return {
                "error": "No components provided for composition",
                "confidence": 0.0
            }
        
        # Mock composition logic
        if len(components) >= 3:
            solution_steps = [
                f"First, use {components[0]} for data loading",
                f"Then, use {components[1]} for processing",
                f"Finally, use {components[2]} for output"
            ]
            dependencies = [
                f"{components[0]} depends on {components[1]}",
                f"{components[1]} depends on {components[2]}"
            ]
        elif len(components) == 2:
            solution_steps = [
                f"First, use {components[0]} for setup",
                f"Then, use {components[1]} for implementation"
            ]
            dependencies = [
                f"{components[0]} integrates with {components[1]}"
            ]
        else:
            solution_steps = [
                f"Use {components[0]} for the complete solution"
            ]
            dependencies = []
        
        composition_result = {
            "type": "composition",
            "components": components,
            "solution_steps": solution_steps,
            "dependencies": dependencies,
            "confidence": 0.7
        }
        
        return composition_result
    
    async def _infer_relationships(self, query: str, step_context: Dict[str, Any], global_context: Dict[str, Any]) -> Dict[str, Any]:
        """Infer relationships between entities."""
        entities = step_context.get("entities", [])
        
        if not entities:
            return {
                "error": "No entities provided for inference",
                "confidence": 0.0
            }
        
        # Mock inference logic
        inference_result = {
            "type": "inference",
            "entities": entities,
            "relationships": [
                {
                    "source": entities[0],
                    "target": entities[1],
                    "type": "depends_on",
                    "confidence": 0.9
                },
                {
                    "source": entities[1],
                    "target": entities[0],
                    "type": "complements",
                    "confidence": 0.7
                }
            ],
            "inferred_patterns": [
                "Common usage pattern: A -> B -> C",
                "Alternative pattern: A -> D -> B"
            ],
            "confidence": 0.75
        }
        
        return inference_result
    
    async def _synthesize_information(self, query: str, step_context: Dict[str, Any], global_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize information from multiple sources."""
        # Mock synthesis logic
        synthesis_result = {
            "type": "synthesis",
            "sources": ["documentation", "examples", "community feedback"],
            "key_points": [
                "Best practice: Use async operations when possible",
                "Performance tip: Reuse connections",
                "Security note: Always validate inputs"
            ],
            "recommendations": [
                "Start with the official documentation",
                "Check community examples for common patterns",
                "Consider performance implications"
            ],
            "confidence": 0.8
        }
        
        return synthesis_result
    
    def _determine_reasoning_type(self, query: str) -> str:
        """Determine the type of reasoning needed for the query."""
        query_lower = query.lower()
        
        # Comparison indicators
        comparison_indicators = [
            "compare", "difference", "similar", "versus", "vs", "better",
            "worse", "advantage", "disadvantage", "pros", "cons"
        ]
        
        # Composition indicators
        composition_indicators = [
            "combine", "integrate", "build", "create", "compose",
            "assemble", "put together", "mix", "merge"
        ]
        
        # Inference indicators
        inference_indicators = [
            "infer", "deduce", "conclude", "imply", "suggest",
            "indicate", "show", "reveal", "demonstrate"
        ]
        
        # Check for indicators
        if any(indicator in query_lower for indicator in comparison_indicators):
            return "comparison"
        elif any(indicator in query_lower for indicator in composition_indicators):
            return "composition"
        elif any(indicator in query_lower for indicator in inference_indicators):
            return "inference"
        else:
            return "synthesis"
    
    def _extract_comparison_entities(self, query: str) -> List[str]:
        """Extract entities to compare from query."""
        import re
        
        # Look for patterns like "A vs B", "compare A and B"
        vs_pattern = r'(\w+)\s+(?:vs|versus)\s+(\w+)'
        compare_pattern = r'compare\s+(\w+)\s+and\s+(\w+)'
        
        matches = re.findall(vs_pattern, query, re.IGNORECASE)
        if matches:
            return list(matches[0])
        
        matches = re.findall(compare_pattern, query, re.IGNORECASE)
        if matches:
            return list(matches[0])
        
        # Fallback: extract any API-like patterns
        api_pattern = r'\b\w+(?:\.\w+)+\b'
        entities = re.findall(api_pattern, query)
        return entities[:2]  # Return first two
    
    def _extract_composition_components(self, query: str) -> List[str]:
        """Extract components for composition from query."""
        import re
        
        # Look for lists of components
        components_pattern = r'(?:using|with|combine|integrate)\s+([\w\s,]+)'
        matches = re.findall(components_pattern, query, re.IGNORECASE)
        
        if matches:
            components_str = matches[0]
            components = [comp.strip() for comp in components_str.split(',')]
            return components
        
        # Fallback: extract API patterns
        api_pattern = r'\b\w+(?:\.\w+)+\b'
        components = re.findall(api_pattern, query)
        return components
    
    def _extract_inference_entities(self, query: str) -> List[str]:
        """Extract entities for inference from query."""
        import re
        
        # Look for entities mentioned in inference context
        api_pattern = r'\b\w+(?:\.\w+)+\b'
        entities = re.findall(api_pattern, query)
        
        # Also look for library names
        library_pattern = r'\b(requests|pandas|numpy|fastapi|django|sqlalchemy)\b'
        libraries = re.findall(library_pattern, query, re.IGNORECASE)
        
        return list(set(entities + libraries))
    
    async def _combine_reasoning_results(self, steps: List[ReasoningStep]) -> str:
        """Combine results from multiple reasoning steps."""
        if not steps:
            return "No reasoning steps completed successfully"
        
        # Extract key information from each step
        key_points = []
        
        for step in steps:
            if step.result and step.confidence > 0.5:
                result = step.result
                
                if result.get("type") == "comparison":
                    recommendation = result.get("recommendation", "")
                    if recommendation:
                        key_points.append(recommendation)
                
                elif result.get("type") == "composition":
                    solution_steps = result.get("solution_steps", [])
                    if solution_steps:
                        key_points.extend(solution_steps)
                
                elif result.get("type") == "inference":
                    patterns = result.get("inferred_patterns", [])
                    if patterns:
                        key_points.extend(patterns)
                
                elif result.get("type") == "synthesis":
                    recommendations = result.get("recommendations", [])
                    if recommendations:
                        key_points.extend(recommendations)
        
        if not key_points:
            return "Unable to synthesize meaningful results"
        
        # Combine into a coherent answer
        return f"Based on the analysis: {'; '.join(key_points[:3])}"  # Limit to 3 key points
    
    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence from reasoning steps."""
        if not steps:
            return 0.0
        
        # Weight by step confidence
        total_confidence = sum(step.confidence for step in steps)
        return total_confidence / len(steps)
