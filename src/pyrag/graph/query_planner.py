"""Query planner for agentic query planning and execution."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryStep:
    """Represents a step in a query execution plan."""
    step_id: int
    step_type: str
    description: str
    parameters: Dict[str, Any]
    dependencies: List[int]
    estimated_cost: float
    timeout: float


@dataclass
class QueryPlan:
    """Complete query execution plan."""
    query: str
    steps: List[QueryStep]
    total_estimated_cost: float
    max_parallel_steps: int
    fallback_strategy: str


@dataclass
class QueryResult:
    """Result of query execution."""
    query: str
    results: List[Dict[str, Any]]
    execution_time: float
    success: bool
    error_message: Optional[str]
    step_results: List[Dict[str, Any]]


class QueryPlanner:
    """Plan complex queries using agentic reasoning."""
    
    def __init__(self):
        """Initialize the query planner."""
        self.logger = get_logger(__name__)
        
        # Query complexity thresholds
        self.simple_query_threshold = 0.3
        self.complex_query_threshold = 0.7
        
        # Step types and their costs
        self.step_costs = {
            "semantic_search": 1.0,
            "graph_traversal": 2.0,
            "multi_hop_reasoning": 3.0,
            "relationship_search": 1.5,
            "entity_extraction": 0.5,
            "result_synthesis": 1.0,
        }
    
    async def plan_query(self, query: str) -> QueryPlan:
        """Create execution plan for complex queries."""
        try:
            self.logger.info(f"Planning query: {query}")
            
            # Analyze query complexity
            complexity = self._analyze_complexity(query)
            
            # Determine query type and required steps
            steps = await self._determine_steps(query, complexity)
            
            # Calculate costs and dependencies
            total_cost = sum(step.estimated_cost for step in steps)
            max_parallel = self._calculate_max_parallel_steps(steps)
            
            # Create fallback strategy
            fallback = self._create_fallback_strategy(query, complexity)
            
            plan = QueryPlan(
                query=query,
                steps=steps,
                total_estimated_cost=total_cost,
                max_parallel_steps=max_parallel,
                fallback_strategy=fallback
            )
            
            self.logger.info(f"Created query plan with {len(steps)} steps, cost: {total_cost}")
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to plan query: {e}")
            # Return simple fallback plan
            return self._create_fallback_plan(query)
    
    async def execute_plan(self, plan: QueryPlan) -> QueryResult:
        """Execute planned query with monitoring."""
        try:
            self.logger.info(f"Executing query plan with {len(plan.steps)} steps")
            
            step_results = []
            all_results = []
            
            # Execute steps in dependency order
            executed_steps = set()
            
            while len(executed_steps) < len(plan.steps):
                # Find steps that can be executed (dependencies satisfied)
                ready_steps = [
                    step for step in plan.steps
                    if step.step_id not in executed_steps and
                    all(dep in executed_steps for dep in step.dependencies)
                ]
                
                if not ready_steps:
                    # Circular dependency or missing step
                    break
                
                # Execute ready steps in parallel
                parallel_results = await self._execute_steps_parallel(ready_steps)
                
                for step, result in zip(ready_steps, parallel_results):
                    step_results.append({
                        "step_id": step.step_id,
                        "step_type": step.step_type,
                        "result": result,
                        "success": result.get("success", False)
                    })
                    
                    if result.get("success", False):
                        all_results.extend(result.get("results", []))
                    
                    executed_steps.add(step.step_id)
            
            # Check if all steps completed successfully
            successful_steps = sum(1 for r in step_results if r["success"])
            success = successful_steps == len(plan.steps)
            
            return QueryResult(
                query=plan.query,
                results=all_results,
                execution_time=0.0,  # TODO: Add timing
                success=success,
                error_message=None if success else "Some steps failed",
                step_results=step_results
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute query plan: {e}")
            return QueryResult(
                query=plan.query,
                results=[],
                execution_time=0.0,
                success=False,
                error_message=str(e),
                step_results=[]
            )
    
    def _analyze_complexity(self, query: str) -> float:
        """Analyze query complexity (0.0 to 1.0)."""
        complexity = 0.0
        
        # Length factor
        length_factor = min(len(query.split()) / 20.0, 1.0)
        complexity += length_factor * 0.2
        
        # Entity count factor
        entities = self._extract_entities(query)
        entity_factor = min(len(entities) / 5.0, 1.0)
        complexity += entity_factor * 0.3
        
        # Relationship indicators
        relationship_indicators = [
            "relationship", "connection", "between", "and", "or",
            "how to", "compare", "difference", "similar"
        ]
        rel_count = sum(1 for indicator in relationship_indicators if indicator in query.lower())
        rel_factor = min(rel_count / 3.0, 1.0)
        complexity += rel_factor * 0.3
        
        # Multi-step indicators
        multi_step_indicators = [
            "first", "then", "next", "finally", "step", "process"
        ]
        step_count = sum(1 for indicator in multi_step_indicators if indicator in query.lower())
        step_factor = min(step_count / 2.0, 1.0)
        complexity += step_factor * 0.2
        
        return min(complexity, 1.0)
    
    async def _determine_steps(self, query: str, complexity: float) -> List[QueryStep]:
        """Determine required steps for the query."""
        steps = []
        step_id = 1
        
        # Always start with entity extraction
        steps.append(QueryStep(
            step_id=step_id,
            step_type="entity_extraction",
            description="Extract key entities from query",
            parameters={"query": query},
            dependencies=[],
            estimated_cost=self.step_costs["entity_extraction"],
            timeout=5.0
        ))
        step_id += 1
        
        if complexity < self.simple_query_threshold:
            # Simple query - just semantic search
            steps.append(QueryStep(
                step_id=step_id,
                step_type="semantic_search",
                description="Perform semantic search",
                parameters={"query": query},
                dependencies=[1],
                estimated_cost=self.step_costs["semantic_search"],
                timeout=10.0
            ))
            
        elif complexity < self.complex_query_threshold:
            # Medium complexity - add relationship search
            steps.append(QueryStep(
                step_id=step_id,
                step_type="relationship_search",
                description="Search for relationships between entities",
                parameters={"query": query},
                dependencies=[1],
                estimated_cost=self.step_costs["relationship_search"],
                timeout=15.0
            ))
            step_id += 1
            
            steps.append(QueryStep(
                step_id=step_id,
                step_type="result_synthesis",
                description="Combine search and relationship results",
                parameters={"query": query},
                dependencies=[step_id - 1],
                estimated_cost=self.step_costs["result_synthesis"],
                timeout=5.0
            ))
            
        else:
            # Complex query - multi-hop reasoning
            steps.append(QueryStep(
                step_id=step_id,
                step_type="graph_traversal",
                description="Traverse knowledge graph",
                parameters={"query": query, "max_hops": 3},
                dependencies=[1],
                estimated_cost=self.step_costs["graph_traversal"],
                timeout=20.0
            ))
            step_id += 1
            
            steps.append(QueryStep(
                step_id=step_id,
                step_type="multi_hop_reasoning",
                description="Perform multi-hop reasoning",
                parameters={"query": query, "max_hops": 3},
                dependencies=[step_id - 1],
                estimated_cost=self.step_costs["multi_hop_reasoning"],
                timeout=30.0
            ))
            step_id += 1
            
            steps.append(QueryStep(
                step_id=step_id,
                step_type="result_synthesis",
                description="Synthesize final results",
                parameters={"query": query},
                dependencies=[step_id - 1],
                estimated_cost=self.step_costs["result_synthesis"],
                timeout=10.0
            ))
        
        return steps
    
    def _calculate_max_parallel_steps(self, steps: List[QueryStep]) -> int:
        """Calculate maximum number of steps that can run in parallel."""
        # Simple heuristic: steps without dependencies can run in parallel
        independent_steps = sum(1 for step in steps if not step.dependencies)
        return max(1, independent_steps)
    
    def _create_fallback_strategy(self, query: str, complexity: float) -> str:
        """Create fallback strategy for query execution."""
        if complexity < self.simple_query_threshold:
            return "simple_semantic_search"
        elif complexity < self.complex_query_threshold:
            return "enhanced_search_with_relationships"
        else:
            return "basic_semantic_search"
    
    def _create_fallback_plan(self, query: str) -> QueryPlan:
        """Create a simple fallback plan."""
        fallback_step = QueryStep(
            step_id=1,
            step_type="semantic_search",
            description="Fallback semantic search",
            parameters={"query": query},
            dependencies=[],
            estimated_cost=1.0,
            timeout=10.0
        )
        
        return QueryPlan(
            query=query,
            steps=[fallback_step],
            total_estimated_cost=1.0,
            max_parallel_steps=1,
            fallback_strategy="simple_semantic_search"
        )
    
    async def _execute_steps_parallel(self, steps: List[QueryStep]) -> List[Dict[str, Any]]:
        """Execute multiple steps in parallel."""
        import asyncio
        
        # Create tasks for each step
        tasks = []
        for step in steps:
            task = asyncio.create_task(self._execute_single_step(step))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "results": []
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_single_step(self, step: QueryStep) -> Dict[str, Any]:
        """Execute a single query step."""
        try:
            self.logger.info(f"Executing step {step.step_id}: {step.step_type}")
            
            # Mock implementation for now
            # In production, this would call actual implementations
            
            if step.step_type == "entity_extraction":
                return await self._execute_entity_extraction(step)
            elif step.step_type == "semantic_search":
                return await self._execute_semantic_search(step)
            elif step.step_type == "relationship_search":
                return await self._execute_relationship_search(step)
            elif step.step_type == "graph_traversal":
                return await self._execute_graph_traversal(step)
            elif step.step_type == "multi_hop_reasoning":
                return await self._execute_multi_hop_reasoning(step)
            elif step.step_type == "result_synthesis":
                return await self._execute_result_synthesis(step)
            else:
                return {
                    "success": False,
                    "error": f"Unknown step type: {step.step_type}",
                    "results": []
                }
                
        except Exception as e:
            self.logger.error(f"Failed to execute step {step.step_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    async def _execute_entity_extraction(self, step: QueryStep) -> Dict[str, Any]:
        """Execute entity extraction step."""
        query = step.parameters["query"]
        entities = self._extract_entities(query)
        
        return {
            "success": True,
            "results": [{"entities": entities}],
            "step_type": "entity_extraction"
        }
    
    async def _execute_semantic_search(self, step: QueryStep) -> Dict[str, Any]:
        """Execute semantic search step."""
        # Mock implementation
        return {
            "success": True,
            "results": [{"search_results": "Mock semantic search results"}],
            "step_type": "semantic_search"
        }
    
    async def _execute_relationship_search(self, step: QueryStep) -> Dict[str, Any]:
        """Execute relationship search step."""
        # Mock implementation
        return {
            "success": True,
            "results": [{"relationships": "Mock relationship results"}],
            "step_type": "relationship_search"
        }
    
    async def _execute_graph_traversal(self, step: QueryStep) -> Dict[str, Any]:
        """Execute graph traversal step."""
        # Mock implementation
        return {
            "success": True,
            "results": [{"graph_paths": "Mock graph traversal results"}],
            "step_type": "graph_traversal"
        }
    
    async def _execute_multi_hop_reasoning(self, step: QueryStep) -> Dict[str, Any]:
        """Execute multi-hop reasoning step."""
        # Mock implementation
        return {
            "success": True,
            "results": [{"reasoning": "Mock multi-hop reasoning results"}],
            "step_type": "multi_hop_reasoning"
        }
    
    async def _execute_result_synthesis(self, step: QueryStep) -> Dict[str, Any]:
        """Execute result synthesis step."""
        # Mock implementation
        return {
            "success": True,
            "results": [{"synthesis": "Mock result synthesis"}],
            "step_type": "result_synthesis"
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query."""
        import re
        
        # Look for API-like patterns
        api_pattern = r'\b\w+(?:\.\w+)+\b'
        entities = re.findall(api_pattern, query)
        
        # Look for library names
        library_pattern = r'\b(requests|pandas|numpy|fastapi|django|sqlalchemy)\b'
        libraries = re.findall(library_pattern, query, re.IGNORECASE)
        
        return list(set(entities + libraries))
