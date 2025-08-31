"""
Auto-scaling capabilities for PyRAG.

This module provides automatic scaling of backend instances based on
load, performance metrics, and resource utilization.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Scaling action types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning."""
    action: ScalingAction
    reason: str
    target_instances: int
    current_instances: int
    metrics: Dict[str, Any]
    timestamp: float


@dataclass
class ScalingPolicy:
    """Scaling policy configuration."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_response_time: float = 200.0  # ms
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: float = 60.0  # seconds
    scale_down_cooldown: float = 300.0  # seconds
    scale_up_step: int = 1
    scale_down_step: int = 1


class AutoScaler:
    """
    Automatic scaling controller for PyRAG backends.
    
    Provides:
    - Load-based scaling decisions
    - Performance-based scaling
    - Resource utilization monitoring
    - Scaling policy management
    - Cooldown periods to prevent thrashing
    """
    
    def __init__(
        self,
        scaling_policy: ScalingPolicy,
        metrics_collector: Optional[Callable] = None,
        scale_up_callback: Optional[Callable] = None,
        scale_down_callback: Optional[Callable] = None
    ):
        """
        Initialize the auto-scaler.
        
        Args:
            scaling_policy: Configuration for scaling behavior
            metrics_collector: Function to collect current metrics
            scale_up_callback: Function to execute scale-up
            scale_down_callback: Function to execute scale-down
        """
        self.policy = scaling_policy
        self.metrics_collector = metrics_collector
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
        
        # State tracking
        self.current_instances = scaling_policy.min_instances
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=100)
        self.scaling_history: List[ScalingDecision] = []
        
        # Background task
        self._scaling_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self, check_interval: float = 30.0):
        """
        Start the auto-scaling background task.
        
        Args:
            check_interval: Interval between scaling checks (seconds)
        """
        if self._running:
            logger.warning("Auto-scaler is already running")
            return
        
        self._running = True
        self._scaling_task = asyncio.create_task(self._scaling_loop(check_interval))
        logger.info("Auto-scaler started")
    
    async def stop(self):
        """Stop the auto-scaling background task."""
        self._running = False
        
        if self._scaling_task and not self._scaling_task.done():
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto-scaler stopped")
    
    async def _scaling_loop(self, check_interval: float):
        """Main scaling decision loop."""
        while self._running:
            try:
                await self._evaluate_scaling()
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(5.0)  # Short delay on error
    
    async def _evaluate_scaling(self):
        """Evaluate if scaling is needed and execute if necessary."""
        if not self.metrics_collector:
            logger.warning("No metrics collector configured for auto-scaling")
            return
        
        try:
            # Collect current metrics
            current_metrics = await self._collect_metrics()
            if not current_metrics:
                return
            
            # Store metrics in history
            self.metrics_history.append({
                "timestamp": time.time(),
                "metrics": current_metrics
            })
            
            # Make scaling decision
            decision = self._make_scaling_decision(current_metrics)
            
            # Execute scaling if needed
            if decision.action != ScalingAction.NO_ACTION:
                await self._execute_scaling(decision)
            
            # Store decision in history
            self.scaling_history.append(decision)
            
            # Keep only recent history
            if len(self.scaling_history) > 50:
                self.scaling_history = self.scaling_history[-25:]
                
        except Exception as e:
            logger.error(f"Scaling evaluation error: {e}")
    
    async def _collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect current system metrics."""
        try:
            if asyncio.iscoroutinefunction(self.metrics_collector):
                return await self.metrics_collector()
            else:
                return self.metrics_collector()
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            return None
    
    def _make_scaling_decision(self, metrics: Dict[str, Any]) -> ScalingDecision:
        """Make scaling decision based on current metrics."""
        current_time = time.time()
        
        # Check cooldown periods
        if (current_time - self.last_scale_up) < self.policy.scale_up_cooldown:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason="Scale-up cooldown period",
                target_instances=self.current_instances,
                current_instances=self.current_instances,
                metrics=metrics,
                timestamp=current_time
            )
        
        if (current_time - self.last_scale_down) < self.policy.scale_down_cooldown:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason="Scale-down cooldown period",
                target_instances=self.current_instances,
                current_instances=self.current_instances,
                metrics=metrics,
                timestamp=current_time
            )
        
        # Analyze metrics for scaling decisions
        cpu_utilization = metrics.get("cpu_utilization", 0.0)
        memory_utilization = metrics.get("memory_utilization", 0.0)
        response_time = metrics.get("response_time_avg", 0.0)
        error_rate = metrics.get("error_rate", 0.0)
        queue_length = metrics.get("queue_length", 0)
        
        # Check for scale-up conditions
        scale_up_needed = (
            cpu_utilization > self.policy.scale_up_threshold or
            memory_utilization > self.policy.scale_up_threshold or
            response_time > self.policy.target_response_time or
            error_rate > 0.05 or  # >5% error rate
            queue_length > 10  # Queue building up
        )
        
        if scale_up_needed and self.current_instances < self.policy.max_instances:
            target_instances = min(
                self.current_instances + self.policy.scale_up_step,
                self.policy.max_instances
            )
            
            reason = self._build_scale_up_reason(metrics, cpu_utilization, memory_utilization, response_time, error_rate, queue_length)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                reason=reason,
                target_instances=target_instances,
                current_instances=self.current_instances,
                metrics=metrics,
                timestamp=current_time
            )
        
        # Check for scale-down conditions
        scale_down_needed = (
            cpu_utilization < self.policy.scale_down_threshold and
            memory_utilization < self.policy.scale_down_threshold and
            response_time < self.policy.target_response_time * 0.5 and  # Well below target
            error_rate < 0.01 and  # Low error rate
            queue_length < 2  # No queue buildup
        )
        
        if scale_down_needed and self.current_instances > self.policy.min_instances:
            target_instances = max(
                self.current_instances - self.policy.scale_down_step,
                self.policy.min_instances
            )
            
            reason = self._build_scale_down_reason(metrics, cpu_utilization, memory_utilization, response_time, error_rate, queue_length)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                reason=reason,
                target_instances=target_instances,
                current_instances=self.current_instances,
                metrics=metrics,
                timestamp=current_time
            )
        
        # No scaling needed
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            reason="Metrics within acceptable ranges",
            target_instances=self.current_instances,
            current_instances=self.current_instances,
            metrics=metrics,
            timestamp=current_time
        )
    
    def _build_scale_up_reason(
        self, 
        metrics: Dict[str, Any], 
        cpu: float, 
        memory: float, 
        response_time: float, 
        error_rate: float, 
        queue_length: int
    ) -> str:
        """Build reason string for scale-up decision."""
        reasons = []
        
        if cpu > self.policy.scale_up_threshold:
            reasons.append(f"CPU utilization {cpu:.1f}% > {self.policy.scale_up_threshold}%")
        
        if memory > self.policy.scale_up_threshold:
            reasons.append(f"Memory utilization {memory:.1f}% > {self.policy.scale_up_threshold}%")
        
        if response_time > self.policy.target_response_time:
            reasons.append(f"Response time {response_time:.1f}ms > {self.policy.target_response_time}ms")
        
        if error_rate > 0.05:
            reasons.append(f"Error rate {error_rate:.2%} > 5%")
        
        if queue_length > 10:
            reasons.append(f"Queue length {queue_length} > 10")
        
        return f"Scale up needed: {', '.join(reasons)}"
    
    def _build_scale_down_reason(
        self, 
        metrics: Dict[str, Any], 
        cpu: float, 
        memory: float, 
        response_time: float, 
        error_rate: float, 
        queue_length: int
    ) -> str:
        """Build reason string for scale-down decision."""
        reasons = []
        
        if cpu < self.policy.scale_down_threshold:
            reasons.append(f"CPU utilization {cpu:.1f}% < {self.policy.scale_down_threshold}%")
        
        if memory < self.policy.scale_down_threshold:
            reasons.append(f"Memory utilization {memory:.1f}% < {self.policy.scale_down_threshold}%")
        
        if response_time < self.policy.target_response_time * 0.5:
            reasons.append(f"Response time {response_time:.1f}ms well below target")
        
        if error_rate < 0.01:
            reasons.append(f"Error rate {error_rate:.2%} < 1%")
        
        if queue_length < 2:
            reasons.append(f"Queue length {queue_length} < 2")
        
        return f"Scale down possible: {', '.join(reasons)}"
    
    async def _execute_scaling(self, decision: ScalingDecision):
        """Execute the scaling decision."""
        try:
            if decision.action == ScalingAction.SCALE_UP:
                await self._execute_scale_up(decision)
            elif decision.action == ScalingAction.SCALE_DOWN:
                await self._execute_scale_down(decision)
                
        except Exception as e:
            logger.error(f"Scaling execution error: {e}")
    
    async def _execute_scale_up(self, decision: ScalingDecision):
        """Execute scale-up operation."""
        logger.info(f"Scaling up: {decision.current_instances} -> {decision.target_instances} instances")
        
        if self.scale_up_callback:
            try:
                if asyncio.iscoroutinefunction(self.scale_up_callback):
                    await self.scale_up_callback(decision.target_instances, decision)
                else:
                    self.scale_up_callback(decision.target_instances, decision)
            except Exception as e:
                logger.error(f"Scale-up callback error: {e}")
                return
        
        # Update state
        self.current_instances = decision.target_instances
        self.last_scale_up = time.time()
        
        logger.info(f"Successfully scaled up to {self.current_instances} instances")
    
    async def _execute_scale_down(self, decision: ScalingDecision):
        """Execute scale-down operation."""
        logger.info(f"Scaling down: {decision.current_instances} -> {decision.target_instances} instances")
        
        if self.scale_down_callback:
            try:
                if asyncio.iscoroutinefunction(self.scale_down_callback):
                    await self.scale_down_callback(decision.target_instances, decision)
                else:
                    self.scale_down_callback(decision.target_instances, decision)
            except Exception as e:
                logger.error(f"Scale-down callback error: {e}")
                return
        
        # Update state
        self.current_instances = decision.target_instances
        self.last_scale_down = time.time()
        
        logger.info(f"Successfully scaled down to {self.current_instances} instances")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and statistics."""
        return {
            "current_instances": self.current_instances,
            "policy": {
                "min_instances": self.policy.min_instances,
                "max_instances": self.policy.max_instances,
                "scale_up_threshold": self.policy.scale_up_threshold,
                "scale_down_threshold": self.policy.scale_down_threshold,
                "target_response_time": self.policy.target_response_time
            },
            "last_scale_up": self.last_scale_up,
            "last_scale_down": self.last_scale_down,
            "recent_decisions": [
                {
                    "action": d.action.value,
                    "reason": d.reason,
                    "target_instances": d.target_instances,
                    "timestamp": d.timestamp
                }
                for d in self.scaling_history[-10:]  # Last 10 decisions
            ],
            "metrics_history_count": len(self.metrics_history)
        }
    
    def update_policy(self, new_policy: ScalingPolicy):
        """Update the scaling policy."""
        self.policy = new_policy
        logger.info("Scaling policy updated")
    
    async def manual_scale(self, target_instances: int):
        """Manually scale to a specific number of instances."""
        if target_instances < self.policy.min_instances or target_instances > self.policy.max_instances:
            raise ValueError(f"Target instances {target_instances} outside policy range [{self.policy.min_instances}, {self.policy.max_instances}]")
        
        if target_instances == self.current_instances:
            logger.info(f"Already at {target_instances} instances")
            return
        
        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP if target_instances > self.current_instances else ScalingAction.SCALE_DOWN,
            reason="Manual scaling request",
            target_instances=target_instances,
            current_instances=self.current_instances,
            metrics={},
            timestamp=time.time()
        )
        
        await self._execute_scaling(decision)
