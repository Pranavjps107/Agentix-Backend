"""
Component Manager Service
"""
import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from uuid import uuid4

from core.registry import ComponentRegistry
from core.exceptions import ExecutionException, ValidationException
from models.component import ComponentResponse, ComponentStats
from models.execution import ExecutionResult, ExecutionStatus
from services.caching import CacheManager

logger = logging.getLogger(__name__)

class ComponentManager:
    """Manages component execution, caching, and monitoring"""
    
    def __init__(self):
        self.execution_cache: Dict[str, Any] = {}
        self.component_instances: Dict[str, Any] = {}
        self.execution_stats: Dict[str, ComponentStats] = {}
        self.cache_manager = CacheManager()
    
    async def execute_component(
        self, 
        component_name: str, 
        inputs: Dict[str, Any],
        component_id: Optional[str] = None,
        timeout: float = 300.0,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Execute a single component with caching and error handling"""
        start_time = time.time()
        execution_id = str(uuid4())
        
        if component_id is None:
            component_id = f"{component_name}-{execution_id}"
        
        logger.info(f"Executing component: {component_name} (ID: {component_id})")
        
        try:
            # Check cache first if enabled
            if use_cache:
                cached_result = await self.cache_manager.get_cached_result(component_name, inputs)
                if cached_result:
                    logger.info(f"Cache hit for component: {component_name}")
                    cached_result["cached"] = True
                    cached_result["execution_time"] = time.time() - start_time
                    return cached_result
            
            # Get component class
            component_class = ComponentRegistry.get_component(component_name)
            if not component_class:
                raise ValueError(f"Component '{component_name}' not found in registry")
            
            # Create or get cached instance
            if component_id in self.component_instances:
                component = self.component_instances[component_id]
            else:
                component = component_class()
                self.component_instances[component_id] = component
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    component._execute_with_error_handling(**inputs),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                raise ExecutionException(f"Component execution timed out after {timeout} seconds")
            
            execution_time = time.time() - start_time
            
            # Prepare response
            response = {
                "outputs": result,
                "execution_time": execution_time,
                "component_name": component_name,
                "component_id": component_id,
                "success": True,
                "cached": False,
                "execution_id": execution_id
            }
            
            # Cache result if successful and caching is enabled
            if use_cache and result:
                await self.cache_manager.cache_result(component_name, inputs, response)
            
            # Update statistics
            await self._update_component_stats(component_name, execution_time, True)
            
            # Cache result in memory
            self.execution_cache[execution_id] = response
            
            logger.info(f"Component {component_name} executed successfully in {execution_time:.2f}s")
            
            return response
            
# In the execute_component method, around line 84-90, update this part:

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Component {component_name} execution failed: {error_msg}")
            
            # Update statistics for failed execution
            await self._update_component_stats(component_name, execution_time, False)
            
            # Fix the ExecutionException call
            response = {
                "outputs": {},
                "execution_time": execution_time,
                "component_name": component_name,
                "component_id": component_id,
                "success": False,
                "error": error_msg,
                "execution_id": execution_id
            }
            
            return response
    
    async def batch_execute_components(
        self,
        executions: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Execute multiple components concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(execution_config):
            async with semaphore:
                return await self.execute_component(**execution_config)
        
        tasks = [execute_with_semaphore(config) for config in executions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "execution_time": 0.0,
                    "component_name": executions[i].get("component_name", "unknown"),
                    "component_id": executions[i].get("component_id", "unknown")
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def get_component_result(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get cached component result by execution ID"""
        return self.execution_cache.get(execution_id)
    
    async def validate_component_inputs(
        self, 
        component_name: str, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate inputs for a component"""
        try:
            component_class = ComponentRegistry.get_component(component_name)
            if not component_class:
                return {
                    "valid": False,
                    "errors": [f"Component '{component_name}' not found"]
                }
            
            instance = component_class()
            instance.validate_inputs(inputs)
            
            return {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
        except ValidationException as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": []
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": []
            }
    
    async def get_component_schema(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a component"""
        instance = ComponentRegistry.get_component_instance(component_name)
        if instance:
            return instance.get_schema()
        return None
    
    async def _update_component_stats(
        self, 
        component_name: str, 
        execution_time: float, 
        success: bool
    ):
        """Update component execution statistics"""
        if component_name not in self.execution_stats:
            self.execution_stats[component_name] = ComponentStats(
                component_name=component_name
            )
        
        stats = self.execution_stats[component_name]
        stats.total_executions += 1
        
        if success:
            stats.successful_executions += 1
        else:
            stats.failed_executions += 1
        
        # Update average execution time
        if stats.total_executions == 1:
            stats.average_execution_time = execution_time
        else:
            stats.average_execution_time = (
                (stats.average_execution_time * (stats.total_executions - 1) + execution_time) /
                stats.total_executions
            )
        
        stats.last_execution = time.time()
    
    def get_component_stats(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Get component execution statistics"""
        if component_name:
            return self.execution_stats.get(component_name, {}).dict() if component_name in self.execution_stats else {}
        
        return {
            name: stats.dict() 
            for name, stats in self.execution_stats.items()
        }
    
    def clear_cache(self, component_id: Optional[str] = None):
        """Clear execution cache"""
        if component_id:
            self.execution_cache.pop(component_id, None)
            self.component_instances.pop(component_id, None)
        else:
            self.execution_cache.clear()
            self.component_instances.clear()
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get component manager statistics"""
        return {
            "cached_results": len(self.execution_cache),
            "active_instances": len(self.component_instances),
            "registered_components": len(ComponentRegistry._components),
            "total_component_executions": sum(
                stats.total_executions for stats in self.execution_stats.values()
            ),
            "successful_executions": sum(
                stats.successful_executions for stats in self.execution_stats.values()
            ),
            "failed_executions": sum(
                stats.failed_executions for stats in self.execution_stats.values()
            )
        }