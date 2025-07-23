"""
Component API Routes
"""
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import time

from core.registry import ComponentRegistry
from services.component_manager import ComponentManager
from models.component import ComponentExecutionRequest, ComponentResponse
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/components", tags=["components"])

class ComponentListResponse(BaseModel):
    components: Dict[str, Any]
    categories: Dict[str, List[str]]
    total_count: int
    success: bool = True

class ComponentExecutionResponse(BaseModel):
    success: bool
    outputs: Dict[str, Any]
    execution_time: float
    component_id: str
    component_name: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ComponentValidationResponse(BaseModel):
    valid: bool
    inputs: Dict[str, Any]
    errors: List[str] = []
    warnings: List[str] = []

class ComponentSearchResponse(BaseModel):
    components: List[str]
    query: str
    total_found: int

# Dependency to get component manager
async def get_component_manager() -> ComponentManager:
    return ComponentManager()

@router.get("/", response_model=ComponentListResponse)
async def list_components(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search components")
):
    """Get all available components with optional filtering"""
    try:
        if search:
            # Search functionality
            found_components = ComponentRegistry.search_components(search)
            all_components = ComponentRegistry.get_all_components()
            filtered_components = {
                name: schema for name, schema in all_components.items()
                if name in found_components
            }
        elif category:
            # Filter by category
            component_names = ComponentRegistry.get_components_by_category(category)
            all_components = ComponentRegistry.get_all_components()
            filtered_components = {
                name: schema for name, schema in all_components.items()
                if name in component_names
            }
        else:
            # Return all components
            filtered_components = ComponentRegistry.get_all_components()
        
        categories = ComponentRegistry.get_categories()
        
        return ComponentListResponse(
            components=filtered_components,
            categories=categories,
            total_count=len(filtered_components),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Failed to list components: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list components: {str(e)}")

@router.get("/categories")
async def get_categories():
    """Get all component categories"""
    try:
        categories = ComponentRegistry.get_categories()
        stats = ComponentRegistry.get_stats()
        
        return {
            "categories": categories,
            "stats": stats,
            "success": True
        }
    except Exception as e:
        logger.error(f"Failed to get categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@router.get("/search")
async def search_components(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results")
) -> ComponentSearchResponse:
    """Search components by name, description, or tags"""
    try:
        found_components = ComponentRegistry.search_components(q)
        limited_results = found_components[:limit]
        
        return ComponentSearchResponse(
            components=limited_results,
            query=q,
            total_found=len(found_components)
        )
    except Exception as e:
        logger.error(f"Component search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/stats")
async def get_component_stats():
    """Get component registry statistics"""
    try:
        stats = ComponentRegistry.get_stats()
        return {
            "stats": stats,
            "success": True
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.get("/{component_name}")
async def get_component_schema(component_name: str):
    """Get schema for a specific component"""
    try:
        component_instance = ComponentRegistry.get_component_instance(component_name)
        if not component_instance:
            raise HTTPException(status_code=404, detail=f"Component '{component_name}' not found")
        
        schema = component_instance.get_schema()
        return {
            "schema": schema,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get component schema: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get component schema: {str(e)}")

@router.post("/{component_name}/execute", response_model=ComponentExecutionResponse)
async def execute_component(
    component_name: str,
    request: ComponentExecutionRequest,
    background_tasks: BackgroundTasks,
    component_manager: ComponentManager = Depends(get_component_manager)
):
    """Execute a single component"""
    start_time = time.time()
    
    try:
        logger.info(f"Executing component: {component_name} with ID: {request.component_id}")
        
        result = await component_manager.execute_component(
            component_name=component_name,
            inputs=request.inputs,
            component_id=request.component_id
        )
        
        execution_time = time.time() - start_time
        
        # Log execution for monitoring
        background_tasks.add_task(
            log_component_execution,
            component_name,
            execution_time,
            result["success"]
        )
        
        return ComponentExecutionResponse(
            success=result["success"],
            outputs=result["outputs"],
            execution_time=result["execution_time"],
            component_id=request.component_id,
            component_name=component_name,
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"Component execution failed: {error_msg}")
        
        # Log failed execution
        background_tasks.add_task(
            log_component_execution,
            component_name,
            execution_time,
            False
        )
        
        return ComponentExecutionResponse(
            success=False,
            outputs={},
            execution_time=execution_time,
            component_id=request.component_id,
            component_name=component_name,
            error=error_msg
        )

@router.post("/{component_name}/validate", response_model=ComponentValidationResponse)
async def validate_component_inputs(
    component_name: str,
    inputs: Dict[str, Any]
):
    """Validate inputs for a component"""
    try:
        component_class = ComponentRegistry.get_component(component_name)
        if not component_class:
            raise HTTPException(status_code=404, detail=f"Component '{component_name}' not found")
        
        instance = component_class()
        
        errors = []
        warnings = []
        
        try:
            is_valid = instance.validate_inputs(inputs)
            
            # Additional validation checks
            for inp in instance.inputs:
                if inp.name in inputs:
                    value = inputs[inp.name]
                    
                    # Check for empty required fields
                    if inp.required and not value and value != 0 and value is not False:
                        errors.append(f"Required field '{inp.name}' is empty")
                    
                    # Type-specific warnings
                    if inp.field_type == "list" and not isinstance(value, list):
                        warnings.append(f"Field '{inp.name}' should be a list")
                    elif inp.field_type == "dict" and not isinstance(value, dict):
                        warnings.append(f"Field '{inp.name}' should be a dictionary")
            
            return ComponentValidationResponse(
                valid=len(errors) == 0,
                inputs=inputs,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as validation_error:
            errors.append(str(validation_error))
            return ComponentValidationResponse(
                valid=False,
                inputs=inputs,
                errors=errors,
                warnings=warnings
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

@router.post("/{component_name}/batch_execute")
async def batch_execute_component(
   component_name: str,
   batch_requests: List[ComponentExecutionRequest],
   component_manager: ComponentManager = Depends(get_component_manager)
):
   """Execute a component multiple times with different inputs"""
   try:
       results = []
       
       for request in batch_requests:
           try:
               result = await component_manager.execute_component(
                   component_name=component_name,
                   inputs=request.inputs,
                   component_id=request.component_id
               )
               results.append({
                   "component_id": request.component_id,
                   "success": True,
                   "outputs": result["outputs"],
                   "execution_time": result["execution_time"]
               })
           except Exception as e:
               results.append({
                   "component_id": request.component_id,
                   "success": False,
                   "error": str(e),
                   "execution_time": 0.0
               })
       
       return {
           "batch_results": results,
           "total_executions": len(batch_requests),
           "successful_executions": sum(1 for r in results if r["success"]),
           "success": True
       }
       
   except Exception as e:
       logger.error(f"Batch execution failed: {str(e)}")
       raise HTTPException(status_code=500, detail=f"Batch execution failed: {str(e)}")

async def log_component_execution(component_name: str, execution_time: float, success: bool):
   """Background task to log component execution metrics"""
   # This would integrate with your monitoring system
   logger.info(f"Component {component_name} executed in {execution_time:.2f}s, success: {success}")