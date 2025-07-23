"""
Flow API Routes
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional,List
import logging
import asyncio

from services.flow_executor import FlowExecutor
from models.flow import FlowDefinition, FlowExecutionRequest, FlowExecutionResponse
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/flows", tags=["flows"])

class FlowValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    flow_id: str

class FlowExportResponse(BaseModel):
    code: str
    language: str = "python"
    flow_id: str
    success: bool = True

# Dependency to get flow executor
async def get_flow_executor() -> FlowExecutor:
    return FlowExecutor()

@router.post("/execute", response_model=FlowExecutionResponse)
async def execute_flow(
    request: FlowExecutionRequest,
    background_tasks: BackgroundTasks,
    flow_executor: FlowExecutor = Depends(get_flow_executor)
):
    """Execute a complete flow"""
    try:
        logger.info(f"Executing flow: {request.flow_definition.name} (ID: {request.flow_definition.id})")
        
        if request.async_execution:
            # Execute in background
            task_id = await flow_executor.execute_flow_async(request.flow_definition, request.inputs)
            
            return FlowExecutionResponse(
                success=True,
                task_id=task_id,
                async_execution=True,
                flow_id=request.flow_definition.id
            )
        else:
            # Execute synchronously
            result = await flow_executor.execute_flow(request.flow_definition, request.inputs)
            
            # Log execution for monitoring
            background_tasks.add_task(
                log_flow_execution,
                request.flow_definition.id,
                result.get("execution_time", 0),
                result["success"]
            )
            
            return FlowExecutionResponse(
                success=result["success"],
                outputs=result.get("outputs", {}),
                execution_time=result.get("execution_time", 0),
                flow_id=request.flow_definition.id,
                component_outputs=result.get("component_outputs", {}),
                error=result.get("error")
            )
            
    except Exception as e:
        logger.error(f"Flow execution failed: {str(e)}")
        return FlowExecutionResponse(
            success=False,
            error=str(e),
            flow_id=request.flow_definition.id
        )

@router.get("/execution/{task_id}")
async def get_execution_status(
    task_id: str,
    flow_executor: FlowExecutor = Depends(get_flow_executor)
):
    """Get status of async flow execution"""
    try:
        status = await flow_executor.get_execution_status(task_id)
        
        return {
            "task_id": task_id,
            "status": status,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to get execution status for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Task not found: {str(e)}")

@router.delete("/execution/{task_id}")
async def cancel_execution(
    task_id: str,
    flow_executor: FlowExecutor = Depends(get_flow_executor)
):
    """Cancel an async flow execution"""
    try:
        result = await flow_executor.cancel_execution(task_id)
        
        return {
            "task_id": task_id,
            "cancelled": result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel execution {task_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Cancellation failed: {str(e)}")

@router.post("/validate", response_model=FlowValidationResponse)
async def validate_flow(
    flow_definition: FlowDefinition,
    flow_executor: FlowExecutor = Depends(get_flow_executor)
):
    """Validate a flow definition"""
    try:
        validation_result = await flow_executor.validate_flow(flow_definition)
        
        return FlowValidationResponse(
            valid=validation_result["valid"],
            errors=validation_result["errors"],
            warnings=validation_result["warnings"],
            flow_id=flow_definition.id
        )
        
    except Exception as e:
        logger.error(f"Flow validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Flow validation failed: {str(e)}")

@router.post("/export", response_model=FlowExportResponse)
async def export_flow_as_langchain(
    flow_definition: FlowDefinition,
    format: str = "python",
    flow_executor: FlowExecutor = Depends(get_flow_executor)
):
    """Export flow as LangChain Python code"""
    try:
        if format == "python":
            code = await flow_executor.export_as_langchain_code(flow_definition)
        elif format == "json":
            code = await flow_executor.export_as_json(flow_definition)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return FlowExportResponse(
            code=code,
            language=format,
            flow_id=flow_definition.id,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Flow export failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Export failed: {str(e)}")

@router.post("/optimize")
async def optimize_flow(
    flow_definition: FlowDefinition,
    flow_executor: FlowExecutor = Depends(get_flow_executor)
):
    """Optimize flow execution order and dependencies"""
    try:
        optimized_flow = await flow_executor.optimize_flow(flow_definition)
        
        return {
            "original_flow": flow_definition.dict(),
            "optimized_flow": optimized_flow.dict(),
            "optimization_applied": True,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Flow optimization failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Optimization failed: {str(e)}")

@router.get("/templates")
async def get_flow_templates():
    """Get predefined flow templates"""
    try:
        templates = await FlowExecutor.get_flow_templates()
        
        return {
            "templates": templates,
            "total_templates": len(templates),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to get flow templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

async def log_flow_execution(flow_id: str, execution_time: float, success: bool):
    """Background task to log flow execution metrics"""
    logger.info(f"Flow {flow_id} executed in {execution_time:.2f}s, success: {success}")