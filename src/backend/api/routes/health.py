"""
Health Check Routes
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import time
import psutil
import logging

from core.registry import ComponentRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/health", tags=["health"])

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    uptime: float
    components: Dict[str, Any]
    system: Dict[str, Any]

class DetailedHealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    uptime: float
    components: Dict[str, Any]
    system: Dict[str, Any]
    dependencies: Dict[str, Any]
    performance: Dict[str, Any]

# Track startup time
startup_time = time.time()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    try:
        current_time = time.time()
        uptime = current_time - startup_time
        
        # Component registry status
        component_stats = ComponentRegistry.get_stats()
        
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        system_info = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=current_time,
            version="1.0.0",
            uptime=uptime,
            components=component_stats,
            system=system_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Detailed health check with dependency status"""
    try:
        current_time = time.time()
        uptime = current_time - startup_time
        
        # Component registry status
        component_stats = ComponentRegistry.get_stats()
        
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        system_info = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
        
        # Check dependencies
        dependencies = await check_dependencies()
        
        # Performance metrics
        performance = {
            "uptime_seconds": uptime,
            "uptime_formatted": format_uptime(uptime),
            "python_version": psutil.version_info,
            "process_id": psutil.Process().pid
        }
        
        return DetailedHealthResponse(
            status="healthy" if all(dep["status"] == "available" for dep in dependencies.values()) else "degraded",
            timestamp=current_time,
            version="1.0.0",
            uptime=uptime,
            components=component_stats,
            system=system_info,
            dependencies=dependencies,
            performance=performance
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        # Check if essential components are loaded
        component_count = len(ComponentRegistry._components)
        
        if component_count == 0:
            raise HTTPException(status_code=503, detail="No components loaded")
        
        return {
            "status": "ready",
            "components_loaded": component_count,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")

@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    try:
        # Simple alive check
        return {
            "status": "alive",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Not alive: {str(e)}")

async def check_dependencies() -> Dict[str, Any]:
    """Check status of external dependencies"""
    dependencies = {}
    
    # Check LangChain packages
    langchain_packages = [
        "langchain_core",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_community"
    ]
    
    for package in langchain_packages:
        try:
            __import__(package)
            dependencies[package] = {
                "status": "available",
                "type": "package"
            }
        except ImportError:
            dependencies[package] = {
                "status": "unavailable",
                "type": "package",
                "error": "Package not installed"
            }
    
    # Check optional dependencies
    optional_packages = [
        "openai",
        "anthropic",
        "chromadb",
        "pinecone",
        "redis"
    ]
    
    for package in optional_packages:
        try:
            __import__(package)
            dependencies[f"{package}_optional"] = {
                "status": "available",
                "type": "optional_package"
            }
        except ImportError:
            dependencies[f"{package}_optional"] = {
                "status": "unavailable",
                "type": "optional_package",
                "error": "Optional package not installed"
            }
    
    return dependencies

def format_uptime(uptime_seconds: float) -> str:
    """Format uptime in human readable format"""
    days = int(uptime_seconds // 86400)
    hours = int((uptime_seconds % 86400) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"