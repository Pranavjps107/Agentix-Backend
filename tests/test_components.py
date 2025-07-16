# tests/test_components.py
import pytest
import asyncio
from src.backend.core.registry import ComponentRegistry
from src.backend.components.llms.base_llm import LLMComponent
from src.backend.services.component_manager import ComponentManager

@pytest.fixture
def component_manager():
    return ComponentManager()

@pytest.mark.asyncio
async def test_llm_component_execution():
    """Test LLM component execution"""
    component = LLMComponent()
    
    inputs = {
        "model_name": "gpt-3.5-turbo",
        "prompt": "Hello, world!",
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    # Mock the LLM execution for testing
    result = await component.execute(**inputs)
    
    assert "response" in result
    assert "usage" in result
    assert isinstance(result["response"], str)

@pytest.mark.asyncio
async def test_component_registry():
    """Test component registry functionality"""
    components = ComponentRegistry.get_all_components()
    assert len(components) > 0
    
    categories = ComponentRegistry.get_categories()
    assert "language_models" in categories

@pytest.mark.asyncio
async def test_component_manager():
    """Test component manager execution"""
    manager = ComponentManager()
    
    result = await manager.execute_component(
        component_name="LLM Model",
        inputs={
            "model_name": "gpt-3.5-turbo",
            "prompt": "Test prompt",
            "temperature": 0.5
        },
        component_id="test-component-1"
    )
    
    assert result["success"] is True
    assert "outputs" in result
    assert "execution_time" in result

# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.backend.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_components():
    """Test component listing endpoint"""
    response = client.get("/api/v1/components/")
    assert response.status_code == 200
    
    data = response.json()
    assert "components" in data
    assert "categories" in data
    assert "total_count" in data

def test_execute_component():
    """Test component execution endpoint"""
    payload = {
        "component_id": "test-component",
        "inputs": {
            "model_name": "gpt-3.5-turbo",
            "prompt": "Hello",
            "temperature": 0.7
        }
    }
    
    response = client.post("/api/v1/components/LLM Model/execute", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "success" in data
    assert "outputs" in data

def test_flow_validation():
    """Test flow validation endpoint"""
    flow_definition = {
        "id": "test-flow",
        "name": "Test Flow",
        "nodes": [
            {
                "id": "node-1",
                "component_type": "LLM Model",
                "position": {"x": 100, "y": 100},
                "data": {}
            }
        ],
        "edges": []
    }
    
    response = client.post("/api/v1/flows/validate", json=flow_definition)
    assert response.status_code == 200
    
    data = response.json()
    assert "valid" in data
    assert "errors" in data