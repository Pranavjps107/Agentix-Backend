#!/usr/bin/env python3
"""
LangChain Platform Server Launcher with Debug
"""
import sys
import os
import uvicorn
import logging

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_component_imports():
    """Test component imports before starting server"""
    logger.info("🧪 Testing component imports before server start...")
    
    try:
        from src.backend.core.registry import ComponentRegistry
        logger.info(f"📊 Initial components: {len(ComponentRegistry._components)}")
        
        # Force import components
        from src.backend.components.llms.base_llm import LLMComponent
        logger.info(f"📦 After LLM import: {len(ComponentRegistry._components)}")
        
        from src.backend.components.chat_models.base_chat import ChatModelComponent  
        logger.info(f"📦 After Chat Model import: {len(ComponentRegistry._components)}")
        
        # Try to create a Text Input component dynamically
        logger.info("📦 Creating Text Input component...")
        from src.backend.core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component
        from typing import Dict, Any
        
        @register_component
        class TextInputComponent(BaseLangChainComponent):
            """Text Input Component for user input"""
            
            def _setup_component(self):
                self.metadata = ComponentMetadata(
                    display_name="Text Input",
                    description="Basic text input component for user input",
                    icon="📝",
                    category="inputs",
                    tags=["input", "text", "user"],
                    version="1.0.0"
                )
                
                self.inputs = [
                    ComponentInput(
                        name="text",
                        display_name="Input Text", 
                        field_type="str",
                        description="The text input from user"
                    )
                ]
                
                self.outputs = [
                    ComponentOutput(
                        name="text",
                        display_name="Output Text",
                        field_type="str", 
                        method="get_text",
                        description="The text that was input"
                    )
                ]
            
            async def execute(self, **kwargs) -> Dict[str, Any]:
                text = kwargs.get("text", "")
                return {"text": text, "length": len(text)}
        
        logger.info(f"📦 After Text Input creation: {len(ComponentRegistry._components)}")
        
        # List all registered components
        components = list(ComponentRegistry._components.keys())
        logger.info(f"✅ Registered components: {components}")
        
        return len(ComponentRegistry._components) > 0
        
    except Exception as e:
        logger.error(f"❌ Component import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting LangChain Drag-and-Drop Platform...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    
    # Test imports first
    if test_component_imports():
        logger.info("✅ Component import test passed - starting server")
    else:
        logger.error("❌ Component import test failed - server may not work properly")
    
    # Run with uvicorn pointing to the module path
    uvicorn.run(
        "src.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to preserve registrations
        log_level="info",
        access_log=True
    )