#!/usr/bin/env python3
"""
Test script for the corrected flow execution
"""
import asyncio
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.services.flow_executor import FlowExecutor
from backend.models.flow import FlowDefinition, FlowNode, FlowEdge
from backend.core.registry import ComponentRegistry

async def test_flow_execution():
    """Test the corrected flow execution"""
    
    # Import components to ensure registration
    from backend.main import _register_missing_components
    _register_missing_components()
    
    # Create flow definition
    flow_definition = FlowDefinition(
        id="ai-agent-flow",
        name="AI Agent with Tools",
        description="Complete AI agent with tools",
        nodes=[
            FlowNode(
                id="input-1",
                component_type="Text Input",
                position={"x": 100, "y": 100},
                data={
                    "placeholder": "Ask me anything...",
                    "text": "What is the weather today?"
                }
            ),
            FlowNode(
                id="chat-model-1",
                component_type="Chat Model",
                position={"x": 300, "y": 100},
                data={
                    "provider": "fake",  # Use fake for testing
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            ),
            FlowNode(
                id="web-search-1",
                component_type="Web Search Tool",
                position={"x": 100, "y": 200},
                data={
                    "search_provider": "ddg",
                    "num_results": 5
                }
            ),
            FlowNode(
                id="agent-1",
                component_type="OpenAI Functions Agent",
                position={"x": 500, "y": 150},
                data={
                    "system_message": "You are a helpful AI assistant",
                    "max_iterations": 10
                }
            ),
            FlowNode(
                id="executor-1",
                component_type="Agent Executor",
                position={"x": 700, "y": 150},
                data={
                    "return_intermediate_steps": True
                }
            )
        ],
        edges=[
            FlowEdge(
                id="edge-1",
                source="chat-model-1",
                target="agent-1",
                source_handle="message_object",
                target_handle="llm"
            ),
            FlowEdge(
                id="edge-2",
                source="web-search-1",
                target="agent-1",
                source_handle="tool",
                target_handle="tools"
            ),
            FlowEdge(
                id="edge-3",
                source="agent-1",
                target="executor-1",
                source_handle="agent",
                target_handle="agent"
            ),
            FlowEdge(
                id="edge-4",
                source="input-1",
                target="executor-1",
                source_handle="text",
                target_handle="input_query"
            )
        ]
    )
    
    # Create flow executor
    executor = FlowExecutor()
    
    # Test flow validation
    print("Testing flow validation...")
    validation_result = await executor.validate_flow(flow_definition)
    print(f"Validation result: {json.dumps(validation_result, indent=2)}")
    
    if validation_result["valid"]:
        print("\n✅ Flow validation passed!")
        
        # Test flow execution
        print("\nTesting flow execution...")
        execution_result = await executor.execute_flow(
            flow_definition, 
            inputs={"user_input": "What is the weather today?"}
        )
        
        print(f"Execution result: {json.dumps(execution_result, indent=2)}")
        
        if execution_result["success"]:
            print("\n✅ Flow execution succeeded!")
        else:
            print(f"\n❌ Flow execution failed: {execution_result.get('error')}")
    else:
        print(f"\n❌ Flow validation failed: {validation_result['errors']}")
    
    # Print registered components
    print(f"\nRegistered components: {len(ComponentRegistry._components)}")
    for name in ComponentRegistry._components.keys():
        print(f"  - {name}")

if __name__ == "__main__":
    asyncio.run(test_flow_execution())