"""
Fake LLM Component for Testing
"""
from typing import Dict, Any
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component

@register_component
class FakeLLMComponent(BaseLangChainComponent):
    """Fake LLM Component for testing and development"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Fake LLM",
            description="Fake LLM for testing and development purposes",
            icon="🎭",
            category="language_models",
            tags=["fake", "testing", "development", "mock"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="prompt",
                display_name="Prompt",
                field_type="str",
                multiline=True,
                description="Input prompt"
            ),
            ComponentInput(
                name="responses",
                display_name="Fake Responses",
                field_type="list",
                default=["This is a fake response", "Another fake response"],
                required=False,
                description="List of responses to cycle through"
            ),
            ComponentInput(
                name="delay",
                display_name="Response Delay (seconds)",
                field_type="float",
                default=1.0,
                required=False,
                description="Artificial delay to simulate API call"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="response",
                display_name="Fake Response",
                field_type="str",
                method="generate_fake_response",
                description="Fake generated response"
            ),
            ComponentOutput(
                name="usage",
                display_name="Fake Usage",
                field_type="dict",
                method="get_fake_usage",
                description="Fake token usage"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        import asyncio
        import random
        
        prompt = kwargs.get("prompt", "")
        responses = kwargs.get("responses", ["This is a fake response"])
        delay = kwargs.get("delay", 1.0)
        
        # Simulate API delay
        await asyncio.sleep(delay)
        
        # Choose a random response
        response = random.choice(responses)
        
        # Add prompt context to response
        if prompt:
            response = f"Fake response to '{prompt[:50]}...': {response}"
        
        # Fake usage statistics
        usage = {
            "prompt_tokens": len(prompt.split()) if prompt else 0,
            "completion_tokens": len(response.split()),
            "total_tokens": len(prompt.split()) + len(response.split()) if prompt else len(response.split())
        }
        
        return {
            "response": response,
            "usage": usage
        }