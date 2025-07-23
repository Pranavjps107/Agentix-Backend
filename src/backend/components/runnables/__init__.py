# src/backend/components/runnables/__init__.py
"""
Runnable Components
"""

from ...core.base import BaseLangChainComponent, ComponentMetadata
from ...core.registry import register_component

@register_component
class RunnableComponent(BaseLangChainComponent):
    """Placeholder Runnable Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Runnable",
            description="Runnable component (placeholder)",
            icon="🏃",
            category="runnables",
            tags=["runnable"]
        )
        self.inputs = []
        self.outputs = []
    
    async def execute(self, **kwargs):
        return {"message": "Runnable component not implemented"}

__all__ = ["RunnableComponent"] 