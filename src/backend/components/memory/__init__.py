# src/backend/components/memory/__init__.py
"""
Memory Components
"""

from ...core.base import BaseLangChainComponent, ComponentMetadata
from ...core.registry import register_component

@register_component
class MemoryComponent(BaseLangChainComponent):
    """Placeholder Memory Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Memory",
            description="Memory component (placeholder)",
            icon="🧠",
            category="memory",
            tags=["memory"]
        )
        self.inputs = []
        self.outputs = []
    
    async def execute(self, **kwargs):
        return {"message": "Memory component not implemented"}

__all__ = ["MemoryComponent"]