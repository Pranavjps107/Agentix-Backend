# src/backend/components/retrievers/__init__.py
"""
Retriever Components
"""

from ...core.base import BaseLangChainComponent, ComponentMetadata
from ...core.registry import register_component

@register_component
class RetrieverComponent(BaseLangChainComponent):
    """Placeholder Retriever Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Retriever",
            description="Retriever component (placeholder)",
            icon="🔍",
            category="retrievers",
            tags=["retrieval"]
        )
        self.inputs = []
        self.outputs = []
    
    async def execute(self, **kwargs):
        return {"message": "Retriever component not implemented"}

__all__ = ["RetrieverComponent"]