# src/backend/components/vectorstores/pinecone.py
"""
Pinecone Vector Store Component
"""
from typing import Dict, Any
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component

@register_component
class PineconeVectorStoreComponent(BaseLangChainComponent):
    """Pinecone Vector Store Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Pinecone Vector Store",
            description="Pinecone cloud vector database",
            icon="🌲",
            category="vectorstores",
            tags=["pinecone", "vectors", "cloud"]
        )
        
        self.inputs = [
            ComponentInput(
                name="index_name",
                display_name="Index Name",
                field_type="str",
                description="Name of the Pinecone index"
            ),
            ComponentInput(
                name="api_key",
                display_name="Pinecone API Key",
                field_type="str",
                password=True,
                required=False,
                description="Pinecone API key"
            ),
            ComponentInput(
                name="environment",
                display_name="Environment",
                field_type="str",
                required=False,
                description="Pinecone environment"
            ),
            ComponentInput(
                name="embeddings",
                display_name="Embeddings Model",
                field_type="embeddings",
                required=False,
                description="Embeddings model for the vector store"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="vectorstore",
                display_name="Pinecone Vector Store",
                field_type="vectorstore",
                method="create_pinecone_store",
                description="Pinecone vector store instance"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        index_name = kwargs.get("index_name")
        api_key = kwargs.get("api_key")
        environment = kwargs.get("environment")
        embeddings = kwargs.get("embeddings")
        
        try:
            from langchain_pinecone import PineconeVectorStore
            
            if not embeddings:
                from langchain_core.embeddings.fake import FakeEmbeddings
                embeddings = FakeEmbeddings(size=1536)
            
            # Set up Pinecone
            import os
            if api_key:
                os.environ["PINECONE_API_KEY"] = api_key
            if environment:
                os.environ["PINECONE_ENVIRONMENT"] = environment
            
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings
            )
            
            return {
                "vectorstore": vectorstore,
                "index_name": index_name,
                "environment": environment
            }
            
        except ImportError:
            return {
                "vectorstore": None,
                "error": "langchain-pinecone package not installed",
                "index_name": index_name
            }
        except Exception as e:
            return {
                "vectorstore": None,
                "error": f"Failed to create Pinecone store: {str(e)}",
                "index_name": index_name
            }