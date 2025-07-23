# src/backend/components/vectorstores/chroma.py
"""
Chroma Vector Store Component
"""
from typing import Dict, Any
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component

@register_component
class ChromaVectorStoreComponent(BaseLangChainComponent):
    """Chroma Vector Store Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Chroma Vector Store",
            description="Chroma vector database for embeddings",
            icon="🟢",
            category="vectorstores",
            tags=["chroma", "vectors", "embeddings"]
        )
        
        self.inputs = [
            ComponentInput(
                name="collection_name",
                display_name="Collection Name",
                field_type="str",
                default="default_collection",
                description="Name of the Chroma collection"
            ),
            ComponentInput(
                name="persist_directory",
                display_name="Persist Directory",
                field_type="str",
                default="./chroma_db",
                required=False,
                description="Directory to persist the database"
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
                display_name="Chroma Vector Store",
                field_type="vectorstore",
                method="create_chroma_store",
                description="Chroma vector store instance"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        collection_name = kwargs.get("collection_name", "default_collection")
        persist_directory = kwargs.get("persist_directory", "./chroma_db")
        embeddings = kwargs.get("embeddings")
        
        try:
            from langchain_chroma import Chroma
            
            if not embeddings:
                from langchain_core.embeddings.fake import FakeEmbeddings
                embeddings = FakeEmbeddings(size=1536)
            
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            
            return {
                "vectorstore": vectorstore,
                "collection_name": collection_name,
                "persist_directory": persist_directory
            }
            
        except ImportError:
            return {
                "vectorstore": None,
                "error": "langchain-chroma package not installed",
                "collection_name": collection_name
            }
        except Exception as e:
            return {
                "vectorstore": None,
                "error": f"Failed to create Chroma store: {str(e)}",
                "collection_name": collection_name
            }