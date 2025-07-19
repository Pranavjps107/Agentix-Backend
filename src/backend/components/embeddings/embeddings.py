# src/backend/components/embeddings/embeddings.py
from langchain_core.embeddings import Embeddings
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component
from typing import Dict, Type,List , Any
@register_component
class EmbeddingsComponent(BaseLangChainComponent):
    """Embeddings Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Embeddings",
            description="Convert text to vector embeddings",
            icon="📊",
            category="embeddings",
            tags=["embeddings", "vectors"]
        )
        
        self.inputs = [
            ComponentInput(
                name="provider",
                display_name="Provider",
                field_type="str",
                options=["openai", "huggingface", "sentence_transformers", "cohere"],
                description="Embedding provider"
            ),
            ComponentInput(
                name="model_name",
                display_name="Model Name",
                field_type="str",
                description="Embedding model name"
            ),
            ComponentInput(
                name="texts",
                display_name="Input Texts",
                field_type="list",
                description="List of texts to embed"
            ),
            ComponentInput(
                name="batch_size",
                display_name="Batch Size",
                field_type="int",
                default=32,
                required=False,
                description="Batch size for processing"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="embeddings",
                display_name="Embeddings",
                field_type="list",
                method="generate_embeddings"
            ),
            ComponentOutput(
                name="dimensions",
                display_name="Dimensions",
                field_type="int",
                method="get_dimensions"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        provider = kwargs.get("provider")
        model_name = kwargs.get("model_name")
        texts = kwargs.get("texts", [])
        batch_size = kwargs.get("batch_size", 32)
        
        # Get embedding instance
        embedding_model = self._get_embedding_instance(provider, model_name)
        
        # Generate embeddings
        embeddings = await embedding_model.aembed_documents(texts)
        
        return {
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings)
        }
    
    def _get_embedding_instance(self, provider: str, model_name: str):
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_name)
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        elif provider == "sentence_transformers":
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            return SentenceTransformerEmbeddings(model_name=model_name)
        # Add more providers...
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")