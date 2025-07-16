"""
Base Embeddings Component
"""
from typing import Dict, Any, List, Optional
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component

@register_component
class EmbeddingsComponent(BaseLangChainComponent):
    """Base Embeddings Component for converting text to vectors"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Embeddings",
            description="Convert text to vector embeddings using various providers",
            icon="📊",
            category="embeddings",
            tags=["embeddings", "vectors", "similarity", "search"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="provider",
                display_name="Provider",
                field_type="str",
                options=["openai", "huggingface", "sentence_transformers", "cohere", "google", "fake"],
                default="openai",
                description="Embedding provider to use"
            ),
            ComponentInput(
                name="model_name",
                display_name="Model Name",
                field_type="str",
                default="text-embedding-ada-002",
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
                description="Batch size for processing multiple texts"
            ),
            ComponentInput(
                name="normalize_embeddings",
                display_name="Normalize Embeddings",
                field_type="bool",
                default=False,
                required=False,
                description="Whether to normalize embeddings to unit length"
            ),
            ComponentInput(
                name="api_key",
                display_name="API Key",
                field_type="str",
                required=False,
                password=True,
                description="API key for the provider"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="embeddings",
                display_name="Embeddings",
                field_type="list",
                method="generate_embeddings",
                description="List of embedding vectors"
            ),
            ComponentOutput(
                name="dimensions",
                display_name="Dimensions",
                field_type="int",
                method="get_dimensions",
                description="Number of dimensions in each embedding"
            ),
            ComponentOutput(
                name="model_info",
                display_name="Model Information",
                field_type="dict",
                method="get_model_info",
                description="Information about the embedding model"
            ),
            ComponentOutput(
                name="processing_stats",
                display_name="Processing Statistics",
                field_type="dict",
                method="get_processing_stats",
                description="Statistics about the embedding process"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        import time
        
        provider = kwargs.get("provider", "openai")
        model_name = kwargs.get("model_name", "text-embedding-ada-002")
        texts = kwargs.get("texts", [])
        batch_size = kwargs.get("batch_size", 32)
        normalize_embeddings = kwargs.get("normalize_embeddings", False)
        api_key = kwargs.get("api_key")
        
        if not texts:
            raise ValueError("At least one text is required for embedding")
        
        # Ensure texts is a list of strings
        if isinstance(texts, str):
            texts = [texts]
        
        text_list = []
        for text in texts:
            if isinstance(text, dict) and "content" in text:
                text_list.append(text["content"])
            elif isinstance(text, str):
                text_list.append(text)
            else:
                text_list.append(str(text))
        
        start_time = time.time()
        
        # Get embedding instance
        embedding_model = self._get_embedding_instance(provider, model_name, api_key)
        
        # Generate embeddings
        try:
            if hasattr(embedding_model, 'aembed_documents'):
                embeddings = await embedding_model.aembed_documents(text_list)
            else:
                import asyncio
                embeddings = await asyncio.to_thread(embedding_model.embed_documents, text_list)
            
            # Normalize if requested
            if normalize_embeddings and embeddings:
                import numpy as np
                embeddings = [
                    (np.array(emb) / np.linalg.norm(emb)).tolist()
                    for emb in embeddings
                ]
            
            processing_time = time.time() - start_time
            
            dimensions = len(embeddings[0]) if embeddings else 0
            
            model_info = {
                "provider": provider,
                "model_name": model_name,
                "normalized": normalize_embeddings
            }
            
            processing_stats = {
                "text_count": len(text_list),
                "batch_size": batch_size,
                "processing_time": processing_time,
                "dimensions": dimensions,
                "total_vectors": len(embeddings)
            }
            
            return {
                "embeddings": embeddings,
                "dimensions": dimensions,
                "model_info": model_info,
                "processing_stats": processing_stats
            }
            
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    def _get_embedding_instance(self, provider: str, model_name: str, api_key: Optional[str]):
        """Factory method to create embedding instances"""
        
        if provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                kwargs = {"model": model_name}
                if api_key:
                    kwargs["openai_api_key"] = api_key
                return OpenAIEmbeddings(**kwargs)
            except ImportError:
                raise ImportError("langchain-openai package required for OpenAI embeddings")
        
        elif provider == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'}
                )
            except ImportError:
                raise ImportError("langchain-huggingface package required for HuggingFace embeddings")
        
        elif provider == "sentence_transformers":
            try:
                from langchain_community.embeddings import SentenceTransformerEmbeddings
                return SentenceTransformerEmbeddings(model_name=model_name)
            except ImportError:
                raise ImportError("sentence-transformers package required for SentenceTransformers")
        
        elif provider == "cohere":
            try:
                from langchain_community.embeddings import CohereEmbeddings
                kwargs = {"model": model_name}
                if api_key:
                    kwargs["cohere_api_key"] = api_key
                return CohereEmbeddings(**kwargs)
            except ImportError:
                raise ImportError("cohere package required for Cohere embeddings")
        
        elif provider == "google":
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                kwargs = {"model": model_name}
                if api_key:
                    kwargs["google_api_key"] = api_key
                return GoogleGenerativeAIEmbeddings(**kwargs)
            except ImportError:
                raise ImportError("langchain-google-genai package required for Google embeddings")
        
        elif provider == "fake":
            from langchain_core.embeddings.fake import FakeEmbeddings
            return FakeEmbeddings(size=1536)  # OpenAI ada-002 size
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")