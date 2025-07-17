# src/backend/components/vectorstores/vectorstore.py
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component
from typing import Dict, Type,List , Any

@register_component
class VectorStoreComponent(BaseLangChainComponent):
    """Vector Store Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Vector Store",
            description="Store and retrieve vector embeddings",
            icon="🗃️",
            category="vectorstores",
            tags=["vectors", "storage", "search"]
        )
        
        self.inputs = [
            ComponentInput(
                name="provider",
                display_name="Provider",
                field_type="str",
                options=["chroma", "pinecone", "weaviate", "qdrant", "faiss", "elasticsearch"],
                description="Vector store provider"
            ),
            ComponentInput(
                name="collection_name",
                display_name="Collection Name",
                field_type="str",
                description="Name of the vector collection"
            ),
            ComponentInput(
                name="embeddings",
                display_name="Embeddings Model",
                field_type="embeddings",
                description="Embeddings model for the vector store"
            ),
            ComponentInput(
                name="connection_string",
                display_name="Connection String",
                field_type="str",
                required=False,
                description="Database connection string"
            ),
            ComponentInput(
                name="documents",
                display_name="Documents",
                field_type="list",
                required=False,
                description="Documents to add to the vector store"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="vectorstore",
                display_name="Vector Store",
                field_type="vectorstore",
                method="create_vectorstore"
            ),
            ComponentOutput(
                name="search_results",
                display_name="Search Results",
                field_type="list",
                method="search_documents"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        provider = kwargs.get("provider")
        collection_name = kwargs.get("collection_name")
        embeddings = kwargs.get("embeddings")
        connection_string = kwargs.get("connection_string")
        documents = kwargs.get("documents", [])
        
        # Create vector store instance
        vectorstore = self._create_vectorstore_instance(
            provider, collection_name, embeddings, connection_string
        )
        
        # Add documents if provided
        if documents:
            doc_objects = [Document(page_content=doc.get("content", ""), 
                                  metadata=doc.get("metadata", {})) 
                          for doc in documents]
            await vectorstore.aadd_documents(doc_objects)
        
        return {
            "vectorstore": vectorstore,
            "collection_name": collection_name,
            "document_count": len(documents)
        }
    
    def _create_vectorstore_instance(self, provider: str, collection_name: str, 
                                   embeddings, connection_string: str):
        if provider == "chroma":
            from langchain_chroma import Chroma
            return Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=connection_string or "./chroma_db"
            )
        elif provider == "pinecone":
            from langchain_pinecone import PineconeVectorStore
            return PineconeVectorStore(
                index_name=collection_name,
                embedding=embeddings
            )
        elif provider == "qdrant":
            from langchain_qdrant import QdrantVectorStore
            return QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=collection_name,
                url=connection_string
            )
        # Add more providers...
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")

@register_component
class VectorStoreRetrieverComponent(BaseLangChainComponent):
    """Vector Store Retriever Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Vector Store Retriever",
            description="Retrieve documents from vector store",
            icon="🔍",
            category="retrievers",
            tags=["retrieval", "search"]
        )
        
        self.inputs = [
            ComponentInput(
                name="vectorstore",
                display_name="Vector Store",
                field_type="vectorstore",
                description="Vector store to search in"
            ),
            ComponentInput(
                name="query",
                display_name="Query",
                field_type="str",
                description="Search query"
            ),
            ComponentInput(
                name="k",
                display_name="Number of Results",
                field_type="int",
                default=4,
                required=False,
                description="Number of documents to retrieve"
            ),
            ComponentInput(
                name="search_type",
                display_name="Search Type",
                field_type="str",
                options=["similarity", "mmr", "similarity_score_threshold"],
                default="similarity",
                required=False,
                description="Type of search to perform"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="documents",
                display_name="Retrieved Documents",
                field_type="list",
                method="retrieve_documents"
            ),
            ComponentOutput(
                name="scores",
                display_name="Similarity Scores",
                field_type="list",
                method="get_scores"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        vectorstore = kwargs.get("vectorstore")
        query = kwargs.get("query")
        k = kwargs.get("k", 4)
        search_type = kwargs.get("search_type", "similarity")
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        
        # Retrieve documents
        documents = await retriever.aget_relevant_documents(query)
        
        # Get similarity scores if available
        scores = []
        if hasattr(vectorstore, 'similarity_search_with_score'):
            scored_docs = await vectorstore.asimilarity_search_with_score(query, k=k)
            scores = [score for _, score in scored_docs]
        
        return {
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ],
            "scores": scores,
            "query": query,
            "count": len(documents)
        }