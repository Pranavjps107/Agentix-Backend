# src/backend/components/vectorstores/vectorstore.py
"""
Vector Store Components
"""
from typing import Dict, Any, List, Optional
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component

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
                options=["chroma", "pinecone", "weaviate", "qdrant", "faiss", "in_memory"],
                default="in_memory",
                description="Vector store provider"
            ),
            ComponentInput(
                name="collection_name",
                display_name="Collection Name",
                field_type="str",
                default="default_collection",
                description="Name of the vector collection"
            ),
            ComponentInput(
                name="embeddings",
                display_name="Embeddings Model",
                field_type="embeddings",
                required=False,
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
                method="create_vectorstore",
                description="Created vector store instance"
            ),
            ComponentOutput(
                name="document_count",
                display_name="Document Count",
                field_type="int",
                method="get_document_count",
                description="Number of documents in the store"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        provider = kwargs.get("provider", "in_memory")
        collection_name = kwargs.get("collection_name", "default_collection")
        embeddings = kwargs.get("embeddings")
        connection_string = kwargs.get("connection_string")
        documents = kwargs.get("documents", [])
        
        # Create vector store instance
        try:
            vectorstore = self._create_vectorstore_instance(
                provider, collection_name, embeddings, connection_string
            )
            
            # Add documents if provided
            document_count = 0
            if documents and vectorstore:
                from langchain_core.documents import Document
                doc_objects = []
                for doc in documents:
                    if isinstance(doc, dict):
                        doc_objects.append(Document(
                            page_content=doc.get("content", ""),
                            metadata=doc.get("metadata", {})
                        ))
                    elif isinstance(doc, str):
                        doc_objects.append(Document(page_content=doc))
                
                if doc_objects and hasattr(vectorstore, 'add_documents'):
                    await vectorstore.aadd_documents(doc_objects)
                    document_count = len(doc_objects)
            
            return {
                "vectorstore": vectorstore,
                "collection_name": collection_name,
                "document_count": document_count,
                "provider": provider
            }
            
        except Exception as e:
            # Return mock vector store for demo
            return {
                "vectorstore": None,
                "collection_name": collection_name,
                "document_count": len(documents),
                "provider": provider,
                "error": f"Vector store creation failed: {str(e)}"
            }
    
    def _create_vectorstore_instance(self, provider: str, collection_name: str, 
                                   embeddings, connection_string: str):
        """Create vector store instance based on provider"""
        
        if provider == "in_memory":
            # Create a simple in-memory vector store
            from langchain_core.vectorstores import InMemoryVectorStore
            if embeddings:
                return InMemoryVectorStore(embeddings)
            else:
                # Use fake embeddings for demo
                from langchain_core.embeddings.fake import FakeEmbeddings
                fake_embeddings = FakeEmbeddings(size=1536)
                return InMemoryVectorStore(fake_embeddings)
        
        elif provider == "chroma":
            try:
                from langchain_chroma import Chroma
                return Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=connection_string or "./chroma_db"
                )
            except ImportError:
                raise ImportError("langchain-chroma package required for Chroma vector store")
        
        elif provider == "pinecone":
            try:
                from langchain_pinecone import PineconeVectorStore
                return PineconeVectorStore(
                    index_name=collection_name,
                    embedding=embeddings
                )
            except ImportError:
                raise ImportError("langchain-pinecone package required for Pinecone vector store")
        
        elif provider == "qdrant":
            try:
                from langchain_qdrant import QdrantVectorStore
                return QdrantVectorStore.from_existing_collection(
                    embedding=embeddings,
                    collection_name=collection_name,
                    url=connection_string
                )
            except ImportError:
                raise ImportError("langchain-qdrant package required for Qdrant vector store")
        
        elif provider == "faiss":
            try:
                from langchain_community.vectorstores import FAISS
                # For FAISS, we need to create it differently
                if embeddings:
                    return FAISS.from_texts([""], embeddings)
                else:
                    from langchain_core.embeddings.fake import FakeEmbeddings
                    fake_embeddings = FakeEmbeddings(size=1536)
                    return FAISS.from_texts([""], fake_embeddings)
            except ImportError:
                raise ImportError("faiss-cpu package required for FAISS vector store")
        
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
                method="retrieve_documents",
                description="Retrieved documents from vector store"
            ),
            ComponentOutput(
                name="scores",
                display_name="Similarity Scores",
                field_type="list",
                method="get_scores",
                description="Similarity scores for retrieved documents"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        vectorstore = kwargs.get("vectorstore")
        query = kwargs.get("query")
        k = kwargs.get("k", 4)
        search_type = kwargs.get("search_type", "similarity")
        
        if not vectorstore:
            # Return mock results for demo
            return {
                "documents": [
                    {
                        "content": f"Mock document 1 for query: {query}",
                        "metadata": {"source": "mock", "score": 0.9}
                    },
                    {
                        "content": f"Mock document 2 for query: {query}",
                        "metadata": {"source": "mock", "score": 0.8}
                    }
                ],
                "scores": [0.9, 0.8],
                "query": query,
                "count": 2
            }
        
        try:
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k}
            )
            
            # Retrieve documents
            if hasattr(retriever, 'aget_relevant_documents'):
                documents = await retriever.aget_relevant_documents(query)
            else:
                import asyncio
                documents = await asyncio.to_thread(retriever.get_relevant_documents, query)
            
            # Get similarity scores if available
            scores = []
            if hasattr(vectorstore, 'similarity_search_with_score'):
                try:
                    if hasattr(vectorstore, 'asimilarity_search_with_score'):
                        scored_docs = await vectorstore.asimilarity_search_with_score(query, k=k)
                    else:
                        import asyncio
                        scored_docs = await asyncio.to_thread(
                            vectorstore.similarity_search_with_score, query, k
                        )
                    scores = [score for _, score in scored_docs]
                except:
                    scores = [0.5] * len(documents)  # Default scores
            
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
            
        except Exception as e:
            return {
                "documents": [],
                "scores": [],
                "query": query,
                "count": 0,
                "error": f"Retrieval failed: {str(e)}"
            }