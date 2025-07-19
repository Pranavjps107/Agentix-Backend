# src/backend/components/document_loaders/loaders.py
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, JSONLoader,
    WebBaseLoader, YoutubeLoader, GitHubIssuesLoader
)
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component
from typing import Dict, Type,List , Any
import asyncio

@register_component
class TextLoaderComponent(BaseLangChainComponent):
    """Text File Loader Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Text Loader",
            description="Load text files as documents",
            icon="📄",
            category="document_loaders",
            tags=["loader", "text", "file"]
        )
        
        self.inputs = [
            ComponentInput(
                name="file_path",
                display_name="File Path",
                field_type="str",
                description="Path to the text file"
            ),
            ComponentInput(
                name="encoding",
                display_name="Encoding",
                field_type="str",
                default="utf-8",
                required=False,
                description="File encoding"
            ),
            ComponentInput(
                name="autodetect_encoding",
                display_name="Auto-detect Encoding",
                field_type="bool",
                default=False,
                required=False,
                description="Automatically detect file encoding"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="documents",
                display_name="Documents",
                field_type="list",
                method="load_documents"
            ),
            ComponentOutput(
                name="document_count",
                display_name="Document Count",
                field_type="int",
                method="count_documents"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        file_path = kwargs.get("file_path")
        encoding = kwargs.get("encoding", "utf-8")
        autodetect_encoding = kwargs.get("autodetect_encoding", False)
        
        # Create text loader
        loader = TextLoader(
            file_path=file_path,
            encoding=encoding,
            autodetect_encoding=autodetect_encoding
        )
        
        # Load documents
        try:
            documents = await asyncio.to_thread(loader.load)
            
            # Convert to serializable format
            doc_list = []
            for doc in documents:
                doc_list.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            
        except Exception as e:
            doc_list = []
            error_doc = {
                "page_content": f"Error loading file: {str(e)}",
                "metadata": {"error": True, "file_path": file_path}
            }
            doc_list.append(error_doc)
        
        return {
            "documents": doc_list,
            "document_count": len(doc_list),
            "file_path": file_path,
            "encoding": encoding
        }

@register_component
class PDFLoaderComponent(BaseLangChainComponent):
    """PDF Loader Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="PDF Loader",
            description="Load PDF files as documents",
            icon="📕",
            category="document_loaders",
            tags=["loader", "pdf", "file"]
        )
        
        self.inputs = [
            ComponentInput(
                name="file_path",
                display_name="File Path",
                field_type="str",
                description="Path to the PDF file"
            ),
            ComponentInput(
                name="extract_images",
                display_name="Extract Images",
                field_type="bool",
                default=False,
                required=False,
                description="Extract images from PDF"
            ),
            ComponentInput(
                name="pages_per_doc",
                display_name="Pages per Document",
                field_type="int",
                default=1,
                required=False,
                description="Number of pages per document chunk"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="documents",
                display_name="Documents",
                field_type="list",
                method="load_pdf"
            ),
            ComponentOutput(
                name="total_pages",
                display_name="Total Pages",
                field_type="int",
                method="count_pages"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        file_path = kwargs.get("file_path")
        extract_images = kwargs.get("extract_images", False)
        pages_per_doc = kwargs.get("pages_per_doc", 1)
        
        # Create PDF loader
        loader = PyPDFLoader(file_path=file_path, extract_images=extract_images)
        
        # Load documents
        try:
            documents = await asyncio.to_thread(loader.load)
            
            # Group pages if needed
            if pages_per_doc > 1:
                grouped_docs = []
                for i in range(0, len(documents), pages_per_doc):
                    chunk = documents[i:i + pages_per_doc]
                    combined_content = "\n\n".join([doc.page_content for doc in chunk])
                    combined_metadata = chunk[0].metadata.copy()
                    combined_metadata.update({
                        "pages": [doc.metadata.get("page", i) for doc in chunk],
                        "page_range": f"{chunk[0].metadata.get('page', i)}-{chunk[-1].metadata.get('page', i + len(chunk) - 1)}"
                    })
                    grouped_docs.append({
                       "page_content": combined_content,
                       "metadata": combined_metadata
                   })
                documents = grouped_docs
            else:
                # Convert to serializable format
                documents = [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
           
        except Exception as e:
            documents = [{
                "page_content": f"Error loading PDF: {str(e)}",
                "metadata": {"error": True, "file_path": file_path}
            }]
        
        return {
            "documents": documents,
            "total_pages": len(documents),
            "file_path": file_path,
            "pages_per_doc": pages_per_doc
        }

@register_component
class WebLoaderComponent(BaseLangChainComponent):
   """Web Page Loader Component"""
   
   def _setup_component(self):
       self.metadata = ComponentMetadata(
           display_name="Web Loader",
           description="Load web pages as documents",
           icon="🌐",
           category="document_loaders",
           tags=["loader", "web", "url"]
       )
       
       self.inputs = [
           ComponentInput(
               name="urls",
               display_name="URLs",
               field_type="list",
               description="List of URLs to load"
           ),
           ComponentInput(
               name="header_template",
               display_name="Header Template",
               field_type="dict",
               required=False,
               description="HTTP headers for requests"
           ),
           ComponentInput(
               name="verify_ssl",
               display_name="Verify SSL",
               field_type="bool",
               default=True,
               required=False,
               description="Verify SSL certificates"
           ),
           ComponentInput(
               name="requests_per_second",
               display_name="Requests per Second",
               field_type="int",
               default=2,
               required=False,
               description="Rate limit for requests"
           )
       ]
       
       self.outputs = [
           ComponentOutput(
               name="documents",
               display_name="Documents",
               field_type="list",
               method="load_web_pages"
           ),
           ComponentOutput(
               name="failed_urls",
               display_name="Failed URLs",
               field_type="list",
               method="get_failed_urls"
           )
       ]
   
   async def execute(self, **kwargs) -> Dict[str, Any]:
       urls = kwargs.get("urls", [])
       header_template = kwargs.get("header_template", {})
       verify_ssl = kwargs.get("verify_ssl", True)
       requests_per_second = kwargs.get("requests_per_second", 2)
       
       # Ensure urls is a list
       if isinstance(urls, str):
           urls = [urls]
       
       # Create web loader
       loader = WebBaseLoader(
           web_paths=urls,
           header_template=header_template,
           verify_ssl=verify_ssl,
           requests_per_second=requests_per_second
       )
       
       # Load documents
       documents = []
       failed_urls = []
       
       try:
           loaded_docs = await asyncio.to_thread(loader.load)
           
           for doc in loaded_docs:
               documents.append({
                   "page_content": doc.page_content,
                   "metadata": doc.metadata
               })
               
       except Exception as e:
           failed_urls = urls
           documents = [{
               "page_content": f"Error loading web pages: {str(e)}",
               "metadata": {"error": True, "urls": urls}
           }]
       
       return {
           "documents": documents,
           "failed_urls": failed_urls,
           "total_urls": len(urls),
           "successful_loads": len(documents) - len(failed_urls)
       }

@register_component
class CSVLoaderComponent(BaseLangChainComponent):
   """CSV Loader Component"""
   
   def _setup_component(self):
       self.metadata = ComponentMetadata(
           display_name="CSV Loader",
           description="Load CSV files as documents",
           icon="📊",
           category="document_loaders",
           tags=["loader", "csv", "data"]
       )
       
       self.inputs = [
           ComponentInput(
               name="file_path",
               display_name="File Path",
               field_type="str",
               description="Path to the CSV file"
           ),
           ComponentInput(
               name="csv_args",
               display_name="CSV Arguments",
               field_type="dict",
               required=False,
               description="Arguments for CSV reader (delimiter, quotechar, etc.)"
           ),
           ComponentInput(
               name="content_columns",
               display_name="Content Columns",
               field_type="list",
               required=False,
               description="Columns to use as document content"
           ),
           ComponentInput(
               name="metadata_columns",
               display_name="Metadata Columns",
               field_type="list",
               required=False,
               description="Columns to use as metadata"
           )
       ]
       
       self.outputs = [
           ComponentOutput(
               name="documents",
               display_name="Documents",
               field_type="list",
               method="load_csv"
           ),
           ComponentOutput(
               name="row_count",
               display_name="Row Count",
               field_type="int",
               method="count_rows"
           )
       ]
   
   async def execute(self, **kwargs) -> Dict[str, Any]:
       file_path = kwargs.get("file_path")
       csv_args = kwargs.get("csv_args", {})
       content_columns = kwargs.get("content_columns")
       metadata_columns = kwargs.get("metadata_columns")
       
       # Create CSV loader
       loader = CSVLoader(
           file_path=file_path,
           csv_args=csv_args,
           content_columns=content_columns,
           metadata_columns=metadata_columns
       )
       
       # Load documents
       try:
           documents = await asyncio.to_thread(loader.load)
           
           doc_list = []
           for doc in documents:
               doc_list.append({
                   "page_content": doc.page_content,
                   "metadata": doc.metadata
               })
               
       except Exception as e:
           doc_list = [{
               "page_content": f"Error loading CSV: {str(e)}",
               "metadata": {"error": True, "file_path": file_path}
           }]
       
       return {
           "documents": doc_list,
           "row_count": len(doc_list),
           "file_path": file_path,
           "content_columns": content_columns
       }