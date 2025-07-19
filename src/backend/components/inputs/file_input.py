"""
File Input Component
"""
from typing import Dict, Any
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component

@register_component
class FileInputComponent(BaseLangChainComponent):
    """File Input Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="File Input", 
            description="Upload and process files",
            icon="📁",
            category="inputs",
            tags=["input", "file", "upload"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="file_path",
                display_name="File Path",
                field_type="str",
                description="Path to the file to process"
            ),
            ComponentInput(
                name="file_type",
                display_name="File Type",
                field_type="str",
                options=["txt", "pdf", "csv", "json", "md"],
                description="Type of file to process"
            ),
            ComponentInput(
                name="encoding",
                display_name="Encoding",
                field_type="str",
                default="utf-8",
                required=False,
                description="File encoding"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="content",
                display_name="File Content",
                field_type="str",
                method="get_content",
                description="Content of the file"
            ),
            ComponentOutput(
                name="file_info",
                display_name="File Information",
                field_type="dict",
                method="get_file_info", 
                description="File metadata and information"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        file_path = kwargs.get("file_path", "")
        file_type = kwargs.get("file_type", "txt")
        encoding = kwargs.get("encoding", "utf-8")
        
        try:
            import os
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read file content based on type
            if file_type in ["txt", "md", "json"]:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
            else:
                content = f"File type {file_type} not yet supported for reading"
            
            file_stats = os.stat(file_path)
            file_info = {
                "file_path": file_path,
                "file_type": file_type,
                "size_bytes": file_stats.st_size,
                "encoding": encoding
            }
            
            return {
                "content": content,
                "file_info": file_info,
                "success": True
            }
            
        except Exception as e:
            return {
                "content": "",
                "file_info": {"error": str(e)},
                "success": False
            }