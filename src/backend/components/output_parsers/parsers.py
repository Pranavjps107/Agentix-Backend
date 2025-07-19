# src/backend/components/output_parsers/parsers.py
from langchain_core.output_parsers import (
    BaseOutputParser, StrOutputParser, JsonOutputParser, 
    ListOutputParser, PydanticOutputParser
)
from langchain_core.output_parsers.openai_functions import (
    JsonOutputFunctionsParser, PydanticOutputFunctionsParser
)
from pydantic import BaseModel
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component
from typing import Dict, Type,List , Any
@register_component
class StringOutputParserComponent(BaseLangChainComponent):
    """String Output Parser Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="String Output Parser",
            description="Parse LLM output as string",
            icon="📄",
            category="output_parsers",
            tags=["parser", "string", "text"]
        )
        
        self.inputs = [
            ComponentInput(
                name="llm_output",
                display_name="LLM Output",
                field_type="str",
                description="Output from language model to parse"
            ),
            ComponentInput(
                name="strip_whitespace",
                display_name="Strip Whitespace",
                field_type="bool",
                default=True,
                required=False,
                description="Remove leading/trailing whitespace"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="parsed_output",
                display_name="Parsed String",
                field_type="str",
                method="parse_string"
            ),
            ComponentOutput(
                name="parser",
                display_name="Parser Object",
                field_type="output_parser",
                method="create_parser"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        llm_output = kwargs.get("llm_output", "")
        strip_whitespace = kwargs.get("strip_whitespace", True)
        
        # Create string parser
        parser = StrOutputParser()
        
        # Parse output
        parsed_output = parser.parse(llm_output)
        
        if strip_whitespace:
            parsed_output = parsed_output.strip()
        
        return {
            "parsed_output": parsed_output,
            "parser": parser,
            "original_length": len(llm_output),
            "parsed_length": len(parsed_output)
        }

@register_component
class JsonOutputParserComponent(BaseLangChainComponent):
    """JSON Output Parser Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="JSON Output Parser",
            description="Parse LLM output as JSON",
            icon="📋",
            category="output_parsers",
            tags=["parser", "json", "structured"]
        )
        
        self.inputs = [
            ComponentInput(
                name="llm_output",
                display_name="LLM Output",
                field_type="str",
                description="JSON string from language model"
            ),
            ComponentInput(
                name="pydantic_object",
                display_name="Pydantic Model",
                field_type="str",
                required=False,
                description="Pydantic model class for validation"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="parsed_json",
                display_name="Parsed JSON",
                field_type="dict",
                method="parse_json"
            ),
            ComponentOutput(
                name="parser",
                display_name="Parser Object",
                field_type="output_parser",
                method="create_parser"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        llm_output = kwargs.get("llm_output", "")
        pydantic_object = kwargs.get("pydantic_object")
        
        # Create JSON parser
        if pydantic_object:
            # Use Pydantic parser if model provided
            parser = PydanticOutputParser(pydantic_object=pydantic_object)
        else:
            parser = JsonOutputParser()
        
        # Parse output
        try:
            parsed_json = parser.parse(llm_output)
        except Exception as e:
            parsed_json = {"error": f"Failed to parse JSON: {str(e)}"}
        
        return {
            "parsed_json": parsed_json,
            "parser": parser,
            "is_valid": "error" not in parsed_json
        }

@register_component
class ListOutputParserComponent(BaseLangChainComponent):
    """List Output Parser Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="List Output Parser",
            description="Parse LLM output as list",
            icon="📝",
            category="output_parsers",
            tags=["parser", "list", "array"]
        )
        
        self.inputs = [
            ComponentInput(
                name="llm_output",
                display_name="LLM Output",
                field_type="str",
                description="List string from language model"
            ),
            ComponentInput(
                name="separator",
                display_name="Separator",
                field_type="str",
                default=",",
                required=False,
                description="Separator for list items"
            ),
            ComponentInput(
                name="strip_items",
                display_name="Strip Items",
                field_type="bool",
                default=True,
                required=False,
                description="Strip whitespace from items"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="parsed_list",
                display_name="Parsed List",
                field_type="list",
                method="parse_list"
            ),
            ComponentOutput(
                name="parser",
                display_name="Parser Object",
                field_type="output_parser",
                method="create_parser"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        llm_output = kwargs.get("llm_output", "")
        separator = kwargs.get("separator", ",")
        strip_items = kwargs.get("strip_items", True)
        
        # Create list parser
        if separator == ",":
            from langchain_core.output_parsers import CommaSeparatedListOutputParser
            parser = CommaSeparatedListOutputParser()
        else:
            parser = ListOutputParser()
        
        # Parse output
        try:
            parsed_list = parser.parse(llm_output)
            if strip_items and isinstance(parsed_list, list):
                parsed_list = [item.strip() if isinstance(item, str) else item 
                              for item in parsed_list]
        except Exception as e:
            parsed_list = [f"Error parsing list: {str(e)}"]
        
        return {
            "parsed_list": parsed_list,
            "parser": parser,
            "item_count": len(parsed_list) if isinstance(parsed_list, list) else 0
        }