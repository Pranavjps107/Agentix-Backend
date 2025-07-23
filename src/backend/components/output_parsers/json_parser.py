"""
Enhanced JSON Output Parser for Structured Data
"""
from typing import Dict, Any, Optional
import json
import logging
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component

logger = logging.getLogger(__name__)

@register_component
class StructuredJSONParserComponent(BaseLangChainComponent):
    """Parse LLM output into structured JSON with validation"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Structured JSON Parser",
            description="Parse LLM output into validated JSON structure",
            icon="📋",
            category="output_parsers",
            tags=["json", "parser", "structured", "validation"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="llm_output",
                display_name="LLM Output",
                field_type="str",
                description="Text output from LLM to parse as JSON"
            ),
            ComponentInput(
                name="schema",
                display_name="Expected Schema",
                field_type="dict",
                required=False,
                description="JSON schema to validate against"
            ),
            ComponentInput(
                name="extract_json_only",
                display_name="Extract JSON Only",
                field_type="bool",
                default=True,
                required=False,
                description="Extract only JSON content from mixed text"
            ),
            ComponentInput(
                name="strict_validation",
                display_name="Strict Validation",
                field_type="bool",
                default=False,
                required=False,
                description="Strict schema validation"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="parsed_json",
                display_name="Parsed JSON",
                field_type="dict",
                method="parse_json",
                description="Successfully parsed JSON object"
            ),
            ComponentOutput(
                name="validation_errors",
                display_name="Validation Errors",
                field_type="list",
                method="get_validation_errors",
                description="List of validation errors if any"
            ),
            ComponentOutput(
                name="parsing_success",
                display_name="Parsing Success",
                field_type="bool",
                method="is_parsing_successful",
                description="Whether parsing was successful"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        llm_output = kwargs.get("llm_output", "")
        schema = kwargs.get("schema")
        extract_json_only = kwargs.get("extract_json_only", True)
        strict_validation = kwargs.get("strict_validation", False)
        
        if not llm_output.strip():
            return {
                "parsed_json": {},
                "validation_errors": ["Empty input provided"],
                "parsing_success": False
            }
        
        try:
            # Extract JSON from mixed content if needed
            if extract_json_only:
                json_content = self._extract_json_content(llm_output)
            else:
                json_content = llm_output
            
            # Parse JSON
            parsed_json = json.loads(json_content)
            
            # Validate against schema if provided
            validation_errors = []
            if schema and strict_validation:
                validation_errors = self._validate_against_schema(parsed_json, schema)
            
            parsing_success = len(validation_errors) == 0
            
            logger.info(f"JSON parsing {'successful' if parsing_success else 'failed'}")
            
            return {
                "parsed_json": parsed_json,
                "validation_errors": validation_errors,
                "parsing_success": parsing_success
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            return {
                "parsed_json": {},
                "validation_errors": [f"JSON decode error: {str(e)}"],
                "parsing_success": False
            }
        except Exception as e:
            logger.error(f"Unexpected parsing error: {str(e)}")
            return {
                "parsed_json": {},
                "validation_errors": [f"Parsing error: {str(e)}"],
                "parsing_success": False
            }
    
    def _extract_json_content(self, text: str) -> str:
        """Extract JSON content from mixed text"""
        import re
        
        # Look for JSON content between ```json and ``` 
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Look for JSON content between { and }
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)
        
        # Look for JSON content between [ and ]
        bracket_match = re.search(r'\[.*\]', text, re.DOTALL)
        if bracket_match:
            return bracket_match.group(0)
        
        # Return original text if no JSON pattern found
        return text
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Basic schema validation"""
        errors = []
        
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in data:
                expected_type = field_schema.get("type")
                actual_value = data[field]
                
                if expected_type == "string" and not isinstance(actual_value, str):
                    errors.append(f"Field '{field}' should be string, got {type(actual_value).__name__}")
                elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                    errors.append(f"Field '{field}' should be number, got {type(actual_value).__name__}")
                elif expected_type == "array" and not isinstance(actual_value, list):
                    errors.append(f"Field '{field}' should be array, got {type(actual_value).__name__}")
                elif expected_type == "object" and not isinstance(actual_value, dict):
                    errors.append(f"Field '{field}' should be object, got {type(actual_value).__name__}")
        
        return errors