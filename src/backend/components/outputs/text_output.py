# src/backend/components/outputs/text_output.py
"""
Text Output Component
"""
from typing import Dict, Any
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component  # Correct import location

@register_component
class TextOutputComponent(BaseLangChainComponent):
    """Text Output Component for displaying results"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Agent Executor",
            description="Execute agent with input query",
            icon="⚡",
            category="agents",
            tags=["agent", "executor", "run"]
        )
        
        self.inputs = [
            ComponentInput(
                name="agent_executor",
                display_name="Agent Executor",
                field_type="agent_executor",
                description="Configured agent executor"
            ),
            ComponentInput(
                name="input_query",
                display_name="Input Query",
                field_type="str",
                description="Query to send to the agent"
            ),
            ComponentInput(
                name="chat_history",
                display_name="Chat History",
                field_type="list",
                required=False,
                description="Previous conversation history"
            ),
            ComponentInput(
                name="return_intermediate_steps",
                display_name="Return Intermediate Steps",
                field_type="bool",
                default=True,
                required=False,
                description="Return reasoning steps"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="response",
                display_name="Agent Response",
                field_type="str",
                method="execute_agent"
            ),
            ComponentOutput(
                name="intermediate_steps",
                display_name="Intermediate Steps",
                field_type="list",
                method="get_steps"
            ),
            ComponentOutput(
                name="execution_metadata",
                display_name="Execution Metadata",
                field_type="dict",
                method="get_metadata"
            ),
            # ADD THIS NEW OUTPUT
            ComponentOutput(
                name="updated_chat_history",
                display_name="Updated Chat History",
                field_type="list",
                method="get_updated_history"
            )
        ]

    # ADD THESE HELPER METHODS
    def get_updated_history(self, **kwargs):
        """Helper method for updated_chat_history output"""
        return kwargs.get("updated_chat_history", [])

    async def execute(self, **kwargs) -> Dict[str, Any]:
        text = kwargs.get("text", "")
        format_type = kwargs.get("format", "plain")
        
        # Format the text based on type
        if format_type == "markdown":
            formatted_text = f"```\n{text}\n```"
        elif format_type == "html":
            formatted_text = f"<div class='output'>{text}</div>"
        else:
            formatted_text = text
        
        metadata = {
            "format": format_type,
            "length": len(text),
            "word_count": len(text.split()) if text else 0,
            "line_count": len(text.split('\n')) if text else 0
        }
        
        return {
            "formatted_text": formatted_text,
            "metadata": metadata
        }