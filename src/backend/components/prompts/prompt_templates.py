# src/backend/components/prompts/prompt_templates.py
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component
from typing import Dict, Type,List , Any

@register_component
class PromptTemplateComponent(BaseLangChainComponent):
    """Prompt Template Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Prompt Template",
            description="Create and format prompt templates",
            icon="📝",
            category="prompts",
            tags=["prompts", "templates"]
        )
        
        self.inputs = [
            ComponentInput(
                name="template",
                display_name="Template",
                field_type="text",
                description="Prompt template with variables in {variable} format"
            ),
            ComponentInput(
                name="input_variables",
                display_name="Input Variables",
                field_type="list",
                description="List of variable names in the template"
            ),
            ComponentInput(
                name="template_format",
                display_name="Template Format",
                field_type="str",
                options=["f-string", "jinja2"],
                default="f-string",
                required=False,
                description="Template formatting style"
            ),
            ComponentInput(
                name="variables",
                display_name="Variables",
                field_type="dict",
                description="Values for template variables"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="formatted_prompt",
                display_name="Formatted Prompt",
                field_type="str",
                method="format_prompt"
            ),
            ComponentOutput(
                name="prompt_template",
                display_name="Prompt Template Object",
                field_type="prompt_template",
                method="create_template"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        template = kwargs.get("template")
        input_variables = kwargs.get("input_variables", [])
        template_format = kwargs.get("template_format", "f-string")
        variables = kwargs.get("variables", {})
        
        # Create prompt template
        prompt_template = PromptTemplate(
            template=template,
            input_variables=input_variables,
            template_format=template_format
        )
        
        # Format prompt with variables
        formatted_prompt = prompt_template.format(**variables)
        
        return {
            "formatted_prompt": formatted_prompt,
            "prompt_template": prompt_template,
            "variables_used": list(variables.keys())
        }

@register_component
class ChatPromptTemplateComponent(BaseLangChainComponent):
    """Chat Prompt Template Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Chat Prompt Template",
            description="Create chat prompt templates with multiple message types",
            icon="💬",
            category="prompts",
            tags=["chat", "prompts", "templates"]
        )
        
        self.inputs = [
            ComponentInput(
                name="system_message",
                display_name="System Message",
                field_type="text",
                required=False,
                description="System message template"
            ),
            ComponentInput(
                name="human_message",
                display_name="Human Message",
                field_type="text",
                description="Human message template"
            ),
            ComponentInput(
                name="assistant_message",
                display_name="Assistant Message",
                field_type="text",
                required=False,
                description="Assistant message template"
            ),
            ComponentInput(
                name="variables",
                display_name="Variables",
                field_type="dict",
                description="Values for template variables"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="chat_prompt",
                display_name="Chat Prompt",
                field_type="chat_prompt",
                method="create_chat_prompt"
            ),
            ComponentOutput(
                name="formatted_messages",
                display_name="Formatted Messages",
                field_type="list",
                method="format_messages"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        system_message = kwargs.get("system_message")
        human_message = kwargs.get("human_message")
        assistant_message = kwargs.get("assistant_message")
        variables = kwargs.get("variables", {})
        
        # Build message templates
        message_templates = []
        
        if system_message:
            message_templates.append(
                SystemMessagePromptTemplate.from_template(system_message)
            )
        
        if human_message:
            message_templates.append(
                HumanMessagePromptTemplate.from_template(human_message)
            )
        
        # Create chat prompt template
        chat_prompt = ChatPromptTemplate.from_messages(message_templates)
        
        # Format messages
        formatted_messages = await chat_prompt.aformat_messages(**variables)
        
        return {
            "chat_prompt": chat_prompt,
            "formatted_messages": [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content
                }
                for msg in formatted_messages
            ],
            "message_count": len(formatted_messages)
        }