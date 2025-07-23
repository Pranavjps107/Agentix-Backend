# src/backend/components/chat_models/openai_chat.py
"""
OpenAI Chat Model Component
"""
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component

@register_component
class OpenAIChatComponent(BaseLangChainComponent):
    """OpenAI Chat Model Component with advanced features"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="OpenAI Chat",
            description="OpenAI Chat models (GPT-3.5, GPT-4) for conversational AI",
            icon="🤖",
            category="language_models",
            tags=["openai", "chat", "gpt", "conversation"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="model",
                display_name="Model",
                field_type="str",
                options=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
                default="gpt-3.5-turbo",
                description="OpenAI chat model to use"
            ),
            ComponentInput(
                name="messages",
                display_name="Messages",
                field_type="list",
                required=False,
                description="List of chat messages"
            ),
            ComponentInput(
                name="system_message",
                display_name="System Message",
                field_type="str",
                required=False,
                multiline=True,
                description="System instruction for the chat model"
            ),
            ComponentInput(
                name="user_message",
                display_name="User Message",
                field_type="str",
                required=False,
                multiline=True,
                description="User message to send to the model"
            ),
            ComponentInput(
                name="temperature",
                display_name="Temperature",
                field_type="float",
                default=0.7,
                required=False,
                description="Controls randomness in responses"
            ),
            ComponentInput(
                name="max_tokens",
                display_name="Max Tokens",
                field_type="int",
                default=512,
                required=False,
                description="Maximum tokens in response"
            ),
            ComponentInput(
                name="api_key",
                display_name="OpenAI API Key",
                field_type="str",
                required=False,
                password=True,
                description="OpenAI API key"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="response",
                display_name="Chat Response",
                field_type="str",
                method="generate_chat_response",
                description="The chat model's response"
            ),
            ComponentOutput(
                name="message_object",
                display_name="Message Object",
                field_type="dict",
                method="get_message_object",
                description="Full message object with metadata"
            ),
            ComponentOutput(
                name="usage",
                display_name="Token Usage",
                field_type="dict",
                method="get_usage_stats",
                description="Token usage statistics"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        model = kwargs.get("model", "gpt-3.5-turbo")
        messages = kwargs.get("messages", [])
        system_message = kwargs.get("system_message")
        user_message = kwargs.get("user_message")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)
        api_key = kwargs.get("api_key")
        
        # Build message list
        chat_messages = []
        
        # Add system message if provided
        if system_message:
            chat_messages.append(SystemMessage(content=system_message))
        
        # Add user message if provided
        if user_message:
            chat_messages.append(HumanMessage(content=user_message))
        
        # Process input messages
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "human")
                content = msg.get("content", "")
                
                if role == "human" or role == "user":
                    chat_messages.append(HumanMessage(content=content))
                elif role == "ai" or role == "assistant":
                    chat_messages.append(AIMessage(content=content))
                elif role == "system":
                    chat_messages.append(SystemMessage(content=content))
        
        if not chat_messages:
            # Default message if none provided
            chat_messages = [HumanMessage(content="Hello")]
        
        try:
            from langchain_openai import ChatOpenAI
            
            # Configure chat model
            chat_kwargs = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if api_key:
                chat_kwargs["openai_api_key"] = api_key
            
            chat_model = ChatOpenAI(**chat_kwargs)
            
            # Generate response
            response = await chat_model.agenerate([chat_messages])
            ai_message = response.generations[0][0].message
            
            # Calculate usage
            total_content = " ".join([msg.content for msg in chat_messages])
            prompt_tokens = len(total_content.split())
            completion_tokens = len(ai_message.content.split())
            
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            
            message_object = {
                "content": ai_message.content,
                "role": "assistant",
                "additional_kwargs": getattr(ai_message, 'additional_kwargs', {}),
                "usage_metadata": getattr(ai_message, 'usage_metadata', {})
            }
            
            return {
                "response": ai_message.content,
                "message_object": message_object,
                "usage": usage,
                "model": model
            }
            
        except ImportError:
            raise ImportError("langchain-openai package required for OpenAI chat models")
        except Exception as e:
            raise Exception(f"OpenAI chat model execution failed: {str(e)}")