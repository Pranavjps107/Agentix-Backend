"""
Base Chat Model Component
"""
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component

@register_component
class ChatModelComponent(BaseLangChainComponent):
    """Base Chat Model Component for conversational AI"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="ChatModel",  # Changed from "Chat Model" to match flow definition
            description="Chat-based language model for conversations",
            icon="💬",
            category="language_models",
            tags=["chat", "conversation", "messages", "ai"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="provider",
                display_name="Provider",
                field_type="str",
                options=["openai", "anthropic", "google", "fake"],
                default="openai",
                description="Chat model provider"
            ),
            ComponentInput(
                name="model",
                display_name="Model",
                field_type="str",
                default="gpt-3.5-turbo",
                description="Chat model name"
            ),
            ComponentInput(
                name="messages",
                display_name="Messages",
                field_type="list",
                required=False,
                description="List of chat messages with role and content"
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
                display_name="API Key",
                field_type="str",
                required=False,
                password=True,
                description="API key for the provider"
            ),
            ComponentInput(
                name="user_input",
                display_name="User Input",
                field_type="str",
                required=False,
                description="Direct user input message"
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
                name="conversation_history",
                display_name="Updated Conversation",
                field_type="list",
                method="get_updated_conversation",
                description="Conversation history including the new response"
            ),
            ComponentOutput(
                name="usage",
                display_name="Token Usage",
                field_type="dict",
                method="get_usage_stats",
                description="Token usage statistics"
            ),
            ComponentOutput(
                name="chat_model",
                display_name="Chat Model Instance",
                field_type="chat_model",
                method="get_chat_model_instance",
                description="The chat model instance for use with agents"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        provider = kwargs.get("provider", "openai")
        model = kwargs.get("model", "gpt-4")  # Changed default to gpt-4
        messages = kwargs.get("messages", [])
        system_message = kwargs.get("system_message")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)
        api_key = kwargs.get("api_key")
        user_input = kwargs.get("user_input")
        
        # Build message list
        chat_messages = []
        
        # Add system message if provided
        if system_message:
            chat_messages.append(SystemMessage(content=system_message))
        
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
            elif isinstance(msg, BaseMessage):
                chat_messages.append(msg)
        
        # Add user input as human message if provided
        if user_input:
            chat_messages.append(HumanMessage(content=user_input))
        
        # Get chat model instance
        chat_model = self._get_chat_model_instance(provider, model, temperature, max_tokens, api_key)
        
        # If no messages, return the chat model instance for use with agents
        if not chat_messages:
            return {
                "response": "",
                "message_object": {},
                "conversation_history": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "chat_model": chat_model
            }
        
        # Generate response
        try:
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
            
            # Update conversation history
            updated_conversation = [
                {"role": self._message_role(msg), "content": msg.content}
                for msg in chat_messages
            ]
            updated_conversation.append({
                "role": "assistant",
                "content": ai_message.content
            })
            
            message_object = {
                "content": ai_message.content,
                "role": "assistant",
                "additional_kwargs": getattr(ai_message, 'additional_kwargs', {}),
                "usage_metadata": getattr(ai_message, 'usage_metadata', {})
            }
            
            return {
                "response": ai_message.content,
                "message_object": message_object,
                "conversation_history": updated_conversation,
                "usage": usage,
                "chat_model": chat_model
            }
            
        except Exception as e:
            raise Exception(f"Chat model execution failed: {str(e)}")
    
    def _get_chat_model_instance(self, provider: str, model: str, temperature: float, max_tokens: int, api_key: Optional[str]):
        """Factory method to create chat model instances"""
        
        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                kwargs = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if api_key:
                    kwargs["openai_api_key"] = api_key
                return ChatOpenAI(**kwargs)
            except ImportError:
                raise ImportError("langchain-openai package required for OpenAI chat models")
        
        elif provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                kwargs = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if api_key:
                    kwargs["anthropic_api_key"] = api_key
                return ChatAnthropic(**kwargs)
            except ImportError:
                raise ImportError("langchain-anthropic package required for Anthropic chat models")
        
        elif provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                kwargs = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if api_key:
                    kwargs["google_api_key"] = api_key
                return ChatGoogleGenerativeAI(**kwargs)
            except ImportError:
                raise ImportError("langchain-google-genai package required for Google chat models")
        
        elif provider == "fake":
            from langchain_core.language_models.fake_chat_models import FakeChatModel
            return FakeChatModel()
        
        else:
            raise ValueError(f"Unsupported chat model provider: {provider}")
    
    def _message_role(self, message: BaseMessage) -> str:
        """Get role string from message type"""
        if isinstance(message, HumanMessage):
            return "human"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            return "unknown"

# Create an alias for backward compatibility
@register_component
class ChatModel(ChatModelComponent):
    """Alias for ChatModelComponent to match flow definitions"""
    
    def _setup_component(self):
        super()._setup_component()
        self.metadata.display_name = "Chat Model"