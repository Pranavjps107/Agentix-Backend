"""
Dedicated Groq Chat Model Component
"""
from typing import Dict, Any, Optional, List
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from ...core.base import (
    BaseLangChainComponent, 
    ComponentInput, 
    ComponentOutput, 
    ComponentMetadata, 
    register_component
)

logger = logging.getLogger(__name__)

@register_component
class GroqChatComponent(BaseLangChainComponent):
    """
    Dedicated Groq Chat Model Component with full support for Groq-specific features
    including reasoning capabilities and ultra-fast inference.
    """
    
    # Popular Groq chat models
    SUPPORTED_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant", 
        "llama-3.1-70b-versatile",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "gemma2-9b-it",
        "llama-3.1-405b-reasoning"  # Reasoning model
    ]
    
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    
    def _setup_component(self) -> None:
        """Initialize component metadata and input/output definitions."""
        self._setup_metadata()
        self._setup_inputs()
        self._setup_outputs()
    
    def _setup_metadata(self) -> None:
        """Configure component metadata."""
        self.metadata = ComponentMetadata(
            display_name="Groq Chat Model",
            description="Groq's ultra-fast chat models with reasoning capabilities", 
            icon="⚡",
            category="language_models",
            tags=["groq", "chat", "llama", "mixtral", "fast", "reasoning"],
            version="1.0.0"
        )
    
    def _setup_inputs(self) -> None:
        """Configure component inputs."""
        self.inputs = [
            ComponentInput(
                name="model",
                display_name="Model",
                field_type="str",
                options=self.SUPPORTED_MODELS,
                default=self.DEFAULT_MODEL,
                description="Groq chat model to use"
            ),
            ComponentInput(
                name="messages",
                display_name="Messages",
                field_type="list",
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
                description="Controls randomness (0.0-1.0)"
            ),
            ComponentInput(
                name="max_tokens",
                display_name="Max Tokens",
                field_type="int",
                default=1024,
                required=False,
                description="Maximum tokens in response"
            ),
            ComponentInput(
                name="reasoning_format",
                display_name="Reasoning Format",
                field_type="str",
                options=["parsed", "raw", "hidden"],
                required=False,
                description="Format for reasoning output (for reasoning models)"
            ),
            ComponentInput(
                name="reasoning_effort",
                display_name="Reasoning Effort",
                field_type="str",
                options=["none", "default"],
                required=False,
                description="Level of reasoning effort"
            ),
            ComponentInput(
                name="service_tier",
                display_name="Service Tier",
                field_type="str",
                options=["on_demand", "flex", "auto"],
                default="on_demand",
                required=False,
                description="Service tier for requests"
            ),
            ComponentInput(
                name="stop_sequences",
                display_name="Stop Sequences",
                field_type="list",
                required=False,
                description="List of stop sequences to halt generation"
            ),
            ComponentInput(
                name="api_key",
                display_name="Groq API Key",
                field_type="str",
                required=False,
                password=True,
                description="Groq API key (optional if set in environment)"
            )
        ]
    
    def _setup_outputs(self) -> None:
        """Configure component outputs."""
        self.outputs = [
            ComponentOutput(
                name="response",
                display_name="Chat Response",
                field_type="str",
                method="generate_response",
                description="The chat model's response"
            ),
            ComponentOutput(
                name="conversation_history",
                display_name="Updated Conversation",
                field_type="list",
                method="get_conversation",
                description="Full conversation history"
            ),
            ComponentOutput(
                name="usage",
                display_name="Token Usage",
                field_type="dict",
                method="get_usage",
                description="Detailed token usage from Groq"
            ),
            ComponentOutput(
                name="reasoning_content",
                display_name="Reasoning Content",
                field_type="str",
                method="get_reasoning",
                description="Reasoning process (for reasoning models)"
            ),
            ComponentOutput(
                name="performance_metrics",
                display_name="Performance Metrics",
                field_type="dict",
                method="get_performance",
                description="Groq performance metrics"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the Groq Chat Model component."""
        try:
            # Extract parameters
            params = self._extract_parameters(**kwargs)
            
            # Build message list
            chat_messages = self._build_messages(params)
            
            # Create Groq instance
            chat_model = self._create_groq_instance(params)
            
            # Generate response
            logger.info(f"Generating chat response with Groq model: {params['model']}")
            
            response = await chat_model.agenerate([chat_messages])
            ai_message = response.generations[0][0].message
            
            # Extract reasoning content
            reasoning_content = self._extract_reasoning_content(ai_message)
            
            # Extract usage and performance metrics
            usage = self._extract_usage(response)
            performance_metrics = self._extract_performance_metrics(response)
            
            # Build updated conversation history
            updated_conversation = [
                {"role": self._message_role(msg), "content": msg.content}
                for msg in chat_messages
            ]
            updated_conversation.append({
                "role": "assistant",
                "content": ai_message.content
            })
            
            logger.info(f"Groq chat response generated successfully. Tokens: {usage.get('total_tokens', 0)}")
            
            return {
                "response": ai_message.content,
                "conversation_history": updated_conversation,
                "usage": usage,
                "reasoning_content": reasoning_content,
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Groq Chat Model execution failed: {str(e)}")
            raise Exception(f"Groq Chat Model execution failed: {str(e)}")
    
    def _extract_parameters(self, **kwargs) -> Dict[str, Any]:
        """Extract and validate parameters."""
        model = kwargs.get("model", self.DEFAULT_MODEL)
        if model not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model} not in supported list, proceeding anyway")
        
        return {
            "model": model,
            "messages": kwargs.get("messages", []),
            "system_message": kwargs.get("system_message"),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "reasoning_format": kwargs.get("reasoning_format"),
            "reasoning_effort": kwargs.get("reasoning_effort"),
            "service_tier": kwargs.get("service_tier", "on_demand"),
            "stop_sequences": kwargs.get("stop_sequences", []),
            "api_key": kwargs.get("api_key")
        }
    
    def _build_messages(self, params: Dict[str, Any]) -> List[BaseMessage]:
        """Build message list from parameters."""
        chat_messages = []
        
        # Add system message if provided
        if params["system_message"]:
            chat_messages.append(SystemMessage(content=params["system_message"]))
        
        # Process input messages
        for msg in params["messages"]:
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
        
        if not chat_messages:
            raise ValueError("At least one message is required")
        
        return chat_messages
    
    def _create_groq_instance(self, params: Dict[str, Any]):
        """Create and configure the Groq Chat Model instance."""
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq package is required. Install with: pip install langchain-groq"
            )
        
        # Prepare chat model configuration
        groq_kwargs = {
            "model": params["model"],
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "service_tier": params["service_tier"]
        }
        
        # Add API key if provided
        if params["api_key"]:
            groq_kwargs["groq_api_key"] = params["api_key"]
        
        # Add reasoning parameters if provided
        if params["reasoning_format"]:
            groq_kwargs["reasoning_format"] = params["reasoning_format"]
        
        if params["reasoning_effort"]:
            groq_kwargs["reasoning_effort"] = params["reasoning_effort"]
        
        # Add stop sequences if provided
        if params["stop_sequences"]:
            groq_kwargs["stop"] = params["stop_sequences"]
        
        return ChatGroq(**groq_kwargs)
    
    def _extract_reasoning_content(self, ai_message) -> str:
        """Extract reasoning content from Groq response."""
        # Check for reasoning content in additional_kwargs
        if hasattr(ai_message, 'additional_kwargs'):
            reasoning = ai_message.additional_kwargs.get('reasoning_content', '')
            if reasoning:
                return reasoning
        
        # Check for reasoning in message content for raw format
        if hasattr(ai_message, 'content'):
            content = ai_message.content
            if '<think>' in content and '</think>' in content:
                import re
                reasoning_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if reasoning_match:
                    return reasoning_match.group(1).strip()
        
        return ""
    
    def _extract_usage(self, response) -> Dict[str, Any]:
        """Extract token usage information."""
        llm_output = getattr(response, 'llm_output', {})
        token_usage = llm_output.get('token_usage', {})
        
        return {
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
            "completion_time": token_usage.get("completion_time", 0),
            "prompt_time": token_usage.get("prompt_time", 0),
            "total_time": token_usage.get("total_time", 0),
            "queue_time": token_usage.get("queue_time")
        }
    
    def _extract_performance_metrics(self, response) -> Dict[str, Any]:
        """Extract Groq-specific performance metrics."""
        llm_output = getattr(response, 'llm_output', {})
        
        return {
            "model_name": llm_output.get("model_name", ""),
            "system_fingerprint": llm_output.get("system_fingerprint", ""),
            "finish_reason": llm_output.get("finish_reason", ""),
            "queue_time": llm_output.get("queue_time"),
            "total_time": llm_output.get("total_time", 0)
        }
    
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