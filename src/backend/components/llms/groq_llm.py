"""
Dedicated Groq LLM Component with full Groq feature support
"""
from typing import Dict, Any, Optional, List
import logging
from ...core.base import (
    BaseLangChainComponent, 
    ComponentInput, 
    ComponentOutput, 
    ComponentMetadata, 
    register_component
)

logger = logging.getLogger(__name__)

@register_component
class GroqLLMComponent(BaseLangChainComponent):
    """
    Dedicated Groq LLM Component with full support for Groq-specific features
    including reasoning capabilities and fast inference.
    """
    
    # Popular Groq models
    SUPPORTED_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant", 
        "llama-3.1-70b-versatile",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "gemma2-9b-it"
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
            display_name="Groq LLM",
            description="Groq's ultra-fast LLM inference with reasoning capabilities", 
            icon="⚡",
            category="language_models",
            tags=["groq", "llama", "mixtral", "fast", "reasoning"],
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
                description="Groq model to use"
            ),
            ComponentInput(
                name="prompt",
                display_name="Prompt",
                field_type="str",
                multiline=True,
                required=True,
                description="Input prompt for Groq"
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
                display_name="Generated Text",
                field_type="str",
                method="generate_text",
                description="The generated text response from Groq"
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
        """Execute the Groq LLM component."""
        try:
            # Extract parameters
            params = self._extract_parameters(**kwargs)
            
            # Create Groq instance
            llm = self._create_groq_instance(params)
            
            # Generate response
            logger.info(f"Generating response with Groq model: {params['model']}")
            
            if params['reasoning_format'] or params['reasoning_effort']:
                # Use chat interface for reasoning models
                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=params["prompt"])]
                response = await llm.agenerate([messages])
                message = response.generations[0][0].message
                generated_text = message.content
                
                # Extract reasoning content
                reasoning_content = message.additional_kwargs.get('reasoning_content', '')
                
            else:
                # Use standard generation
                response = await llm.agenerate([params["prompt"]])
                generation = response.generations[0][0]
                generated_text = generation.text
                reasoning_content = ''
            
            # Extract usage and performance metrics
            usage = self._extract_usage(response)
            performance_metrics = self._extract_performance_metrics(response)
            
            logger.info(f"Groq response generated successfully. Tokens: {usage.get('total_tokens', 0)}")
            
            return {
                "response": generated_text,
                "usage": usage,
                "reasoning_content": reasoning_content,
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Groq LLM execution failed: {str(e)}")
            raise Exception(f"Groq LLM execution failed: {str(e)}")
    
    def _extract_parameters(self, **kwargs) -> Dict[str, Any]:
        """Extract and validate parameters."""
        model = kwargs.get("model", self.DEFAULT_MODEL)
        if model not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model} not in supported list, proceeding anyway")
        
        return {
            "model": model,
            "prompt": kwargs.get("prompt", "").strip(),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "reasoning_format": kwargs.get("reasoning_format"),
            "reasoning_effort": kwargs.get("reasoning_effort"),
            "service_tier": kwargs.get("service_tier", "on_demand"),
            "stop_sequences": kwargs.get("stop_sequences", []),
            "api_key": kwargs.get("api_key")
        }
    
    def _create_groq_instance(self, params: Dict[str, Any]):
        """Create and configure the Groq LLM instance."""
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq package is required. Install with: pip install langchain-groq"
            )
        
        # Prepare LLM configuration
        llm_kwargs = {
            "model": params["model"],
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "service_tier": params["service_tier"]
        }
        
        # Add API key if provided
        if params["api_key"]:
            llm_kwargs["groq_api_key"] = params["api_key"]
        
        # Add reasoning parameters if provided
        if params["reasoning_format"]:
            llm_kwargs["reasoning_format"] = params["reasoning_format"]
        
        if params["reasoning_effort"]:
            llm_kwargs["reasoning_effort"] = params["reasoning_effort"]
        
        # Add stop sequences if provided
        if params["stop_sequences"]:
            llm_kwargs["stop"] = params["stop_sequences"]
        
        return ChatGroq(**llm_kwargs)
    
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
            "total_time": token_usage.get("total_time", 0)
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