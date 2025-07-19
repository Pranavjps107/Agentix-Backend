"""
Anthropic Claude LLM Component for LangChain Integration
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
class AnthropicLLMComponent(BaseLangChainComponent):
    """
    Anthropic Claude LLM Component for text generation using various Claude models.
    
    Supports Claude 3 (Opus, Sonnet, Haiku) and Claude 2.x models with configurable
    parameters for temperature, token limits, and sampling methods.
    """
    
    # Model configurations
    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0"
    ]
    
    DEFAULT_MODEL = "claude-3-sonnet-20240229"
    
    def _setup_component(self) -> None:
        """Initialize component metadata and input/output definitions."""
        self._setup_metadata()
        self._setup_inputs()
        self._setup_outputs()
    
    def _setup_metadata(self) -> None:
        """Configure component metadata."""
        self.metadata = ComponentMetadata(
            display_name="Anthropic Claude",
            description="Anthropic Claude language models for text generation", 
            icon="🎭",
            category="language_models",
            tags=["anthropic", "claude", "llm", "generation"],
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
                description="Anthropic Claude model to use"
            ),
            ComponentInput(
                name="prompt",
                display_name="Prompt",
                field_type="str",
                multiline=True,
                required=True,
                description="Input prompt for Claude"
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
                default=256,
                required=False,
                description="Maximum tokens in response"
            ),
            ComponentInput(
                name="top_p",
                display_name="Top P",
                field_type="float",
                default=1.0,
                required=False,
                description="Nucleus sampling parameter (0.0-1.0)"
            ),
            ComponentInput(
                name="top_k",
                display_name="Top K",
                field_type="int",
                default=250,
                required=False,
                description="Top-k sampling parameter"
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
                display_name="Anthropic API Key",
                field_type="str",
                required=False,
                password=True,
                description="Anthropic API key (optional if set in environment)"
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
                description="The generated text response from Claude"
            ),
            ComponentOutput(
                name="usage",
                display_name="Token Usage",
                field_type="dict",
                method="get_usage",
                description="Token usage statistics"
            ),
            ComponentOutput(
                name="model_info",
                display_name="Model Info",
                field_type="dict",
                method="get_model_info",
                description="Information about the Claude model used"
            )
        ]
    
    def _validate_inputs(self, **kwargs) -> None:
        """Validate input parameters."""
        prompt = kwargs.get("prompt", "")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        model = kwargs.get("model", self.DEFAULT_MODEL)
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}. Supported models: {self.SUPPORTED_MODELS}")
        
        temperature = kwargs.get("temperature", 0.7)
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        
        top_p = kwargs.get("top_p", 1.0)
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("Top P must be between 0.0 and 1.0")
        
        max_tokens = kwargs.get("max_tokens", 256)
        if max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
    
    def _extract_parameters(self, **kwargs) -> Dict[str, Any]:
        """Extract and prepare parameters for the LLM."""
        return {
            "model": kwargs.get("model", self.DEFAULT_MODEL),
            "prompt": kwargs.get("prompt", "").strip(),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 256),
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", 250),
            "stop_sequences": kwargs.get("stop_sequences", []),
            "api_key": kwargs.get("api_key")
        }
    
    def _create_llm_instance(self, params: Dict[str, Any]) -> Any:
        """Create and configure the Anthropic LLM instance."""
        try:
            from langchain_anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic package is required for Anthropic component. "
                "Install with: pip install langchain-anthropic"
            )
        
        # Prepare LLM configuration
        llm_kwargs = {
            "model": params["model"],
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "top_p": params["top_p"],
            "top_k": params["top_k"]
        }
        
        # Add API key if provided
        if params["api_key"]:
            llm_kwargs["anthropic_api_key"] = params["api_key"]
        
        # Add stop sequences if provided
        if params["stop_sequences"]:
            llm_kwargs["stop_sequences"] = params["stop_sequences"]
        
        return Anthropic(**llm_kwargs)
    
    def _calculate_usage(self, prompt: str, response_text: str) -> Dict[str, int]:
        """Calculate approximate token usage."""
        # Note: This is a rough approximation. For accurate token counting,
        # consider using tiktoken or similar tokenization libraries
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response_text.split())
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    
    def _create_model_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create model information dictionary."""
        return {
            "model": params["model"],
            "provider": "anthropic",
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "top_p": params["top_p"],
            "top_k": params["top_k"]
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the Anthropic LLM component.
        
        Args:
            **kwargs: Component input parameters
            
        Returns:
            Dict containing response, usage, and model_info
            
        Raises:
            ValueError: For invalid input parameters
            ImportError: If required dependencies are missing
            Exception: For LLM execution errors
        """
        try:
            # Validate inputs
            self._validate_inputs(**kwargs)
            
            # Extract parameters
            params = self._extract_parameters(**kwargs)
            
            # Create LLM instance
            llm = self._create_llm_instance(params)
            
            # Generate response
            logger.info(f"Generating response with model: {params['model']}")
            response = await llm.agenerate([params["prompt"]])
            generation = response.generations[0][0]
            
            # Calculate usage and prepare results
            usage = self._calculate_usage(params["prompt"], generation.text)
            model_info = self._create_model_info(params)
            
            logger.info(f"Response generated successfully. Tokens used: {usage['total_tokens']}")
            
            return {
                "response": generation.text,
                "usage": usage,
                "model_info": model_info
            }
            
        except ValueError as e:
            logger.error(f"Input validation error: {str(e)}")
            raise
        except ImportError as e:
            logger.error(f"Dependency error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Anthropic LLM execution failed: {str(e)}")
            raise Exception(f"Anthropic LLM execution failed: {str(e)}")
    
    # Output method implementations
    async def generate_text(self, **kwargs) -> str:
        """Generate text using the configured model."""
        result = await self.execute(**kwargs)
        return result["response"]
    
    async def get_usage(self, **kwargs) -> Dict[str, int]:
        """Get token usage statistics."""
        result = await self.execute(**kwargs)
        return result["usage"]
    
    async def get_model_info(self, **kwargs) -> Dict[str, Any]:
        """Get model information."""
        result = await self.execute(**kwargs)
        return result["model_info"]