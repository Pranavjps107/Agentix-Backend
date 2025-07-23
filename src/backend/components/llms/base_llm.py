"""
Base LLM Component Implementation
"""
import asyncio
from typing import Dict, Any, Optional
from langchain_core.language_models.llms import BaseLLM
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component
@register_component
class LLMComponent(BaseLangChainComponent):
    """Generic LLM Component for text generation"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="LLM Model",
            description="Language Model for text generation with support for multiple providers",
            icon="🤖",
            category="language_models",
            tags=["llm", "generation", "text", "ai"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="provider",
                display_name="Provider",
                field_type="str",
                options=["openai", "anthropic", "huggingface", "ollama", "fake"],
                description="LLM provider to use",
                default="openai"
            ),
            ComponentInput(
                name="model_name",
                display_name="Model Name",
                field_type="str",
                description="Name of the LLM model (e.g., gpt-3.5-turbo, claude-3-sonnet)",
                default="gpt-3.5-turbo"
            ),
            ComponentInput(
                name="prompt",
                display_name="Prompt",
                field_type="str",
                multiline=True,
                description="Input prompt for the model"
            ),
            ComponentInput(
                name="temperature",
                display_name="Temperature",
                field_type="float",
                default=0.7,
                required=False,
                description="Sampling temperature (0.0-2.0)"
            ),
            ComponentInput(
                name="max_tokens",
                display_name="Max Tokens",
                field_type="int",
                default=256,
                required=False,
                description="Maximum tokens to generate"
            ),
            ComponentInput(
                name="api_key",
                display_name="API Key",
                field_type="str",
                required=False,
                password=True,
                description="API key for the provider (optional if set in environment)"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="response",
                display_name="Generated Text",
                field_type="str",
                method="generate_text",
                description="The generated text response"
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
                display_name="Model Information",
                field_type="dict", 
                method="get_model_info",
                description="Information about the used model"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        provider = kwargs.get("provider", "openai")
        model_name = kwargs.get("model_name", "gpt-3.5-turbo")
        prompt = kwargs.get("prompt", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 256)
        api_key = kwargs.get("api_key")
        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Initialize LLM based on provider
        llm = self._get_llm_instance(provider, model_name, temperature, max_tokens, api_key)
        
        # Generate response
        try:
            if hasattr(llm, 'agenerate'):
                response = await llm.agenerate([prompt])
                generated_text = response.generations[0][0].text
            else:
                # Fallback to sync method wrapped in async
                response = await asyncio.to_thread(llm.generate, [prompt])
                generated_text = response.generations[0][0].text
            
            # Calculate token usage (approximate)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(generated_text.split())
            total_tokens = prompt_tokens + completion_tokens
            
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            
            model_info = {
                "provider": provider,
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            return {
                "response": generated_text,
                "usage": usage,
                "model_info": model_info
            }
            
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def _get_llm_instance(self, provider: str, model_name: str, temperature: float, max_tokens: int, api_key: Optional[str]):
        """Factory method to create LLM instances based on provider"""
        
        if provider == "openai":
            try:
                from langchain_openai import OpenAI
                kwargs = {
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if api_key:
                    kwargs["openai_api_key"] = api_key
                return OpenAI(**kwargs)
            except ImportError:
                raise ImportError("langchain-openai package required for OpenAI provider")
        
        elif provider == "anthropic":
            try:
                from langchain_anthropic import Anthropic
                kwargs = {
                    "model": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if api_key:
                    kwargs["anthropic_api_key"] = api_key
                return Anthropic(**kwargs)
            except ImportError:
                raise ImportError("langchain-anthropic package required for Anthropic provider")
        
        elif provider == "huggingface":
            try:
                from langchain_community.llms import HuggingFacePipeline
                return HuggingFacePipeline.from_model_id(
                    model_id=model_name,
                    task="text-generation",
                    model_kwargs={"temperature": temperature, "max_length": max_tokens}
                )
            except ImportError:
                raise ImportError("transformers package required for HuggingFace provider")
        
        elif provider == "ollama":
            try:
                from langchain_community.llms import Ollama
                return Ollama(
                    model=model_name,
                    temperature=temperature
                )
            except ImportError:
                raise ImportError("ollama package required for Ollama provider")
        
        elif provider == "fake":
            from langchain_core.language_models.fake import FakeListLLM
            return FakeListLLM(responses=[f"Fake response to: {model_name}"])
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")