"""
OpenAI LLM Component
"""
from typing import Dict, Any, Optional
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component

@register_component  
class OpenAILLMComponent(BaseLangChainComponent):
    """OpenAI-specific LLM Component with advanced features"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="OpenAI LLM",
            description="OpenAI language models with advanced configuration options",
            icon="🔥",
            category="language_models",
            tags=["openai", "gpt", "llm", "generation"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="model",
                display_name="Model",
                field_type="str",
                options=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "text-davinci-003"],
                default="gpt-3.5-turbo",
                description="OpenAI model to use"
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
                description="Controls randomness (0.0-2.0)"
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
                description="Nucleus sampling parameter"
            ),
            ComponentInput(
                name="frequency_penalty",
                display_name="Frequency Penalty",
                field_type="float",
                default=0.0,
                required=False,
                description="Penalty for token frequency"
            ),
            ComponentInput(
                name="presence_penalty",
                display_name="Presence Penalty", 
                field_type="float",
                default=0.0,
                required=False,
                description="Penalty for token presence"
            ),
            ComponentInput(
                name="stop_sequences",
                display_name="Stop Sequences",
                field_type="list",
                required=False,
                description="List of stop sequences"
            ),
            ComponentInput(
                name="api_key",
                display_name="OpenAI API Key",
                field_type="str",
                required=False,
                password=True,
                description="OpenAI API key (optional if set in environment)"
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
                description="Detailed token usage from OpenAI"
            ),
            ComponentOutput(
                name="finish_reason",
                display_name="Finish Reason",
                field_type="str",
                method="get_finish_reason", 
                description="Why the model stopped generating"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        model = kwargs.get("model", "gpt-3.5-turbo")
        prompt = kwargs.get("prompt", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 256)
        top_p = kwargs.get("top_p", 1.0)
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        presence_penalty = kwargs.get("presence_penalty", 0.0)
        stop_sequences = kwargs.get("stop_sequences", [])
        api_key = kwargs.get("api_key")
        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            from langchain_openai import OpenAI
            
            # Prepare parameters
            llm_kwargs = {
                "model_name": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            
            if api_key:
                llm_kwargs["openai_api_key"] = api_key
            
            if stop_sequences:
                llm_kwargs["stop"] = stop_sequences
            
            # Create LLM instance
            llm = OpenAI(**llm_kwargs)
            
            # Generate response
            response = await llm.agenerate([prompt])
            generation = response.generations[0][0]
            
            # Extract usage information if available
            usage = {}
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                usage = {
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0)
                }
            
            # Get finish reason
            finish_reason = getattr(generation, 'finish_reason', 'unknown')
            
            return {
                "response": generation.text,
                "usage": usage,
                "finish_reason": finish_reason,
                "model": model
            }
            
        except ImportError:
            raise ImportError("langchain-openai package is required for OpenAI LLM component")
        except Exception as e:
            raise Exception(f"OpenAI LLM execution failed: {str(e)}")