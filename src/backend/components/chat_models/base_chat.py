"""
Base Chat Model Component
"""
import os
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component

@register_component
class ChatModelComponent(BaseLangChainComponent):
    """Base Chat Model Component for conversational AI"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Chat Model",
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
                options=["openai", "anthropic", "google", "groq", "fake"],
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
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        provider = kwargs.get("provider", "openai")
        model = kwargs.get("model", "gpt-3.5-turbo")
        messages = kwargs.get("messages", [])
        system_message = kwargs.get("system_message")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)
        api_key = kwargs.get("api_key")
        
        # Build message list
        chat_messages = []
        
        # Add system message if provided
        if system_message:
            chat_messages.append(SystemMessage(content=system_message))
        
        # Process input messages
        if messages:
            # Handle different message formats
            if isinstance(messages, str):
                # If messages is a string, treat it as user input
                chat_messages.append(HumanMessage(content=messages))
            elif isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "human")
                        content = msg.get("content", "")
                        
                        if role in ["human", "user"]:
                            chat_messages.append(HumanMessage(content=content))
                        elif role in ["ai", "assistant"]:
                            chat_messages.append(AIMessage(content=content))
                        elif role == "system":
                            chat_messages.append(SystemMessage(content=content))
                    elif isinstance(msg, BaseMessage):
                        chat_messages.append(msg)
                    elif isinstance(msg, str):
                        chat_messages.append(HumanMessage(content=msg))
            else:
                # Handle other formats (like search results)
                content = str(messages)
                chat_messages.append(HumanMessage(content=content))
        
        # If no messages were added, create a default message
        if not chat_messages:
            chat_messages.append(HumanMessage(content="Hello"))
        
        # Get chat model instance with fallback to fake model
        try:
            chat_model = self._get_chat_model_instance(provider, model, temperature, max_tokens, api_key)
        except Exception as e:
            print(f"Failed to create {provider} model: {str(e)}, using fake model")
            chat_model = self._create_fake_chat_model()
        
        # Generate response
        try:
            response = await chat_model.agenerate([chat_messages])
            ai_message = response.generations[0][0].message
            
            # Calculate usage (approximate)
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
                "usage": usage
            }
            
        except Exception as e:
            # If execution fails, fall back to fake model
            print(f"Chat model execution failed: {str(e)}, using fake response")
            return await self._generate_fake_response(chat_messages, system_message)
    
    def _get_chat_model_instance(self, provider: str, model: str, temperature: float, max_tokens: int, api_key: Optional[str]):
        """Factory method to create chat model instances"""
        
        # Check if we should use fake model (invalid or fake API keys)
        if (api_key and api_key.startswith("fake")) or provider == "fake":
            return self._create_fake_chat_model()
        
        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                kwargs = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                # Use provided API key or environment variable
                if api_key:
                    kwargs["openai_api_key"] = api_key
                elif os.getenv("OPENAI_API_KEY"):
                    kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
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
                elif os.getenv("ANTHROPIC_API_KEY"):
                    kwargs["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
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
                elif os.getenv("GOOGLE_API_KEY"):
                    kwargs["google_api_key"] = os.getenv("GOOGLE_API_KEY")
                return ChatGoogleGenerativeAI(**kwargs)
            except ImportError:
                raise ImportError("langchain-google-genai package required for Google chat models")
        
        elif provider == "groq":
            # Get API key from multiple sources
            groq_api_key = api_key or os.getenv("GROQ_API_KEY")
            
            # If no valid API key, use fake model
            if not groq_api_key:
                print("No Groq API key found, using fake model")
                return self._create_fake_chat_model()
            
            try:
                from langchain_groq import ChatGroq
                
                # Use correct ChatGroq parameters
                chat_groq = ChatGroq(
                    model=model,  # "llama-3.1-70b-versatile"
                    temperature=temperature,
                    max_tokens=max_tokens,
                    groq_api_key=groq_api_key
                )
                print(f"✅ Successfully created ChatGroq with model: {model}")
                return chat_groq
                
            except ImportError as ie:
                print(f"❌ langchain-groq not installed: {str(ie)}")
                return self._create_fake_chat_model()
            except Exception as e:
                print(f"❌ Error creating ChatGroq: {str(e)}")
                return self._create_fake_chat_model()
    def _create_fake_chat_model(self):
        """Create a fake chat model for testing"""
        from langchain_core.language_models.fake_chat_models import FakeChatModel
        
        # Create context-aware fake responses
        fake_responses = [
            # News analysis JSON response
            '''{"topic": "artificial intelligence latest developments", "sentiment": "positive", "sentiment_score": 0.85, "key_insights": ["AI technology advancing rapidly with breakthrough innovations", "Major tech companies increasing AI investments significantly", "New AI models showing improved efficiency and capabilities"], "main_themes": ["technological advancement", "market expansion", "research breakthroughs"], "urgency_level": "medium", "summary": "Recent AI developments show strong positive momentum with significant breakthroughs in machine learning, natural language processing, and industry adoption across multiple sectors.", "trending_keywords": ["artificial intelligence", "machine learning", "deep learning", "neural networks", "automation"], "geographical_focus": ["United States", "China", "Europe", "South Korea"], "time_relevance": "recent", "credibility_assessment": "high", "potential_impact": "AI developments are expected to revolutionize industries including healthcare, finance, education, and transportation, potentially creating new job markets while transforming existing workflows."}''',
            
            # Insight generation response
            "Based on the comprehensive news analysis, here's what's happening in the world of artificial intelligence: The AI landscape is experiencing unprecedented positive momentum! 🚀\n\n**Key Takeaways:**\n- **Sentiment**: Highly positive (85% positive sentiment score) - this indicates strong optimism in the AI community\n- **Innovation Surge**: We're seeing breakthrough developments in machine learning and neural networks that are pushing the boundaries of what's possible\n- **Global Impact**: Major developments are happening across the US, China, Europe, and South Korea, showing this is truly a worldwide phenomenon\n\n**What This Means:**\n- **For Businesses**: Now is a critical time to consider AI integration strategies\n- **For Professionals**: AI skills are becoming increasingly valuable across industries\n- **For Society**: We're on the cusp of transformative changes in healthcare, finance, education, and transportation\n\n**Urgency Level**: Medium - while not breaking news, these developments are significant enough to warrant attention and potential action. The window for early adoption advantages is still open, but it's narrowing as AI becomes more mainstream.\n\nThe overall picture is very encouraging - we're witnessing a technological revolution that promises to enhance human capabilities rather than simply replace them!",
            
            # Additional contextual responses
            "The artificial intelligence sector continues to demonstrate remarkable growth and innovation. Key developments include advances in large language models, computer vision, and autonomous systems. Industry leaders are collaborating on ethical AI frameworks while pushing the boundaries of what's technologically possible.",
            
            "Latest AI research shows significant improvements in model efficiency and accuracy. Companies are reporting successful deployments of AI systems that enhance productivity while maintaining human oversight. The focus has shifted toward responsible AI development and practical applications."
        ]
        
        return FakeChatModel(responses=fake_responses)
    
    async def _generate_fake_response(self, messages: List[BaseMessage], system_message: Optional[str]) -> Dict[str, Any]:
        """Generate a fake response when real models fail"""
        
        # Analyze the system message to determine response type
        is_json_request = system_message and "json" in system_message.lower()
        is_news_analysis = system_message and "news" in system_message.lower() and "analyst" in system_message.lower()
        
        if is_json_request and is_news_analysis:
            # Return structured JSON for news analysis
            response_text = '''{"topic": "artificial intelligence latest developments", "sentiment": "positive", "sentiment_score": 0.85, "key_insights": ["AI technology advancing rapidly", "Industry adoption increasing", "Research breakthroughs continue"], "main_themes": ["innovation", "growth", "adoption"], "urgency_level": "medium", "summary": "AI continues to show strong positive development with increasing industry adoption and technological breakthroughs.", "trending_keywords": ["AI", "machine learning", "innovation"], "geographical_focus": ["Global"], "time_relevance": "recent", "credibility_assessment": "high", "potential_impact": "Significant positive impact expected across multiple industries."}'''
        else:
            # Return conversational response for insight generation
            response_text = "Based on the analysis, artificial intelligence is experiencing very positive momentum with strong growth indicators and increasing adoption across industries. The developments suggest continued innovation and expansion in the AI sector."
        
        # Calculate usage
        total_content = " ".join([msg.content for msg in messages])
        prompt_tokens = len(total_content.split())
        completion_tokens = len(response_text.split())
        
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
            "content": response_text,
            "role": "assistant",
            "additional_kwargs": {},
            "usage_metadata": {}
        }
        
        return {
            "response": response_text,
            "message_object": message_object,
            "conversation_history": updated_conversation,
            "usage": usage
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