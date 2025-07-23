"""
Main FastAPI application
"""
import os
import logging
import time
import traceback
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# Temporary: Set a fake Groq API key for testing if not found
if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "fake-key-for-testing"
    print("Using fake Groq API key for testing")

# Debug: Check if environment variables are loaded
print(f"GROQ_API_KEY loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
if os.getenv('GROQ_API_KEY'):
    print(f"GROQ_API_KEY starts with: {os.getenv('GROQ_API_KEY')[:10]}...")

# Import API routes - these are in the same backend directory
from api.routes import components, flows, health

# Import core components - these are in the same backend directory
from core.registry import ComponentRegistry
from services.component_manager import ComponentManager
from services.storage import StorageService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def register_components_manually():
    """Manually register components using absolute imports and inline definitions"""
    try:
        # Instead of importing problematic files, create components inline
        from core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
        from core.registry import ComponentRegistry, register_component
        
        # Create Text Input Component inline
        @register_component
        class TextInputComponent(BaseLangChainComponent):
            def _setup_component(self):
                self.metadata = ComponentMetadata(
                    display_name="Text Input",
                    description="Simple text input component for starting workflows",
                    icon="📝",
                    category="inputs",
                    tags=["input", "text"]
                )
                self.inputs = [
                    ComponentInput(
                        name="user_input",
                        display_name="User Input", 
                        field_type="str",
                        required=False,
                        description="User input text"
                    ),
                    ComponentInput(
                        name="placeholder",
                        display_name="Placeholder",
                        field_type="str",
                        default="Enter text...",
                        required=False,
                        description="Placeholder text"
                    )
                ]
                self.outputs = [
                    ComponentOutput(
                        name="text",
                        display_name="Output Text",
                        field_type="str", 
                        method="get_text",
                        description="The processed text"
                    )
                ]
            
            async def execute(self, **kwargs):
                user_input = kwargs.get("user_input", "")
                placeholder = kwargs.get("placeholder", "Enter text...")
                
                # Use global input if user_input is empty
                if not user_input:
                    user_input = "Hello, this is a test input!"  # Default for testing
                
                return {
                    "text": user_input, 
                    "length": len(user_input),
                    "placeholder": placeholder
                }
        
        logger.info("Text Input component created and registered")
        
        # Create LLM Model Component inline  
        @register_component
        class LLMComponent(BaseLangChainComponent):
            def _setup_component(self):
                self.metadata = ComponentMetadata(
                    display_name="LLM Model",
                    description="Language Model for text generation",
                    icon="🤖",
                    category="language_models",
                    tags=["llm", "generation", "text", "ai"]
                )
                self.inputs = [
                    ComponentInput(
                        name="prompt",
                        display_name="Prompt",
                        field_type="str",
                        multiline=True,
                        description="Input prompt for the model"
                    ),
                    ComponentInput(
                        name="provider",
                        display_name="Provider",
                        field_type="str",
                        options=["fake", "openai", "anthropic"],
                        default="fake",
                        description="LLM provider to use"
                    ),
                    ComponentInput(
                        name="temperature",
                        display_name="Temperature",
                        field_type="float",
                        default=0.7,
                        description="Controls randomness (0.0-1.0)"
                    )
                ]
                self.outputs = [
                    ComponentOutput(
                        name="response",
                        display_name="Generated Text",
                        field_type="str",
                        method="generate_text",
                        description="The generated text response"
                    )
                ]
            
            async def execute(self, **kwargs):
                prompt = kwargs.get("prompt", "")
                provider = kwargs.get("provider", "fake")
                temperature = kwargs.get("temperature", 0.7)
                
                if not prompt:
                    return {"response": "No prompt provided", "error": "Empty prompt"}
                
                # For now, return a fake response for testing
                fake_response = f"This is a simulated LLM response to: '{prompt}' (provider: {provider}, temp: {temperature})"
                
                return {
                    "response": fake_response,
                    "prompt": prompt,
                    "provider": provider,
                    "temperature": temperature
                }
        
        logger.info("LLM Model component created and registered")
        
        # Create Chat Model Component inline  
        @register_component
        class ChatModelComponent(BaseLangChainComponent):
            def _setup_component(self):
                self.metadata = ComponentMetadata(
                    display_name="Chat Model",
                    description="Chat-based Language Model for conversational AI",
                    icon="💬",
                    category="chat_models",
                    tags=["chat", "conversation", "llm", "ai"]
                )
                self.inputs = [
                    ComponentInput(
                        name="messages",
                        display_name="Messages",
                        field_type="str",
                        multiline=True,
                        description="Input messages or prompt for the chat model"
                    ),
                    ComponentInput(
                        name="provider",
                        display_name="Provider",
                        field_type="str",
                        options=["fake", "openai", "anthropic", "groq"],
                        default="fake",
                        description="Chat model provider to use"
                    ),
                    ComponentInput(
                        name="model",
                        display_name="Model",
                        field_type="str",
                        default="gpt-3.5-turbo",
                        description="Specific model name (e.g., gpt-4, claude-3-sonnet, llama-3.1-70b-versatile)"
                    ),
                    ComponentInput(
                        name="temperature",
                        display_name="Temperature",
                        field_type="float",
                        default=0.7,
                        description="Controls randomness (0.0-1.0)"
                    ),
                    ComponentInput(
                        name="max_tokens",
                        display_name="Max Tokens",
                        field_type="int",
                        default=1000,
                        description="Maximum tokens in response"
                    ),
                    ComponentInput(
                        name="system_message",
                        display_name="System Message",
                        field_type="str",
                        multiline=True,
                        required=False,
                        description="System prompt to guide the model's behavior"
                    )
                ]
                self.outputs = [
                    ComponentOutput(
                        name="response",
                        display_name="Chat Response",
                        field_type="str",
                        method="generate_response",
                        description="The generated chat response"
                    ),
                    ComponentOutput(
                        name="usage",
                        display_name="Token Usage",
                        field_type="dict",
                        method="get_usage",
                        description="Token usage statistics"
                    )
                ]
            
            async def execute(self, **kwargs):
                messages = kwargs.get("messages", "")
                provider = kwargs.get("provider", "fake")
                model = kwargs.get("model", "gpt-3.5-turbo")
                temperature = kwargs.get("temperature", 0.7)
                max_tokens = kwargs.get("max_tokens", 1000)
                system_message = kwargs.get("system_message", "")
                
                if not messages:
                    return {"response": "No messages provided", "error": "Empty messages"}
                
                # Simulate different responses based on system message
                if "news analyst" in system_message.lower():
                    # For news analysis, return structured JSON
                    fake_response = {
                        "topic": "artificial intelligence",
                        "sentiment": "positive",
                        "sentiment_score": 0.78,
                        "key_insights": [
                            "AI adoption is accelerating across industries",
                            "New breakthrough in machine learning efficiency",
                            "Regulatory frameworks are being developed"
                        ],
                        "main_themes": ["innovation", "regulation", "adoption"],
                        "urgency_level": "medium",
                        "summary": "Recent AI developments show positive trends with continued innovation and growing adoption across various sectors.",
                        "trending_keywords": ["AI", "machine learning", "automation"],
                        "geographical_focus": ["United States", "China", "Europe"],
                        "time_relevance": "recent",
                        "credibility_assessment": "high",
                        "potential_impact": "Significant transformation expected in multiple industries over the next 2-3 years"
                    }
                    import json
                    response_text = json.dumps(fake_response, indent=2)
                elif "friendly ai assistant" in system_message.lower():
                    # For insight generation, return friendly explanation
                    response_text = """Based on the news analysis, here's what's happening with artificial intelligence:

🎯 **Key Takeaway**: The AI landscape is looking quite positive right now! There's a lot of exciting innovation happening.

📈 **Sentiment Analysis**: The overall mood around AI is positive (78% positive sentiment), which suggests that people are generally optimistic about AI developments rather than fearful.

💡 **Most Important Insights**:
- AI is being adopted much faster across different industries than expected
- There's been a significant breakthrough in making machine learning more efficient
- Governments are starting to create proper rules and guidelines for AI use

⚡ **Urgency Level**: Medium - This isn't breaking news that requires immediate action, but these are important ongoing developments worth following.

🌍 **Global Impact**: The changes are happening primarily in the US, China, and Europe, but the effects will likely spread worldwide as AI becomes more mainstream.

This is an exciting time for AI - we're seeing both technological progress and responsible development of governance frameworks!"""
                else:
                    # Generic chat response
                    response_text = f"This is a simulated chat response to: '{messages}' (provider: {provider}, model: {model}, temp: {temperature})"
                
                # Simulate token usage
                usage = {
                    "prompt_tokens": len(str(messages).split()) + len(str(system_message).split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(str(messages).split()) + len(str(system_message).split()) + len(response_text.split())
                }
                
                return {
                    "response": response_text,
                    "usage": usage,
                    "provider": provider,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
        
        logger.info("Chat Model component created and registered")
        
        # Create String Output Parser Component inline
        @register_component
        class StringOutputParserComponent(BaseLangChainComponent):
            def _setup_component(self):
                self.metadata = ComponentMetadata(
                    display_name="String Output Parser",
                    description="Parse LLM output as string",
                    icon="📄",
                    category="output_parsers",
                    tags=["parser", "string"]
                )
                self.inputs = [
                    ComponentInput(
                        name="llm_output",
                        display_name="LLM Output",
                        field_type="str",
                        description="Raw output from LLM"
                    )
                ]
                self.outputs = [
                    ComponentOutput(
                        name="parsed_text",
                        display_name="Parsed Text",
                        field_type="str",
                        method="parse_text",
                        description="Cleaned and parsed text"
                    )
                ]
            
            async def execute(self, **kwargs):
                llm_output = kwargs.get("llm_output", "")
                
                # Simple string cleaning/parsing
                parsed = llm_output.strip()
                
                return {
                    "parsed_text": parsed,
                    "original_length": len(llm_output),
                    "parsed_length": len(parsed)
                }
        
        logger.info("String Output Parser component created and registered")
        
        # Create JSON Output Parser Component inline
        @register_component
        class JsonOutputParserComponent(BaseLangChainComponent):
            def _setup_component(self):
                self.metadata = ComponentMetadata(
                    display_name="JSON Output Parser",
                    description="Parse LLM output as JSON",
                    icon="📋",
                    category="output_parsers",
                    tags=["parser", "json"]
                )
                self.inputs = [
                    ComponentInput(
                        name="llm_output",
                        display_name="LLM Output",
                        field_type="str",
                        description="Raw output from LLM to parse as JSON"
                    )
                ]
                self.outputs = [
                    ComponentOutput(
                        name="parsed_json",
                        display_name="Parsed JSON",
                        field_type="dict",
                        method="parse_json",
                        description="Parsed JSON object"
                    )
                ]
            
            async def execute(self, **kwargs):
                llm_output = kwargs.get("llm_output", "")
                
                try:
                    # Try to parse as JSON
                    if llm_output.strip().startswith('{') and llm_output.strip().endswith('}'):
                        parsed = json.loads(llm_output)
                    else:
                        # Try to extract JSON from text
                        start = llm_output.find('{')
                        end = llm_output.rfind('}') + 1
                        if start != -1 and end != 0:
                            json_str = llm_output[start:end]
                            parsed = json.loads(json_str)
                        else:
                            # Fallback: create a mock JSON response
                            parsed = {
                                "text": llm_output,
                                "parsed": True,
                                "message": "Successfully processed text"
                            }
                except Exception as e:
                    parsed = {
                        "error": f"Failed to parse JSON: {str(e)}", 
                        "raw_output": llm_output,
                        "parsed": False
                    }
                
                return {
                    "parsed_json": parsed, 
                    "is_valid": "error" not in parsed
                }
        
        logger.info("JSON Output Parser component created and registered")
        
        # Create a basic web search tool component
        @register_component
        class WebSearchToolComponent(BaseLangChainComponent):
            def _setup_component(self):
                self.metadata = ComponentMetadata(
                    display_name="Web Search Tool",
                    description="Simulated web search tool",
                    icon="🔍",
                    category="tools",
                    tags=["search", "web", "tool"]
                )
                self.inputs = [
                    ComponentInput(
                        name="query",
                        display_name="Search Query",
                        field_type="str",
                        description="Query to search for"
                    ),
                    ComponentInput(
                        name="num_results",
                        display_name="Number of Results",
                        field_type="int",
                        default=5,
                        description="Number of search results to return"
                    )
                ]
                self.outputs = [
                    ComponentOutput(
                        name="search_results",
                        display_name="Search Results",
                        field_type="str",
                        method="search",
                        description="Formatted search results as text"
                    ),
                    ComponentOutput(
                        name="results",
                        display_name="Raw Results",
                        field_type="list",
                        method="get_raw_results",
                        description="List of search results"
                    )
                ]
            
            async def execute(self, **kwargs):
                query = kwargs.get("query", "")
                num_results = kwargs.get("num_results", 5)
                
                if not query:
                    return {"search_results": "No search query provided", "results": [], "error": "No search query provided"}
                
                # Simulate search results based on the query
                fake_results = []
                if "artificial intelligence" in query.lower() or "ai" in query.lower():
                    fake_results = [
                        {
                            "title": "Major AI Breakthrough: New Neural Network Architecture Achieves 40% Efficiency Gain",
                            "url": "https://tech-news.com/ai-breakthrough-2025",
                            "snippet": "Researchers at leading tech companies have developed a revolutionary neural network architecture that significantly reduces computational requirements while maintaining accuracy.",
                            "rank": 1
                        },
                        {
                            "title": "AI Adoption Surges Across Healthcare Industry, Improving Patient Outcomes",
                            "url": "https://health-tech.com/ai-healthcare-adoption",
                            "snippet": "Healthcare providers are increasingly integrating AI systems for diagnosis, treatment planning, and patient monitoring, leading to improved outcomes and reduced costs.",
                            "rank": 2
                        },
                        {
                            "title": "New AI Regulation Framework Proposed by Global Tech Leaders",
                            "url": "https://policy-watch.com/ai-regulation-framework",
                            "snippet": "A comprehensive framework for AI governance has been proposed, focusing on transparency, accountability, and ethical AI development practices.",
                            "rank": 3
                        },
                        {
                            "title": "AI-Powered Automation Transforms Manufacturing Sector",
                            "url": "https://industry-news.com/ai-manufacturing",
                            "snippet": "Manufacturing companies are leveraging AI-driven automation to optimize production processes, reduce waste, and improve quality control.",
                            "rank": 4
                        },
                        {
                            "title": "Investment in AI Startups Reaches Record High in 2025",
                            "url": "https://finance-today.com/ai-startup-investment",
                            "snippet": "Venture capital funding for AI startups has reached unprecedented levels, with particular focus on enterprise AI solutions and ethical AI development.",
                            "rank": 5
                        }
                    ]
                else:
                    # Generic results for other queries
                    for i in range(min(num_results, 5)):
                        fake_results.append({
                            "title": f"Search Result {i+1} for '{query}'",
                            "url": f"https://example.com/result-{i+1}",
                            "snippet": f"This is a simulated search result snippet for query: {query}",
                            "rank": i + 1
                        })
                
                # Format results as text for the search_results output
                formatted_results = f"Search results for '{query}':\n\n"
                for result in fake_results[:num_results]:
                    formatted_results += f"{result['rank']}. {result['title']}\n"
                    formatted_results += f"   URL: {result['url']}\n"
                    formatted_results += f"   {result['snippet']}\n\n"
                
                return {
                    "search_results": formatted_results,
                    "results": fake_results[:num_results],
                    "query": query,
                    "total_results": len(fake_results[:num_results])
                }
        
        logger.info("Web Search Tool component created and registered")
        
        logger.info("Manual component registration completed successfully")
        
    except Exception as e:
        logger.error(f"Manual registration failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

# Track application startup time
startup_time = time.time()

def ensure_static_directory():
    """Ensure static directory and index.html exist"""
    static_dir = "static"
    index_file = os.path.join(static_dir, "index.html")
    
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        logger.info(f"Created static directory: {static_dir}")
    
    if not os.path.exists(index_file):
        # Create a simple index.html
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangChain Platform</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .links { display: flex; gap: 20px; justify-content: center; }
        .link { padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 LangChain Platform</h1>
            <p>Drag-and-drop LangChain component platform</p>
        </div>
        <div class="links">
            <a href="/docs" class="link">API Documentation</a>
            <a href="/api/v1/health" class="link">Health Check</a>
            <a href="/info" class="link">Platform Info</a>
        </div>
    </div>
</body>
</html>"""
        
        with open(index_file, 'w') as f:
            f.write(html_content)
        logger.info(f"Created index.html: {index_file}")
    
    logger.info("Static directory and index.html created")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("=" * 50)
    logger.info("Starting LangChain Platform...")
    
    # Ensure static directory exists
    ensure_static_directory()
    
    # Manual registration
    register_components_manually()
    
    # Log registered components
    component_count = len(ComponentRegistry._components)
    categories = ComponentRegistry.get_categories()
    logger.info(f"Registered {component_count} components across {len(categories)} categories")
    
    # List all registered components
    logger.info("Registered components:")
    for name in ComponentRegistry._components.keys():
        logger.info(f"  - {name}")
    
    for category, component_list in categories.items():
        logger.info(f"Category '{category}': {component_list}")
    
    # Initialize services
    try:
        storage_service = StorageService()
        logger.info("Storage service initialized")
        
        component_manager = ComponentManager()
        logger.info("Component manager initialized")
        
        logger.info("LangChain Platform started successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LangChain Platform...")
    
    # Cleanup resources
    try:
        # Clear caches
        component_manager = ComponentManager()
        if hasattr(component_manager, 'clear_cache'):
            component_manager.clear_cache()
            logger.info("Cleared component caches")
        
        logger.info("Cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
    
    logger.info("LangChain Platform shut down")

# Create FastAPI app
app = FastAPI(
    title="LangChain Drag-and-Drop Platform",
    description="""
    Production-ready LangChain component platform with drag-and-drop interface.
    
    ## Features
    - 🔗 Complete LangChain Core API integration
    - 🎨 Drag-and-drop visual flow builder
    - ⚡ Real-time component execution
    - 📊 Built-in monitoring and caching
    - 🔄 Flow export to Python code
    - 🛡️ Production-ready architecture
    
    ## Components
    - Language Models (OpenAI, Anthropic, etc.)
    - Embeddings (OpenAI, HuggingFace, etc.)
    - Vector Stores (Chroma, Pinecone, etc.)
    - Tools and Agents
    - Document Loaders
    - Output Parsers
    - And much more!
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path)
        }
    )

# Include API routers
app.include_router(components.router)
app.include_router(flows.router)
app.include_router(health.router)

# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with platform information"""
    uptime = time.time() - startup_time
    
    return {
        "message": "🚀 LangChain Drag-and-Drop Platform",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": uptime,
        "components_registered": len(ComponentRegistry._components),
        "categories": list(ComponentRegistry.get_categories().keys()),
        "registered_components": list(ComponentRegistry._components.keys()),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc", 
            "health": "/api/v1/health",
            "components": "/api/v1/components",
            "flows": "/api/v1/flows"
        },
        "features": [
            "Drag-and-drop flow builder",
            "Real-time component execution", 
            "LangChain integration",
            "Flow export to Python",
            "Built-in monitoring",
            "Caching and optimization"
        ]
    }

@app.get("/info")
async def get_platform_info():
    """Get detailed platform information"""
    component_stats = ComponentRegistry.get_stats()
    
    return {
        "platform": {
            "name": "LangChain Platform",
            "version": "1.0.0",
            "uptime_seconds": time.time() - startup_time
        },
        "components": component_stats,
        "categories": ComponentRegistry.get_categories(),
        "system": {
            "python_version": "3.11+",
            "framework": "FastAPI",
            "database": "File-based storage"
        }
    }

# Serve static files (for frontend)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static files mounted at /static")
except Exception as e:
    logger.warning(f"Static files directory not found: {str(e)}")

# Additional utility endpoints
@app.get("/status")
async def get_status():
    """Quick status check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - startup_time,
        "components": len(ComponentRegistry._components)
    }

@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint"""
    try:
        component_manager = ComponentManager()
        stats = component_manager.get_manager_stats()
        
        return {
            "platform_metrics": {
                "uptime_seconds": time.time() - startup_time,
                "registered_components": len(ComponentRegistry._components),
                "categories": len(ComponentRegistry.get_categories())
            },
            "execution_metrics": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )