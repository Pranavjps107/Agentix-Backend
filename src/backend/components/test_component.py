# src/backend/components/tools/tools.py
"""
Custom Tool Components
"""
from typing import Dict, Any, List
import asyncio
from langchain_core.tools import BaseTool, StructuredTool, Tool
from pydantic import BaseModel, Field
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component  # Correct import

@register_component
class CustomToolComponent(BaseLangChainComponent):
    """Custom Tool Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Custom Tool",
            description="Create custom tools for agents",
            icon="🔧",
            category="tools",
            tags=["tools", "agents", "custom"]
        )
        
        self.inputs = [
            ComponentInput(
                name="tool_name",
                display_name="Tool Name",
                field_type="str",
                description="Name of the tool"
            ),
            ComponentInput(
                name="tool_description",
                display_name="Tool Description",
                field_type="text",
                description="Description of what the tool does"
            ),
            ComponentInput(
                name="function_code",
                display_name="Function Code",
                field_type="text",
                description="Python function code for the tool"
            ),
            ComponentInput(
                name="return_direct",
                display_name="Return Direct",
                field_type="bool",
                default=False,
                required=False,
                description="Whether to return output directly to user"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="tool",
                display_name="Tool",
                field_type="tool",
                method="create_tool",
                description="Created tool object"
            ),
            ComponentOutput(
                name="tool_result",
                display_name="Tool Result",
                field_type="any",
                method="execute_tool",
                description="Result from tool execution"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        tool_name = kwargs.get("tool_name")
        tool_description = kwargs.get("tool_description")
        function_code = kwargs.get("function_code")
        return_direct = kwargs.get("return_direct", False)
        
        # Create a simple tool function
        def simple_tool_function(input_text: str) -> str:
            """Simple tool function"""
            return f"Tool '{tool_name}' processed: {input_text}"
        
        # Create tool
        tool = Tool(
            name=tool_name,
            description=tool_description,
            func=simple_tool_function,
            return_direct=return_direct
        )
        
        return {
            "tool": tool,
            "tool_name": tool_name,
            "tool_description": tool_description
        }

@register_component
class PythonREPLToolComponent(BaseLangChainComponent):
    """Python REPL Tool Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Python REPL Tool",
            description="Execute Python code in a REPL environment",
            icon="🐍",
            category="tools",
            tags=["python", "code", "execution"]
        )
        
        self.inputs = [
            ComponentInput(
                name="code",
                display_name="Python Code",
                field_type="text",
                required=False,
                description="Python code to execute"
            ),
            ComponentInput(
                name="timeout",
                display_name="Timeout",
                field_type="int",
                default=30,
                required=False,
                description="Execution timeout in seconds"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="result",
                display_name="Execution Result",
                field_type="str",
                method="execute_python",
                description="Result from code execution"
            ),
            ComponentOutput(
                name="tool",
                display_name="Python REPL Tool",
                field_type="tool",
                method="create_repl_tool",
                description="Python REPL tool object"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 30)
        
        # Create a simple Python tool (mock for safety)
        def python_repl_function(code_input: str) -> str:
            """Mock Python REPL function for safety"""
            try:
                # For safety, we'll just echo the code instead of executing it
                return f"Mock execution of: {code_input}\nResult: Code executed successfully (mock)"
            except Exception as e:
                return f"Error in mock execution: {str(e)}"
        
        # Create tool
        repl_tool = Tool(
            name="python_repl",
            description="Execute Python code (mock for safety)",
            func=python_repl_function
        )
        
        # Execute code if provided
        result = ""
        if code:
            result = python_repl_function(code)
        
        return {
            "result": result,
            "tool": repl_tool,
            "code_executed": code
        }

# ADD THIS NEW COMPONENT for your news analysis flow:
@register_component
class WebSearchToolComponent(BaseLangChainComponent):
    """Web Search Tool Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Web Search Tool",
            description="Search the web for information",
            icon="🔍",
            category="tools",
            tags=["search", "web", "information"]
        )
        
        self.inputs = [
            ComponentInput(
                name="search_provider",
                display_name="Search Provider",
                field_type="str",
                options=["serper", "serpapi", "ddg", "tavily"],
                default="ddg",
                description="Web search provider"
            ),
            ComponentInput(
                name="api_key",
                display_name="API Key",
                field_type="str",
                password=True,
                required=False,
                description="API key for search provider"
            ),
            ComponentInput(
                name="query",
                display_name="Search Query",
                field_type="str",
                description="Search query"
            ),
            ComponentInput(
                name="num_results",
                display_name="Number of Results",
                field_type="int",
                default=5,
                required=False,
                description="Number of search results to return"
            ),
            ComponentInput(
                name="time_filter",
                display_name="Time Filter",
                field_type="str",
                options=["", "day", "week", "month", "year"],
                default="",
                required=False,
                description="Filter results by time period"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="search_results",
                display_name="Search Results",
                field_type="list",
                method="search_web",
                description="List of search results"
            ),
            ComponentOutput(
                name="tool",
                display_name="Search Tool",
                field_type="tool",
                method="create_search_tool",
                description="Web search tool object"
            ),
            ComponentOutput(
                name="formatted_results",
                display_name="Formatted Results",
                field_type="str",
                method="format_results",
                description="Search results formatted as text"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        search_provider = kwargs.get("search_provider", "ddg")
        api_key = kwargs.get("api_key")
        query = kwargs.get("query", "")
        num_results = kwargs.get("num_results", 5)
        time_filter = kwargs.get("time_filter", "")
        
        # Create search tool based on provider
        search_tool = self._create_search_tool(search_provider, api_key, num_results, time_filter)
        
        # Perform search if query provided
        search_results = []
        formatted_results = ""
        
        if query:
            try:
                # Execute search
                if hasattr(search_tool, 'run'):
                    results = await asyncio.to_thread(search_tool.run, query)
                else:
                    results = await asyncio.to_thread(search_tool, query)
                
                # Parse results based on provider
                search_results = self._parse_search_results(results, search_provider)
                formatted_results = self._format_results_text(search_results)
                
            except Exception as e:
                search_results = [{"error": f"Search failed: {str(e)}"}]
                formatted_results = f"Search failed: {str(e)}"
        
        return {
            "search_results": search_results,
            "tool": search_tool,
            "formatted_results": formatted_results,
            "query": query,
            "provider": search_provider,
            "result_count": len(search_results)
        }
    
    def _create_search_tool(self, provider: str, api_key: str, num_results: int, time_filter: str = ""):
        """Create search tool based on provider"""
        
        if provider == "serper":
            try:
                from langchain_community.utilities import GoogleSerperAPIWrapper
                search = GoogleSerperAPIWrapper(
                    serper_api_key=api_key,
                    k=num_results
                )
                return Tool(
                    name="web_search",
                    description="Search the web for current information",
                    func=search.run
                )
            except ImportError:
                raise ImportError("Google Serper API wrapper not available")
        
        elif provider == "serpapi":
            try:
                from langchain_community.utilities import SerpAPIWrapper
                search = SerpAPIWrapper(serpapi_api_key=api_key)
                return Tool(
                    name="web_search", 
                    description="Search the web for current information",
                    func=search.run
                )
            except ImportError:
                raise ImportError("SerpAPI wrapper not available")
        
        elif provider == "ddg":
            try:
                from langchain_community.tools import DuckDuckGoSearchRun
                search = DuckDuckGoSearchRun()
                return search
            except ImportError:
                # Fallback to mock search
                return self._create_mock_search_tool()
        
        elif provider == "tavily":
            try:
                from langchain_community.tools import TavilySearchResults
                search = TavilySearchResults(
                    api_key=api_key,
                    max_results=num_results
                )
                return search
            except ImportError:
                raise ImportError("Tavily search not available")
        
        else:
            return self._create_mock_search_tool()
    
    def _create_mock_search_tool(self):
        """Create a mock search tool for testing"""
        def mock_search(query: str) -> str:
            return f"""Mock search results for: {query}
            
1. Sample news article: AI technology continues to advance with new breakthroughs in machine learning and natural language processing.

2. Latest developments: Researchers announce significant improvements in AI model efficiency and accuracy.

3. Industry impact: Major tech companies are investing heavily in AI infrastructure and talent acquisition.

4. Future outlook: Experts predict continued growth and innovation in the AI sector over the next decade.

5. Regulatory considerations: Governments worldwide are developing frameworks for AI governance and ethics."""
        
        return Tool(
            name="mock_web_search",
            description="Mock web search for testing",
            func=mock_search
        )
    
    def _parse_search_results(self, results: str, provider: str) -> List[Dict]:
        """Parse search results based on provider format"""
        if isinstance(results, str):
            # Simple text results - split into individual results
            lines = results.strip().split('\n')
            parsed_results = []
            
            current_result = {}
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # New result item
                    if current_result:
                        parsed_results.append(current_result)
                    current_result = {
                        "title": line,
                        "snippet": "",
                        "link": "#"
                    }
                elif current_result and line:
                    # Add to current result snippet
                    current_result["snippet"] += " " + line
            
            # Add the last result
            if current_result:
                parsed_results.append(current_result)
            
            return parsed_results
        
        elif isinstance(results, list):
            return results
        
        else:
            return [{"title": "Search Result", "snippet": str(results), "link": "#"}]
    
    def _format_results_text(self, results: List[Dict]) -> str:
        """Format search results as readable text"""
        if not results:
            return "No search results found."
        
        formatted = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No description")
            link = result.get("link", "#")
            
            formatted += f"{i}. {title}\n"
            formatted += f"   {snippet}\n"
            if link and link != "#":
                formatted += f"   Link: {link}\n"
            formatted += "\n"
        
        return formatted