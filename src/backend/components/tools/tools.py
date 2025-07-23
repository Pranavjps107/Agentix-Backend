# src/backend/components/tools/tools.py
"""
Custom Tool Components
"""
from typing import Dict, Any, List
import asyncio
from langchain_core.tools import BaseTool, StructuredTool, Tool
from pydantic import BaseModel, Field
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component

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
                name="tool_info",
                display_name="Tool Info",
                field_type="dict",
                method="create_tool",
                description="Information about the created tool"
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
        
        # Return serializable tool information instead of the actual tool object
        return {
            "tool_info": {
                "name": tool_name,
                "description": tool_description,
                "return_direct": return_direct,
                "type": "custom_tool"
            },
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
                name="tool_info",
                display_name="Python REPL Tool Info",
                field_type="dict",
                method="create_repl_tool",
                description="Information about the Python REPL tool"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 30)
        
        # Execute code if provided (mock execution for safety)
        result = ""
        if code:
            result = f"Mock execution of: {code}\nResult: Code executed successfully (mock)"
        
        return {
            "result": result,
            "tool_info": {
                "name": "python_repl",
                "description": "Execute Python code (mock for safety)",
                "type": "python_repl"
            },
            "code_executed": code
        }

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
        
        # Perform search (using mock search for now)
        search_results = []
        formatted_results = ""
        
        if query:
            try:
                # Use mock search results for testing
                search_results = self._get_mock_search_results(query, num_results)
                formatted_results = self._format_results_text(search_results)
                
            except Exception as e:
                search_results = [{"error": f"Search failed: {str(e)}"}]
                formatted_results = f"Search failed: {str(e)}"
        
        return {
            "search_results": search_results,
            "formatted_results": formatted_results,
            "query": query,
            "provider": search_provider,
            "result_count": len(search_results)
        }
    
    def _get_mock_search_results(self, query: str, num_results: int) -> List[Dict]:
        """Generate mock search results for testing"""
        mock_results = [
            {
                "title": f"Latest {query} Development: Revolutionary Breakthrough Announced",
                "snippet": f"Scientists and researchers have made significant progress in {query}, with new innovations promising to transform the industry. The latest developments show remarkable potential for widespread adoption.",
                "link": "https://example.com/news1",
                "source": "Tech News Daily"
            },
            {
                "title": f"{query} Market Analysis: Growth Projections for 2024",
                "snippet": f"Market analysts predict substantial growth in the {query} sector, with investments reaching record levels. Industry leaders are optimistic about future prospects and continued innovation.",
                "link": "https://example.com/analysis2",
                "source": "Market Research Today"
            },
            {
                "title": f"Expert Opinion: The Future of {query}",
                "snippet": f"Leading experts discuss the implications of recent {query} advances and their potential impact on society. The consensus points to transformative changes across multiple sectors.",
                "link": "https://example.com/expert3",
                "source": "Expert Insights"
            },
            {
                "title": f"Global {query} Initiative Gains Momentum",
                "snippet": f"International collaboration on {query} research has intensified, with major funding announcements and new partnerships forming. The global community is working together to accelerate progress.",
                "link": "https://example.com/global4",
                "source": "Global Tech Tribune"
            },
            {
                "title": f"{query} Regulatory Framework Updated",
                "snippet": f"Governments worldwide are updating regulations related to {query} to ensure safe and ethical development. New guidelines aim to balance innovation with responsible deployment.",
                "link": "https://example.com/regulation5",
                "source": "Policy Watch"
            }
        ]
        
        return mock_results[:num_results]
    
    def _format_results_text(self, results: List[Dict]) -> str:
        """Format search results as readable text"""
        if not results:
            return "No search results found."
        
        formatted = f"Search Results ({len(results)} found):\n\n"
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No description")
            source = result.get("source", "Unknown source")
            
            formatted += f"{i}. {title}\n"
            formatted += f"   Source: {source}\n"
            formatted += f"   {snippet}\n\n"
        
        return formatted