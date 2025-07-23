# src/backend/components/tools/web_search.py
"""
Web Search Tool Component
"""
from typing import Dict, Any, List
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component  # Correct import

@register_component
class WebSearchToolComponent(BaseLangChainComponent):
    """Web Search Tool Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Web Search Tool",
            description="Search the web for information",
            icon="🔍",
            category="tools",
            tags=["search", "web", "information"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="search_provider",
                display_name="Search Provider",
                field_type="str",
                options=["ddg", "serper", "serpapi", "tavily"],
                default="ddg",
                description="Web search provider"
            ),
            ComponentInput(
                name="query",
                display_name="Search Query",
                field_type="str",
                required=False,
                description="Search query"
            ),
            ComponentInput(
                name="num_results",
                display_name="Number of Results",
                field_type="int",
                default=5,
                required=False,
                description="Number of search results to return"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="search_results",
                display_name="Search Results",
                field_type="list",
                method="search_web",
                description="Search results from the web"
            ),
            ComponentOutput(
                name="tool",
                display_name="Search Tool",
                field_type="tool",
                method="create_search_tool",
                description="LangChain tool for web search"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        search_provider = kwargs.get("search_provider", "ddg")
        query = kwargs.get("query", "")
        num_results = kwargs.get("num_results", 5)
        
        # Create mock search tool for now
        from langchain_core.tools import Tool
        
        def mock_search(query: str) -> str:
            return f"Mock search results for '{query}' using {search_provider}"
        
        search_tool = Tool(
            name="web_search",
            description="Search the web for current information",
            func=mock_search
        )
        
        # Perform search if query provided
        search_results = []
        if query:
            result = mock_search(query)
            search_results = [{"content": result, "provider": search_provider}]
        
        return {
            "search_results": search_results,
            "tool": search_tool,
            "query": query,
            "provider": search_provider
        }