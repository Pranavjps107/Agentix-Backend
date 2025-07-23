"""
Enhanced Web Search Tool Component for Real-time Information
"""
from typing import Dict, Any, List, Optional
import logging
import asyncio
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component

logger = logging.getLogger(__name__)

@register_component
class WebSearchToolComponent(BaseLangChainComponent):
    """Real-time web search tool for current information"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Web Search",
            description="Search the web for real-time information",
            icon="🔍",
            category="tools",
            tags=["search", "web", "real-time", "news"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="query",
                display_name="Search Query",
                field_type="str",
                description="What to search for"
            ),
            ComponentInput(
                name="search_provider",
                display_name="Search Provider",
                field_type="str",
                options=["serper", "tavily", "duckduckgo"],
                default="duckduckgo",
                description="Search engine provider"
            ),
            ComponentInput(
                name="num_results",
                display_name="Number of Results",
                field_type="int",
                default=5,
                required=False,
                description="How many results to return"
            ),
            ComponentInput(
                name="api_key",
                display_name="API Key",
                field_type="str",
                required=False,
                password=True,
                description="API key for paid search providers"
            ),
            ComponentInput(
                name="time_filter",
                display_name="Time Filter",
                field_type="str",
                options=["any", "day", "week", "month", "year"],
                default="day",
                required=False,
                description="Filter results by time"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="search_results",
                display_name="Search Results",
                field_type="list",
                method="search_web",
                description="List of search results with titles, snippets, and URLs"
            ),
            ComponentOutput(
                name="formatted_content",
                display_name="Formatted Content",
                field_type="str",
                method="get_formatted_content",
                description="Search results formatted as text"
            ),
            ComponentOutput(
                name="search_summary",
                display_name="Search Summary",
                field_type="dict",
                method="get_search_summary",
                description="Summary of search operation"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        query = kwargs.get("query", "")
        search_provider = kwargs.get("search_provider", "duckduckgo")
        num_results = kwargs.get("num_results", 5)
        api_key = kwargs.get("api_key")
        time_filter = kwargs.get("time_filter", "day")
        
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        
        logger.info(f"Searching for: '{query}' using {search_provider}")
        
        try:
            # Perform search based on provider
            search_results = await self._perform_search(
                query, search_provider, num_results, api_key, time_filter
            )
            
            # Format content for LLM consumption
            formatted_content = self._format_search_results(search_results)
            
            # Create search summary
            search_summary = {
                "query": query,
                "provider": search_provider,
                "results_count": len(search_results),
                "time_filter": time_filter,
                "success": True
            }
            
            logger.info(f"Search completed: {len(search_results)} results found")
            
            return {
                "search_results": search_results,
                "formatted_content": formatted_content,
                "search_summary": search_summary
            }
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {
                "search_results": [],
                "formatted_content": f"Search failed: {str(e)}",
                "search_summary": {
                    "query": query,
                    "provider": search_provider,
                    "results_count": 0,
                    "error": str(e),
                    "success": False
                }
            }
    
    async def _perform_search(self, query: str, provider: str, num_results: int, api_key: str, time_filter: str) -> List[Dict[str, Any]]:
        """Perform the actual search"""
        
        if provider == "duckduckgo":
            return await self._duckduckgo_search(query, num_results)
        elif provider == "tavily":
            return await self._tavily_search(query, num_results, api_key)
        elif provider == "serper":
            return await self._serper_search(query, num_results, api_key)
        else:
            raise ValueError(f"Unsupported search provider: {provider}")
    
    async def _duckduckgo_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (free, no API key required)"""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=num_results)
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "url": result.get("href", ""),
                        "source": "duckduckgo"
                    })
            
            return results
            
        except ImportError:
            # Fallback to a simple mock result
            return [{
                "title": f"Mock result for: {query}",
                "snippet": f"This is a mock search result for the query '{query}'. Install duckduckgo-search for real results.",
                "url": "https://example.com",
                "source": "mock"
            }]
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []
    
    async def _tavily_search(self, query: str, num_results: int, api_key: str) -> List[Dict[str, Any]]:
        """Search using Tavily API"""
        if not api_key:
            raise ValueError("Tavily API key required")
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": num_results
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                for result in data.get("results", []):
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("content", ""),
                        "url": result.get("url", ""),
                        "source": "tavily"
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            return []
    
    async def _serper_search(self, query: str, num_results: int, api_key: str) -> List[Dict[str, Any]]:
        """Search using Serper API"""
        if not api_key:
            raise ValueError("Serper API key required")
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    json={"q": query, "num": num_results},
                    headers={"X-API-KEY": api_key}
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                for result in data.get("organic", []):
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "url": result.get("link", ""),
                        "source": "serper"
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Serper search failed: {str(e)}")
            return []
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM consumption"""
        if not results:
            return "No search results found."
        
        formatted = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   URL: {result['url']}\n\n"
        
        return formatted