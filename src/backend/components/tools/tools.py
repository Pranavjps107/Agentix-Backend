# src/backend/components/tools/tools.py
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools.simple import Tool
from pydantic import BaseModel, Field
from core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component
from typing import Dict, Type,List , Any
import asyncio  # Add this line

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
                field_type="code",
                description="Python function code for the tool"
            ),
            ComponentInput(
                name="input_schema",
                display_name="Input Schema",
                field_type="dict",
                required=False,
                description="Pydantic schema for tool inputs"
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
                method="create_tool"
            ),
            ComponentOutput(
                name="tool_result",
                display_name="Tool Result",
                field_type="any",
                method="execute_tool"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        tool_name = kwargs.get("tool_name")
        tool_description = kwargs.get("tool_description")
        function_code = kwargs.get("function_code")
        input_schema = kwargs.get("input_schema", {})
        return_direct = kwargs.get("return_direct", False)
        
        # Create function from code
        tool_function = self._create_function_from_code(function_code)
        
        # Create tool
        if input_schema:
            # Create Pydantic model for structured tool
            schema_class = self._create_schema_class(input_schema)
            tool = StructuredTool.from_function(
                func=tool_function,
                name=tool_name,
                description=tool_description,
                args_schema=schema_class,
                return_direct=return_direct
            )
        else:
            tool = Tool(
                name=tool_name,
                description=tool_description,
                func=tool_function,
                return_direct=return_direct
           )
        return {
           "tool": tool,
           "tool_name": tool_name,
           "tool_description": tool_description
       }
   
    def _create_function_from_code(self, function_code: str):
       """Create a function from code string"""
       # Execute the function code in a safe environment
       namespace = {}
       exec(function_code, namespace)
       
       # Find the function in the namespace
       for name, obj in namespace.items():
           if callable(obj) and not name.startswith('__'):
               return obj
       
       raise ValueError("No function found in the provided code")
   
    def _create_schema_class(self, input_schema: dict):
       """Create Pydantic model from schema dictionary"""
       fields = {}
       for field_name, field_config in input_schema.items():
           field_type = field_config.get("type", str)
           field_description = field_config.get("description", "")
           field_default = field_config.get("default", ...)
           
           fields[field_name] = (field_type, Field(
               default=field_default,
               description=field_description
           ))
       
       return type("ToolInputSchema", (BaseModel,), {"__annotations__": {k: v[0] for k, v in fields.items()}})

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
               field_type="code",
               description="Python code to execute"
           ),
           ComponentInput(
               name="timeout",
               display_name="Timeout",
               field_type="int",
               default=30,
               required=False,
               description="Execution timeout in seconds"
           ),
           ComponentInput(
               name="globals_dict",
               display_name="Global Variables",
               field_type="dict",
               required=False,
               description="Global variables for code execution"
           )
       ]
       
       self.outputs = [
           ComponentOutput(
               name="result",
               display_name="Execution Result",
               field_type="str",
               method="execute_python"
           ),
           ComponentOutput(
               name="tool",
               display_name="Python REPL Tool",
               field_type="tool",
               method="create_repl_tool"
           )
       ]
   
   async def execute(self, **kwargs) -> Dict[str, Any]:
       code = kwargs.get("code", "")
       timeout = kwargs.get("timeout", 30)
       globals_dict = kwargs.get("globals_dict", {})
       
       # Create Python REPL tool
       from langchain_experimental.tools import PythonREPLTool
       repl_tool = PythonREPLTool()
       
       # Execute code if provided
       result = ""
       if code:
           try:
               result = await asyncio.wait_for(
                   asyncio.to_thread(repl_tool.run, code),
                   timeout=timeout
               )
           except asyncio.TimeoutError:
               result = f"Code execution timed out after {timeout} seconds"
           except Exception as e:
               result = f"Error executing code: {str(e)}"
       
       return {
           "result": result,
           "tool": repl_tool,
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
           )
       ]
       
       self.outputs = [
           ComponentOutput(
               name="search_results",
               display_name="Search Results",
               field_type="list",
               method="search_web"
           ),
           ComponentOutput(
               name="tool",
               display_name="Search Tool",
               field_type="tool",
               method="create_search_tool"
           )
       ]
   
   async def execute(self, **kwargs) -> Dict[str, Any]:
       search_provider = kwargs.get("search_provider")
       api_key = kwargs.get("api_key")
       query = kwargs.get("query", "")
       num_results = kwargs.get("num_results", 5)
       
       # Create search tool based on provider
       search_tool = self._create_search_tool(search_provider, api_key, num_results)
       
       # Perform search if query provided
       search_results = []
       if query:
           try:
               results = await asyncio.to_thread(search_tool.run, query)
               search_results = self._parse_search_results(results, search_provider)
           except Exception as e:
               search_results = [{"error": f"Search failed: {str(e)}"}]
       
       return {
           "search_results": search_results,
           "tool": search_tool,
           "query": query,
           "provider": search_provider
       }
   
   def _create_search_tool(self, provider: str, api_key: str, num_results: int):
       if provider == "serper":
           from langchain_community.tools import GoogleSerperAPIWrapper
           search = GoogleSerperAPIWrapper(serper_api_key=api_key, k=num_results)
           return Tool(
               name="web_search",
               description="Search the web for current information",
               func=search.run
           )
       elif provider == "serpapi":
           from langchain_community.tools import SerpAPIWrapper
           search = SerpAPIWrapper(serpapi_api_key=api_key)
           return Tool(
               name="web_search",
               description="Search the web for current information",
               func=search.run
           )
       elif provider == "ddg":
           from langchain_community.tools import DuckDuckGoSearchRun
           search = DuckDuckGoSearchRun()
           return search
       elif provider == "tavily":
           from langchain_community.tools import TavilySearchResults
           search = TavilySearchResults(api_key=api_key, max_results=num_results)
           return search
       else:
           raise ValueError(f"Unsupported search provider: {provider}")
   
   def _parse_search_results(self, results: str, provider: str) -> List[Dict]:
       """Parse search results based on provider format"""
       # Implementation would depend on the specific format of each provider
       # This is a simplified version
       if isinstance(results, str):
           return [{"content": results}]
       elif isinstance(results, list):
           return results
       else:
           return [{"raw_result": str(results)}]