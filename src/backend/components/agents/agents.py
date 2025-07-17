"""
Agent Components with proper LangChain imports
"""
from typing import Dict, Any, List, Optional
import logging
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component

logger = logging.getLogger(__name__)

@register_component
class OpenAIFunctionsAgentComponent(BaseLangChainComponent):
    """OpenAI Functions Agent Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="OpenAI Functions Agent",
            description="Agent that uses OpenAI function calling",
            icon="🤖",
            category="agents",
            tags=["agent", "openai", "functions"]
        )
        
        self.inputs = [
            ComponentInput(
                name="llm",
                display_name="Language Model",
                field_type="chat_model",
                description="Chat model for the agent"
            ),
            ComponentInput(
                name="tools",
                display_name="Tools",
                field_type="list",
                description="List of tools available to the agent"
            ),
            ComponentInput(
                name="system_message",
                display_name="System Message",
                field_type="str",
                required=False,
                default="You are a helpful assistant.",
                description="System prompt for the agent"
            ),
            ComponentInput(
                name="max_iterations",
                display_name="Max Iterations",
                field_type="int",
                default=10,
                required=False,
                description="Maximum number of agent iterations"
            ),
            ComponentInput(
                name="verbose",
                display_name="Verbose",
                field_type="bool",
                default=False,
                required=False,
                description="Enable verbose logging"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="agent",
                display_name="Agent",
                field_type="agent",
                method="create_agent",
                description="Configured agent"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        llm = kwargs.get("llm")
        tools = kwargs.get("tools", [])
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        max_iterations = kwargs.get("max_iterations", 10)
        verbose = kwargs.get("verbose", False)
        
        try:
            # Import LangChain components - using the available imports
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            
            # For now, return a simplified agent structure
            agent_config = {
                "type": "openai_functions",
                "llm": llm,
                "tools": tools,
                "prompt": prompt,
                "max_iterations": max_iterations,
                "verbose": verbose
            }
            
            return {
                "agent": agent_config,
                "agent_type": "openai_functions",
                "tool_count": len(tools),
                "max_iterations": max_iterations
            }
            
        except Exception as e:
            logger.error(f"Failed to create OpenAI Functions Agent: {str(e)}")
            # Return a mock agent for now
            return {
                "agent": {
                    "type": "mock_openai_functions",
                    "error": str(e)
                },
                "agent_type": "mock_openai_functions",
                "tool_count": len(tools),
                "max_iterations": max_iterations
            }

@register_component
class ReActAgentComponent(BaseLangChainComponent):
    """ReAct Agent Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="ReAct Agent",
            description="ReAct (Reasoning + Acting) agent",
            icon="🧠",
            category="agents",
            tags=["agent", "react", "reasoning"]
        )
        
        self.inputs = [
            ComponentInput(
                name="llm",
                display_name="Language Model",
                field_type="llm",
                description="Language model for the agent"
            ),
            ComponentInput(
                name="tools",
                display_name="Tools",
                field_type="list",
                description="List of tools available to the agent"
            ),
            ComponentInput(
                name="system_message",
                display_name="System Message",
                field_type="str",
                required=False,
                description="System prompt for the agent"
            ),
            ComponentInput(
                name="max_iterations",
                display_name="Max Iterations",
                field_type="int",
                default=10,
                required=False,
                description="Maximum number of agent iterations"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="agent",
                display_name="Agent",
                field_type="agent",
                method="create_react_agent",
                description="ReAct agent"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        llm = kwargs.get("llm")
        tools = kwargs.get("tools", [])
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        max_iterations = kwargs.get("max_iterations", 10)
        
        try:
            # Create ReAct prompt template
            from langchain_core.prompts import PromptTemplate
            
            prompt = PromptTemplate.from_template(
                "Answer the following questions as best you can. You have access to the following tools:\n\n"
                "{tools}\n\n"
                "Use the following format:\n\n"
                "Question: the input question you must answer\n"
                "Thought: you should always think about what to do\n"
                "Action: the action to take, should be one of [{tool_names}]\n"
                "Action Input: the input to the action\n"
                "Observation: the result of the action\n"
                "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
                "Thought: I now know the final answer\n"
                "Final Answer: the final answer to the original input question\n\n"
                "Begin!\n\n"
                "Question: {input}\n"
                "Thought:{agent_scratchpad}"
            )
            
            agent_config = {
                "type": "react",
                "llm": llm,
                "tools": tools,
                "prompt": prompt,
                "max_iterations": max_iterations
            }
            
            return {
                "agent": agent_config,
                "agent_type": "react",
                "tool_count": len(tools),
                "max_iterations": max_iterations
            }
            
        except Exception as e:
            logger.error(f"Failed to create ReAct Agent: {str(e)}")
            return {
                "agent": {
                    "type": "mock_react",
                    "error": str(e)
                },
                "agent_type": "mock_react",
                "tool_count": len(tools),
                "max_iterations": max_iterations
            }

@register_component
class AgentExecutorComponent(BaseLangChainComponent):
    """Agent Executor Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Agent Executor",
            description="Execute agent with input query",
            icon="⚡",
            category="agents",
            tags=["agent", "executor", "run"]
        )
        
        self.inputs = [
            ComponentInput(
                name="agent",
                display_name="Agent",
                field_type="agent",
                description="Configured agent"
            ),
            ComponentInput(
                name="input_query",
                display_name="Input Query",
                field_type="str",
                description="Query to send to the agent"
            ),
            ComponentInput(
                name="chat_history",
                display_name="Chat History",
                field_type="list",
                required=False,
                description="Previous conversation history"
            ),
            ComponentInput(
                name="return_intermediate_steps",
                display_name="Return Intermediate Steps",
                field_type="bool",
                default=True,
                required=False,
                description="Return reasoning steps"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="response",
                display_name="Agent Response",
                field_type="str",
                method="execute_agent",
                description="Agent response"
            ),
            ComponentOutput(
                name="intermediate_steps",
                display_name="Intermediate Steps",
                field_type="list",
                method="get_steps",
                description="Reasoning steps"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        agent = kwargs.get("agent")
        input_query = kwargs.get("input_query")
        chat_history = kwargs.get("chat_history", [])
        return_intermediate_steps = kwargs.get("return_intermediate_steps", True)
        
        try:
            # For now, return a mock response
            response = f"Agent processed query: {input_query}"
            
            intermediate_steps = [
                {
                    "action": {
                        "tool": "thinking",
                        "tool_input": input_query,
                        "log": f"Processing query: {input_query}"
                    },
                    "observation": "Query processed successfully"
                }
            ]
            
            execution_metadata = {
                "total_steps": len(intermediate_steps),
                "input_length": len(input_query),
                "output_length": len(response),
                "success": True
            }
            
            return {
                "response": response,
                "intermediate_steps": intermediate_steps,
                "execution_metadata": execution_metadata,
                "input_query": input_query
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            return {
                "response": f"Agent execution failed: {str(e)}",
                "intermediate_steps": [],
                "execution_metadata": {
                    "error": str(e),
                    "success": False
                },
                "input_query": input_query
            }