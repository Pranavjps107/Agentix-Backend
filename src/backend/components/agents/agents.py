# src/backend/components/agents/agents.py
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_react_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component
from typing import Dict, Type,List , Any

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
                field_type="text",
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
                name="agent_executor",
                display_name="Agent Executor",
                field_type="agent_executor",
                method="create_agent"
            ),
            ComponentOutput(
                name="agent_response",
                display_name="Agent Response",
                field_type="dict",
                method="run_agent"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        llm = kwargs.get("llm")
        tools = kwargs.get("tools", [])
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        max_iterations = kwargs.get("max_iterations", 10)
        verbose = kwargs.get("verbose", False)
        
        # Create agent prompt
        from langchain.agents import create_openai_functions_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_openai_functions_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=max_iterations,
            verbose=verbose,
            return_intermediate_steps=True
        )
        
        return {
            "agent_executor": agent_executor,
            "agent_type": "openai_functions",
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
                field_type="text",
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
                name="agent_executor",
                display_name="Agent Executor",
                field_type="agent_executor",
                method="create_react_agent"
            ),
            ComponentOutput(
                name="agent_response",
                display_name="Agent Response",
                field_type="dict",
                method="run_agent"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        llm = kwargs.get("llm")
        tools = kwargs.get("tools", [])
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        max_iterations = kwargs.get("max_iterations", 10)
        
        # Create ReAct prompt
        from langchain import hub
        
        try:
            prompt = hub.pull("hwchase17/react")
        except:
            # Fallback prompt if hub is not available
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
        
        # Create ReAct agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=max_iterations,
            verbose=True,
            return_intermediate_steps=True
        )
        
        return {
            "agent_executor": agent_executor,
            "agent_type": "react",
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
                name="agent_executor",
                display_name="Agent Executor",
                field_type="agent_executor",
                description="Configured agent executor"
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
                method="execute_agent"
            ),
            ComponentOutput(
                name="intermediate_steps",
                display_name="Intermediate Steps",
                field_type="list",
                method="get_steps"
            ),
            ComponentOutput(
                name="execution_metadata",
                display_name="Execution Metadata",
                field_type="dict",
                method="get_metadata"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        agent_executor = kwargs.get("agent_executor")
        input_query = kwargs.get("input_query")
        chat_history = kwargs.get("chat_history", [])
        return_intermediate_steps = kwargs.get("return_intermediate_steps", True)
        
        # Prepare input
        agent_input = {
            "input": input_query,
            "chat_history": chat_history
        }
        
        # Execute agent
        try:
            result = await agent_executor.ainvoke(
                agent_input,
                return_only_outputs=False,
                include_run_info=True
            )
            
            response = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Format intermediate steps
            formatted_steps = []
            for step in intermediate_steps:
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    formatted_steps.append({
                        "action": {
                            "tool": action.tool if hasattr(action, 'tool') else str(action),
                            "tool_input": action.tool_input if hasattr(action, 'tool_input') else "",
                            "log": action.log if hasattr(action, 'log') else ""
                        },
                        "observation": str(observation)
                    })
            
            execution_metadata = {
                "total_steps": len(formatted_steps),
                "input_length": len(input_query),
                "output_length": len(response),
                "success": True
            }
            
        except Exception as e:
            response = f"Agent execution failed: {str(e)}"
            formatted_steps = []
            execution_metadata = {
                "error": str(e),
                "success": False
            }
        
        return {
            "response": response,
            "intermediate_steps": formatted_steps,
            "execution_metadata": execution_metadata,
            "input_query": input_query
        }