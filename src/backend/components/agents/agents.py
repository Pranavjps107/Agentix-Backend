# src/backend/components/agents/agents.py
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_react_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component
from typing import Dict, Type, List, Any, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import Dict, Type, List, Any, Union

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
        
        try:
            # Create agent prompt with chat history support
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            
            # Create agent using modern approach
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
            
        except Exception as e:
            # Fallback to a simple mock agent for demo purposes
            return {
                "agent_executor": None,
                "agent_type": "mock",
                "tool_count": len(tools),
                "max_iterations": max_iterations,
                "error": f"Agent creation failed: {str(e)}"
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
        
        try:
            # Create ReAct prompt with chat history support
            from langchain import hub
            
            try:
                # Try to get the standard ReAct prompt and modify it for chat history
                base_prompt = hub.pull("hwchase17/react")
                
                # Create a chat-enabled version of the ReAct prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"{system_message}\n\nYou have access to the following tools: {{tools}}\n\nUse the ReAct format (Thought/Action/Action Input/Observation)."),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "Question: {input}\nThought:"),
                    MessagesPlaceholder("agent_scratchpad"),
                ])
            except:
                # Fallback prompt with chat history support
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"{system_message}\n\nAnswer the following questions as best you can. You have access to the following tools:\n\n{{tools}}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{{tool_names}}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question"),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "Question: {input}\nThought:"),
                    MessagesPlaceholder("agent_scratchpad"),
                ])
            
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
            
        except Exception as e:
            # Fallback for demo purposes
            return {
                "agent_executor": None,
                "agent_type": "mock",
                "tool_count": len(tools),
                "max_iterations": max_iterations,
                "error": f"Agent creation failed: {str(e)}"
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
            ),
            ComponentOutput(
                name="updated_chat_history",
                display_name="Updated Chat History",
                field_type="list",
                method="get_updated_history"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        agent_executor = kwargs.get("agent_executor")
        input_query = kwargs.get("input_query")
        chat_history = kwargs.get("chat_history", [])
        return_intermediate_steps = kwargs.get("return_intermediate_steps", True)
        
        # Convert chat history to LangChain messages format
        formatted_chat_history = self._convert_to_langchain_messages(chat_history)
        
        # Handle mock agents for demo
        if agent_executor is None:
            mock_updated_history = formatted_chat_history + [
                HumanMessage(content=input_query),
                AIMessage(content=f"Mock response for query: {input_query}")
            ]
            return {
                "response": f"Mock response for query: {input_query}",
                "intermediate_steps": [],
                "execution_metadata": {
                    "total_steps": 0,
                    "input_length": len(input_query or ""),
                    "output_length": 20,
                    "success": True,
                    "agent_type": "mock",
                    "chat_history_length": len(mock_updated_history)
                },
                "input_query": input_query,
                "updated_chat_history": self._convert_to_serializable(mock_updated_history)
            }
        
        # Prepare input with formatted chat history
        agent_input = {
            "input": input_query,
            "chat_history": formatted_chat_history
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
            
            # Create updated chat history with new conversation turn
            updated_chat_history = formatted_chat_history + [
                HumanMessage(content=input_query),
                AIMessage(content=response)
            ]
            
            execution_metadata = {
                "total_steps": len(formatted_steps),
                "input_length": len(input_query),
                "output_length": len(response),
                "success": True,
                "chat_history_length": len(updated_chat_history),
                "previous_history_length": len(formatted_chat_history)
            }
            
        except Exception as e:
            response = f"Agent execution failed: {str(e)}"
            formatted_steps = []
            
            # Still create updated history even on error for continuity
            updated_chat_history = formatted_chat_history + [
                HumanMessage(content=input_query),
                AIMessage(content=response)
            ]
            
            execution_metadata = {
                "error": str(e),
                "success": False,
                "chat_history_length": len(updated_chat_history),
                "previous_history_length": len(formatted_chat_history)
            }
        
        return {
            "response": response,
            "intermediate_steps": formatted_steps,
            "execution_metadata": execution_metadata,
            "input_query": input_query,
            "updated_chat_history": self._convert_to_serializable(updated_chat_history)
        }
    
    def _convert_to_langchain_messages(self, chat_history: List[Union[Dict, str, BaseMessage]]) -> List[BaseMessage]:
        """Convert various chat history formats to LangChain BaseMessage objects"""
        messages = []
        
        for item in chat_history:
            if isinstance(item, BaseMessage):
                # Already a LangChain message
                messages.append(item)
            elif isinstance(item, dict):
                # Convert dict format to message
                role = item.get("role", "human").lower()
                content = item.get("content", "")
                
                if role in ["human", "user"]:
                    messages.append(HumanMessage(content=content))
                elif role in ["ai", "assistant", "bot"]:
                    messages.append(AIMessage(content=content))
            elif isinstance(item, str):
                # Treat strings as human messages
                messages.append(HumanMessage(content=item))
        
        return messages
    
    def _convert_to_serializable(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages back to serializable format"""
        serialized = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                serialized.append({
                    "role": "human",
                    "content": message.content,
                    "type": "human"
                })
            elif isinstance(message, AIMessage):
                serialized.append({
                    "role": "assistant", 
                    "content": message.content,
                    "type": "ai"
                })
        
        return serialized
    
    def get_updated_history(self, **kwargs):
        """Helper method for updated_chat_history output"""
        return kwargs.get("updated_chat_history", [])
    
    def execute_agent(self, **kwargs):
        """Helper method for response output"""
        return kwargs.get("response", "")
    
    def get_steps(self, **kwargs):
        """Helper method for intermediate_steps output"""
        return kwargs.get("intermediate_steps", [])
    
    def get_metadata(self, **kwargs):
        """Helper method for execution_metadata output"""
        return kwargs.get("execution_metadata", {})