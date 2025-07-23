"""
Flow Execution Service
"""
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Set
from uuid import uuid4

from models.flow import FlowDefinition, FlowNode, FlowEdge, FlowStatus
from models.execution import ExecutionResult, ExecutionStatus, TaskInfo, ExecutionStep
from core.registry import ComponentRegistry
from core.exceptions import FlowException, FlowValidationException, FlowExecutionException
from services.component_manager import ComponentManager

logger = logging.getLogger(__name__)

class FlowExecutor:
    """Execute complete flows with dependency resolution and optimization"""
    
    def __init__(self):
        self.component_manager = ComponentManager()
        self.execution_cache: Dict[str, TaskInfo] = {}
        self.active_executions: Set[str] = set()
    
    async def execute_flow(
        self, 
        flow_definition: FlowDefinition, 
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a complete flow synchronously"""
        start_time = time.time()
        execution_id = str(uuid4())
        
        logger.info(f"Starting flow execution: {flow_definition.name} (ID: {flow_definition.id})")
        
        try:
            # Validate flow
            validation_result = await self.validate_flow(flow_definition)
            if not validation_result["valid"]:
                raise FlowValidationException(f"Flow validation failed: {validation_result['errors']}")
            
            # Build execution graph and order
            execution_order = self._build_execution_order(flow_definition)
            logger.info(f"Execution order: {execution_order}")
            
            # Execute components in topological order
            component_outputs = {}
            execution_steps = []
            
            for node_id in execution_order:
                node = self._get_node_by_id(flow_definition, node_id)
                if not node:
                    logger.warning(f"Node {node_id} not found, skipping")
                    continue
                
                step_start_time = time.time()
                
                try:
                    # Prepare inputs for this node
                    node_inputs = await self._prepare_node_inputs(
                        node, flow_definition, component_outputs, inputs
                    )
                    
                    logger.info(f"Executing node {node_id} ({node.component_type})")
                    
                    # Execute component
                    result = await self.component_manager.execute_component(
                        component_name=node.component_type,
                        inputs=node_inputs,
                        component_id=node.id
                    )
                    
                    if result["success"]:
                        component_outputs[node_id] = result["outputs"]
                        logger.info(f"Node {node_id} completed successfully")
                    else:
                        logger.error(f"Node {node_id} failed: {result.get('error', 'Unknown error')}")
                        raise FlowExecutionException(f"Node {node_id} execution failed: {result.get('error')}")
                    
                    # Record execution step
                    step = ExecutionStep(
                        component_id=node.id,
                        component_name=node.component_type,
                        status=ExecutionStatus.COMPLETED,
                        inputs=node_inputs,
                        outputs=result["outputs"],
                        execution_time=result["execution_time"],
                        start_time=step_start_time,
                        end_time=time.time()
                    )
                    execution_steps.append(step)
                    
                except Exception as e:
                    logger.error(f"Node {node_id} execution failed: {str(e)}")
                    
                    # Record failed step
                    step = ExecutionStep(
                        component_id=node.id,
                        component_name=node.component_type,
                        status=ExecutionStatus.FAILED,
                        inputs=node_inputs if 'node_inputs' in locals() else {},
                        error=str(e),
                        execution_time=time.time() - step_start_time,
                        start_time=step_start_time,
                        end_time=time.time()
                    )
                    execution_steps.append(step)
                    raise
            
            # Get final outputs
            final_outputs = self._get_final_outputs(flow_definition, component_outputs)
            
            execution_time = time.time() - start_time
            
            logger.info(f"Flow {flow_definition.name} completed successfully in {execution_time:.2f}s")
            
            return {
                "outputs": final_outputs,
                "execution_time": execution_time,
                "component_outputs": component_outputs,
                "execution_steps": [step.dict() for step in execution_steps],
                "execution_order": execution_order,
                "success": True,
                "flow_id": flow_definition.id,
                "execution_id": execution_id
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Flow execution failed: {error_msg}")
            
            return {
                "outputs": {},
                "execution_time": execution_time,
                "error": error_msg,
                "success": False,
                "flow_id": flow_definition.id,
                "execution_id": execution_id
            }
    
    async def execute_flow_async(
        self, 
        flow_definition: FlowDefinition, 
        inputs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute flow asynchronously and return task ID"""
        task_id = str(uuid4())
        
        # Store task info
        task_info = TaskInfo(
            task_id=task_id,
            status=ExecutionStatus.PENDING,
            metadata={
                "flow_id": flow_definition.id,
                "flow_name": flow_definition.name
            }
        )
        self.execution_cache[task_id] = task_info
        self.active_executions.add(task_id)
        
        # Start execution in background
        asyncio.create_task(self._execute_flow_background(task_id, flow_definition, inputs))
        
        logger.info(f"Started async flow execution: {task_id}")
        return task_id
    
    async def _execute_flow_background(
        self, 
        task_id: str, 
        flow_definition: FlowDefinition, 
        inputs: Optional[Dict[str, Any]]
    ):
        """Background execution of flow"""
        task_info = self.execution_cache[task_id]
        
        try:
            task_info.status = ExecutionStatus.RUNNING
            result = await self.execute_flow(flow_definition, inputs)
            
            # Create execution result
            execution_result = ExecutionResult(
                status=ExecutionStatus.COMPLETED if result["success"] else ExecutionStatus.FAILED,
                outputs=result.get("outputs", {}),
                error=result.get("error"),
                execution_time=result.get("execution_time", 0),
                metadata=result
            )
            
            task_info.result = execution_result
            task_info.status = execution_result.status
            task_info.progress = 100.0
            task_info.end_time = time.time()
            
        except Exception as e:
            logger.error(f"Background flow execution failed: {str(e)}")
            
            execution_result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=0
            )
            
            task_info.result = execution_result
            task_info.status = ExecutionStatus.FAILED
            task_info.error = str(e)
            task_info.end_time = time.time()
        
        finally:
            self.active_executions.discard(task_id)
    
    async def get_execution_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of async execution"""
        if task_id not in self.execution_cache:
            raise ValueError(f"Task {task_id} not found")
        
        task_info = self.execution_cache[task_id]
        return task_info.dict()
    
    async def cancel_execution(self, task_id: str) -> bool:
        """Cancel an async flow execution"""
        if task_id not in self.execution_cache:
            return False
        
        task_info = self.execution_cache[task_id]
        
        if task_info.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
            return False
        
        task_info.status = ExecutionStatus.CANCELLED
        task_info.end_time = time.time()
        self.active_executions.discard(task_id)
        
        logger.info(f"Cancelled execution: {task_id}")
        return True
    
    def _build_execution_order(self, flow_definition: FlowDefinition) -> List[str]:
        """Build topological execution order using Kahn's algorithm"""
        # Create adjacency list and in-degree count
        dependencies = {}  # node -> list of dependencies
        in_degree = {}     # node -> number of incoming edges
        all_nodes = set()
        
        # Initialize
        for node in flow_definition.nodes:
            dependencies[node.id] = []
            in_degree[node.id] = 0
            all_nodes.add(node.id)
        
        # Build dependency graph
        for edge in flow_definition.edges:
            if edge.target not in dependencies:
                dependencies[edge.target] = []
            dependencies[edge.target].append(edge.source)
            in_degree[edge.target] += 1
        
        # Kahn's algorithm for topological sort
        queue = [node_id for node_id in all_nodes if in_degree[node_id] == 0]
        execution_order = []
        
        while queue:
            node_id = queue.pop(0)
            execution_order.append(node_id)
            
            # Check all nodes that depend on this node
            for target_node in all_nodes:
                if node_id in dependencies[target_node]:
                    in_degree[target_node] -= 1
                    if in_degree[target_node] == 0:
                        queue.append(target_node)
        
        # Check for cycles
        if len(execution_order) != len(all_nodes):
            raise FlowValidationException("Circular dependency detected in flow")
        
        return execution_order
    
    async def _prepare_node_inputs(
        self, 
        node: FlowNode, 
        flow_definition: FlowDefinition, 
        component_outputs: Dict[str, Any],
        global_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare inputs for a node from connected outputs and global inputs"""
        node_inputs = {}
        
        # Start with global inputs
        if global_inputs:
            # Map flow-level inputs to component inputs
            for key, value in global_inputs.items():
                if key == "topic" and node.id == "topic-input":
                    node_inputs["user_input"] = value
                else:
                    node_inputs[key] = value
        
        # Add node's own data
        if node.data:
            node_inputs.update(node.data)
        
        # Add inputs from connected nodes
        for edge in flow_definition.edges:
            if edge.target == node.id:
                source_outputs = component_outputs.get(edge.source, {})
                
                if edge.source_handle and edge.target_handle:
                    # Specific output to specific input mapping
                    if edge.source_handle in source_outputs:
                        value = source_outputs[edge.source_handle]
                        # Convert lists to strings for chat models
                        if edge.target_handle == "messages" and isinstance(value, list):
                            # Convert search results to readable text
                            if all(isinstance(item, dict) for item in value):
                                text_content = ""
                                for item in value:
                                    if "title" in item and "snippet" in item:
                                        text_content += f"Title: {item['title']}\nContent: {item['snippet']}\n\n"
                                    else:
                                        text_content += str(item) + "\n"
                                node_inputs[edge.target_handle] = text_content
                            else:
                                node_inputs[edge.target_handle] = "\n".join(str(item) for item in value)
                        else:
                            node_inputs[edge.target_handle] = value
                else:
                    # Merge all outputs (be careful of key conflicts)
                    for key, value in source_outputs.items():
                        if key not in node_inputs:  # Don't override existing keys
                            node_inputs[key] = value
        
        return node_inputs
    def _get_node_by_id(self, flow_definition: FlowDefinition, node_id: str) -> Optional[FlowNode]:
        """Get node by ID"""
        for node in flow_definition.nodes:
            if node.id == node_id:
                return node
        return None
    
    def _get_final_outputs(
        self, 
        flow_definition: FlowDefinition, 
        component_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get final outputs from terminal nodes (nodes with no outgoing edges)"""
        final_outputs = {}
        
        # Find nodes that have no outgoing edges
        nodes_with_outgoing = set()
        for edge in flow_definition.edges:
            nodes_with_outgoing.add(edge.source)
        
        terminal_nodes = []
        for node in flow_definition.nodes:
            if node.id not in nodes_with_outgoing:
                terminal_nodes.append(node.id)
        
        # If no terminal nodes found, use all nodes
        if not terminal_nodes:
            terminal_nodes = [node.id for node in flow_definition.nodes]
        
        # Collect outputs from terminal nodes
        for node_id in terminal_nodes:
            node_outputs = component_outputs.get(node_id, {})
            if node_outputs:
                final_outputs[node_id] = node_outputs
        
        return final_outputs
    
    async def validate_flow(self, flow_definition: FlowDefinition) -> Dict[str, Any]:
        """Validate flow definition"""
        errors = []
        warnings = []
        
        try:
            # Check for missing components
            for node in flow_definition.nodes:
                component_class = ComponentRegistry.get_component(node.component_type)
                if not component_class:
                    errors.append(f"Component '{node.component_type}' not found for node {node.id}")
            
            # Check for circular dependencies
            try:
                self._build_execution_order(flow_definition)
            except FlowValidationException as e:
                errors.append(str(e))
            
            # Check for disconnected nodes (only if flow has multiple nodes)
            if len(flow_definition.nodes) > 1:
                connected_nodes = set()
                for edge in flow_definition.edges:
                    connected_nodes.add(edge.source)
                    connected_nodes.add(edge.target)
                
                for node in flow_definition.nodes:
                    if node.id not in connected_nodes:
                        warnings.append(f"Node {node.id} ({node.component_type}) is not connected to any other nodes")
            
            # Validate edge connections
            node_ids = {node.id for node in flow_definition.nodes}
            for edge in flow_definition.edges:
                if edge.source not in node_ids:
                    errors.append(f"Edge source '{edge.source}' references non-existent node")
                if edge.target not in node_ids:
                    errors.append(f"Edge target '{edge.target}' references non-existent node")
            
            # Check for input/output compatibility
            for edge in flow_definition.edges:
                if edge.source_handle and edge.target_handle:
                    source_node = self._get_node_by_id(flow_definition, edge.source)
                    target_node = self._get_node_by_id(flow_definition, edge.target)
                    
                    if source_node and target_node:
                        source_component = ComponentRegistry.get_component_instance(source_node.component_type)
                        target_component = ComponentRegistry.get_component_instance(target_node.component_type)
                        
                        if source_component and target_component:
                            # Check if output exists
                            source_output = source_component.get_output_by_name(edge.source_handle)
                            if not source_output:
                                warnings.append(f"Output '{edge.source_handle}' not found in component '{source_node.component_type}'")
                            
                            # Check if input exists
                            target_input = target_component.get_input_by_name(edge.target_handle)
                            if not target_input:
                                warnings.append(f"Input '{edge.target_handle}' not found in component '{target_node.component_type}'")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "node_count": len(flow_definition.nodes),
            "edge_count": len(flow_definition.edges)
        }
    
    async def export_as_langchain_code(self, flow_definition: FlowDefinition) -> str:
        """Export flow as LangChain Python code"""
        code_lines = [
            "# Generated LangChain code",
            f"# Flow: {flow_definition.name}",
            f"# Description: {flow_definition.description or 'Auto-generated flow'}",
            "import asyncio",
            "from typing import Dict, Any",
            "",
            "# Import required LangChain modules",
            "from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda",
            "from langchain_core.output_parsers import StrOutputParser",
            "",
        ]
        
        try:
            # Build execution order
            execution_order = self._build_execution_order(flow_definition)
            
            # Generate component initialization code
            code_lines.append("async def create_flow():")
            code_lines.append('    """Create and return the LangChain flow"""')
            
            component_vars = {}
            
            for i, node_id in enumerate(execution_order):
                node = self._get_node_by_id(flow_definition, node_id)
                if not node:
                    continue
                    
                var_name = f"component_{i}"
                component_vars[node_id] = var_name
                
                # Generate component creation code based on type
                creation_code = self._generate_component_creation_code(node, var_name)
                code_lines.extend([f"    {line}" for line in creation_code])
                code_lines.append("")
            
            # Generate flow composition
            if len(component_vars) > 1:
                chain_components = list(component_vars.values())
                code_lines.append("    # Create the chain")
                code_lines.append(f"    flow = {' | '.join(chain_components)}")
            elif len(component_vars) == 1:
                code_lines.append(f"    flow = {list(component_vars.values())[0]}")
            else:
                code_lines.append("    flow = RunnablePassthrough()  # Empty flow")
            
            code_lines.extend([
                "",
                "    return flow",
                "",
                "# Example usage:",
                "# flow = await create_flow()",
                "# result = await flow.ainvoke({'input': 'your input here'})",
                "",
                f"# Original flow ID: {flow_definition.id}",
                f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            ])
            
            return "\n".join(code_lines)
            
        except Exception as e:
            logger.error(f"Code export failed: {str(e)}")
            raise FlowException(f"Failed to export flow as code: {str(e)}")
    
    def _generate_component_creation_code(self, node: FlowNode, var_name: str) -> List[str]:
        """Generate code for creating a specific component"""
        component_type = node.component_type
        node_data = node.data or {}
        
        if "LLM" in component_type or "llm" in component_type.lower():
            model_name = node_data.get('model_name', 'gpt-3.5-turbo')
            temperature = node_data.get('temperature', 0.7)
            max_tokens = node_data.get('max_tokens', 256)
            
            return [
                f"# {component_type}",
                f"from langchain_openai import ChatOpenAI",
                f"{var_name} = ChatOpenAI(",
                f"    model='{model_name}',",
                f"    temperature={temperature},",
                f"    max_tokens={max_tokens}",
                f")"
            ]
        
        elif "prompt" in component_type.lower():
            template = node_data.get('template', '{input}')
            return [
                f"# {component_type}",
                f"from langchain_core.prompts import PromptTemplate",
                f"{var_name} = PromptTemplate.from_template('{template}')"
            ]
        
        elif "parser" in component_type.lower():
            return [
                f"# {component_type}",
                f"from langchain_core.output_parsers import StrOutputParser",
                f"{var_name} = StrOutputParser()"
            ]
        
        elif "embeddings" in component_type.lower():
           model_name = node_data.get('model_name', 'text-embedding-ada-002')
           provider = node_data.get('provider', 'openai')
           
           if provider == 'openai':
               return [
                   f"# {component_type}",
                   f"from langchain_openai import OpenAIEmbeddings",
                   f"{var_name} = OpenAIEmbeddings(model='{model_name}')"
               ]
           else:
               return [
                   f"# {component_type}",
                   f"from langchain_community.embeddings import HuggingFaceEmbeddings",
                   f"{var_name} = HuggingFaceEmbeddings(model_name='{model_name}')"
               ]
       
        elif "vector" in component_type.lower():
           return [
               f"# {component_type}",
               f"# Vector store setup would require additional configuration",
               f"{var_name} = RunnablePassthrough()  # Placeholder for vector store"
           ]
       
        elif "retriever" in component_type.lower():
           return [
               f"# {component_type}",
               f"# Retriever setup would require vector store configuration", 
               f"{var_name} = RunnablePassthrough()  # Placeholder for retriever"
           ]
       
        else:
           return [
               f"# {component_type} - Custom implementation needed",
               f"{var_name} = RunnablePassthrough()  # Placeholder"
           ]
   
    async def export_as_json(self, flow_definition: FlowDefinition) -> str:
       """Export flow as JSON"""
       import json
       return json.dumps(flow_definition.dict(), indent=2)
   
    async def optimize_flow(self, flow_definition: FlowDefinition) -> FlowDefinition:
       """Optimize flow execution order and remove redundant connections"""
       # This is a placeholder for flow optimization logic
       # Could include:
       # - Removing redundant edges
       # - Optimizing execution order
       # - Parallel execution opportunities
       # - Component fusion where possible
       
       optimized_flow = FlowDefinition(**flow_definition.dict())
       optimized_flow.metadata = optimized_flow.metadata or {}
       optimized_flow.metadata["optimized"] = True
       optimized_flow.metadata["optimization_timestamp"] = time.time()
       
       return optimized_flow
   
    @staticmethod
    async def get_flow_templates() -> List[Dict[str, Any]]:
       """Get predefined flow templates"""
       templates = [
           {
               "id": "simple-chat",
               "name": "Simple Chatbot",
               "description": "A basic chatbot using LLM",
               "category": "conversational",
               "difficulty": "beginner",
               "flow_definition": {
                   "id": "template-simple-chat",
                   "name": "Simple Chatbot Template",
                   "nodes": [
                       {
                           "id": "input-1",
                           "component_type": "Text Input",
                           "position": {"x": 100, "y": 100},
                           "data": {"placeholder": "Enter your message..."}
                       },
                       {
                           "id": "llm-1",
                           "component_type": "OpenAI LLM",
                           "position": {"x": 400, "y": 100},
                           "data": {
                               "model": "gpt-3.5-turbo",
                               "temperature": 0.7,
                               "max_tokens": 200
                           }
                       },
                       {
                           "id": "output-1",
                           "component_type": "Text Output",
                           "position": {"x": 700, "y": 100}
                       }
                   ],
                   "edges": [
                       {
                           "id": "edge-1",
                           "source": "input-1",
                           "target": "llm-1",
                           "source_handle": "text",
                           "target_handle": "prompt"
                       },
                       {
                           "id": "edge-2", 
                           "source": "llm-1",
                           "target": "output-1",
                           "source_handle": "response",
                           "target_handle": "text"
                       }
                   ]
               }
           },
           {
               "id": "rag-pipeline",
               "name": "RAG Pipeline",
               "description": "Retrieval-Augmented Generation pipeline",
               "category": "rag",
               "difficulty": "intermediate",
               "flow_definition": {
                   "id": "template-rag-pipeline",
                   "name": "RAG Pipeline Template", 
                   "nodes": [
                       {
                           "id": "query-1",
                           "component_type": "Text Input",
                           "position": {"x": 100, "y": 100}
                       },
                       {
                           "id": "embeddings-1",
                           "component_type": "Embeddings",
                           "position": {"x": 300, "y": 100},
                           "data": {"provider": "openai"}
                       },
                       {
                           "id": "retriever-1",
                           "component_type": "Vector Store Retriever",
                           "position": {"x": 500, "y": 100},
                           "data": {"k": 5}
                       },
                       {
                           "id": "prompt-1",
                           "component_type": "Prompt Template",
                           "position": {"x": 300, "y": 300},
                           "data": {
                               "template": "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                           }
                       },
                       {
                           "id": "llm-1",
                           "component_type": "OpenAI LLM",
                           "position": {"x": 500, "y": 300}
                       }
                   ],
                   "edges": [
                       {
                           "id": "edge-1",
                           "source": "query-1",
                           "target": "retriever-1"
                       },
                       {
                           "id": "edge-2",
                           "source": "retriever-1", 
                           "target": "prompt-1",
                           "source_handle": "documents",
                           "target_handle": "context"
                       },
                       {
                           "id": "edge-3",
                           "source": "query-1",
                           "target": "prompt-1",
                           "source_handle": "text",
                           "target_handle": "question"
                       },
                       {
                           "id": "edge-4",
                           "source": "prompt-1",
                           "target": "llm-1"
                       }
                   ]
               }
           }
       ]
       
       return templates
   
    def get_execution_stats(self) -> Dict[str, Any]:
       """Get flow execution statistics"""
       active_count = len(self.active_executions)
       total_executions = len(self.execution_cache)
       
       completed_count = sum(
           1 for task in self.execution_cache.values()
           if task.status == ExecutionStatus.COMPLETED
       )
       
       failed_count = sum(
           1 for task in self.execution_cache.values() 
           if task.status == ExecutionStatus.FAILED
       )
       
       return {
           "active_executions": active_count,
           "total_executions": total_executions,
           "completed_executions": completed_count,
           "failed_executions": failed_count,
           "success_rate": completed_count / total_executions if total_executions > 0 else 0
       }