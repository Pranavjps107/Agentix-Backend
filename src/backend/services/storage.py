"""
Storage Service for persisting flows and execution results
"""
import json
import os
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from models.flow import FlowDefinition
from models.execution import ExecutionResult, ExecutionLog

logger = logging.getLogger(__name__)

class StorageService:
    """Service for storing and retrieving flows and execution data"""
    
    def __init__(self, storage_path: str = "./data"):
        self.storage_path = storage_path
        self.flows_path = os.path.join(storage_path, "flows")
        self.executions_path = os.path.join(storage_path, "executions")
        self.templates_path = os.path.join(storage_path, "templates")
        
        # Create directories if they don't exist
        os.makedirs(self.flows_path, exist_ok=True)
        os.makedirs(self.executions_path, exist_ok=True)
        os.makedirs(self.templates_path, exist_ok=True)
    
    async def save_flow(self, flow_definition: FlowDefinition) -> bool:
        """Save a flow definition to storage"""
        try:
            flow_file = os.path.join(self.flows_path, f"{flow_definition.id}.json")
            
            # Update timestamp
            flow_definition.updated_at = datetime.utcnow()
            
            # Save to file
            with open(flow_file, 'w') as f:
                json.dump(flow_definition.dict(), f, indent=2, default=str)
            
            logger.info(f"Saved flow: {flow_definition.name} ({flow_definition.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save flow {flow_definition.id}: {str(e)}")
            return False
    
    async def load_flow(self, flow_id: str) -> Optional[FlowDefinition]:
        """Load a flow definition from storage"""
        try:
            flow_file = os.path.join(self.flows_path, f"{flow_id}.json")
            
            if not os.path.exists(flow_file):
                return None
            
            with open(flow_file, 'r') as f:
                flow_data = json.load(f)
            
            return FlowDefinition(**flow_data)
            
        except Exception as e:
            logger.error(f"Failed to load flow {flow_id}: {str(e)}")
            return None
    
    async def list_flows(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List all saved flows"""
        try:
            flow_files = [f for f in os.listdir(self.flows_path) if f.endswith('.json')]
            flow_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.flows_path, x)), reverse=True)
            
            flows = []
            for flow_file in flow_files[offset:offset + limit]:
                flow_path = os.path.join(self.flows_path, flow_file)
                try:
                    with open(flow_path, 'r') as f:
                        flow_data = json.load(f)
                    
                    # Return summary info
                    flows.append({
                        "id": flow_data["id"],
                        "name": flow_data["name"],
                        "description": flow_data.get("description"),
                        "node_count": len(flow_data.get("nodes", [])),
                        "edge_count": len(flow_data.get("edges", [])),
                        "created_at": flow_data.get("created_at"),
                        "updated_at": flow_data.get("updated_at")
                    })
                except Exception as e:
                    logger.warning(f"Failed to load flow summary from {flow_file}: {str(e)}")
            
            return flows
            
        except Exception as e:
            logger.error(f"Failed to list flows: {str(e)}")
            return []
    
    async def delete_flow(self, flow_id: str) -> bool:
        """Delete a flow from storage"""
        try:
            flow_file = os.path.join(self.flows_path, f"{flow_id}.json")
            
            if os.path.exists(flow_file):
                os.remove(flow_file)
                logger.info(f"Deleted flow: {flow_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete flow {flow_id}: {str(e)}")
            return False
    
    async def save_execution_log(self, execution_log: ExecutionLog) -> bool:
        """Save execution log to storage"""
        try:
            date_folder = datetime.now().strftime("%Y-%m-%d")
            execution_folder = os.path.join(self.executions_path, date_folder)
            os.makedirs(execution_folder, exist_ok=True)
            
            execution_file = os.path.join(execution_folder, f"{execution_log.execution_id}.json")
            
            with open(execution_file, 'w') as f:
                json.dump(execution_log.dict(), f, indent=2, default=str)
            
            logger.info(f"Saved execution log: {execution_log.execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save execution log {execution_log.execution_id}: {str(e)}")
            return False
    
    async def load_execution_log(self, execution_id: str) -> Optional[ExecutionLog]:
        """Load execution log from storage"""
        try:
            # Search in recent date folders
            for days_back in range(30):  # Search last 30 days
                date = datetime.now() - timedelta(days=days_back)
                date_folder = date.strftime("%Y-%m-%d")
                execution_file = os.path.join(self.executions_path, date_folder, f"{execution_id}.json")
                
                if os.path.exists(execution_file):
                    with open(execution_file, 'r') as f:
                        execution_data = json.load(f)
                    
                    return ExecutionLog(**execution_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load execution log {execution_id}: {str(e)}")
            return None
    
    async def get_execution_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get execution statistics for the last N days"""
        try:
            stats = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0.0,
                "executions_by_day": {}
            }
            
            total_time = 0.0
            
            for days_back in range(days):
                date = datetime.now() - timedelta(days=days_back)
                date_str = date.strftime("%Y-%m-%d")
                date_folder = os.path.join(self.executions_path, date_str)
                
                day_stats = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0
                }
                
                if os.path.exists(date_folder):
                    execution_files = [f for f in os.listdir(date_folder) if f.endswith('.json')]
                    
                    for execution_file in execution_files:
                        try:
                            execution_path = os.path.join(date_folder, execution_file)
                            with open(execution_path, 'r') as f:
                                execution_data = json.load(f)
                            
                            stats["total_executions"] += 1
                            day_stats["total"] += 1
                            
                            if execution_data.get("status") == "completed":
                                stats["successful_executions"] += 1
                                day_stats["successful"] += 1
                            else:
                                stats["failed_executions"] += 1
                                day_stats["failed"] += 1
                            
                            total_time += execution_data.get("total_execution_time", 0)
                            
                        except Exception as e:
                            logger.warning(f"Failed to process execution file {execution_file}: {str(e)}")
                
                stats["executions_by_day"][date_str] = day_stats
            
            if stats["total_executions"] > 0:
                stats["average_execution_time"] = total_time / stats["total_executions"]
                stats["success_rate"] = stats["successful_executions"] / stats["total_executions"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get execution stats: {str(e)}")
            return {}
    
    async def cleanup_old_executions(self, days_to_keep: int = 30) -> int:
        """Clean up old execution logs"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            if not os.path.exists(self.executions_path):
                return 0
            
            for folder_name in os.listdir(self.executions_path):
                try:
                    folder_date = datetime.strptime(folder_name, "%Y-%m-%d")
                    if folder_date < cutoff_date:
                        folder_path = os.path.join(self.executions_path, folder_name)
                        if os.path.isdir(folder_path):
                            import shutil
                            shutil.rmtree(folder_path)
                            deleted_count += 1
                            logger.info(f"Deleted old execution logs for {folder_name}")
                except ValueError:
                    # Skip folders that don't match date format
                    continue
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old executions: {str(e)}")
            return 0