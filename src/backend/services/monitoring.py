# src/backend/services/monitoring.py
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from typing import Dict, Any

class MetricsCollector:
    """Collect and expose metrics for monitoring"""
    
    def __init__(self):
        # Prometheus metrics
        self.component_executions = Counter(
            'component_executions_total',
            'Total number of component executions',
            ['component_type', 'status']
        )
        
        self.execution_duration = Histogram(
            'component_execution_duration_seconds',
            'Time spent executing components',
            ['component_type']
        )
        
        self.active_flows = Gauge(
            'active_flows',
            'Number of currently executing flows'
        )
        
        self.system_memory = Gauge(
            'system_memory_usage_bytes',
            'System memory usage'
        )
        
        self.system_cpu = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )
    
    def record_component_execution(self, component_type: str, duration: float, success: bool):
        """Record component execution metrics"""
        status = 'success' if success else 'error'
        self.component_executions.labels(component_type=component_type, status=status).inc()
        self.execution_duration.labels(component_type=component_type).observe(duration)
    
    def update_system_metrics(self):
        """Update system metrics"""
        self.system_memory.set(psutil.virtual_memory().used)
        self.system_cpu.set(psutil.cpu_percent())
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        self.update_system_metrics()
        return generate_latest()

