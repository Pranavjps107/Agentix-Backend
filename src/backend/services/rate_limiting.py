
# src/backend/services/rate_limiting.py
from fastapi import HTTPException, Request
from typing import Dict
import time

class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
        self.max_requests = 100  # requests per window
        self.window_size = 60  # seconds
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        current_time = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests outside the window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < self.window_size
        ]
        
        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(current_time)
        return True
    
    def rate_limit_middleware(self, request: Request):
        """Middleware function for rate limiting"""
        client_ip = request.client.host
        
        if not self.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )