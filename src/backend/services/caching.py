"""
Caching Service using Redis
"""
import json
import hashlib
import logging
from typing import Any, Optional, Dict
from datetime import timedelta
import asyncio

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis-based caching for component results and flow data"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = timedelta(seconds=default_ttl)
        self.redis_client = None
        self._memory_cache = {} 
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            import redis.asyncio as redis
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            logger.info("Redis cache manager initialized")
        except ImportError:
            logger.warning("Redis not available, using in-memory cache fallback")
            self.redis_client = None
            self._memory_cache = {}
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None
            self._memory_cache = {}
    
    def _generate_cache_key(self, prefix: str, data: dict) -> str:
        """Generate cache key from prefix and data"""
        data_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        return f"{prefix}:{data_hash}"
    
    async def get_cached_result(self, component_type: str, inputs: dict) -> Optional[dict]:
        """Get cached component result"""
        cache_key = self._generate_cache_key(f"component:{component_type}", inputs)
        
        try:
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                # Fallback to memory cache
                return self._memory_cache.get(cache_key)
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def cache_result(
        self, 
        component_type: str, 
        inputs: dict, 
        result: dict,
        ttl: Optional[timedelta] = None
    ):
        """Cache component result"""
        cache_key = self._generate_cache_key(f"component:{component_type}", inputs)
        ttl = ttl or self.default_ttl
        
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    int(ttl.total_seconds()),
                    json.dumps(result, default=str)
                )
            else:
                # Fallback to memory cache (no TTL in memory)
                self._memory_cache[cache_key] = result
            
            logger.debug(f"Cached result for {component_type}")
            
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
    
    async def cache_flow_result(self, flow_id: str, inputs: dict, result: dict, ttl: Optional[timedelta] = None):
        """Cache flow execution result"""
        cache_key = self._generate_cache_key(f"flow:{flow_id}", inputs)
        ttl = ttl or self.default_ttl
        
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    int(ttl.total_seconds()),
                    json.dumps(result, default=str)
                )
            else:
                self._memory_cache[cache_key] = result
            
            logger.debug(f"Cached flow result for {flow_id}")
            
        except Exception as e:
            logger.error(f"Flow cache error: {str(e)}")
    
    async def get_cached_flow_result(self, flow_id: str, inputs: dict) -> Optional[dict]:
        """Get cached flow result"""
        cache_key = self._generate_cache_key(f"flow:{flow_id}", inputs)
        
        try:
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                return self._memory_cache.get(cache_key)
            
            return None
            
        except Exception as e:
            logger.error(f"Flow cache get error: {str(e)}")
            return None
    
    async def clear_cache(self, pattern: str = "*"):
        """Clear cache entries matching pattern"""
        try:
            if self.redis_client:
                if pattern == "*":
                    await self.redis_client.flushdb()
                else:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)
            else:
                if pattern == "*":
                    self._memory_cache.clear()
                else:
                    # Simple pattern matching for memory cache
                    keys_to_delete = [
                        key for key in self._memory_cache.keys()
                        if pattern.replace("*", "") in key
                    ]
                    for key in keys_to_delete:
                        del self._memory_cache[key]
            
            logger.info(f"Cleared cache with pattern: {pattern}")
            
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if self.redis_client:
                info = await self.redis_client.info()
                return {
                    "type": "redis",
                    "connected": True,
                    "keys": info.get("db0", {}).get("keys", 0),
                    "memory_usage": info.get("used_memory_human", "unknown"),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0)
                }
            else:
                return {
                    "type": "memory",
                    "connected": True,
                    "keys": len(self._memory_cache),
                    "memory_usage": "unknown"
                }
        except Exception as e:
            logger.error(f"Cache stats error: {str(e)}")
            return {
                "type": "unknown",
                "connected": False,
                "error": str(e)
            }
    
    async def health_check(self) -> bool:
        """Check if cache is healthy"""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                return True
            else:
                return True  # Memory cache is always "healthy"
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            return False