"""
Component registry for managing all available components
"""
from typing import Dict, Type, List, Optional, Any
import logging
from .base import BaseLangChainComponent
from .exceptions import RegistrationException

logger = logging.getLogger(__name__)

class ComponentRegistry:
    """Registry for all available components"""
    
    _components: Dict[str, Type[BaseLangChainComponent]] = {}
    _categories: Dict[str, List[str]] = {}
    _instances: Dict[str, BaseLangChainComponent] = {}
    
    @classmethod
    def register(cls, component_class: Type[BaseLangChainComponent]) -> None:
        """Register a component"""
        try:
            instance = component_class()
            component_name = instance.metadata.display_name
            
            if component_name in cls._components:
                logger.warning(f"Component '{component_name}' already registered. Overwriting...")
            
            cls._components[component_name] = component_class
            cls._instances[component_name] = instance
            
            category = instance.metadata.category
            if category not in cls._categories:
                cls._categories[category] = []
            
            if component_name not in cls._categories[category]:
                cls._categories[category].append(component_name)
            
            logger.info(f"Registered component: {component_name} in category: {category}")
            
        except Exception as e:
            logger.error(f"Failed to register component {component_class.__name__}: {str(e)}")
            raise RegistrationException(f"Failed to register component {component_class.__name__}: {str(e)}")
    
    @classmethod
    def get_component(cls, name: str) -> Optional[Type[BaseLangChainComponent]]:
        """Get component class by name"""
        return cls._components.get(name)
    
    @classmethod
    def get_component_instance(cls, name: str) -> Optional[BaseLangChainComponent]:
        """Get component instance by name"""
        return cls._instances.get(name)
    
    @classmethod
    def create_component_instance(cls, name: str) -> Optional[BaseLangChainComponent]:
        """Create new component instance"""
        component_class = cls._components.get(name)
        if component_class:
            return component_class()
        return None
    
    @classmethod
    def get_all_components(cls) -> Dict[str, Any]:
        """Get all registered components with their schemas"""
        result = {}
        for name, instance in cls._instances.items():
            try:
                result[name] = instance.get_schema()
            except Exception as e:
                logger.error(f"Error getting schema for component {name}: {str(e)}")
                result[name] = {
                    "error": f"Schema error: {str(e)}",
                    "metadata": {
                        "display_name": name,
                        "description": "Error loading component",
                        "category": "error"
                    }
                }
        return result
    
    @classmethod
    def get_categories(cls) -> Dict[str, List[str]]:
        """Get all categories with their components"""
        return cls._categories.copy()
    
    @classmethod
    def get_components_by_category(cls, category: str) -> List[str]:
        """Get components in a specific category"""
        return cls._categories.get(category, [])
    
    @classmethod
    def search_components(cls, query: str) -> List[str]:
        """Search components by name or description"""
        results = []
        query_lower = query.lower()
        
        for name, instance in cls._instances.items():
            if (query_lower in name.lower() or 
                query_lower in instance.metadata.description.lower() or
                any(query_lower in tag.lower() for tag in instance.metadata.tags)):
                results.append(name)
        
        return results
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_components": len(cls._components),
            "categories": len(cls._categories),
            "components_by_category": {cat: len(comps) for cat, comps in cls._categories.items()}
        }
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a component"""
        if name in cls._components:
            instance = cls._instances.get(name)
            if instance:
                category = instance.metadata.category
                if category in cls._categories and name in cls._categories[category]:
                    cls._categories[category].remove(name)
                    if not cls._categories[category]:
                        del cls._categories[category]
            
            del cls._components[name]
            if name in cls._instances:
                del cls._instances[name]
            
            logger.info(f"Unregistered component: {name}")
            return True
        return False
    
    @classmethod
    def clear_registry(cls):
        """Clear all registered components"""
        cls._components.clear()
        cls._categories.clear()
        cls._instances.clear()
        logger.info("Cleared component registry")

def register_component(component_class: Type[BaseLangChainComponent]):
    """Decorator to auto-register components"""
    ComponentRegistry.register(component_class)
    return component_class