#!/usr/bin/env python3
"""Test if all imports work"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_import(module_path, description):
    try:
        exec(f"from {module_path} import *")
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description}: {e}")
        return False

print("🧪 Testing imports...")
print("=" * 50)

# Test core imports
test_import("src.backend.core.registry", "Core Registry")
test_import("src.backend.core.base", "Core Base")

# Test services
test_import("src.backend.services.component_manager", "Component Manager")
test_import("src.backend.services.flow_executor", "Flow Executor")
test_import("src.backend.services.caching", "Caching Service")

# Test models
test_import("src.backend.models.component", "Component Models")
test_import("src.backend.models.flow", "Flow Models")

# Test routes (this will test the full chain)
test_import("src.backend.api.routes.components", "Components Routes")
test_import("src.backend.api.routes.flows", "Flows Routes")
test_import("src.backend.api.routes.health", "Health Routes")

print("=" * 50)
print("🏁 Import testing completed!")
