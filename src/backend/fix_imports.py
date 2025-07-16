#!/usr/bin/env python3
"""
Fix all relative imports in the codebase
"""
import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix relative imports in a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace relative imports with absolute imports
        patterns = [
            # Three dots (going up two levels)
            (r'from \.\.\.core\.', 'from core.'),
            (r'from \.\.\.services\.', 'from services.'),
            (r'from \.\.\.models\.', 'from models.'),
            (r'from \.\.\.api\.', 'from api.'),
            (r'from \.\.\.components\.', 'from components.'),
            
            # Two dots (going up one level)
            (r'from \.\.core\.', 'from core.'),
            (r'from \.\.services\.', 'from services.'),
            (r'from \.\.models\.', 'from models.'),
            (r'from \.\.api\.', 'from api.'),
            (r'from \.\.components\.', 'from components.'),
            
            # One dot (same level)
            (r'from \.core\.', 'from core.'),
            (r'from \.services\.', 'from services.'),
            (r'from \.models\.', 'from models.'),
            (r'from \.api\.', 'from api.'),
            (r'from \.components\.', 'from components.'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in {file_path}")
            return True
        else:
            print(f"⏭️ No changes needed in {file_path}")
            return False
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def fix_all_imports():
    """Fix imports in all Python files"""
    backend_dir = Path('/workspaces/Agentix-Backend/src/backend')
    
    if not backend_dir.exists():
        print(f"❌ Backend directory not found: {backend_dir}")
        return
    
    total_files = 0
    fixed_files = 0
    
    # Walk through all Python files
    for py_file in backend_dir.rglob('*.py'):
        total_files += 1
        if fix_imports_in_file(py_file):
            fixed_files += 1
    
    print(f"\n📊 Summary: Fixed {fixed_files} out of {total_files} Python files")

if __name__ == "__main__":
    print("🔧 Fixing all relative imports...")
    fix_all_imports()
    print("✅ Import fixing complete!")