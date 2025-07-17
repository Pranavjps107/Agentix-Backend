#!/usr/bin/env python3
"""
Complete import fixer for LangChain Platform
"""
import os
import re

def fix_file_imports(file_path):
    """Fix imports based on file location"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Determine relative path from backend root
        rel_path = os.path.relpath(file_path, 'src/backend')
        depth = rel_path.count(os.sep)
        
        print(f"Processing: {file_path} (depth: {depth})")
        
        # Fix imports based on file location
        if 'api/routes' in file_path:
            # Routes are 3 levels deep: api/routes/file.py
            content = re.sub(r'^(\s*)from core\.', r'\1from ...core.', content, flags=re.MULTILINE)
            content = re.sub(r'^(\s*)from services\.', r'\1from ...services.', content, flags=re.MULTILINE) 
            content = re.sub(r'^(\s*)from models\.', r'\1from ...models.', content, flags=re.MULTILINE)
            
        elif 'services' in file_path:
            # Services are 2 levels deep: services/file.py
            content = re.sub(r'^(\s*)from core\.', r'\1from ..core.', content, flags=re.MULTILINE)
            content = re.sub(r'^(\s*)from models\.', r'\1from ..models.', content, flags=re.MULTILINE)
            content = re.sub(r'^(\s*)from services\.', r'\1from .', content, flags=re.MULTILINE)  # Same level
            
        elif 'components' in file_path and depth > 1:
            # Component subdirectories are 3+ levels deep
            content = re.sub(r'^(\s*)from core\.', r'\1from ...core.', content, flags=re.MULTILINE)
            content = re.sub(r'^(\s*)from models\.', r'\1from ...models.', content, flags=re.MULTILINE)
            content = re.sub(r'^(\s*)from services\.', r'\1from ...services.', content, flags=re.MULTILINE)
            
        # Fix Pydantic V2 config
        content = re.sub(r'schema_extra', 'json_schema_extra', content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"  ✅ Fixed imports")
        else:
            print(f"  ⚪ No changes needed")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

def main():
    print("🔧 Fixing all import issues in LangChain Platform...")
    
    # Process all Python files
    for root, dirs, files in os.walk('src/backend'):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                fix_file_imports(file_path)
    
    print("\n✅ Import fixing completed!")

if __name__ == "__main__":
    main()
