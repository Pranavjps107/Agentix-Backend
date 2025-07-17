#!/usr/bin/env python3
import os
import re

def fix_imports_in_file(file_path):
    """Fix imports in a file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original = content
        
        # Fix imports based on file location
        if 'api/routes' in file_path:
            # Routes files need ... to go up to backend/
            content = re.sub(r'^from services\.', 'from ...services.', content, flags=re.MULTILINE)
            content = re.sub(r'^from models\.', 'from ...models.', content, flags=re.MULTILINE)
            content = re.sub(r'^from core\.', 'from ...core.', content, flags=re.MULTILINE)
            
        elif 'services' in file_path:
            # Services files need .. to go up one level
            content = re.sub(r'^from models\.', 'from ..models.', content, flags=re.MULTILINE)
            content = re.sub(r'^from core\.', 'from ..core.', content, flags=re.MULTILINE)
            
        elif 'components' in file_path and file_path.count(os.sep) > 3:
            # Component files in subdirectories
            content = re.sub(r'^from core\.', 'from ...core.', content, flags=re.MULTILINE)
            
        # Fix Pydantic V2 warnings
        content = re.sub(r'schema_extra', 'json_schema_extra', content)
            
        if content != original:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
        else:
            print(f"⚪ No changes needed: {file_path}")
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")

# Fix all Python files
for root, dirs, files in os.walk('src/backend'):
    for file in files:
        if file.endswith('.py'):
            fix_imports_in_file(os.path.join(root, file))

print("\n🎉 Import fixes completed!")
