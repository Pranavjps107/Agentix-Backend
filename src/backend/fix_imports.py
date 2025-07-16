import os
import re

def fix_imports_in_file(file_path):
    """Fix relative imports in a Python file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace relative imports with absolute imports
    patterns = [
        (r'from \.\.\.core\.', 'from core.'),
        (r'from \.\.core\.', 'from core.'),
        (r'from \.core\.', 'from core.'),
        (r'from \.\.\.services\.', 'from services.'),
        (r'from \.\.services\.', 'from services.'),
        (r'from \.services\.', 'from services.'),
        (r'from \.\.\.models\.', 'from models.'),
        (r'from \.\.models\.', 'from models.'),
        (r'from \.models\.', 'from models.'),
        (r'from \.\.\.api\.', 'from api.'),
        (r'from \.\.api\.', 'from api.'),
        (r'from \.api\.', 'from api.'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed imports in {file_path}")

def fix_all_imports():
    """Fix imports in all Python files"""
    backend_dir = '/workspaces/Agentix-Backend/src/backend'
    
    for root, dirs, files in os.walk(backend_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    fix_imports_in_file(file_path)
                except Exception as e:
                    print(f"Error fixing {file_path}: {e}")

if __name__ == "__main__":
    fix_all_imports()
    print("✅ All imports fixed!")