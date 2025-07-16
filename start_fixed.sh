#!/bin/bash
echo "🚀 Starting LangChain Platform with fixed imports..."

# Navigate to project root
cd /workspaces/Agentix-Backend

# Set Python path to include both backend and project root
export PYTHONPATH="/workspaces/Agentix-Backend/src/backend:/workspaces/Agentix-Backend:/workspaces/Agentix-Backend/src"

# Set environment variables
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"

# Check if main.py exists and has correct imports
echo "🔍 Checking main.py..."
if python -c "
import sys
sys.path.append('/workspaces/Agentix-Backend/src/backend')
try:
    import main
    print('✅ main.py imports successfully')
except Exception as e:
    print(f'❌ main.py import error: {e}')
    sys.exit(1)
"; then
    echo "✅ All imports working!"
    echo "🌐 Starting server..."
    uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "❌ Import issues detected. Please run the import fixer first."
fi