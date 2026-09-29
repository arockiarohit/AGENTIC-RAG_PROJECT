import sys
import os

# Add the current directory to path so it can find 'app'
sys.path.append(os.getcwd())

try:
    from app.vector_store import vector_store
    print("SUCCESS: vector_store imported correctly!")
    print(f"Stats: {vector_store.get_stats()}")
except Exception as e:
    print(f"FAILURE: Could not import vector_store.")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
