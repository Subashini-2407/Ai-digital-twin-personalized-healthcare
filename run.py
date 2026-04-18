import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize database if needed
from src.backend.database.db_setup import DatabaseManager
db = DatabaseManager()
db.init_database()

# Run the app
if __name__ == "__main__":
    # Ensure streamlit is executed with the same Python interpreter
    try:
        # Use sys.executable to make sure the venv interpreter is used
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app/integrated_dashboard.py"], check=True)
    except ModuleNotFoundError:
        print("Error: Streamlit is not installed in the current environment.")
        print("Please install it with `pip install streamlit` and try again.")
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with code {e.returncode}")
        print("Make sure all dependencies are installed and the path is correct.")