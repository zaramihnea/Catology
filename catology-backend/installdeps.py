import subprocess
import sys
from pathlib import Path

def install_requirements():
    requirements = [
        'flask',
        'flask-cors',
        'pandas',
        'numpy',
        'google-generativeai',
        'scikit-learn',
        'matplotlib',
        'openpyxl'
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    try:
        install_requirements()
        print("\nAll dependencies installed successfully!")
    except Exception as e:
        print(f"Error installing dependencies: {str(e)}")
        sys.exit(1)