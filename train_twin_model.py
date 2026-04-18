"""
Train the Twin Health Model
Run this script once to train and save the model
"""

import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backend.twin_health_model import train_and_save_model

if __name__ == "__main__":
    print("=" * 50)
    print("Twin Health Model Training")
    print("=" * 50)
    
    # Train and save the model
    model = train_and_save_model()
    
    print("\n" + "=" * 50)
    print("✅ Training complete! Model saved to models/twin_health_model.pkl")
    print("=" * 50)