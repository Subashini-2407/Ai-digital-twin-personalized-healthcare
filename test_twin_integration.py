import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.twin_health_model import TwinHealthModel

# Test the model
model = TwinHealthModel()

# Try to load from the correct path
model_path = r'C:\Users\acer\OneDrive\Documents\working project\src\app\models\twin_health_model.pkl'

if model.load_model(model_path):
    print("✅ Model loaded successfully!")
    
    # Test prediction
    twin_a = {'age': 45, 'bmi': 27, 'bp': 130, 'cholesterol': 210}
    twin_b = {'age': 45, 'bmi': 26, 'bp': 128, 'cholesterol': 205}
    
    risk = model.predict_risk(twin_a, twin_b)
    category, conf = model.predict_category(twin_a, twin_b)
    
    print(f"Predicted risk: {risk*100:.1f}%")
    print(f"Risk category: {category}")
    print(f"Confidence: {conf*100:.0f}%")
    
    # Test individual prediction
    individual_risk = model.predict_risk(twin_a)
    print(f"Individual risk: {individual_risk*100:.1f}%")
    
else:
    print("❌ Failed to load model")