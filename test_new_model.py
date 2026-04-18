import pickle
import os
import numpy as np

model_path = r'C:\Users\acer\OneDrive\Documents\working project\src\app\models\twin_health_model.pkl'
scaler_path = r'C:\Users\acer\OneDrive\Documents\working project\src\app\models\scaler.pkl'

print("Testing the new model...")
print("=" * 50)

try:
    # Load model and scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✅ Model and scaler loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    
    # Test prediction
    print("\n📊 Making test prediction...")
    
    # Example twin data
    test_data = np.array([[
        45,  # age_A
        27,  # bmi_A
        130, # bp_A
        210, # cholesterol_A
        45,  # age_B
        26,  # bmi_B
        128, # bp_B
        205, # cholesterol_B
        0.92 # genetic_similarity
    ]])
    
    # Scale and predict
    test_scaled = scaler.transform(test_data)
    prediction = model.predict(test_scaled)
    
    print(f"   Input features: {test_data[0].tolist()}")
    print(f"   Predicted 5-year health risk: {prediction[0]:.1f}/100")
    print("\n✅ Model is working correctly!")
    
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    print("Please run train_fresh_model.py first")
except Exception as e:
    print(f"❌ Error: {e}")