import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def train_fresh_model():
    """Train a fresh model with proper pickle serialization"""
    
    print("=" * 50)
    print("Training Fresh Twin Health Model")
    print("=" * 50)
    
    print("\n📊 Generating training data...")
    np.random.seed(42)
    n_samples = 10000
    
    # Generate features for twin pairs (9 features)
    X = np.random.randn(n_samples, 9)
    
    # Create target (health risk score 0-100)
    y = (
        30 +  # baseline risk
        X[:, 0] * 15 +  # age_A
        X[:, 1] * 12 +  # bmi_A
        X[:, 2] * 8 +   # bp_A
        X[:, 3] * 5 +   # cholesterol_A
        X[:, 4] * 15 +  # age_B
        X[:, 5] * 12 +  # bmi_B
        X[:, 6] * 8 +   # bp_B
        X[:, 7] * 5 +   # cholesterol_B
        X[:, 8] * 20 +  # genetic_similarity
        np.random.randn(n_samples) * 5  # noise
    )
    
    # Clip to 0-100 range
    y = np.clip(y, 0, 100)
    
    # Split into train/test
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Scale features
    print("🔄 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("🤖 Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"\n📈 Model Performance:")
    print(f"   Train R² Score: {train_score:.3f}")
    print(f"   Test R² Score:  {test_score:.3f}")
    
    # Save model and scaler
    model_dir = r'C:\Users\acer\OneDrive\Documents\working project\src\app\models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Backup old model if exists
    old_model_path = os.path.join(model_dir, 'twin_health_model.pkl')
    if os.path.exists(old_model_path):
        backup_path = os.path.join(model_dir, 'twin_health_model_backup.pkl')
        os.rename(old_model_path, backup_path)
        print(f"\n📦 Backed up old model to: {backup_path}")
    
    model_path = os.path.join(model_dir, 'twin_health_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    # Save with protocol 4 for better compatibility
    print("\n💾 Saving model and scaler...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, protocol=4)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f, protocol=4)
    
    print(f"   ✅ Model saved to: {model_path}")
    print(f"   ✅ Scaler saved to: {scaler_path}")
    
    # Verify the saved model works
    print("\n🔍 Verifying saved model...")
    try:
        with open(model_path, 'rb') as f:
            test_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            test_scaler = pickle.load(f)
        
        # Test prediction
        test_sample = np.random.randn(1, 9)
        test_sample_scaled = test_scaler.transform(test_sample)
        test_pred = test_model.predict(test_sample_scaled)
        print(f"   ✅ Model verified! Test prediction: {test_pred[0]:.2f}")
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return None, None
    
    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETE! Model is ready to use.")
    print("=" * 50)
    
    return model, scaler

if __name__ == "__main__":
    train_fresh_model()