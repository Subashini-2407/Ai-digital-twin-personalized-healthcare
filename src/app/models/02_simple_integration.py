# src/models/02_simple_integration.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("=" * 60)
print("SIMPLE INTEGRATION - WORKING WITH AVAILABLE DATA")
print("=" * 60)

# Load data
nhanes = pd.read_csv('data/processed/nhanes_all.csv')
framingham = pd.read_csv('data/raw/health_data.csv')

print(f"NHANES: {len(nhanes)} rows")
print(f"Framingham: {len(framingham)} rows")

# 1. CREATE CLEAN NHANES DATASET WITH BASIC FEATURES
print("\n" + "=" * 60)
print("STEP 1: CREATING NHANES FEATURE SET")
print("=" * 60)

nhanes_features = pd.DataFrame()
nhanes_features['age'] = nhanes['RIDAGEYR']

# BMI (fill missing with median)
nhanes_features['bmi'] = nhanes['BMXBMI'].fillna(nhanes['BMXBMI'].median())

# Blood Pressure (fill missing with median)
nhanes_features['bp'] = nhanes['BPXOSY1'].fillna(nhanes['BPXOSY1'].median())

# Cholesterol (fill missing with median)
if 'LBXTC' in nhanes.columns:
    nhanes_features['cholesterol'] = nhanes['LBXTC'].fillna(nhanes['LBXTC'].median())
else:
    nhanes_features['cholesterol'] = 200  # default value

# Smoking (create from available data)
if 'SMQ856' in nhanes.columns:
    # Ever smoked
    nhanes_features['smoker'] = (nhanes['SMQ856'] == 1).astype(float)
else:
    nhanes_features['smoker'] = 0

print(f"NHANES features shape: {nhanes_features.shape}")
print(f"Features: {list(nhanes_features.columns)}")

# 2. TRAIN ON FRAMINGHAM
print("\n" + "=" * 60)
print("STEP 2: TRAINING ON FRAMINGHAM")
print("=" * 60)

# Prepare Framingham data
X_fram = framingham[['age', 'bmi', 'bp', 'cholesterol', 'smoker']]
y_fram = framingham['outcome']

# Train model
model_fram = RandomForestRegressor(n_estimators=100, random_state=42)
model_fram.fit(X_fram, y_fram)
print("✓ Trained on Framingham")

# 3. PREDICT RISK FOR NHANES
print("\n" + "=" * 60)
print("STEP 3: PREDICTING RISK FOR NHANES")
print("=" * 60)

# Use the same features order
X_nhanes = nhanes_features[['age', 'bmi', 'bp', 'cholesterol', 'smoker']]
nhanes_risk = model_fram.predict(X_nhanes)

# Add risk to dataframe
nhanes_features['framingham_risk'] = nhanes_risk
nhanes_features['high_risk'] = (nhanes_risk > 0.5).astype(int)

print(f"Risk range: {nhanes_risk.min():.3f} - {nhanes_risk.max():.3f}")
print(f"High risk patients: {nhanes_features['high_risk'].sum()} ({nhanes_features['high_risk'].mean()*100:.1f}%)")

# 4. ADD LIFESTYLE FACTORS
print("\n" + "=" * 60)
print("STEP 4: ADDING LIFESTYLE FACTORS")
print("=" * 60)

# Add exercise if available
if 'PAQ605' in nhanes.columns:
    nhanes_features['exercise'] = (nhanes['PAQ605'] == 1).astype(float).fillna(0)
    print("✓ Added exercise")

# Add diabetes if available
if 'DIQ010' in nhanes.columns:
    nhanes_features['diabetes'] = (nhanes['DIQ010'] == 1).astype(float).fillna(0)
    print("✓ Added diabetes")

print(f"\nFinal dataset: {len(nhanes_features)} rows")
print(f"Columns: {list(nhanes_features.columns)}")

# 5. TRAIN FINAL MODEL WITH LIFESTYLE
print("\n" + "=" * 60)
print("STEP 5: TRAINING FINAL MODEL WITH LIFESTYLE")
print("=" * 60)

# Features for final model (include lifestyle if available)
final_features = ['age', 'bmi', 'bp', 'cholesterol', 'smoker']
if 'exercise' in nhanes_features.columns:
    final_features.append('exercise')
if 'diabetes' in nhanes_features.columns:
    final_features.append('diabetes')

X_final = nhanes_features[final_features]
y_final = nhanes_features['framingham_risk']

# Train final model
final_model = RandomForestRegressor(n_estimators=150, random_state=42)
final_model.fit(X_final, y_final)

# Feature importance
importance = pd.DataFrame({
    'feature': final_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n🔍 Feature Importance:")
print(importance)

# Save everything
nhanes_features.to_csv('data/processed/nhanes_with_risk.csv', index=False)
joblib.dump(final_model, 'models/digital_twin_final.pkl')
joblib.dump(final_features, 'models/final_features.pkl')
print("\n✅ Saved:")
print("   - data/processed/nhanes_with_risk.csv")
print("   - models/digital_twin_final.pkl")

# Show sample
print("\n📊 Sample of final data:")
print(nhanes_features.head())