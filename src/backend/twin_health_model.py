import pickle
import os
import numpy as np
import pandas as pd

class TwinHealthModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = None
        
    def load_model(self, model_path=None):
        """Load the trained twin health model"""
        try:
            if model_path is None:
                # Try to find the model in the models directory
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(script_dir, 'app', 'models', 'twin_health_model.pkl')
                scaler_path = os.path.join(script_dir, 'app', 'models', 'scaler.pkl')
            else:
                # If model_path is provided, assume scaler is in the same directory
                model_dir = os.path.dirname(model_path)
                scaler_path = os.path.join(model_dir, 'scaler.pkl')
            
            print(f"Looking for model at: {model_path}")
            
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            
            # Load model and scaler
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.model_path = model_path
            print("Twin model loaded successfully!")
            return True
            
        except FileNotFoundError as e:
            print(f"Model load failed: {e}")
            return False
        except Exception as e:
            print(f"Failed to load twin model: {str(e)}")
            return False
    
    def predict_risk(self, twin_a_data, twin_b_data=None):
        """
        Predict health risk for twin pair or individual
        
        Parameters:
        twin_a_data: dict with keys 'age', 'bmi', 'bp', 'cholesterol', etc.
        twin_b_data: dict with keys 'age', 'bmi', 'bp', 'cholesterol', etc. (optional)
        
        Returns:
        risk_score: float between 0-1 (0-100% risk)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # If twin_b_data is None, use twin_a_data for both (individual prediction)
        if twin_b_data is None:
            twin_b_data = twin_a_data.copy()
            genetic_similarity = 1.0  # Same person
        else:
            # Calculate genetic similarity based on how similar the inputs are
            genetic_similarity = self._calculate_genetic_similarity(twin_a_data, twin_b_data)
        
        # Extract features with defaults
        features = np.array([[
            float(twin_a_data.get('age', 50)),
            float(twin_a_data.get('bmi', 25)),
            float(twin_a_data.get('bp', 120)),
            float(twin_a_data.get('cholesterol', 200)),
            float(twin_b_data.get('age', 50)),
            float(twin_b_data.get('bmi', 25)),
            float(twin_b_data.get('bp', 120)),
            float(twin_b_data.get('cholesterol', 200)),
            float(genetic_similarity)
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        risk = self.model.predict(features_scaled)[0]
        
        # Convert to 0-1 range if needed (model outputs 0-100 or 0-1)
        if risk > 1:
            risk = risk / 100.0
        
        # Ensure risk is within 0-1 range
        risk = np.clip(risk, 0.01, 0.5)
        
        return risk
    
    def predict_category(self, twin_a_data, twin_b_data=None):
        """
        Predict risk category for twin pair
        
        Returns:
        category: string ('LOW', 'MODERATE', 'HIGH')
        confidence: float between 0-1
        """
        risk = self.predict_risk(twin_a_data, twin_b_data)
        
        if risk < 0.1:
            return "LOW", 0.9
        elif risk < 0.2:
            return "MODERATE", 0.8
        else:
            return "HIGH", 0.85
    
    def predict_with_confidence(self, twin_a_data, twin_b_data=None):
        """
        Predict risk with confidence interval
        
        Returns:
        risk: float
        confidence_interval: tuple (lower, upper)
        confidence: float
        """
        risk = self.predict_risk(twin_a_data, twin_b_data)
        
        # Simple confidence interval based on risk level
        confidence_margin = 0.02 if risk < 0.1 else 0.03 if risk < 0.2 else 0.04
        lower = max(0.01, risk - confidence_margin)
        upper = min(0.5, risk + confidence_margin)
        
        confidence = 0.85 if risk < 0.2 else 0.75
        
        return risk, (lower, upper), confidence
    
    def get_feature_importance(self):
        """
        Get feature importance from the model
        
        Returns:
        DataFrame with feature names and importance scores
        """
        if self.model is None:
            return None
        
        feature_names = [
            'Age (Twin A)', 'BMI (Twin A)', 'Blood Pressure (Twin A)', 'Cholesterol (Twin A)',
            'Age (Twin B)', 'BMI (Twin B)', 'Blood Pressure (Twin B)', 'Cholesterol (Twin B)',
            'Genetic Similarity'
        ]
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return df
        
        return None
    
    def _calculate_genetic_similarity(self, twin_a_data, twin_b_data):
        """
        Calculate genetic similarity based on health metric similarities
        """
        # Key health metrics to compare
        metrics = ['bmi', 'bp', 'cholesterol', 'age']
        
        similarities = []
        for metric in metrics:
            val_a = twin_a_data.get(metric, 0)
            val_b = twin_b_data.get(metric, 0)
            
            if val_a == 0 and val_b == 0:
                similarity = 1.0
            else:
                # Calculate relative similarity
                diff_percent = abs(val_a - val_b) / max(val_a, val_b, 1)
                similarity = max(0, 1 - diff_percent)
            
            similarities.append(similarity)
        
        # Weighted average of similarities
        weights = [0.3, 0.3, 0.3, 0.1]  # Age less important for genetic similarity
        genetic_similarity = sum(s * w for s, w in zip(similarities, weights))
        
        # Clamp between 0.7 and 1.0 for twins (they should be fairly similar)
        genetic_similarity = np.clip(genetic_similarity, 0.7, 1.0)
        
        return genetic_similarity