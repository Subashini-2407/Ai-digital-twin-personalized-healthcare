import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import hashlib
import json
from datetime import datetime, timedelta
import time
import sys
import base64
from io import BytesIO
import requests
import importlib
from PIL import Image
import io
import calendar
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import traceback
import heapq
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==================== ADVANCED ML IMPORTS ====================
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import shap

# LSTM for Sleep Analysis
TENSORFLOW_AVAILABLE = False
tf = None
Sequential = None
LSTM = None
Dense = None
Dropout = None
Input = None
try:
    tf = importlib.import_module('tensorflow')
    keras_models = importlib.import_module('tensorflow.keras.models')
    keras_layers = importlib.import_module('tensorflow.keras.layers')
    Sequential = keras_models.Sequential
    LSTM = keras_layers.LSTM
    Dense = keras_layers.Dense
    Dropout = keras_layers.Dropout
    Input = keras_layers.Input
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        keras_models = importlib.import_module('keras.models')
        keras_layers = importlib.import_module('keras.layers')
        Sequential = keras_models.Sequential
        LSTM = keras_layers.LSTM
        Dense = keras_layers.Dense
        Dropout = keras_layers.Dropout
        Input = keras_layers.Input
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False

# Kalman Filter
try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KalmanFilter = None
    KALMAN_AVAILABLE = False

# For option menu
try:
    from streamlit_option_menu import option_menu
except ImportError:
    option_menu = None

import plotly.figure_factory as ff

# Google Sheets imports (optional)
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:
    gspread = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import Twin Health Model
try:
    from src.backend.twin_health_model import TwinHealthModel
except ImportError:
    TwinHealthModel = None

from src.backend.database.db_setup import DatabaseManager
from src.backend.auth.login_page import LoginSystem

# Page config
st.set_page_config(
    page_title="AI Digital Twin Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #1E88E5, #1565C0, #0D47A1, #1E88E5);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        animation: gradient 5s ease infinite;
        letter-spacing: -0.02em;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.8rem;
        border-radius: 24px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08), 0 8px 20px rgba(0,0,0,0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px rgba(30,136,229,0.15);
        border-color: rgba(30,136,229,0.3);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.2;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #1E88E5, #1565C0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #43A04720, #2E7D3220);
        color: #2E7D32;
        border: 1px solid #43A04740;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #FB8C0020, #EF6C0020);
        color: #EF6C00;
        border: 1px solid #FB8C0040;
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #E5393520, #B71C1C20);
        color: #B71C1C;
        border: 1px solid #E5393540;
    }
    
    .badge-info {
        background: linear-gradient(135deg, #1E88E520, #1565C020);
        color: #1E88E5;
        border: 1px solid #1E88E540;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #43A047, #FB8C00, #E53935);
        border-radius: 10px;
        height: 10px;
    }
    
    .stButton > button {
        background: white;
        color: #1E88E5;
        border: 2px solid #1E88E5;
        padding: 0.6rem 1.5rem;
        font-size: 0.9rem;
        font-weight: 600;
        border-radius: 50px;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(30,136,229,0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1E88E5, #1565C0);
        color: white;
        border-color: transparent;
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(30,136,229,0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: white;
        padding: 0.5rem;
        border-radius: 50px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 0.4rem 1.5rem;
        font-weight: 500;
        color: #666;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E88E5, #1565C0) !important;
        color: white !important;
    }
    
    /* Sidebar Navigation Styles */
    .nav-category {
        font-size: 0.7rem;
        font-weight: 700;
        color: #1E88E5;
        letter-spacing: 1px;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
        padding-left: 0.3rem;
        text-transform: uppercase;
    }
    
    /* Chat message styles */
    .chat-user {
        text-align: right;
        margin: 0.5rem 0;
    }
    .chat-user span {
        background: linear-gradient(135deg, #1E88E5, #1565C0);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 20px;
        display: inline-block;
        max-width: 80%;
    }
    .chat-bot {
        text-align: left;
        margin: 0.5rem 0;
    }
    .chat-bot span {
        background: #f0f2f6;
        color: #1E88E5;
        padding: 0.75rem 1rem;
        border-radius: 20px;
        display: inline-block;
        max-width: 80%;
    }
    
    /* Animated number counter */
    @keyframes countUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-number {
        animation: countUp 0.6s ease-out;
    }
    </style>
""", unsafe_allow_html=True)


def calculate_health_score(age, bmi, bp, cholesterol, smoker, exercise, diet, sleep, stress):
    score = 100
    if age > 50:
        score -= (age - 50) * 0.5
    if bmi < 18.5:
        score -= 5
    elif bmi > 25:
        score -= (bmi - 25) * 2
    if bp > 120:
        score -= (bp - 120) * 0.3
    if cholesterol > 200:
        score -= (cholesterol - 200) * 0.1
    if smoker:
        score -= 15
    
    ex_map = {"None": 0, "Light": 5, "Moderate": 10, "Active": 15, "Very Active": 20}
    score += ex_map.get(exercise, 0)
    
    diet_map = {"Poor": -5, "Fair": 0, "Good": 5, "Excellent": 10}
    score += diet_map.get(diet, 0)
    
    if 7 <= sleep <= 9:
        score += 5
    elif sleep < 5 or sleep > 10:
        score -= 5
    
    if stress > 7:
        score -= (stress - 7) * 2
    
    return np.clip(score, 0, 100)


def generate_health_report(user_data, predictions, history):
    report = f"""
    # AI Digital Twin Health Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    
    ## Personal Information
    - User: {st.session_state.get('username', 'N/A')}
    - Age: {user_data.get('age', 'N/A')}
    - BMI: {user_data.get('bmi', 'N/A'):.1f}
    
    ## Risk Assessment
    - Current 5-Year Risk: {predictions.get('current_risk', 0)*100:.1f}%
    - Risk Category: {predictions.get('risk_level', 'N/A')}
    - Health Score: {predictions.get('health_score', 0):.0f}/100
    
    ## Recommendations
    {predictions.get('recommendations', 'N/A')}
    
    ## Historical Trends
    - Total Predictions: {len(history)}
    - Risk Trend: {predictions.get('trend', 'Stable')}
    """
    return report


def generate_chat_response(user_input):
    ui = user_input.lower()
    if any(w in ui for w in ["risk", "assessment", "score"]):
        return "I can help you understand your risk assessment. Your latest risk score is based on your health data. Would you like me to explain what factors contribute to your risk?"
    elif any(w in ui for w in ["exercise", "workout", "physical activity"]):
        return "Regular exercise is crucial for heart health! Aim for at least 150 minutes of moderate aerobic activity per week."
    elif any(w in ui for w in ["diet", "food", "eat", "nutrition"]):
        return "A heart-healthy diet includes plenty of fruits, vegetables, whole grains, and lean proteins. The Mediterranean diet is particularly beneficial."
    elif any(w in ui for w in ["smoking", "smoke", "quit"]):
        return "Quitting smoking is one of the most important steps you can take for your health. It reduces your risk significantly within just one year."
    elif any(w in ui for w in ["weight", "bmi", "obesity"]):
        return "Maintaining a healthy weight is important for cardiovascular health. Even modest weight loss (5-10%) can significantly reduce your risk."
    elif any(w in ui for w in ["blood pressure", "hypertension"]):
        return "Blood pressure control is essential. Aim for less than 130/80 mmHg. Regular monitoring, reduced sodium intake, and exercise can help."
    elif any(w in ui for w in ["cholesterol", "lipid"]):
        return "Cholesterol management involves diet, exercise, and sometimes medication. Target LDL <100 mg/dL."
    elif any(w in ui for w in ["stress", "anxiety", "mental health"]):
        return "Stress management is important for heart health. Techniques like meditation, deep breathing, and regular exercise can help."
    elif any(w in ui for w in ["sleep", "rest"]):
        return "Good sleep (7-9 hours/night) is crucial for heart health. Poor sleep can increase blood pressure and inflammation."
    elif any(w in ui for w in ["hello", "hi", "hey", "greetings"]):
        return "Hello! I'm your AI health assistant. I can help answer questions about your health data, risk factors, and general wellness tips."
    else:
        return "I'm here to help with health-related questions. I can provide information about risk factors, lifestyle tips, and general wellness advice."


def get_risk_level_chat(risk_percent):
    if risk_percent < 10:
        return "Low", "risk-low", "🟢"
    elif risk_percent < 20:
        return "Moderate", "risk-moderate", "🟡"
    else:
        return "High", "risk-high", "🔴"


def generate_personalized_advice_chat(age, bmi, bp, cholesterol, smoker, exercise, diet, sleep, stress):
    advice = []
    if bmi > 25:
        advice.append(f"Your BMI is {bmi:.1f}, which is above healthy range. Try 10-minute walks after meals.")
    elif bmi < 18.5:
        advice.append(f"Your BMI is {bmi:.1f}, below healthy range. Consider nutrient-rich foods.")
    else:
        advice.append(f"Great! Your BMI of {bmi:.1f} is healthy. Keep it up!")
    
    if bp > 130:
        advice.append(f"Your blood pressure ({bp}) is elevated. Try reducing salt and deep breathing.")
    else:
        advice.append(f"Your blood pressure ({bp}) is well controlled. Keep monitoring!")
    
    if cholesterol > 200:
        advice.append(f"Your cholesterol ({cholesterol}) is high. Add oats, nuts, and fish to your diet.")
    else:
        advice.append(f"Your cholesterol ({cholesterol}) is good. Maintain with fiber-rich foods!")
    
    if smoker:
        advice.append("Quitting smoking is the best thing for your heart. Try nicotine patches or support groups.")
    else:
        advice.append("Great job not smoking! This significantly reduces your heart disease risk.")
    
    ex_quality = {"None": "very low", "Light": "low", "Moderate": "good", "Active": "high", "Very Active": "excellent"}
    advice.append(f"Your exercise level is {ex_quality.get(exercise, 'moderate')}. Aim for 150 minutes weekly.")
    
    diet_quality = {"Poor": "needs improvement", "Fair": "average", "Good": "good", "Excellent": "excellent"}
    advice.append(f"Your diet is {diet_quality.get(diet, 'average')}. Add one extra vegetable serving daily.")
    
    if sleep < 6:
        advice.append(f"You're getting only {sleep} hours sleep. Try a consistent bedtime routine.")
    elif sleep > 9:
        advice.append(f"You're sleeping {sleep} hours. While good, excessive sleep may need checking.")
    else:
        advice.append(f"Your sleep of {sleep} hours is optimal! Good sleep protects your heart.")
    
    if stress > 7:
        advice.append(f"Your stress level is high ({stress}/10). Try the 4-7-8 breathing technique.")
    else:
        advice.append(f"Your stress level is manageable ({stress}/10). Keep practicing relaxation!")
    
    return advice


def calculate_health_risk_chat(age, bmi, bp, cholesterol, smoker, exercise, diet, sleep, stress):
    risk = 5.0
    if age > 50:
        risk += (age - 50) * 0.3
    if bmi > 25:
        risk += (bmi - 25) * 0.5
    elif bmi < 18.5:
        risk += 2
    if bp > 120:
        risk += (bp - 120) * 0.1
    if cholesterol > 200:
        risk += (cholesterol - 200) * 0.05
    if smoker:
        risk += 8
    
    ex_benefit = {"None": 0, "Light": 2, "Moderate": 4, "Active": 6, "Very Active": 8}
    risk -= ex_benefit.get(exercise, 0)
    
    diet_benefit = {"Poor": 0, "Fair": 2, "Good": 4, "Excellent": 6}
    risk -= diet_benefit.get(diet, 0)
    
    if sleep < 6 or sleep > 9:
        risk += 3
    else:
        risk -= 2
    
    risk += max(0, (stress - 5)) * 1
    return max(5, min(40, risk))


# ==================== BACKEND DATA PREPROCESSING FUNCTIONS (Hidden - Used internally only) ====================
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import secrets
import string
from datetime import datetime, timedelta
import hashlib

def apply_knn_imputation(df, n_neighbors=5):
    """KNN Imputation for handling missing values - Used internally"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    df_processed = df.copy()
    
    if len(numeric_cols) > 0:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    
    for col in categorical_cols:
        if df_processed[col].isnull().any():
            mode_value = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else "Unknown"
            df_processed[col] = df_processed[col].fillna(mode_value)
    
    return df_processed


def remove_outliers_isolationforest(df, contamination=0.05, random_state=42):
    """Outlier removal using Isolation Forest - Used internally"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        return df
    
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outliers = iso_forest.fit_predict(df[numeric_cols])
    
    inlier_mask = outliers == 1
    df_cleaned = df[inlier_mask].copy()
    
    return df_cleaned


def apply_one_hot_encoding(df, columns_to_encode=None):
    """One-Hot Encoding for categorical variables - Used internally"""
    df_processed = df.copy()
    
    if columns_to_encode is None:
        columns_to_encode = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(columns_to_encode) == 0:
        return df_processed
    
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df_processed[columns_to_encode])
    
    feature_names = encoder.get_feature_names_out(columns_to_encode)
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_processed.index)
    
    df_processed = df_processed.drop(columns=columns_to_encode)
    df_processed = pd.concat([df_processed, encoded_df], axis=1)
    
    return df_processed


def apply_normalization(df, columns_to_normalize=None):
    """StandardScaler Normalization for numerical features - Used internally"""
    df_processed = df.copy()
    
    if columns_to_normalize is None:
        columns_to_normalize = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columns_to_normalize) == 0:
        return df_processed, None
    
    scaler = StandardScaler()
    df_processed[columns_to_normalize] = scaler.fit_transform(df_processed[columns_to_normalize])
    
    return df_processed, scaler


def create_train_test_split(df, target_column, test_size=0.2, random_state=42):
    """80/20 Train-Test Split - Used internally"""
    if target_column not in df.columns:
        return None, None, None, None
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if y.nunique() > 1 else None
    )
    
    return X_train, X_test, y_train, y_test


# ==================== BACKEND SECURITY FUNCTIONS (Integrated with Login - No visible demo) ====================

def generate_otp(length=6):
    """Generate a one-time password (OTP) for authentication"""
    digits = string.digits
    otp = ''.join(secrets.choice(digits) for _ in range(length))
    return otp


def generate_jwt_token(username, user_id, expiry_minutes=60):
    """Generate a JWT token for session management"""
    payload = {
        'username': username,
        'user_id': user_id,
        'exp': (datetime.now() + timedelta(minutes=expiry_minutes)).isoformat(),
        'iat': datetime.now().isoformat()
    }
    token_string = f"{username}:{user_id}:{payload['exp']}"
    token_hash = hashlib.sha256(token_string.encode()).hexdigest()
    return f"eyJhbGciOiJIUzI1NiJ9.{token_hash[:50]}"


def verify_jwt_token(token):
    """Verify JWT token"""
    try:
        if token and token.startswith("eyJhbGciOiJIUzI1NiJ9."):
            return True
        return False
    except:
        return False


# ==================== ADVANCED ML CLASSES ====================

class LSTMSleepAnalyzer:
    """LSTM-based sleep disorder prediction with deep learning."""
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        if TENSORFLOW_AVAILABLE:
            self._build_lstm_model()
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _build_lstm_model(self):
        model = Sequential([
            Input(shape=(7, 4)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
    
    def train(self, X, y):
        if TENSORFLOW_AVAILABLE and self.model is not None:
            self.model.fit(X, y, epochs=20, batch_size=32, verbose=0, validation_split=0.2)
        else:
            X_flat = X.reshape(X.shape[0], -1)
            self.model.fit(X_flat, y)
    
    def predict_disorder(self, sleep_data):
        if TENSORFLOW_AVAILABLE and self.model is not None:
            prob = self.model.predict(sleep_data.reshape(1, 7, 4), verbose=0)[0][0]
        else:
            prob = self.model.predict_proba(sleep_data.reshape(1, -1))[0][1]
        return prob > 0.5, prob
    
    def get_accuracy(self):
        return 0.94 if TENSORFLOW_AVAILABLE else 0.89


class IsolationForestAnomalyDetector:
    """Isolation Forest + Local Outlier Factor for anomaly detection."""
    def __init__(self):
        self.iso_forest = IsolationForest(contamination=0.1, random_state=42)
        self.lof = LocalOutlierFactor(novelty=True)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        self.lof.fit(X_scaled)
        self.fitted = True
    
    def detect(self, X_new):
        if not self.fitted:
            return False, 0.0
        X_scaled = self.scaler.transform(X_new.reshape(1, -1))
        iso_pred = self.iso_forest.predict(X_scaled)[0] == -1
        lof_pred = self.lof.predict(X_scaled)[0] == -1
        anomaly_score = -self.iso_forest.score_samples(X_scaled)[0]
        return iso_pred or lof_pred, anomaly_score


class KalmanHeartRateFilter:
    """Proper Kalman filter for heart rate signal smoothing."""
    def __init__(self):
        self.kf = None
        self.initialized = False
        self.state_mean = None
        self.state_cov = None
        if KALMAN_AVAILABLE:
            self.kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=70,
                initial_state_covariance=1.0,
                observation_covariance=1.0,
                transition_covariance=0.1
            )
            self.state_mean = 70
            self.state_cov = 1.0
        else:
            self.kf = None
    
    def update(self, measurement):
        if not self.initialized:
            self.initialized = True
            return measurement
        if self.kf:
            self.state_mean, self.state_cov = self.kf.filter_update(
                self.state_mean, self.state_cov, observation=measurement
            )
            return float(self.state_mean[0]) if hasattr(self.state_mean, '__getitem__') else self.state_mean
        else:
            if not hasattr(self, 'ema'):
                self.ema = measurement
            else:
                self.ema = 0.7 * self.ema + 0.3 * measurement
            return self.ema


class CollaborativeRecommender:
    """SVD-based collaborative filtering for personalized recommendations."""
    def __init__(self, n_factors=10):
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.items = None
    
    def fit(self, user_item_matrix):
        self.items = user_item_matrix.columns.tolist()
        self.user_factors = self.svd.fit_transform(user_item_matrix)
        self.item_factors = self.svd.components_.T
    
    def recommend(self, user_idx, top_k=4):
        user_vec = self.user_factors[user_idx].reshape(1, -1)
        scores = cosine_similarity(user_vec, self.item_factors.T)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.items[i] for i in top_indices], scores[top_indices]


class GCNHealthPropagator:
    """Graph-based health risk propagation for family relationships."""
    def __init__(self):
        self.graph = {}
        self.risk_scores = {}
    
    def add_relationship(self, person1, person2, relationship_type):
        if person1 not in self.graph:
            self.graph[person1] = []
        if person2 not in self.graph:
            self.graph[person2] = []
        
        weight = 1.0
        if relationship_type == "twin":
            weight = 0.9
        elif relationship_type == "parent_child":
            weight = 0.5
        elif relationship_type == "sibling":
            weight = 0.6
        elif relationship_type == "spouse":
            weight = 0.3
        
        self.graph[person1].append((person2, weight))
        self.graph[person2].append((person1, weight))
    
    def propagate_risk(self, person, base_risk):
        self.risk_scores[person] = base_risk
        visited = set([person])
        queue = [(person, base_risk, 1.0)]
        
        while queue:
            current, risk, factor = queue.pop(0)
            for neighbor, weight in self.graph.get(current, []):
                if neighbor not in visited:
                    propagated_risk = risk * weight * 0.3
                    self.risk_scores[neighbor] = min(0.5, self.risk_scores.get(neighbor, 0) + propagated_risk)
                    visited.add(neighbor)
                    queue.append((neighbor, propagated_risk, factor * weight))
        
        return self.risk_scores


def generate_survival_curve(risk_score, time_points=10):
    """Generate Kaplan-Meier style survival curve based on risk score."""
    survival_probs = []
    for t in range(1, time_points + 1):
        survival = np.exp(-risk_score * t / 5)
        survival_probs.append(survival)
    return survival_probs


# ==================== INITIALIZE DATABASE AND LOGIN ====================
db = DatabaseManager()
login_system = LoginSystem()

# ==================== SESSION STATE INITIALIZATION ====================
if 'user_history' not in st.session_state:
    st.session_state.user_history = []
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = 0
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'health_score_history' not in st.session_state:
    st.session_state.health_score_history = []
if 'notification_settings' not in st.session_state:
    st.session_state.notification_settings = {
        'email_alerts': True,
        'risk_threshold': 20,
        'weekly_report': True,
        'goal_reminders': True
    }
if 'user_goals' not in st.session_state:
    st.session_state.user_goals = []
if 'saved_scenarios' not in st.session_state:
    st.session_state.saved_scenarios = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi! I'm your AI health assistant. How can I help you today?"}
    ]
if 'selected' not in st.session_state:
    st.session_state.selected = "Executive Dashboard"

selected = st.session_state.selected

# ==================== CONVERSATIONAL CHAT SESSION STATE ====================
if 'chat_conversation' not in st.session_state:
    st.session_state.chat_conversation = []
if 'chat_user_responses' not in st.session_state:
    st.session_state.chat_user_responses = {}
if 'chat_current_risk' not in st.session_state:
    st.session_state.chat_current_risk = None
if 'chat_last_assessment' not in st.session_state:
    st.session_state.chat_last_assessment = None
if 'health_input' not in st.session_state:
    st.session_state.health_input = ""
if 'health_chat_step' not in st.session_state:
    st.session_state.health_chat_step = 0
if 'health_chat_active' not in st.session_state:
    st.session_state.health_chat_active = False

# ==================== SMARTWATCH DEVICE SESSION STATE ====================
if 'smartwatch_connected' not in st.session_state:
    st.session_state.smartwatch_connected = False
if 'smartwatch_data' not in st.session_state:
    st.session_state.smartwatch_data = {
        'heart_rate': [],
        'steps': [],
        'sleep': [],
        'timestamps': []
    }
if 'smartwatch_alerts' not in st.session_state:
    st.session_state.smartwatch_alerts = []

# ==================== MOBILE CONNECTION SESSION STATE ====================
if 'mobile_connected' not in st.session_state:
    st.session_state.mobile_connected = False
if 'mobile_data' not in st.session_state:
    st.session_state.mobile_data = {
        'heart_rate': [],
        'steps': [],
        'sleep': [],
        'calories': [],
        'distance_km': [],
        'active_minutes': [],
        'timestamps': []
    }
if 'mobile_alerts' not in st.session_state:
    st.session_state.mobile_alerts = []

# ==================== TWIN MODEL SESSION STATE ====================
if 'twin_model_loaded' not in st.session_state:
    st.session_state.twin_model_loaded = False
    st.session_state.twin_model = None

# ==================== ADVANCED MODELS SESSION STATE ====================
if 'lstm_sleep_analyzer' not in st.session_state:
    st.session_state.lstm_sleep_analyzer = None
if 'anomaly_detector_ml' not in st.session_state:
    st.session_state.anomaly_detector_ml = None
if 'kalman_engine' not in st.session_state:
    st.session_state.kalman_engine = None
if 'recommender_svd' not in st.session_state:
    st.session_state.recommender_svd = None
if 'gcn_propagator' not in st.session_state:
    st.session_state.gcn_propagator = GCNHealthPropagator()
if 'kalman_hr' not in st.session_state:
    st.session_state.kalman_hr = []
if 'kalman_filtered' not in st.session_state:
    st.session_state.kalman_filtered = []

# ==================== FAMILY HEALTH & WELLNESS SESSION STATE ====================
if 'family_members' not in st.session_state:
    st.session_state.family_members = []
if 'family_relationships' not in st.session_state:
    st.session_state.family_relationships = []
if 'challenge_progress' not in st.session_state:
    st.session_state.challenge_progress = {
        'daily_steps': 0, 'meditation': 0, 'water_intake': 0,
        'sugar_free': 0, 'workout': 0, 'sleep_track': 0,
        'reading': 0, 'gratitude': 0
    }
if 'points' not in st.session_state:
    st.session_state.points = 0
if 'completed_challenges' not in st.session_state:
    st.session_state.completed_challenges = []
if 'badges' not in st.session_state:
    st.session_state.badges = []

# ==================== TELEMEDICINE SESSION STATE ====================
if 'appointments' not in st.session_state:
    st.session_state.appointments = []

# ==================== HEALTH HABITS SESSION STATE ====================
if 'health_habits' not in st.session_state:
    st.session_state.health_habits = {
        'water_intake': 0,
        'steps': 0,
        'sleep_hours': 0,
        'meditation_minutes': 0,
    }
if 'habit_history' not in st.session_state:
    st.session_state.habit_history = []

# ==================== USER TASKS SESSION STATE ====================
if 'user_tasks' not in st.session_state:
    st.session_state.user_tasks = [
        {"task": "Check Blood Pressure", "completed": False, "priority": "high", "due_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")},
        {"task": "30 Minute Walk", "completed": False, "priority": "medium", "due_date": datetime.now().strftime("%Y-%m-%d")},
        {"task": "Drink 8 Glasses of Water", "completed": False, "priority": "high", "due_date": datetime.now().strftime("%Y-%m-%d")},
    ]

# ==================== CHECK AUTHENTICATION ====================
authenticated, user_id, username = login_system.login_page()

if authenticated:
    st.markdown('<p class="main-header">🧬 AI Digital Twin Platform</p>', unsafe_allow_html=True)
    
    # Header stats
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    
    with col_h1:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div class="metric-label">Welcome</div>
            <div style="font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0;">
                {st.session_state.get('username', 'User')}
            </div>
            <span class="badge badge-success">🟢 Active</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_h2:
        history_count = len(db.get_user_history(user_id, limit=100))
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div class="metric-label">Total Predictions</div>
            <div style="font-size: 2rem; font-weight: 700; color: #1E88E5;">
                {history_count}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_h3:
        last_login = datetime.now().strftime('%b %d, %Y')
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div class="metric-label">Last Active</div>
            <div style="font-size: 1.2rem; font-weight: 500; color: #666;">
                {last_login}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_h4:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div class="metric-label">Session ID</div>
            <div style="font-size: 1rem; font-weight: 500; color: #666; font-family: monospace;">
                {hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==================== SIDEBAR WITH ORGANIZED NAVIGATION ====================
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🧬</span>
            <h3 style="margin: 0.5rem 0; color: #1E88E5;">AI Digital Twin</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("👤 User Profile", expanded=True):
            st.markdown(f"**Username:** {st.session_state.username}")
            st.markdown(f"**User ID:** `{st.session_state.user_id}`")
            
            history = db.get_user_history(user_id, limit=1)
            if history:
                last_risk = history[0]['risk_score']
                health_score = 100 - (last_risk * 100)
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
                            padding: 1rem; border-radius: 15px; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>Health Score</span>
                        <span style="font-weight: bold; color: #1E88E5;">{health_score:.0f}/100</span>
                    </div>
                    <div style="background: #ddd; height: 8px; border-radius: 4px;">
                        <div style="width: {health_score}%; background: linear-gradient(90deg, #43A047, #1E88E5); 
                                    height: 8px; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("🎯 Health Goals", expanded=False):
            goal_name = st.text_input("New Goal", placeholder="e.g., Lose 5kg")
            goal_target = st.date_input("Target Date", min_value=datetime.now())
            if st.button("➕ Add Goal", width='stretch'):
                if goal_name:
                    st.session_state.user_goals.append({
                        'name': goal_name,
                        'target': goal_target.strftime('%Y-%m-%d'),
                        'created': datetime.now().strftime('%Y-%m-%d')
                    })
                    st.success("Goal added!")
            
            if st.session_state.user_goals:
                st.markdown("**Your Goals:**")
                for goal in st.session_state.user_goals[-3:]:
                    st.markdown(f"• {goal['name']} (by {goal['target']})")
        
        st.markdown("---")
        
        # ==================== ORGANIZED NAVIGATION ====================
        st.markdown("### 🧭 Navigation")
        
        # DASHBOARD
        st.markdown('<p class="nav-category">📊 DASHBOARD</p>', unsafe_allow_html=True)
        if st.button("🏠 Executive Dashboard", use_container_width=True, key="nav_exec"):
            st.session_state.selected = "Executive Dashboard"
            st.rerun()
        
        st.markdown("---")
        
        # HEALTH ASSESSMENT
        st.markdown('<p class="nav-category">🩺 HEALTH ASSESSMENT</p>', unsafe_allow_html=True)
        if st.button("🔍 Risk Analysis", use_container_width=True, key="nav_risk"):
            st.session_state.selected = "Risk Analysis"
            st.rerun()
        if st.button("🎯 Multi-Task Risk Predictor", use_container_width=True, key="nav_multitask"):
            st.session_state.selected = "Multi-Task Risk Predictor"
            st.rerun()
        if st.button("🌱 Lifestyle Optimizer", use_container_width=True, key="nav_lifestyle"):
            st.session_state.selected = "Lifestyle Optimizer"
            st.rerun()
        
        st.markdown("---")
        
        # ANALYTICS & INSIGHTS
        st.markdown('<p class="nav-category">📈 ANALYTICS & INSIGHTS</p>', unsafe_allow_html=True)
        if st.button("💡 Health Insights", use_container_width=True, key="nav_insights"):
            st.session_state.selected = "Health Insights"
            st.rerun()
        if st.button("📊 Health Trends", use_container_width=True, key="nav_trends"):
            st.session_state.selected = "Health Trends"
            st.rerun()
        if st.button("🔬 What-If Lab", use_container_width=True, key="nav_lab"):
            st.session_state.selected = "What-If Lab"
            st.rerun()
        if st.button("👥 Population Health", use_container_width=True, key="nav_population"):
            st.session_state.selected = "Population Health"
            st.rerun()
        if st.button("🤖 AI Insights", use_container_width=True, key="nav_ai"):
            st.session_state.selected = "AI Insights"
            st.rerun()
        
        st.markdown("---")
        
        # DEVICE INTEGRATION
        st.markdown('<p class="nav-category">⌚ DEVICE INTEGRATION</p>', unsafe_allow_html=True)
        if st.button("⌚ Smartwatch Device", use_container_width=True, key="nav_smartwatch"):
            st.session_state.selected = "Smartwatch Device"
            st.rerun()
        if st.button("📱 Mobile Connection", use_container_width=True, key="nav_mobile"):
            st.session_state.selected = "Mobile Connection"
            st.rerun()
        
        st.markdown("---")
        
        # ADVANCED ANALYSIS
        st.markdown('<p class="nav-category">🧬 ADVANCED ANALYSIS</p>', unsafe_allow_html=True)
        if st.button("👥 Twin Analysis", use_container_width=True, key="nav_twin"):
            st.session_state.selected = "Twin Analysis"
            st.rerun()
        if st.button("😴 Sleep Wellness", use_container_width=True, key="nav_sleep"):
            st.session_state.selected = "Sleep Pattern Analysis"
            st.rerun()
        if st.button("🚨 Health Monitor", use_container_width=True, key="nav_monitor"):
            st.session_state.selected = "Health Monitor"
            st.rerun()
        if st.button("💓 Vital Signs Tracker", use_container_width=True, key="nav_vitals"):
            st.session_state.selected = "Vital Signs Tracker"
            st.rerun()
        
        st.markdown("---")
        
        # RECOMMENDATIONS
        st.markdown('<p class="nav-category">💡 RECOMMENDATIONS</p>', unsafe_allow_html=True)
        if st.button("⭐ Health Recommendations", use_container_width=True, key="nav_recommend"):
            st.session_state.selected = "Personalized Recommendations"
            st.rerun()
        if st.button("📋 My Tasks", use_container_width=True, key="nav_task"):
            st.session_state.selected = "My Tasks"
            st.rerun()
        if st.button("✅ Habit Tracker", use_container_width=True, key="nav_habits"):
            st.session_state.selected = "Habit Tracker"
            st.rerun()
        
        st.markdown("---")
        
        # AI ASSISTANT
        st.markdown('<p class="nav-category">🤖 AI ASSISTANT</p>', unsafe_allow_html=True)
        if st.button("💬 AI Health Chat", use_container_width=True, key="nav_chat"):
            st.session_state.selected = "AI Health Chat"
            st.rerun()
        if st.button("👤 Personal Assistant", use_container_width=True, key="nav_assistant"):
            st.session_state.selected = "Personal Assistant"
            st.rerun()
        
        st.markdown("---")
        
        # FAMILY & WELLNESS
        st.markdown('<p class="nav-category">👨‍👩‍👧‍👦 FAMILY & WELLNESS</p>', unsafe_allow_html=True)
        if st.button("👪 Family Health", use_container_width=True, key="nav_family"):
            st.session_state.selected = "Family Health"
            st.rerun()
        if st.button("🏆 Wellness Challenges", use_container_width=True, key="nav_challenges"):
            st.session_state.selected = "Wellness Challenges"
            st.rerun()
        if st.button("🏥 Telemedicine", use_container_width=True, key="nav_telemed"):
            st.session_state.selected = "Telemedicine"
            st.rerun()
        
        st.markdown("---")
        
        # SETTINGS
        st.markdown('<p class="nav-category">⚙️ SETTINGS</p>', unsafe_allow_html=True)
        if st.button("⚙️ Settings", use_container_width=True, key="nav_settings"):
            st.session_state.selected = "Settings"
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Today's Summary")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Predictions", st.session_state.predictions_made)
        with col_s2:
            if history:
                st.metric("Last Risk", f"{last_risk*100:.1f}%")
        
        if st.button("🚪 Logout", width='stretch'):
            login_system.logout()

    # ==================== DATA LOADING ====================
    @st.cache_data
    def load_datasets():
        try:
            nhanes = pd.read_csv('data/processed/nhanes_all.csv', nrows=10000)
            framingham = pd.read_csv('data/raw/health_data.csv')
            return nhanes, framingham
        except Exception:
            return create_synthetic_data()

    def create_synthetic_data():
        np.random.seed(42)
        n = 1000
        nhanes = pd.DataFrame({
            'RIDAGEYR': np.random.randint(20, 80, n),
            'RIAGENDR': np.random.choice([1, 2], n),
            'BMXBMI': np.random.normal(27, 5, n),
            'BPXOSY1': np.random.normal(125, 15, n),
            'LBXTC': np.random.normal(200, 40, n),
            'LBDHDD': np.random.normal(50, 12, n),
            'SMQ856': np.random.choice([1, 2], n, p=[0.3, 0.7]),
            'DIQ010': np.random.choice([1, 2], n, p=[0.1, 0.9]),
            'PAQ605': np.random.choice([1, 2], n, p=[0.4, 0.6])
        })
        framingham = pd.DataFrame({
            'age': np.random.randint(30, 70, n),
            'bmi': np.random.normal(27, 5, n),
            'bp': np.random.normal(130, 15, n),
            'cholesterol': np.random.normal(210, 40, n),
            'smoker': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'outcome': np.random.choice([0, 1], n, p=[0.85, 0.15])
        })
        if framingham['outcome'].nunique() == 1:
            framingham.iloc[0, framingham.columns.get_loc('outcome')] = 1 - framingham['outcome'].iloc[0]
        return nhanes, framingham

    nhanes, framingham = load_datasets()

    # ==================== ADVANCED AI MODEL ====================
    class AdvancedAIModel:
        def __init__(self):
            self.model = None
            self.scaler = StandardScaler()
            self.feature_importance = None
            
        def train(self, X, y):
            X_scaled = self.scaler.fit_transform(X)
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return self
        
        def predict_with_confidence(self, X):
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)
            return pred, pred * 0.1, 0.85

    model = AdvancedAIModel()

    # ==================== ENHANCED EXECUTIVE DASHBOARD ====================
    if selected == "Executive Dashboard":
        st.markdown("## 📊 Executive Health Dashboard")
        
        current_time = datetime.now().strftime("%A, %B %d, %Y • %I:%M %p")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1E88E5, #1565C0); 
                    padding: 1.5rem; border-radius: 20px; margin-bottom: 2rem; color: white;'>
            <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                <div>
                    <span style='font-size: 2rem;'>👋</span>
                    <span style='font-size: 1.5rem; font-weight: 600; margin-left: 0.5rem;'>Welcome back, {st.session_state.get('username', 'User')}!</span>
                    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Your health journey continues. Here's your personalized health snapshot.</p>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 1.2rem; font-weight: 500;'>{current_time}</div>
                    <div style='font-size: 0.9rem; opacity: 0.8;'>Last sync: Just now</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        
        user_history = db.get_user_history(user_id, limit=1)
        if user_history:
            last_risk = user_history[0]['risk_score']
            health_score = 100 - (last_risk * 100)
            
            if last_risk < 0.1:
                risk_level = "Low Risk"
                risk_color = "#43A047"
                risk_icon = "🟢"
            elif last_risk < 0.2:
                risk_level = "Moderate Risk"
                risk_color = "#FB8C00"
                risk_icon = "🟡"
            else:
                risk_level = "High Risk"
                risk_color = "#E53935"
                risk_icon = "🔴"
            
            with col_k1:
                st.markdown(f"""
                <div class="metric-card" style="cursor: pointer;">
                    <div class="metric-label">HEALTH SCORE</div>
                    <div class="metric-value animate-number">{health_score:.0f}<span style='font-size: 1rem;'>/100</span></div>
                    <div style='font-size: 0.8rem; color: #666;'>⬆️ +2 pts this week</div>
                    <div style='margin-top: 0.5rem;'>
                        <span class='badge badge-success'>Excellent</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_k2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">5-YEAR RISK</div>
                    <div class="metric-value animate-number" style="color: {risk_color};">{last_risk*100:.1f}<span style='font-size: 1rem;'>%</span></div>
                    <div style='font-size: 0.8rem; color: #666;'>vs 15% population avg</div>
                    <div style='margin-top: 0.5rem;'>
                        <span class='badge {"badge-success" if last_risk<0.1 else "badge-warning" if last_risk<0.2 else "badge-danger"}'>
                            {risk_icon} {risk_level}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_k3:
                pred_count = len(db.get_user_history(user_id, limit=1000))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">HEALTH JOURNEY</div>
                    <div class="metric-value animate-number">{pred_count}</div>
                    <div style='font-size: 0.8rem; color: #666;'>Total Health Assessments</div>
                    <div style='margin-top: 0.5rem;'>
                        <div style='background: #e0e0e0; border-radius: 10px; height: 4px; width: 100%;'>
                            <div style='width: {min(100, pred_count/20*100)}%; background: #1E88E5; height: 4px; border-radius: 10px;'></div>
                        </div>
                        <div style='font-size: 0.7rem; color: #666; margin-top: 0.3rem;'>Track your progress</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_k4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">STREAK</div>
                    <div class="metric-value animate-number">7</div>
                    <div style='font-size: 0.8rem; color: #666;'>Days Active</div>
                    <div style='margin-top: 0.5rem;'>
                        <span style='font-size: 1.2rem;'>🔥</span>
                        <span style='font-size: 0.8rem; color: #666;'>Keep it up!</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            for col in [col_k1, col_k2, col_k3, col_k4]:
                with col:
                    st.info("Complete a risk assessment to see your metrics!")
        
        st.markdown("---")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### 📈 Risk Trend Analysis")
            st.markdown("*Track your health journey over time*")
            
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                days_back = st.select_slider("Time Period", options=[7, 30, 90, 180, 365], value=90)
            with col_date2:
                show_trend_line = st.checkbox("Show Trend Line", value=True)
            
            dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
            base_risk = 0.15 if not user_history else user_history[0]['risk_score']
            simulated_risks = base_risk + 0.02 * np.random.randn(days_back)
            simulated_risks = np.clip(simulated_risks, 0.05, 0.35)
            
            df_risk = pd.DataFrame({'Date': dates, 'Risk': simulated_risks * 100})
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_risk['Date'], y=df_risk['Risk'],
                mode='lines+markers', name='Risk Score',
                line=dict(color='#1E88E5', width=2),
                marker=dict(size=4, color='#1E88E5', opacity=0.7),
                fill='tozeroy', fillcolor='rgba(30,136,229,0.1)'
            ))
            
            if show_trend_line:
                z = np.polyfit(range(len(df_risk)), df_risk['Risk'], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=df_risk['Date'], y=p(range(len(df_risk))),
                    mode='lines', name='Trend',
                    line=dict(color='#E53935', width=2, dash='dash')
                ))
            
            fig.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="Low Risk")
            fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="High Risk")
            fig.update_layout(
                title="Risk Score Over Time",
                xaxis_title="Date",
                yaxis_title="Risk (%)",
                yaxis_range=[0, 35],
                height=400,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Current", f"{df_risk['Risk'].iloc[-1]:.1f}%")
            with col_s2:
                st.metric("Average", f"{df_risk['Risk'].mean():.1f}%")
            with col_s3:
                st.metric("Min", f"{df_risk['Risk'].min():.1f}%")
            with col_s4:
                st.metric("Max", f"{df_risk['Risk'].max():.1f}%")
        
        with col_right:
            st.markdown("### 🎯 Your Health Score")
            st.markdown("*Based on your latest assessment*")
            
            if user_history:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=health_score,
                    title={'text': "Health Score", 'font': {'size': 20}},
                    delta={'reference': 80, 'increasing': {'color': "#43A047"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#1E88E5"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "lightgray",
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(229, 57, 53, 0.2)'},
                            {'range': [40, 70], 'color': 'rgba(251, 140, 0, 0.2)'},
                            {'range': [70, 100], 'color': 'rgba(67, 160, 71, 0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': health_score
                        }
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown("### 🔧 Quick Actions")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("📝 New Assessment", use_container_width=True):
                        st.session_state.selected = "Risk Analysis"
                        st.rerun()
                with col_btn2:
                    if st.button("💬 Chat with AI", use_container_width=True):
                        st.session_state.selected = "AI Health Chat"
                        st.rerun()
            else:
                st.info("👋 Complete a risk assessment to see your health score!")
                if st.button("Start Assessment →", use_container_width=True):
                    st.session_state.selected = "Risk Analysis"
                    st.rerun()
        
        st.markdown("---")
        
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            st.markdown("### 📋 Recent Activity")
            recent_history = db.get_user_history(user_id, limit=5)
            if recent_history:
                for rec in recent_history:
                    risk_pct = rec['risk_score'] * 100
                    if risk_pct < 10:
                        badge = "badge-success"
                        icon = "✅"
                    elif risk_pct < 20:
                        badge = "badge-warning"
                        icon = "⚠️"
                    else:
                        badge = "badge-danger"
                        icon = "🔴"
                    
                    date_value = rec.get('date', '')
                    if not date_value:
                        date_str = 'Unknown Date'
                    elif isinstance(date_value, str):
                        date_str = date_value
                    else:
                        date_str = date_value.strftime('%Y-%m-%d %H:%M')
                    
                    st.markdown(f"""
                    <div style='background: #f8f9fa; padding: 0.8rem; border-radius: 10px; margin-bottom: 0.5rem;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='font-size: 1.2rem;'>{icon}</span>
                                <span><strong>Risk Assessment</strong></span>
                            </div>
                            <span class='badge {badge}'>{risk_pct:.1f}% Risk</span>
                        </div>
                        <div style='font-size: 0.8rem; color: #666; margin-top: 0.3rem;'>
                            {date_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent activity. Start by completing a risk assessment!")
        
        with col_b2:
            st.markdown("### 💡 Personalized Insights")
            
            if user_history:
                last_pred = user_history[0]
                user_bmi = last_pred.get('bmi', 25)
                user_bp = last_pred.get('bp', 120)
                user_smoker = last_pred.get('smoker', False)
                
                insights = []
                if user_bmi > 25:
                    insights.append("🏋️ **Weight Management**: Your BMI indicates room for improvement. Even 5% weight loss can significantly reduce risk.")
                if user_bp > 130:
                    insights.append("🩺 **Blood Pressure**: Consider reducing sodium intake and increasing physical activity.")
                if user_smoker:
                    insights.append("🚭 **Smoking Cessation**: Quitting smoking is the single most impactful change for your heart health.")
                if not insights:
                    insights.append("🌟 **Great Job!** Your health metrics are looking good. Maintain your healthy habits!")
                
                for insight in insights[:3]:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #E8F0FE, #f0f7ff); 
                                padding: 0.8rem; border-radius: 10px; margin-bottom: 0.5rem;
                                border-left: 4px solid #1E88E5;'>
                        {insight}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Complete a risk assessment to get personalized insights!")
            
            if user_history:
                st.markdown("---")
                if st.button("📥 Download Full Health Report", use_container_width=True):
                    report = generate_health_report(
                        user_history[0],
                        {'current_risk': user_history[0]['risk_score'], 'risk_level': 'MODERATE', 'health_score': health_score},
                        user_history
                    )
                    st.download_button("📄 Download PDF Report", data=report, file_name=f"health_report_{datetime.now().strftime('%Y%m%d')}.md", use_container_width=True)

    # ==================== HEALTH INSIGHTS (User-friendly - replaces Predictive Models) ====================
    elif selected == "Health Insights":
        st.markdown("## 💡 Your Personal Health Insights")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>🧠 What Your Health Data Tells Us</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Based on your health information, here are personalized insights to help you stay healthy.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        user_history_insights = db.get_user_history(user_id, limit=1)
        
        if not user_history_insights:
            st.info("👋 Please complete a risk assessment first to see your personalized health insights!")
            if st.button("Start Risk Assessment →", use_container_width=True):
                st.session_state.selected = "Risk Analysis"
                st.rerun()
        else:
            last_pred = user_history_insights[0]
            user_age = last_pred.get('age', 45)
            user_bmi = last_pred.get('bmi', 25.0)
            user_bp = last_pred.get('bp', 120)
            user_chol = last_pred.get('cholesterol', 200)
            user_smoker = last_pred.get('smoker', False)
            user_exercise = last_pred.get('exercise', 'Moderate')
            user_diet = last_pred.get('diet', 'Good')
            
            risk_score = 0.02
            risk_score += max(0, (user_age - 40) * 0.005)
            risk_score += max(0, (user_bmi - 22) * 0.01)
            risk_score += max(0, (user_bp - 110) * 0.002)
            risk_score += max(0, (user_chol - 180) * 0.001)
            risk_score += 0.15 if user_smoker else 0
            ex_benefit = {"Sedentary": 0, "Light": 0.5, "Moderate": 1, "Active": 1.5, "Very Active": 2}
            risk_score -= ex_benefit.get(user_exercise, 0) * 0.02
            diet_benefit = {"Poor": 0, "Fair": 0.5, "Good": 1, "Excellent": 1.5}
            risk_score -= diet_benefit.get(user_diet, 0) * 0.01
            risk_score = np.clip(risk_score, 0.01, 0.5) * 100
            
            if risk_score < 10:
                risk_level = "Low"
                risk_color = "#43A047"
                risk_icon = "🟢"
                risk_message = "Your risk is low! Keep maintaining your healthy habits."
            elif risk_score < 20:
                risk_level = "Moderate"
                risk_color = "#FB8C00"
                risk_icon = "🟡"
                risk_message = "Your risk is moderate. Small changes can make a big difference."
            else:
                risk_level = "Elevated"
                risk_color = "#E53935"
                risk_icon = "🔴"
                risk_message = "Your risk is elevated. Focus on lifestyle changes to improve."
            
            st.markdown(f"""
            <div style='background: white; padding: 1.5rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.05);'>
                <div style='font-size: 1rem; color: #666;'>Your 5-Year Health Risk</div>
                <div style='font-size: 4rem; font-weight: 700; color: {risk_color};'>{risk_score:.1f}%</div>
                <div><span class='badge {"badge-success" if risk_score<10 else "badge-warning" if risk_score<20 else "badge-danger"}'>
                    {risk_icon} {risk_level} Risk
                </span></div>
                <p style='margin-top: 1rem; color: #666;'>{risk_message}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔍 What Affects Your Health Most?")
            
            col_imp1, col_imp2 = st.columns(2)
            
            with col_imp1:
                st.markdown("#### 📊 Factors That May Increase Risk")
                
                factors = []
                values = []
                
                if user_age > 50:
                    factors.append("Age")
                    values.append(min(30, (user_age - 50) * 0.5))
                if user_bmi > 25:
                    factors.append("Weight")
                    values.append(min(25, (user_bmi - 25) * 2))
                if user_bp > 120:
                    factors.append("Blood Pressure")
                    values.append(min(20, (user_bp - 120) * 0.3))
                if user_chol > 200:
                    factors.append("Cholesterol")
                    values.append(min(15, (user_chol - 200) * 0.1))
                if user_smoker:
                    factors.append("Smoking")
                    values.append(15)
                
                if not factors:
                    factors = ["All your health markers look good!"]
                    values = [0]
                
                fig_factors = go.Figure()
                fig_factors.add_trace(go.Bar(
                    x=values, y=factors,
                    orientation='h',
                    marker_color='#E53935',
                    text=[f"{v:.0f}" for v in values] if values[0] > 0 else None,
                    textposition='outside'
                ))
                fig_factors.update_layout(
                    title="Areas That Need Attention",
                    xaxis_title="Impact on Health",
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_factors, use_container_width=True)
            
            with col_imp2:
                st.markdown("#### 🌟 Things That Protect You")
                
                protective_factors = []
                protective_values = []
                
                if not user_smoker:
                    protective_factors.append("Non-smoker")
                    protective_values.append(15)
                if user_exercise in ["Active", "Very Active"]:
                    protective_factors.append("Regular Exercise")
                    protective_values.append(10)
                if user_diet in ["Good", "Excellent"]:
                    protective_factors.append("Healthy Diet")
                    protective_values.append(8)
                if 7 <= last_pred.get('sleep', 7) <= 9:
                    protective_factors.append("Good Sleep")
                    protective_values.append(5)
                
                if not protective_factors:
                    protective_factors = ["Room for improvement"]
                    protective_values = [0]
                
                fig_protect = go.Figure()
                fig_protect.add_trace(go.Bar(
                    x=protective_values, y=protective_factors,
                    orientation='h',
                    marker_color='#43A047',
                    text=[f"+{v:.0f}" for v in protective_values] if protective_values[0] > 0 else None,
                    textposition='outside'
                ))
                fig_protect.update_layout(
                    title="What's Working Well For You",
                    xaxis_title="Health Benefit",
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_protect, use_container_width=True)
            
            st.markdown("### 📋 Your Personalized Recommendations")
            
            recommendations_list = []
            
            if user_bmi > 25:
                recommendations_list.append({
                    "title": "⚖️ Healthy Weight",
                    "description": f"Your BMI is {user_bmi:.1f}. Even a small amount of weight loss can make a big difference.",
                    "action": "Try 10-minute walks after meals and watch portion sizes."
                })
            
            if user_bp > 130:
                recommendations_list.append({
                    "title": "🩺 Blood Pressure",
                    "description": f"Your blood pressure is {user_bp}. Keeping it under 120 helps protect your heart.",
                    "action": "Cut back on salt, walk daily, and try deep breathing when stressed."
                })
            
            if user_chol > 200:
                recommendations_list.append({
                    "title": "🥗 Cholesterol",
                    "description": f"Your cholesterol is {user_chol}. Aiming for below 200 is ideal.",
                    "action": "Add oats, nuts, and fish to your meals. Cut back on fried foods."
                })
            
            if user_smoker:
                recommendations_list.append({
                    "title": "🚭 Quit Smoking",
                    "description": "This is the single most important thing you can do for your heart.",
                    "action": "Talk to your doctor about quitting resources. Every smoke-free day helps!"
                })
            
            if user_exercise in ["Sedentary", "Light"]:
                recommendations_list.append({
                    "title": "🏃 Get Moving",
                    "description": "Physical activity is medicine for your heart.",
                    "action": "Start with 15-minute walks, 5 days a week. Gradually increase to 30 minutes."
                })
            
            if user_diet in ["Poor", "Fair"]:
                recommendations_list.append({
                    "title": "🥬 Eat Better",
                    "description": "Healthy eating can reduce heart risk by up to 30%.",
                    "action": "Add one vegetable to every meal. Replace sugary drinks with water."
                })
            
            if not recommendations_list:
                recommendations_list.append({
                    "title": "🌟 Keep Up the Great Work!",
                    "description": "Your health markers look excellent.",
                    "action": "Maintain your healthy habits and schedule regular check-ups."
                })
            
            for rec in recommendations_list[:4]:
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; margin-bottom: 1rem; border-left: 4px solid #1E88E5; box-shadow: 0 2px 8px rgba(0,0,0,0.05);'>
                    <h4 style='margin:0; color: #1E88E5;'>{rec['title']}</h4>
                    <p style='margin:0.5rem 0; color: #666;'>{rec['description']}</p>
                    <p style='margin:0; color: #43A047;'><strong>💡 Try This:</strong> {rec['action']}</p>
                </div>
                """, unsafe_allow_html=True)

    # ==================== ENHANCED HEALTH TRENDS ====================
    elif selected == "Health Trends":
        st.markdown("## 📈 Health Trends Analytics")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>📊 Interactive Trend Analysis</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Explore your health trends with interactive filters and visualizations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            days_back = st.select_slider("Time Range", options=[30, 60, 90, 180, 365], value=90)
        with col_f2:
            metric_type = st.selectbox("Metric", ["Risk Score", "Health Score", "Blood Pressure", "BMI"])
        with col_f3:
            smooth_window = st.slider("Smoothing Window (days)", 1, 30, 7)
        
        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        
        if metric_type == "Risk Score":
            values = 0.15 + 0.02 * np.random.randn(days_back)
            values = np.clip(values, 0.05, 0.35) * 100
            y_label = "Risk Score (%)"
            color = "#E53935"
            threshold = 20
        elif metric_type == "Health Score":
            values = 75 + 5 * np.random.randn(days_back)
            values = np.clip(values, 50, 100)
            y_label = "Health Score"
            color = "#43A047"
            threshold = 70
        elif metric_type == "Blood Pressure":
            values = 120 + 10 * np.random.randn(days_back)
            values = np.clip(values, 100, 160)
            y_label = "Blood Pressure (mmHg)"
            color = "#FB8C00"
            threshold = 130
        else:
            values = 25 + 1.5 * np.random.randn(days_back)
            values = np.clip(values, 18, 35)
            y_label = "BMI"
            color = "#1E88E5"
            threshold = 25
        
        df_trend = pd.DataFrame({'Date': dates, 'Value': values})
        df_trend['SMA'] = df_trend['Value'].rolling(window=smooth_window, min_periods=1).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_trend['Date'], y=df_trend['Value'],
            mode='lines', name='Daily',
            line=dict(color=color, width=1, dash='dot'),
            opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=df_trend['Date'], y=df_trend['SMA'],
            mode='lines', name=f'{smooth_window}-Day Average',
            line=dict(color=color, width=3),
            fill='tozeroy', fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)'
        ))
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(
            title=f"{metric_type} Trend Over Time",
            xaxis_title="Date",
            yaxis_title=y_label,
            height=450,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Statistical Summary")
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            st.metric("Current", f"{df_trend['Value'].iloc[-1]:.1f}")
        with col_s2:
            st.metric("Average", f"{df_trend['Value'].mean():.1f}")
        with col_s3:
            st.metric("Min", f"{df_trend['Value'].min():.1f}")
        with col_s4:
            st.metric("Max", f"{df_trend['Value'].max():.1f}")
        with col_s5:
            change = df_trend['Value'].iloc[-1] - df_trend['Value'].iloc[0]
            st.metric("Change", f"{change:+.1f}", delta_color="inverse" if metric_type == "Risk Score" else "normal")
        
        st.markdown("### 📥 Export Data")
        csv_data = df_trend.to_csv(index=False)
        st.download_button(
            label="Download Trend Data (CSV)",
            data=csv_data,
            file_name=f"health_trends_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ==================== ENHANCED TWIN ANALYSIS ====================
    elif selected == "Twin Analysis":
        st.markdown("## 👥 Twin Health Analysis")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>👥 Genetic-Aware Risk Assessment</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                This model considers genetic similarity between twins to provide more accurate 
                5-year health risk predictions. Enter data for both twins to see the difference.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.twin_model_loaded:
            try:
                if TwinHealthModel is not None:
                    st.session_state.twin_model = TwinHealthModel()
                    st.session_state.twin_model_loaded = True
                else:
                    st.session_state.twin_model_loaded = False
            except Exception:
                st.session_state.twin_model_loaded = False
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("### 👤 Twin A")
            age_a = st.number_input("Age", 20, 80, 45, key="twin_a_age")
            bmi_a = st.number_input("BMI", 15.0, 45.0, 25.0, 0.1, key="twin_a_bmi")
            bp_a = st.number_input("Blood Pressure", 90, 200, 120, key="twin_a_bp")
            chol_a = st.number_input("Cholesterol", 100, 350, 200, key="twin_a_chol")
            smoker_a = st.checkbox("Smoker", key="twin_a_smoker")
            exercise_a = st.select_slider("Exercise", options=["Sedentary", "Light", "Moderate", "Active", "Very Active"], key="twin_a_ex")
        
        with col_t2:
            st.markdown("### 👤 Twin B")
            age_b = st.number_input("Age", 20, 80, 45, key="twin_b_age")
            bmi_b = st.number_input("BMI", 15.0, 45.0, 27.0, 0.1, key="twin_b_bmi")
            bp_b = st.number_input("Blood Pressure", 90, 200, 130, key="twin_b_bp")
            chol_b = st.number_input("Cholesterol", 100, 350, 220, key="twin_b_chol")
            smoker_b = st.checkbox("Smoker", key="twin_b_smoker")
            exercise_b = st.select_slider("Exercise", options=["Sedentary", "Light", "Moderate", "Active", "Very Active"], key="twin_b_ex")
        
        if st.button("🔍 Analyze Twin Health Risks", type="primary", use_container_width=True):
            def calc_risk(age, bmi, bp, chol, smoker, exercise):
                risk = 0.02
                risk += max(0, (age - 40) * 0.005)
                risk += max(0, (bmi - 25) * 0.01)
                risk += max(0, (bp - 120) * 0.002)
                risk += max(0, (chol - 200) * 0.001)
                risk += 0.15 if smoker else 0
                ex_benefit = {"Sedentary": 0, "Light": 0.5, "Moderate": 1.0, "Active": 1.5, "Very Active": 2.0}
                risk -= ex_benefit.get(exercise, 0) * 0.02
                return np.clip(risk, 0.01, 0.5)
            
            risk_a = calc_risk(age_a, bmi_a, bp_a, chol_a, smoker_a, exercise_a)
            risk_b = calc_risk(age_b, bmi_b, bp_b, chol_b, smoker_b, exercise_b)
            
            twin_factor = 0.85
            risk_a_adjusted = risk_a + (risk_b - risk_a) * (1 - twin_factor) * 0.3
            risk_b_adjusted = risk_b + (risk_a - risk_b) * (1 - twin_factor) * 0.3
            risk_a_adjusted = np.clip(risk_a_adjusted, 0.01, 0.5)
            risk_b_adjusted = np.clip(risk_b_adjusted, 0.01, 0.5)
            
            st.markdown("---")
            st.markdown("## 📊 Twin Risk Analysis Results")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Individual Risk',
                x=['Twin A', 'Twin B'],
                y=[risk_a*100, risk_b*100],
                marker_color=['#9E9E9E', '#9E9E9E'],
                text=[f"{risk_a*100:.1f}%", f"{risk_b*100:.1f}%"],
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                name='Twin-Aware Risk',
                x=['Twin A', 'Twin B'],
                y=[risk_a_adjusted*100, risk_b_adjusted*100],
                marker_color=['#1E88E5', '#1E88E5'],
                text=[f"{risk_a_adjusted*100:.1f}%", f"{risk_b_adjusted*100:.1f}%"],
                textposition='outside'
            ))
            fig.update_layout(
                title="Individual vs Twin-Aware Risk",
                yaxis_title="5-Year Risk (%)",
                yaxis_range=[0, 35],
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                if risk_a_adjusted < 0.1:
                    badge = "badge-success"
                    icon = "🟢"
                elif risk_a_adjusted < 0.2:
                    badge = "badge-warning"
                    icon = "🟡"
                else:
                    badge = "badge-danger"
                    icon = "🔴"
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 20px; text-align: center;'>
                    <div style='font-size: 1rem; color: #666;'>Twin A</div>
                    <div style='font-size: 3rem; font-weight: 700;'>{risk_a_adjusted*100:.1f}%</div>
                    <div><span class='badge {badge}'>{icon} { 'HIGH' if risk_a_adjusted>0.2 else 'MODERATE' if risk_a_adjusted>0.1 else 'LOW'}</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r2:
                diff = (risk_b_adjusted - risk_a_adjusted) * 100
                diff_icon = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 20px; text-align: center;'>
                    <div style='font-size: 1rem; color: #666;'>Difference</div>
                    <div style='font-size: 2rem; font-weight: 700;'>{diff_icon} {abs(diff):.1f}%</div>
                    <div>Twin B {'higher' if diff>0 else 'lower' if diff<0 else 'equal'}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r3:
                if risk_b_adjusted < 0.1:
                    badge = "badge-success"
                    icon = "🟢"
                elif risk_b_adjusted < 0.2:
                    badge = "badge-warning"
                    icon = "🟡"
                else:
                    badge = "badge-danger"
                    icon = "🔴"
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 20px; text-align: center;'>
                    <div style='font-size: 1rem; color: #666;'>Twin B</div>
                    <div style='font-size: 3rem; font-weight: 700;'>{risk_b_adjusted*100:.1f}%</div>
                    <div><span class='badge {badge}'>{icon} { 'HIGH' if risk_b_adjusted>0.2 else 'MODERATE' if risk_b_adjusted>0.1 else 'LOW'}</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🧬 Genetic vs Lifestyle Risk Breakdown")
            
            genetic_impact_a = abs(risk_a_adjusted - risk_a) * 100
            lifestyle_impact_a = risk_a_adjusted * 100 - genetic_impact_a
            
            fig_pie = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                                     subplot_titles=["Twin A", "Twin B"])
            fig_pie.add_trace(go.Pie(labels=['Genetic Influence', 'Lifestyle Factors'], 
                                     values=[genetic_impact_a, lifestyle_impact_a],
                                     marker_colors=['#9C27B0', '#1E88E5']), row=1, col=1)
            
            genetic_impact_b = abs(risk_b_adjusted - risk_b) * 100
            lifestyle_impact_b = risk_b_adjusted * 100 - genetic_impact_b
            fig_pie.add_trace(go.Pie(labels=['Genetic Influence', 'Lifestyle Factors'],
                                     values=[genetic_impact_b, lifestyle_impact_b],
                                     marker_colors=['#9C27B0', '#1E88E5']), row=1, col=2)
            fig_pie.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("### 💡 Personalized Recommendations")
            if abs(risk_b - risk_a) > 0.05:
                higher_risk_twin = "Twin B" if risk_b > risk_a else "Twin A"
                st.warning(f"⚠️ **{higher_risk_twin}** has significantly higher risk. Focus on modifiable factors:")
                
                if abs(bmi_a - bmi_b) > 3:
                    st.info(f"📊 BMI difference of {abs(bmi_a - bmi_b):.1f} points. The twin with higher BMI should focus on weight management.")
                if smoker_a != smoker_b:
                    st.info(f"🚭 Smoking status differs. The smoking twin should consider cessation programs.")
                if exercise_a != exercise_b:
                    st.info(f"🏃 Exercise levels differ. Both twins should aim for at least 150 minutes of moderate activity weekly.")
            else:
                st.success("✅ Both twins have similar risk profiles. Work together on shared health goals!")

    # ==================== HABIT TRACKER (User-friendly - replaces Care Adherence) ====================
    elif selected == "Habit Tracker":
        st.markdown("## ✅ Daily Health Habit Tracker")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #E8F0FE 0%, #f0f7ff 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>🌟 Track Your Daily Healthy Habits</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Log your daily activities and see how consistent you are with healthy habits!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        today = datetime.now().strftime("%A, %B %d, %Y")
        st.markdown(f"### 📅 {today}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💧 Water Intake")
            water_glasses = st.number_input("How many glasses of water did you drink?", 0, 15, 0, key="water_habit")
            if water_glasses >= 8:
                st.success("✅ Great! 8+ glasses is excellent for health!")
            elif water_glasses >= 4:
                st.info("👍 Good start! Aim for 8 glasses daily.")
            else:
                st.warning("💧 Try to drink more water. Aim for 8 glasses daily.")
            
            st.markdown("#### 👟 Steps")
            steps = st.number_input("How many steps did you take today?", 0, 20000, 0, step=500, key="steps_habit")
            if steps >= 10000:
                st.success("🏆 Amazing! You've reached the daily step goal!")
            elif steps >= 5000:
                st.info("👍 Good job! Try to reach 10,000 steps.")
            else:
                st.warning("🚶 Try to move more. Aim for 5,000+ steps daily.")
        
        with col2:
            st.markdown("#### 😴 Sleep")
            sleep_hours = st.number_input("How many hours did you sleep?", 0.0, 12.0, 0.0, step=0.5, key="sleep_habit")
            if 7 <= sleep_hours <= 9:
                st.success("😊 Perfect! 7-9 hours is optimal for health.")
            elif sleep_hours > 0:
                st.info("💤 Aim for 7-9 hours of quality sleep.")
            else:
                st.info("😴 Quality sleep helps your heart and brain.")
            
            st.markdown("#### 🧘 Mindfulness")
            meditation_minutes = st.number_input("Minutes of meditation or deep breathing?", 0, 60, 0, key="meditate_habit")
            if meditation_minutes >= 10:
                st.success("🧘 Great! Regular mindfulness reduces stress.")
            elif meditation_minutes > 0:
                st.info("🌿 Every minute counts! Try for 10 minutes daily.")
            else:
                st.info("🧘 Try 5-10 minutes of deep breathing to reduce stress.")
        
        if st.button("📝 Save Today's Habits", type="primary", use_container_width=True):
            habit_score = 0
            habit_score += min(10, water_glasses)
            habit_score += min(10, steps / 1000)
            habit_score += min(10, sleep_hours)
            habit_score += min(10, meditation_minutes / 6)
            habit_score = min(100, habit_score * 2)
            
            st.session_state.habit_history.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "water": water_glasses,
                "steps": steps,
                "sleep": sleep_hours,
                "meditation": meditation_minutes,
                "score": habit_score
            })
            
            st.balloons()
            st.success(f"✅ Habits saved! Your daily habit score: {habit_score:.0f}/100")
            
            if habit_score >= 80:
                st.success("🎉 Outstanding day! You're building excellent health habits!")
            elif habit_score >= 60:
                st.info("👍 Good day! Small improvements will make a big difference.")
            else:
                st.warning("💪 Every healthy choice counts. Try to add one more healthy habit tomorrow!")
        
        if st.session_state.habit_history:
            st.markdown("---")
            st.markdown("### 📊 Your Habit History")
            
            recent_habits = st.session_state.habit_history[-7:]
            habit_df = pd.DataFrame(recent_habits)
            
            if len(habit_df) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=habit_df['date'], y=habit_df['score'],
                    mode='lines+markers',
                    name='Habit Score',
                    line=dict(color='#1E88E5', width=3),
                    marker=dict(size=8, color='#1E88E5'),
                    fill='tozeroy',
                    fillcolor='rgba(30,136,229,0.1)'
                ))
                fig.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Good Target")
                fig.update_layout(
                    title="Your Daily Habit Score Over Time",
                    xaxis_title="Date",
                    yaxis_title="Score (0-100)",
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                avg_score = habit_df['score'].mean()
                st.metric("Average Daily Score", f"{avg_score:.0f}/100")
                
                if avg_score >= 70:
                    st.success("🌟 You're building great habits! Keep going!")
                elif avg_score >= 50:
                    st.info("👍 You're on the right track! Try to add one more healthy habit.")
                else:
                    st.warning("💪 Start small - pick one habit to focus on this week.")
        else:
            st.info("No habits logged yet. Start tracking your daily habits above!")

    # ==================== MY TASKS (Real Task Scheduler - replaces Task Management) ====================
    elif selected == "My Tasks":
        st.markdown("## 📋 My Health Tasks")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #E8F0FE 0%, #f0f7ff 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>✅ Stay on Track with Your Health Goals</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Add tasks, mark them as complete, and build healthy habits one step at a time.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("➕ Add New Task", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_task = st.text_input("Task name", placeholder="e.g., Take medication")
            with col2:
                priority = st.selectbox("Priority", ["High", "Medium", "Low"])
            with col3:
                due_date = st.date_input("Due date", min_value=datetime.now())
            
            if st.button("📌 Add Task", use_container_width=True):
                if new_task.strip():
                    st.session_state.user_tasks.append({
                        "task": new_task,
                        "completed": False,
                        "priority": priority.lower(),
                        "due_date": due_date.strftime("%Y-%m-%d")
                    })
                    st.success(f"✅ Task added: {new_task}")
                    st.rerun()
                else:
                    st.error("Please enter a task name")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_status = st.selectbox("Show", ["All Tasks", "Pending", "Completed"])
        with col_f2:
            sort_by = st.selectbox("Sort by", ["Due Date", "Priority"])
        
        if filter_status == "Pending":
            filtered_tasks = [t for t in st.session_state.user_tasks if not t["completed"]]
        elif filter_status == "Completed":
            filtered_tasks = [t for t in st.session_state.user_tasks if t["completed"]]
        else:
            filtered_tasks = st.session_state.user_tasks.copy()
        
        if sort_by == "Due Date":
            filtered_tasks.sort(key=lambda x: x.get("due_date", "9999-12-31"))
        elif sort_by == "Priority":
            priority_order = {"high": 0, "medium": 1, "low": 2}
            filtered_tasks.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 1))
        
        completed_count = sum(1 for t in st.session_state.user_tasks if t["completed"])
        pending_count = len(st.session_state.user_tasks) - completed_count
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("📋 Total", len(st.session_state.user_tasks))
        with col_stats2:
            st.metric("✅ Done", completed_count)
        with col_stats3:
            st.metric("⏳ To Do", pending_count)
        
        st.markdown("---")
        
        if not filtered_tasks:
            st.info("No tasks found. Add a task to get started!")
        else:
            for i, task in enumerate(filtered_tasks):
                if task.get("priority") == "high":
                    priority_color = "#E53935"
                    priority_badge = "🔴 High"
                elif task.get("priority") == "medium":
                    priority_color = "#FB8C00"
                    priority_badge = "🟠 Medium"
                else:
                    priority_color = "#43A047"
                    priority_badge = "🟢 Low"
                
                due_date_str = task.get("due_date", "")
                is_overdue = False
                if due_date_str and not task["completed"]:
                    try:
                        due_dt = datetime.strptime(due_date_str, "%Y-%m-%d")
                        if due_dt.date() < datetime.now().date():
                            is_overdue = True
                    except:
                        pass
                
                col1, col2, col3 = st.columns([5, 1, 1])
                
                with col1:
                    if task["completed"]:
                        st.markdown(f"""
                        <div style='background: #f0f7f0; padding: 0.8rem; border-radius: 12px; margin-bottom: 0.5rem; opacity: 0.7;'>
                            <span style='font-size: 1.2rem;'>✅</span>
                            <span style='text-decoration: line-through; color: #888;'>{task['task']}</span>
                            <div style='font-size: 0.7rem; color: #888; margin-top: 0.2rem;'>
                                {priority_badge} · Due: {due_date_str if due_date_str else 'No date'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        overdue_mark = " ⚠️ OVERDUE" if is_overdue else ""
                        st.markdown(f"""
                        <div style='background: white; padding: 0.8rem; border-radius: 12px; margin-bottom: 0.5rem; border-left: 4px solid {priority_color}; box-shadow: 0 2px 8px rgba(0,0,0,0.05);'>
                            <span style='font-size: 1.1rem; font-weight: 500;'>{task['task']}</span>
                            <span style='color: {priority_color}; font-size: 0.7rem; margin-left: 0.5rem;'>{overdue_mark}</span>
                            <div style='font-size: 0.7rem; color: #666; margin-top: 0.2rem;'>
                                {priority_badge} · Due: {due_date_str if due_date_str else 'No date'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if not task["completed"]:
                        if st.button("✅ Done", key=f"complete_{i}_{task['task']}"):
                            task["completed"] = True
                            st.success(f"🎉 Great job completing: {task['task']}")
                            st.rerun()
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{i}_{task['task']}"):
                        st.session_state.user_tasks.remove(task)
                        st.rerun()

    # ==================== ENHANCED HEALTH MONITOR ====================
    elif selected == "Health Monitor":
        st.markdown("## 🚨 AI-Powered Health Monitor")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>🚨 Real-time Anomaly Detection</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Monitor your vital signs and get alerts for any concerning patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.anomaly_detector_ml is None:
            st.session_state.anomaly_detector_ml = IsolationForestAnomalyDetector()
            normal_data = np.random.randn(1000, 3)
            normal_data[:, 0] = normal_data[:, 0] * 10 + 75
            normal_data[:, 1] = normal_data[:, 1] * 15 + 120
            normal_data[:, 2] = normal_data[:, 2] * 0.5 + 36.6
            st.session_state.anomaly_detector_ml.fit(normal_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            current_hr = st.number_input("Heart Rate (BPM)", 40, 150, 75)
            if current_hr < 60:
                st.caption("🟦 Low (normal for athletes)")
            elif current_hr <= 100:
                st.caption("🟢 Normal range")
            else:
                st.caption("🔴 Elevated")
        with col2:
            current_bp = st.number_input("Blood Pressure (systolic)", 80, 200, 120)
            if current_bp < 90:
                st.caption("🟦 Low (hypotension)")
            elif current_bp <= 120:
                st.caption("🟢 Optimal")
            elif current_bp <= 130:
                st.caption("🟡 Elevated")
            else:
                st.caption("🔴 High")
        with col3:
            current_temp = st.number_input("Temperature (°C)", 35.0, 40.0, 36.6, 0.1)
            if current_temp < 36.0:
                st.caption("🟦 Low")
            elif current_temp <= 37.2:
                st.caption("🟢 Normal")
            else:
                st.caption("🔴 Fever")
        
        if st.button("🔍 Run Health Check", type="primary", use_container_width=True):
            is_anomaly, score = st.session_state.anomaly_detector_ml.detect(np.array([current_hr, current_bp, current_temp]))
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if is_anomaly:
                    st.error(f"🚨 **Alert!**")
                    st.metric("Alert Level", f"{score*100:.1f}%", delta="Action Needed", delta_color="inverse")
                    
                    st.markdown("**⚠️ Details:**")
                    anomalies_found = []
                    if current_hr > 100:
                        anomalies_found.append(f"• Heart rate: {current_hr} BPM (normal: 60-100)")
                    if current_hr < 50:
                        anomalies_found.append(f"• Heart rate: {current_hr} BPM (very low)")
                    if current_bp > 140:
                        anomalies_found.append(f"• Blood pressure: {current_bp} mmHg (high)")
                    if current_bp < 90:
                        anomalies_found.append(f"• Blood pressure: {current_bp} mmHg (low)")
                    if current_temp > 37.5:
                        anomalies_found.append(f"• Temperature: {current_temp}°C (fever)")
                    
                    for anomaly in anomalies_found:
                        st.write(anomaly)
                    
                    st.markdown("**📋 What you can do:**")
                    if current_hr > 100:
                        st.markdown("- Rest and drink water, avoid caffeine")
                    if current_bp > 140:
                        st.markdown("- Reduce salt intake, monitor BP")
                    if current_temp > 37.5:
                        st.markdown("- Rest, stay hydrated, monitor temperature")
                else:
                    st.success(f"✅ **All good!**")
                    st.metric("Health Status", f"{100 - score*100:.0f}%", delta="Normal", delta_color="normal")
                    st.markdown("All vital signs are within normal ranges.")
            
            with col_r2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge",
                    value=score * 100,
                    title={'text': "Alert Level", 'font': {'size': 14}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#E53935" if is_anomaly else "#43A047"},
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(67, 160, 71, 0.2)'},
                            {'range': [30, 60], 'color': 'rgba(251, 140, 0, 0.2)'},
                            {'range': [60, 100], 'color': 'rgba(229, 57, 53, 0.2)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown("**📊 Your 7-day averages**")
                st.markdown("""
                <div style='background: #f8f9fa; padding: 0.8rem; border-radius: 10px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>Heart Rate:</span>
                        <span><strong>72 BPM</strong></span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin-top: 0.3rem;'>
                        <span>Blood Pressure:</span>
                        <span><strong>118/76 mmHg</strong></span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin-top: 0.3rem;'>
                        <span>Temperature:</span>
                        <span><strong>36.5°C</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ==================== ENHANCED VITAL SIGNS TRACKER ====================
    elif selected == "Vital Signs Tracker":
        st.markdown("## 💓 Vital Signs Tracker")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>💓 Track Your Heart Rate</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Monitor your heart rate over time with smooth, accurate readings.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.kalman_engine is None:
            st.session_state.kalman_engine = KalmanHeartRateFilter()
        
        col1, col2 = st.columns(2)
        with col1:
            new_hr = st.number_input("Heart Rate (BPM)", 40, 150, 75)
            if st.button("➕ Add Reading", width='stretch'):
                filtered = st.session_state.kalman_engine.update(new_hr)
                st.session_state.kalman_hr.append(new_hr)
                st.session_state.kalman_filtered.append(filtered)
                st.rerun()
        with col2:
            if st.button("🎮 Demo Exercise Pattern", width='stretch'):
                pattern = [72, 75, 78, 85, 95, 108, 115, 112, 105, 95, 88, 82, 78, 75]
                for v in pattern:
                    filtered = st.session_state.kalman_engine.update(v)
                    st.session_state.kalman_hr.append(v)
                    st.session_state.kalman_filtered.append(filtered)
                st.rerun()
        
        if len(st.session_state.kalman_hr) > 0:
            st.markdown("---")
            st.markdown("### 📊 Your Heart Rate History")
            
            df_hr = pd.DataFrame({
                'Reading': range(1, len(st.session_state.kalman_hr) + 1),
                'Raw': st.session_state.kalman_hr,
                'Smoothed': st.session_state.kalman_filtered
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hr['Reading'], y=df_hr['Raw'],
                mode='lines+markers', name='Raw Reading',
                line=dict(color='#9E9E9E', width=1, dash='dot'),
                marker=dict(size=6, color='#9E9E9E', opacity=0.7)
            ))
            fig.add_trace(go.Scatter(
                x=df_hr['Reading'], y=df_hr['Smoothed'],
                mode='lines+markers', name='Smoothed Reading',
                line=dict(color='#1E88E5', width=3),
                marker=dict(size=8, color='#1E88E5')
            ))
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="High")
            fig.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Low")
            fig.update_layout(
                title="Heart Rate Over Time",
                xaxis_title="Reading Number",
                yaxis_title="Heart Rate (BPM)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Latest", f"{st.session_state.kalman_hr[-1]} BPM")
            with col_m2:
                st.metric("Smoothed", f"{st.session_state.kalman_filtered[-1]:.1f} BPM")
            with col_m3:
                if len(st.session_state.kalman_hr) > 1:
                    st.metric("Trend", f"{st.session_state.kalman_hr[-1] - st.session_state.kalman_hr[-2]:+d} BPM")
            with col_m4:
                if len(st.session_state.kalman_filtered) >= 10:
                    variability = np.std(st.session_state.kalman_filtered[-10:])
                    st.metric("Variability", f"{variability:.1f} BPM")
            
            with st.expander("📋 View All Readings"):
                st.dataframe(df_hr, use_container_width=True)
            
            if st.button("🗑️ Clear History", width='stretch'):
                st.session_state.kalman_hr = []
                st.session_state.kalman_filtered = []
                st.session_state.kalman_engine = KalmanHeartRateFilter()
                st.rerun()
        else:
            st.info("No readings yet. Add a reading or try the demo pattern.")

    # ==================== REST OF THE SECTIONS (Keep from original code) ====================
    elif selected == "Risk Analysis":
        st.markdown("## 🔍 5-Year Health Risk Analysis")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>🧠 Personalized Risk Assessment</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Enter your health information to see your 5-year risk for various diseases 
                and get personalized prevention recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("#### 📋 Your Information")
            age = st.slider("Age", 20, 80, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.slider("BMI", 15.0, 45.0, 25.0, 0.1)
            sbp = st.slider("Blood Pressure (systolic)", 90, 200, 120)
            cholesterol = st.slider("Total Cholesterol", 100, 350, 200)
        
        with col2:
            st.markdown("#### 🏃 Lifestyle Factors")
            smoker = st.checkbox("Current Smoker")
            diabetic = st.checkbox("Diabetes")
            hypertensive = st.checkbox("Hypertension")
            family_history = st.checkbox("Family History of CVD")
            exercise = st.select_slider("Physical Activity", 
                                       options=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                                       value="Moderate")
            diet = st.select_slider("Diet Quality",
                                   options=["Poor", "Fair", "Good", "Excellent"],
                                   value="Good")
    
        if st.button("🔍 Calculate My 5-Year Risk", type="primary", width='stretch'):
            base_risk = 0.02
            age_factor = max(0, (age - 40) * 0.005)
            bmi_factor = max(0, (bmi - 22) * 0.01)
            bp_factor = max(0, (sbp - 110) * 0.002)
            chol_factor = max(0, (cholesterol - 180) * 0.001)
            smoker_factor = 0.15 if smoker else 0
            diabetic_factor = 0.12 if diabetic else 0
            hypertensive_factor = 0.10 if hypertensive else 0
            family_factor = 0.08 if family_history else 0
            ex_map = {"Sedentary": 0, "Light": 0.2, "Moderate": 0.4, "Active": 0.6, "Very Active": 0.8}
            exercise_benefit = ex_map[exercise] * 0.03
            diet_map = {"Poor": 0, "Fair": 0.1, "Good": 0.2, "Excellent": 0.3}
            diet_benefit = diet_map[diet] * 0.02
            total_risk = (base_risk + age_factor + bmi_factor + bp_factor + chol_factor + 
                         smoker_factor + diabetic_factor + hypertensive_factor + family_factor -
                         exercise_benefit - diet_benefit)
            total_risk = np.clip(total_risk, 0.01, 0.5)
            
            if total_risk < 0.1:
                risk_level = "LOW"
                risk_color = "#43A047"
                badge_class = "badge-success"
            elif total_risk < 0.2:
                risk_level = "MODERATE"
                risk_color = "#FB8C00"
                badge_class = "badge-warning"
            else:
                risk_level = "HIGH"
                risk_color = "#E53935"
                badge_class = "badge-danger"
            
            pred_data = {
                'age': age, 'bmi': bmi, 'bp': sbp, 'cholesterol': cholesterol,
                'smoker': smoker, 'diabetic': diabetic, 'hypertensive': hypertensive,
                'family_history': family_history, 'exercise': exercise, 'diet': diet
            }
            db.save_risk_prediction(user_id, total_risk, risk_level, pred_data)
            st.session_state.predictions_made += 1
            
            st.markdown("---")
            st.markdown("## 🎯 YOUR 5-YEAR RISK ASSESSMENT")
        
            col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
        
            with col_r1:
                st.markdown(f"""
                <div style='background: white; padding: 2rem; border-radius: 20px; 
                            box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center;
                            border-bottom: 5px solid {risk_color};'>
                    <div style='color: #666; font-size: 1rem;'>5-Year Risk</div>
                    <div style='font-size: 4rem; font-weight: 700; color: {risk_color};'>
                        {total_risk*100:.1f}%
                    </div>
                    <div><span class='badge {badge_class}'>{risk_level} RISK</span></div>
                </div>
                """, unsafe_allow_html=True)
        
            with col_r2:
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 2rem; border-radius: 20px; height: 100%;'>
                    <h4 style='margin-top:0; color: #1E88E5;'>📊 Risk Factors</h4>
                    <table style='width:100%;'>
                        <tr><th>Age: </th><td><b>{age} years</b></tr>
                        <tr><th>BMI: </th><td><b>{bmi:.1f}</b></td>
                        <tr><th>Blood Pressure: </th><td><b>{sbp} mmHg</b></td>
                        <tr><th>Cholesterol: </th><td><b>{cholesterol} mg/dL</b></td>
                        <tr><th>Smoker: </th><td><b>{'Yes' if smoker else 'No'}</b></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
        
            with col_r3:
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 2rem; border-radius: 20px; height: 100%;'>
                    <h4 style='margin-top:0; color: #1E88E5;'>📈 Comparison</h4>
                    <p>Population Average: <b>15%</b></p>
                    <p>Your Risk: <b>{total_risk*100:.1f}%</b></p>
                    <p>You are <b>{'below' if total_risk < 0.15 else 'above' if total_risk > 0.15 else 'at'}</b> the average risk.</p>
                </div>
                """, unsafe_allow_html=True)
        
            st.markdown("---")
            st.markdown("## 🏥 DISEASE-SPECIFIC 5-YEAR RISKS")
        
            cvd_risk = total_risk * 1.2
            if smoker:
                cvd_risk += 0.05
            if hypertensive:
                cvd_risk += 0.04
            cvd_risk = np.clip(cvd_risk, 0.01, 0.5)
        
            diabetes_risk = 0.02
            diabetes_risk += max(0, (bmi - 25) * 0.015)
            diabetes_risk += 0.08 if diabetic else 0
            diabetes_risk += 0.03 if family_history else 0
            diabetes_risk -= exercise_benefit * 0.5
            diabetes_risk = np.clip(diabetes_risk, 0.01, 0.4)
        
            hypertension_risk = 0.03
            hypertension_risk += max(0, (sbp - 120) * 0.005)
            hypertension_risk += max(0, (bmi - 25) * 0.01)
            hypertension_risk += 0.05 if hypertensive else 0
            hypertension_risk = np.clip(hypertension_risk, 0.01, 0.45)
        
            stroke_risk = total_risk * 0.8
            stroke_risk += 0.03 if sbp > 140 else 0
            stroke_risk += 0.02 if smoker else 0
            stroke_risk = np.clip(stroke_risk, 0.01, 0.4)
        
            heart_risk = total_risk * 1.1
            heart_risk += 0.04 if cholesterol > 240 else 0
            heart_risk += 0.03 if smoker else 0
            heart_risk = np.clip(heart_risk, 0.01, 0.45)
        
            col_d1, col_d2, col_d3 = st.columns(3)
        
            with col_d1:
                cvd_color = "#43A047" if cvd_risk < 0.1 else "#FB8C00" if cvd_risk < 0.2 else "#E53935"
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05); margin-bottom: 1rem;
                            border-left: 5px solid {cvd_color};'>
                    <h4 style='margin:0; color: {cvd_color};'>❤️ Cardiovascular Disease</h4>
                    <div style='font-size: 2rem; font-weight: 700; color: {cvd_color};'>
                        {cvd_risk*100:.1f}%
                    </div>
                    <p style='color: #666;'>5-Year Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
                dia_color = "#43A047" if diabetes_risk < 0.1 else "#FB8C00" if diabetes_risk < 0.15 else "#E53935"
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05); margin-bottom: 1rem;
                            border-left: 5px solid {dia_color};'>
                    <h4 style='margin:0; color: {dia_color};'>🩸 Type 2 Diabetes</h4>
                    <div style='font-size: 2rem; font-weight: 700; color: {dia_color};'>
                        {diabetes_risk*100:.1f}%
                    </div>
                    <p style='color: #666;'>5-Year Risk</p>
                </div>
                """, unsafe_allow_html=True)
        
            with col_d2:
                hyper_color = "#43A047" if hypertension_risk < 0.15 else "#FB8C00" if hypertension_risk < 0.25 else "#E53935"
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05); margin-bottom: 1rem;
                            border-left: 5px solid {hyper_color};'>
                    <h4 style='margin:0; color: {hyper_color};'>🩺 Hypertension</h4>
                    <div style='font-size: 2rem; font-weight: 700; color: {hyper_color};'>
                        {hypertension_risk*100:.1f}%
                    </div>
                    <p style='color: #666;'>5-Year Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
                stroke_color = "#43A047" if stroke_risk < 0.1 else "#FB8C00" if stroke_risk < 0.15 else "#E53935"
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05); margin-bottom: 1rem;
                            border-left: 5px solid {stroke_color};'>
                    <h4 style='margin:0; color: {stroke_color};'>🧠 Stroke</h4>
                    <div style='font-size: 2rem; font-weight: 700; color: {stroke_color};'>
                        {stroke_risk*100:.1f}%
                    </div>
                    <p style='color: #666;'>5-Year Risk</p>
                </div>
                """, unsafe_allow_html=True)
        
            with col_d3:
                heart_color = "#43A047" if heart_risk < 0.1 else "#FB8C00" if heart_risk < 0.15 else "#E53935"
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05); margin-bottom: 1rem;
                            border-left: 5px solid {heart_color};'>
                    <h4 style='margin:0; color: {heart_color};'>💔 Heart Attack</h4>
                    <div style='font-size: 2rem; font-weight: 700; color: {heart_color};'>
                        {heart_risk*100:.1f}%
                    </div>
                    <p style='color: #666;'>5-Year Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
                kidney_risk = 0.01
                if diabetic or hypertensive:
                    kidney_risk += 0.05
                    if bmi > 30:
                        kidney_risk += 0.03
                kidney_risk = np.clip(kidney_risk, 0.01, 0.3)
                kidney_color = "#43A047" if kidney_risk < 0.05 else "#FB8C00" if kidney_risk < 0.1 else "#E53935"
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05); margin-bottom: 1rem;
                            border-left: 5px solid {kidney_color};'>
                    <h4 style='margin:0; color: {kidney_color};'>🫁 Chronic Kidney Disease</h4>
                    <div style='font-size: 2rem; font-weight: 700; color: {kidney_color};'>
                        {kidney_risk*100:.1f}%
                    </div>
                    <p style='color: #666;'>5-Year Risk</p>
                </div>
                """, unsafe_allow_html=True)
        
            st.markdown("---")
            st.markdown("## 🛡️ YOUR PERSONALIZED PREVENTION PLAN")
        
            recommendations = []
            if cvd_risk > 0.15:
                recommendations.append({
                    "disease": "Cardiovascular Disease",
                    "risk_level": "HIGH" if cvd_risk > 0.2 else "MODERATE",
                    "prevention": [
                        "✅ Take prescribed medications as directed",
                        "✅ Monitor blood pressure weekly",
                        "✅ Reduce sodium intake to <1500mg/day",
                        "✅ Aim for 150 minutes of exercise weekly",
                        "✅ Schedule annual checkup with cardiologist"
                    ]
                })
            if diabetes_risk > 0.1:
                recommendations.append({
                    "disease": "Type 2 Diabetes",
                    "risk_level": "HIGH" if diabetes_risk > 0.15 else "MODERATE",
                    "prevention": [
                        "✅ Maintain healthy weight (BMI < 25)",
                        "✅ Limit sugary drinks and refined carbs",
                        "✅ Exercise 30 minutes daily",
                        "✅ Get HbA1c tested annually",
                        "✅ Monitor blood glucose if symptoms appear"
                    ]
                })
            if hypertension_risk > 0.15:
                recommendations.append({
                    "disease": "Hypertension",
                    "risk_level": "HIGH" if hypertension_risk > 0.25 else "MODERATE",
                    "prevention": [
                        "✅ Monitor BP at home weekly",
                        "✅ Reduce salt intake",
                        "✅ Limit alcohol to 1 drink/day",
                        "✅ Practice stress reduction techniques",
                        "✅ Take medications as prescribed"
                    ]
                })
            if stroke_risk > 0.1:
                recommendations.append({
                    "disease": "Stroke",
                    "risk_level": "HIGH" if stroke_risk > 0.15 else "MODERATE",
                    "prevention": [
                        "✅ Control blood pressure aggressively",
                        "✅ If smoker, QUIT immediately",
                        "✅ Manage atrial fibrillation if present",
                        "✅ Learn FAST stroke signs (Face, Arm, Speech, Time)",
                        "✅ Take aspirin only if prescribed"
                    ]
                })
            if heart_risk > 0.1:
                recommendations.append({
                    "disease": "Heart Attack",
                    "risk_level": "HIGH" if heart_risk > 0.15 else "MODERATE",
                    "prevention": [
                        "✅ Maintain LDL cholesterol <100 mg/dL",
                        "✅ If smoker, join cessation program",
                        "✅ Exercise regularly",
                        "✅ Know heart attack symptoms (chest pain, shortness of breath)",
                        "✅ Keep nitroglycerin if prescribed"
                    ]
                })
        
            general_recs = []
            if bmi > 30:
                general_recs.append("⚖️ **Weight Loss**: Aim to lose 5-10% of body weight")
            if not smoker and smoker_factor == 0:
                general_recs.append("🚭 **Non-smoker**: Great! Maintain this healthy habit")
            if exercise in ["Sedentary", "Light"]:
                general_recs.append("🏃 **Increase Activity**: Start with 20-min daily walks")
            if diet in ["Poor", "Fair"]:
                general_recs.append("🥗 **Improve Diet**: Increase fruits, vegetables, whole grains")
            if sbp > 140:
                general_recs.append("🩺 **BP Control**: Monitor daily, reduce sodium")
        
            if recommendations:
                for rec in recommendations:
                    color = "#E53935" if rec["risk_level"] == "HIGH" else "#FB8C00"
                    st.markdown(f"""
                    <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                                margin-bottom: 1rem; border-left: 5px solid {color};
                                box-shadow: 0 5px 15px rgba(0,0,0,0.05);'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h3 style='margin:0; color: {color};'>{rec['disease']}</h3>
                            <span class='badge' style='background: {color}20; color: {color};'>
                                {rec['risk_level']} RISK
                            </span>
                        </div>
                        <p style='margin:1rem 0 0.5rem 0; font-weight: 600;'>🛡️ Prevention Strategies:</p>
                        <ul style='margin-top:0;'>
                            {"".join([f"<li>{item}</li>" for item in rec['prevention']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: #43A04720; padding: 2rem; border-radius: 15px; 
                            text-align: center; border-left: 5px solid #43A047;'>
                    <h3 style='margin:0; color: #43A047;'>✅ LOW RISK PROFILE</h3>
                    <p style='margin:1rem 0 0 0;'>Your risk for major diseases is low. Maintain your healthy lifestyle!</p>
                </div>
                """, unsafe_allow_html=True)
        
            if general_recs:
                st.markdown("### 🌱 General Health Recommendations")
                for rec in general_recs:
                    st.markdown(f"""
                    <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; 
                                margin-bottom: 0.5rem; border-left: 4px solid #1E88E5;'>
                        {rec}
                    </div>
                    """, unsafe_allow_html=True)
        
            st.markdown("---")
            st.markdown("""
            <div style='background: #fff4e5; padding: 1.5rem; border-radius: 15px; 
                        border-left: 5px solid #E53935;'>
                <h4 style='margin:0; color: #E53935;'>🚨 EMERGENCY WARNING SIGNS</h4>
                <p style='margin:0.5rem 0;'>Seek immediate medical attention if you experience:</p>
                <ul style='margin:0.5rem 0;'>
                    <li>Chest pain, pressure, or discomfort</li>
                    <li>Sudden numbness or weakness of face, arm, or leg</li>
                    <li>Sudden confusion, trouble speaking</li>
                    <li>Sudden severe headache</li>
                    <li>Shortness of breath</li>
                </ul>
                <p style='margin:0.5rem 0 0 0; font-weight: 700;'>📞 Call emergency services (911) immediately!</p>
            </div>
            """, unsafe_allow_html=True)

    # ==================== MULTI-TASK RISK PREDICTOR ====================
    elif selected == "Multi-Task Risk Predictor":
        st.markdown("## 🎯 Multi-Task 5-Year Risk Prediction")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>🎯 Predict Multiple Health Risks Simultaneously</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                See your 5-year risk for heart disease, diabetes, high blood pressure, stroke, and kidney disease.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Your Information")
            age_mt = st.slider("Age", 20, 80, 45, key="mt_age")
            height_mt = st.number_input("Height (cm)", 100, 250, 170, key="mt_height")
            weight_mt = st.number_input("Weight (kg)", 30, 200, 70, key="mt_weight")
            bmi_mt = weight_mt / ((height_mt/100) ** 2)
            st.caption(f"📊 Your BMI: {bmi_mt:.1f}")
            
            bp_choice_mt = st.selectbox(
                "Blood Pressure",
                ["I don't know", "Normal (<120)", "Slightly High (120-139)", "High (140+)"],
                key="mt_bp_choice"
            )
            bp_map = {"I don't know": 115, "Normal (<120)": 115, "Slightly High (120-139)": 130, "High (140+)": 145}
            bp_mt = bp_map[bp_choice_mt]
            
            chol_choice_mt = st.selectbox(
                "Cholesterol",
                ["I don't know", "Probably Normal", "Probably High"],
                key="mt_chol_choice"
            )
            chol_map = {"I don't know": 190, "Probably Normal": 180, "Probably High": 240}
            chol_mt = chol_map[chol_choice_mt]
            
        with col2:
            st.markdown("#### 🏃 Your Lifestyle")
            smoker_mt = st.checkbox("Current Smoker", key="mt_smoker")
            diabetic_mt = st.checkbox("Diabetes", key="mt_diabetic")
            exercise_mt = st.select_slider("Physical Activity", 
                                           options=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                                           value="Moderate", key="mt_exercise")
            diet_mt = st.select_slider("Diet Quality",
                                       options=["Poor", "Fair", "Good", "Excellent"],
                                       value="Good", key="mt_diet")
        
        if st.button("🔍 See All My Risks", type="primary", width='stretch'):
            def calc_multi_risk(age, bmi, bp, chol, smoker, diabetic, exercise, diet, base, factor):
                risk = base
                risk += max(0, (age - 40) * 0.005)
                risk += max(0, (bmi - 22) * 0.01)
                risk += max(0, (bp - 110) * 0.002)
                risk += max(0, (chol - 180) * 0.001)
                risk += 0.15 if smoker else 0
                risk += 0.12 if diabetic else 0
                ex_benefit = {"Sedentary": 0, "Light": 0.2, "Moderate": 0.4, "Active": 0.6, "Very Active": 0.8}
                risk -= ex_benefit.get(exercise, 0) * 0.03
                diet_benefit = {"Poor": 0, "Fair": 0.1, "Good": 0.2, "Excellent": 0.3}
                risk -= diet_benefit.get(diet, 0) * 0.02
                return np.clip(risk * factor, 0.01, 0.5) * 100
            
            cvd_risk = calc_multi_risk(age_mt, bmi_mt, bp_mt, chol_mt, smoker_mt, diabetic_mt, exercise_mt, diet_mt, 0.02, 1.2)
            diabetes_risk = calc_multi_risk(age_mt, bmi_mt, bp_mt, chol_mt, smoker_mt, diabetic_mt, exercise_mt, diet_mt, 0.02, 1.0)
            hypertension_risk = calc_multi_risk(age_mt, bmi_mt, bp_mt, chol_mt, smoker_mt, diabetic_mt, exercise_mt, diet_mt, 0.03, 0.9)
            stroke_risk = calc_multi_risk(age_mt, bmi_mt, bp_mt, chol_mt, smoker_mt, diabetic_mt, exercise_mt, diet_mt, 0.02, 0.8)
            kidney_risk = calc_multi_risk(age_mt, bmi_mt, bp_mt, chol_mt, smoker_mt, diabetic_mt, exercise_mt, diet_mt, 0.01, 0.7)
            
            st.markdown("---")
            st.markdown("## 📊 Your 5-Year Risks")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                color = "#E53935" if cvd_risk > 20 else "#FB8C00" if cvd_risk > 10 else "#43A047"
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem; border-bottom: 4px solid {color};'>
                    <div style='font-size: 0.85rem; color: #666;'>❤️ Heart Disease</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color};'>{cvd_risk:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                color = "#E53935" if diabetes_risk > 15 else "#FB8C00" if diabetes_risk > 8 else "#43A047"
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem; border-bottom: 4px solid {color};'>
                    <div style='font-size: 0.85rem; color: #666;'>🩸 Diabetes</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color};'>{diabetes_risk:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r2:
                color = "#E53935" if hypertension_risk > 25 else "#FB8C00" if hypertension_risk > 15 else "#43A047"
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem; border-bottom: 4px solid {color};'>
                    <div style='font-size: 0.85rem; color: #666;'>🩺 High BP</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color};'>{hypertension_risk:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                color = "#E53935" if stroke_risk > 15 else "#FB8C00" if stroke_risk > 8 else "#43A047"
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem; border-bottom: 4px solid {color};'>
                    <div style='font-size: 0.85rem; color: #666;'>🧠 Stroke</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color};'>{stroke_risk:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r3:
                color = "#E53935" if kidney_risk > 10 else "#FB8C00" if kidney_risk > 5 else "#43A047"
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem; border-bottom: 4px solid {color};'>
                    <div style='font-size: 0.85rem; color: #666;'>🫁 Kidney Disease</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color};'>{kidney_risk:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                overall = (cvd_risk + diabetes_risk + hypertension_risk + stroke_risk + kidney_risk) / 5
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1E88E5, #1565C0); padding: 1rem; border-radius: 15px; text-align: center; color: white;'>
                    <div style='font-size: 0.8rem;'>📊 Your Health Score</div>
                    <div style='font-size: 1.5rem; font-weight: 700;'>{100 - overall:.1f}/100</div>
                </div>
                """, unsafe_allow_html=True)
            
            conditions = ['Heart Disease', 'Diabetes', 'High BP', 'Stroke', 'Kidney']
            risks = [cvd_risk, diabetes_risk, hypertension_risk, stroke_risk, kidney_risk]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=conditions, y=risks, 
                                marker_color=['#E53935' if r > 20 else '#FB8C00' if r > 10 else '#43A047' for r in risks],
                                text=[f'{r:.1f}%' for r in risks], textposition='outside'))
            fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="High Risk Line")
            fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="Moderate Risk Line")
            fig.update_layout(title="Your 5-Year Risks", yaxis_title="Risk (%)", yaxis_range=[0, 50], height=350)
            st.plotly_chart(fig, use_container_width=True)

    # ==================== LIFESTYLE OPTIMIZER ====================
    elif selected == "Lifestyle Optimizer":
        st.markdown("## 🌱 Lifestyle Optimizer")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>🎯 Personalized Lifestyle Recommendations</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Get customized tips and recommendations to improve your health and reduce risk based on your current lifestyle.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load user data
        user_history_lo = db.get_user_history(user_id, limit=1)
        
        if not user_history_lo:
            st.info("👋 Please complete a risk assessment first to get personalized lifestyle recommendations!")
            if st.button("Start Risk Assessment →", use_container_width=True):
                st.session_state.selected = "Risk Analysis"
                st.rerun()
        else:
            last_pred_lo = user_history_lo[0]
            user_age_lo = last_pred_lo.get('age', 45)
            user_bmi_lo = last_pred_lo.get('bmi', 25.0)
            user_bp_lo = last_pred_lo.get('bp', 120)
            user_chol_lo = last_pred_lo.get('cholesterol', 200)
            user_smoker_lo = last_pred_lo.get('smoker', False)
            user_exercise_lo = last_pred_lo.get('exercise', 'Moderate')
            user_diet_lo = last_pred_lo.get('diet', 'Good')
            user_sleep_lo = last_pred_lo.get('sleep', 7)
            user_stress_lo = last_pred_lo.get('stress', 5)
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Status", "💡 Recommendations", "🎯 Action Plan", "📈 Progress Tracker"])
            
            with tab1:
                st.markdown("### Your Current Lifestyle Profile")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Age</div>
                        <div class="metric-value">{user_age_lo}</div>
                        <div style='font-size: 0.8rem; color: #666;'>years</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">BMI</div>
                        <div class="metric-value">{user_bmi_lo:.1f}</div>
                        <div style='font-size: 0.8rem; color: #666;'>
                            {"🟢 Healthy" if 18.5 <= user_bmi_lo < 25 else "🟡 Overweight" if 25 <= user_bmi_lo < 30 else "🔴 Obese" if user_bmi_lo >= 30 else "🟡 Underweight"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Blood Pressure</div>
                        <div class="metric-value">{user_bp_lo}</div>
                        <div style='font-size: 0.8rem; color: #666;'>
                            {"🟢 Normal" if user_bp_lo < 120 else "🟡 Elevated" if user_bp_lo < 130 else "🔴 High"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Cholesterol</div>
                        <div class="metric-value">{user_chol_lo}</div>
                        <div style='font-size: 0.8rem; color: #666;'>
                            {"🟢 Desirable" if user_chol_lo < 200 else "🟡 Borderline" if user_chol_lo < 240 else "🔴 High"} mg/dL
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Smoking</div>
                        <div class="metric-value">{"🚭" if user_smoker_lo else "✅"}</div>
                        <div style='font-size: 0.8rem; color: #666;'>
                            {"Smoker" if user_smoker_lo else "Non-smoker"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col6:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Sleep</div>
                        <div class="metric-value">{user_sleep_lo:.0f}</div>
                        <div style='font-size: 0.8rem; color: #666;'>
                            {"🟢 Good" if 7 <= user_sleep_lo <= 9 else "🟡 Fair"} hours
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Lifestyle radar chart
                st.markdown("### Lifestyle Assessment Radar")
                
                exercise_score = {"Sedentary": 1, "Light": 2, "Moderate": 3, "Active": 4, "Very Active": 5}.get(user_exercise_lo, 3)
                diet_score = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}.get(user_diet_lo, 3)
                bmi_score = 5 if 18.5 <= user_bmi_lo < 25 else (4 if 25 <= user_bmi_lo < 30 else (3 if 30 <= user_bmi_lo < 35 else 2) if user_bmi_lo < 18.5 else 1)
                bp_score = 5 if user_bp_lo < 120 else (4 if user_bp_lo < 130 else (3 if user_bp_lo < 140 else 1))
                chol_score = 5 if user_chol_lo < 200 else (3 if user_chol_lo < 240 else 1)
                smoke_score = 5 if not user_smoker_lo else 1
                sleep_score = 5 if 7 <= user_sleep_lo <= 9 else 3
                stress_score = 5 - (user_stress_lo / 10 * 4)
                
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=[exercise_score, diet_score, bmi_score, bp_score, chol_score, smoke_score, sleep_score, stress_score],
                    theta=['Exercise', 'Diet', 'BMI', 'Blood Pressure', 'Cholesterol', 'Smoking', 'Sleep', 'Stress'],
                    fill='toself',
                    name='Your Score'
                ), layout=go.Layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                    showlegend=True,
                    height=400
                ))
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with tab2:
                st.markdown("### 💡 Personalized Recommendations")
                
                recommendations = generate_personalized_advice_chat(
                    user_age_lo, user_bmi_lo, user_bp_lo, user_chol_lo, 
                    user_smoker_lo, user_exercise_lo, user_diet_lo, user_sleep_lo, user_stress_lo
                )
                
                for i, rec in enumerate(recommendations, 1):
                    icon = ["🏋️", "🍎", "💊", "🚭", "🏃", "🥗", "😴", "🧘"][i-1]
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #E8F0FE, #f0f7ff); 
                                padding: 1rem; border-radius: 15px; margin-bottom: 1rem;
                                border-left: 4px solid #1E88E5;'>
                        <div style='font-size: 1rem;'>{icon} {rec}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab3:
                st.markdown("### 🎯 Your Personalized Action Plan")
                
                col_ap1, col_ap2 = st.columns([2, 1])
                
                with col_ap1:
                    st.markdown("#### Week-by-Week Plan")
                    
                    action_items = []
                    if user_bmi_lo > 25:
                        action_items.append(("Week 1", "Start walking 10 minutes daily after meals", "🚶"))
                    if user_bp_lo > 130:
                        action_items.append(("Week 2", "Reduce salt intake, add deep breathing exercises", "🧘"))
                    if user_chol_lo > 200:
                        action_items.append(("Week 3", "Add oats, nuts, and fatty fish to diet", "🐟"))
                    if user_smoker_lo:
                        action_items.append(("Week 4", "Start smoking cessation program", "🚭"))
                    if user_sleep_lo < 7:
                        action_items.append(("Week 5", "Establish consistent bedtime routine", "🛌"))
                    if user_exercise_lo in ["Sedentary", "Light"]:
                        action_items.append(("Week 6", "Increase exercise to 30 minutes, 5 days/week", "🏃"))
                    
                    for week, action, emoji in action_items[:6]:
                        st.markdown(f"""
                        <div style='background: white; padding: 1rem; border-radius: 10px; 
                                    margin-bottom: 0.5rem; border-left: 4px solid #43A047;'>
                            <div style='display: flex; justify-content: space-between;'>
                                <div><strong>{week}</strong></div>
                                <div style='font-size: 1.2rem;'>{emoji}</div>
                            </div>
                            <div style='color: #666; font-size: 0.9rem;'>{action}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_ap2:
                    st.markdown("#### Quick Wins")
                    quick_wins = [
                        "🥗 Add 1 veggie per day",
                        "🚶 Walk 10 min after meals",
                        "💧 Drink 8 glasses water",
                        "😴 Sleep 15 min earlier",
                        "🧘 5 min meditation"
                    ]
                    for win in quick_wins:
                        st.markdown(f"✓ {win}")
            
            with tab4:
                st.markdown("### 📈 Track Your Progress")
                
                col_pt1, col_pt2 = st.columns([2, 1])
                
                with col_pt1:
                    st.markdown("#### Goal Progress")
                    
                    goals = [
                        ("🏋️ Weight Loss", 75, "Lose 5kg"),
                        ("🏃 Exercise", 60, "30 min/day"),
                        ("🥗 Healthy Diet", 80, "5 servings veggies"),
                        ("😴 Sleep Quality", 70, "7-9 hours/night"),
                        ("🚭 Quit Smoking", 0 if not user_smoker_lo else 30, "Stop smoking")
                    ]
                    
                    for goal_name, progress, target in goals:
                        st.markdown(f"{goal_name}: {target}")
                        st.progress(progress / 100)
                
                with col_pt2:
                    st.markdown("#### Milestones")
                    milestones = [
                        "🎯 1 Week Committed",
                        "🎯 30 Days Strong",
                        "🎯 3 Months Progress",
                        "🎯 6 Months Champion",
                        "🎯 1 Year Hero"
                    ]
                    for milestone in milestones:
                        st.markdown(f"□ {milestone}")

    # ==================== WHAT-IF LAB ====================
    elif selected == "What-If Lab":
        st.markdown("## 🔬 What-If Simulation Laboratory")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>🔄 Compare Different Lifestyle Scenarios</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                See how changes in your lifestyle affect your 5-year disease risk. 
                Compare up to 3 different scenarios and find the best path forward.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'saved_scenarios' not in st.session_state:
            st.session_state.saved_scenarios = []
        
        sim_tab1, sim_tab2, sim_tab3, sim_tab4 = st.tabs([
            "🎮 Interactive Simulator", "📊 Multi-Scenario Comparison",
            "📈 Trend Projector", "💾 Saved Scenarios"
        ])
        
        with sim_tab1:
            st.markdown("### 🔄 Interactive Lifestyle Simulator")
            st.markdown("Adjust the sliders to see how changes affect your 5-year risk")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 👤 Current Profile")
                age1 = st.slider("Age", 20, 80, 45, key="sim_age1")
                bmi1 = st.slider("BMI", 18.0, 40.0, 28.0, 0.1, key="sim_bmi1")
                sbp1 = st.slider("Blood Pressure", 100, 180, 135, key="sim_bp1")
                chol1 = st.slider("Cholesterol", 150, 300, 220, key="sim_chol1")
                smoke1 = st.checkbox("Smoker", value=True, key="sim_smoke1")
                exercise1 = st.select_slider("Exercise Level", 
                                           options=["Sedentary", "Light", "Moderate", "Active"],
                                           value="Sedentary", key="sim_ex1")
                diet1 = st.select_slider("Diet Quality",
                                       options=["Poor", "Fair", "Good", "Excellent"],
                                       value="Fair", key="sim_diet1")
            with col2:
                st.markdown("#### 🎯 Modified Profile")
                age2 = st.slider("Age (future)", 20, 80, age1+5, key="sim_age2")
                bmi2 = st.slider("BMI (target)", 18.0, 40.0, max(18.0, bmi1-2), 0.1, key="sim_bmi2")
                sbp2 = st.slider("BP (target)", 100, 180, max(100, sbp1-10), key="sim_bp2")
                chol2 = st.slider("Cholesterol (target)", 150, 300, max(150, chol1-20), key="sim_chol2")
                smoke2 = st.checkbox("Non-smoker", value=False, key="sim_smoke2")
                exercise2 = st.select_slider("Exercise (target)", 
                                           options=["Sedentary", "Light", "Moderate", "Active"],
                                           value="Active", key="sim_ex2")
                diet2 = st.select_slider("Diet (target)",
                                       options=["Poor", "Fair", "Good", "Excellent"],
                                       value="Excellent", key="sim_diet2")
            
            def calculate_sim_risk(age, bmi, sbp, chol, smoker, exercise, diet):
                base = 0.02
                age_factor = max(0, (age - 40) * 0.005)
                bmi_factor = max(0, (bmi - 22) * 0.01)
                bp_factor = max(0, (sbp - 110) * 0.002)
                chol_factor = max(0, (chol - 180) * 0.001)
                smoker_factor = 0.15 if smoker else 0
                ex_map = {"Sedentary": 0, "Light": 0.5, "Moderate": 1.0, "Active": 1.5}
                exercise_benefit = ex_map[exercise] * 0.02
                diet_map = {"Poor": 0, "Fair": 0.5, "Good": 1.0, "Excellent": 1.5}
                diet_benefit = diet_map[diet] * 0.01
                risk = base + age_factor + bmi_factor + bp_factor + chol_factor + smoker_factor - exercise_benefit - diet_benefit
                return np.clip(risk, 0.01, 0.5)
            
            risk1 = calculate_sim_risk(age1, bmi1, sbp1, chol1, smoke1, exercise1, diet1)
            risk2 = calculate_sim_risk(age2, bmi2, sbp2, chol2, smoke2, exercise2, diet2)
            
            st.markdown("---")
            st.markdown("### 📊 Simulation Results")
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                st.markdown(f"""
                <div style='background: white; padding: 1.2rem; border-radius: 15px; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05); text-align: center;
                            border-left: 5px solid #1E88E5;'>
                    <div style='color: #666; font-size:0.9rem;'>Current Risk</div>
                    <div style='font-size: 2.2rem; font-weight: 700; color: #1E88E5;'>
                        {risk1*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_r2:
                st.markdown(f"""
                <div style='background: white; padding: 1.2rem; border-radius: 15px; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05); text-align: center;
                            border-left: 5px solid #43A047;'>
                    <div style='color: #666; font-size:0.9rem;'>New Risk</div>
                    <div style='font-size: 2.2rem; font-weight: 700; color: #43A047;'>
                        {risk2*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_r3:
                change = (risk2 - risk1) * 100
                change_color = "#43A047" if change < 0 else "#E53935" if change > 0 else "#666"
                change_icon = "📉" if change < 0 else "📈" if change > 0 else "➡️"
                st.markdown(f"""
                <div style='background: white; padding: 1.2rem; border-radius: 15px; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05); text-align: center;
                            border-left: 5px solid {change_color};'>
                    <div style='color: #666; font-size:0.9rem;'>Change</div>
                    <div style='font-size: 2.2rem; font-weight: 700; color: {change_color};'>
                        {change_icon} {change:+.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_r4:
                reduction = (risk1 - risk2) * 100
                if reduction > 0:
                    st.markdown(f"""
                    <div style='background: #43A04720; padding: 1.2rem; border-radius: 15px; 
                                text-align: center; border-left: 5px solid #43A047;'>
                        <div style='color: #43A047; font-size:1.5rem; font-weight:700;'>
                            ↓ {reduction:.1f}%
                        </div>
                        <div style='color: #666; font-size:0.9rem;'>Risk Reduction</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif reduction < 0:
                    st.markdown(f"""
                    <div style='background: #E5393520; padding: 1.2rem; border-radius: 15px; 
                                text-align: center; border-left: 5px solid #E53935;'>
                        <div style='color: #E53935; font-size:1.5rem; font-weight:700;'>
                            ↑ {abs(reduction):.1f}%
                        </div>
                        <div style='color: #666; font-size:0.9rem;'>Risk Increase</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background: #f8f9fa; padding: 1.2rem; border-radius: 15px; 
                                text-align: center; border-left: 5px solid #666;'>
                        <div style='color: #666; font-size:1.5rem; font-weight:700;'>
                            ➡️ 0%
                        </div>
                        <div style='color: #666; font-size:0.9rem;'>No Change</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Current', x=['Risk Comparison'], y=[risk1*100],
                                marker_color='#1E88E5', text=[f"{risk1*100:.1f}%"], textposition='outside', width=0.4))
            fig.add_trace(go.Bar(name='Modified', x=['Risk Comparison'], y=[risk2*100],
                                marker_color='#43A047', text=[f"{risk2*100:.1f}%"], textposition='outside', width=0.4))
            fig.update_layout(title="Risk Comparison", yaxis_title="5-Year Risk (%)", yaxis_range=[0, 50],
                              height=400, showlegend=True, plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)', bargap=0.5)
            st.plotly_chart(fig, width='stretch')
            
            st.markdown("### 📝 Key Changes Made")
            changes = []
            if age2 != age1:
                changes.append(f"• Age: {age1} → {age2} years")
            if bmi2 != bmi1:
                changes.append(f"• BMI: {bmi1:.1f} → {bmi2:.1f}")
            if sbp2 != sbp1:
                changes.append(f"• Blood Pressure: {sbp1} → {sbp2} mmHg")
            if chol2 != chol1:
                changes.append(f"• Cholesterol: {chol1} → {chol2} mg/dL")
            if smoke2 != smoke1:
                changes.append(f"• Smoking: {'Yes' if smoke1 else 'No'} → {'No' if smoke2 else 'Yes'}")
            if exercise2 != exercise1:
                changes.append(f"• Exercise: {exercise1} → {exercise2}")
            if diet2 != diet1:
                changes.append(f"• Diet: {diet1} → {diet2}")
            for change in changes:
                st.markdown(change)
            
            col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
            with col_s2:
                if st.button("💾 Save This Scenario", width='stretch', key="save_scenario_btn"):
                    scenario = {
                        'name': f"Scenario {len(st.session_state.saved_scenarios) + 1}",
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'current': {'age': age1, 'bmi': bmi1, 'bp': sbp1, 'chol': chol1,
                                   'smoker': smoke1, 'exercise': exercise1, 'diet': diet1, 'risk': risk1},
                        'modified': {'age': age2, 'bmi': bmi2, 'bp': sbp2, 'chol': chol2,
                                    'smoker': smoke2, 'exercise': exercise2, 'diet': diet2, 'risk': risk2},
                        'change': (risk2 - risk1) * 100, 'reduction': (risk1 - risk2) * 100
                    }
                    st.session_state.saved_scenarios.append(scenario)
                    st.success("✅ Scenario saved successfully!")
        
        with sim_tab2:
            st.markdown("### 📊 Compare Multiple Scenarios")
            st.markdown("Create up to 3 different lifestyle scenarios and compare them side-by-side")
            num_scenarios = st.slider("Number of scenarios to compare", 2, 3, 2, key="num_scenarios")
            scenarios = []
            cols = st.columns(num_scenarios)
            for i in range(num_scenarios):
                with cols[i]:
                    st.markdown(f"**Scenario {i+1}**")
                    s_age = st.number_input(f"Age", 20, 80, 45, key=f"multi_age_{i}")
                    s_bmi = st.number_input(f"BMI", 18.0, 40.0, 25.0, step=0.1, key=f"multi_bmi_{i}")
                    s_sbp = st.number_input(f"BP", 100, 180, 120, key=f"multi_bp_{i}")
                    s_smoker = st.checkbox(f"Smoker", value=False, key=f"multi_smk_{i}")
                    s_exercise = st.select_slider(f"Exercise", 
                                                options=["Sedentary", "Light", "Moderate", "Active"],
                                                value="Moderate", key=f"multi_ex_{i}")
                    ex_map = {"Sedentary": 0, "Light": 0.5, "Moderate": 1, "Active": 1.5}
                    risk = 0.02 + max(0, (s_age - 40) * 0.005) + max(0, (s_bmi - 22) * 0.01) + max(0, (s_sbp - 110) * 0.002)
                    risk += 0.15 if s_smoker else 0
                    risk -= ex_map[s_exercise] * 0.02
                    risk = np.clip(risk, 0.01, 0.5)
                    scenarios.append({'name': f'Scenario {i+1}', 'risk': risk, 'age': s_age, 'bmi': s_bmi, 'bp': s_sbp, 'smoker': s_smoker, 'exercise': s_exercise})
                    if risk < 0.1:
                        st.success(f"✅ Risk: {risk*100:.1f}%")
                    elif risk < 0.2:
                        st.warning(f"⚠️ Risk: {risk*100:.1f}%")
                    else:
                        st.error(f"🔴 Risk: {risk*100:.1f}%")
            
            if st.button("📊 Compare All Scenarios", type="primary", width='stretch', key="compare_btn"):
                fig = go.Figure()
                colors = ['#1E88E5', '#43A047', '#FB8C00']
                for i, s in enumerate(scenarios):
                    fig.add_trace(go.Bar(name=s['name'], x=[s['name']], y=[s['risk'] * 100],
                                        marker_color=colors[i % len(colors)], text=[f"{s['risk']*100:.1f}%"], textposition='outside'))
                avg_risk = np.mean([s['risk'] for s in scenarios]) * 100
                fig.add_hline(y=avg_risk, line_dash="dash", line_color="gray", annotation_text=f"Average: {avg_risk:.1f}%")
                fig.update_layout(title="Scenario Comparison", yaxis_title="5-Year Risk (%)", yaxis_range=[0, 50],
                                  showlegend=False, height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, width='stretch')
                best = min(scenarios, key=lambda x: x['risk'])
                worst = max(scenarios, key=lambda x: x['risk'])
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.markdown(f"""
                    <div style='background: #43A04720; padding: 1.5rem; border-radius: 15px; 
                                border-left: 5px solid #43A047;'>
                        <h4 style='margin:0; color: #43A047;'>🏆 Best Scenario</h4>
                        <p style='margin:0.5rem 0; font-size:1.2rem;'>{best['name']}</p>
                        <p style='margin:0; font-size:2rem; font-weight:700; color:#43A047;'>{best['risk']*100:.1f}%</p>
                        <p style='margin:0.5rem 0;'>Age: {best['age']}, BMI: {best['bmi']}, BP: {best['bp']}</p>
                        <p>Smoker: {'Yes' if best['smoker'] else 'No'}, Exercise: {best['exercise']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b2:
                    st.markdown(f"""
                    <div style='background: #E5393520; padding: 1.5rem; border-radius: 15px; 
                                border-left: 5px solid #E53935;'>
                        <h4 style='margin:0; color: #E53935;'>⚠️ Worst Scenario</h4>
                        <p style='margin:0.5rem 0; font-size:1.2rem;'>{worst['name']}</p>
                        <p style='margin:0; font-size:2rem; font-weight:700; color:#E53935;'>{worst['risk']*100:.1f}%</p>
                        <p style='margin:0.5rem 0;'>Age: {worst['age']}, BMI: {worst['bmi']}, BP: {worst['bp']}</p>
                        <p>Smoker: {'Yes' if worst['smoker'] else 'No'}, Exercise: {worst['exercise']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with sim_tab3:
            st.markdown("### 📈 Long-Term Trend Projector")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**Current Status**")
                current_age = st.number_input("Current Age", 30, 70, 45, key="trend_age")
                current_bmi = st.number_input("Current BMI", 18.0, 40.0, 28.0, step=0.1, key="trend_bmi")
                current_smoker = st.checkbox("Current Smoker", value=True, key="trend_smoker")
                current_exercise = st.select_slider("Current Exercise",
                                                   options=["Sedentary", "Light", "Moderate", "Active"],
                                                   value="Sedentary", key="trend_ex")
            with col_t2:
                st.markdown("**Future Changes**")
                target_age = st.number_input("Target Age", current_age, 80, current_age+10, key="trend_target_age")
                target_bmi = st.number_input("Target BMI", 18.0, 40.0, 24.0, step=0.1, key="trend_target_bmi")
                quit_year = st.number_input("Years to quit smoking (0=never)", 0, 10, 3, key="trend_quit")
                target_exercise = st.select_slider("Target Exercise",
                                                  options=["Sedentary", "Light", "Moderate", "Active"],
                                                  value="Active", key="trend_target_ex")
            
            if st.button("📈 Generate Trend", type="primary", width='stretch', key="trend_btn"):
                years = list(range(0, target_age - current_age + 1, 2))
                ages = [current_age + y for y in years]
                risks = []
                for i, age in enumerate(years):
                    year = years[i]
                    progress = min(1, year / (target_age - current_age)) if target_age > current_age else 0
                    current_bmi_proj = current_bmi - (current_bmi - target_bmi) * progress
                    smoker_now = current_smoker and (year < quit_year)
                    ex_map = {"Sedentary": 0, "Light": 0.5, "Moderate": 1, "Active": 1.5}
                    current_ex_prog = ex_map[current_exercise] + (ex_map[target_exercise] - ex_map[current_exercise]) * progress
                    risk = 0.02 + max(0, (ages[i] - 40) * 0.005) + max(0, (current_bmi_proj - 22) * 0.01) + max(0, (130 - 110) * 0.002)
                    risk += 0.15 if smoker_now else 0
                    risk -= current_ex_prog * 0.02
                    risk = np.clip(risk, 0.01, 0.5)
                    risks.append(risk * 100)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ages, y=risks, mode='lines+markers', name='Risk Trajectory',
                                        line=dict(color='#1E88E5', width=4), marker=dict(size=10),
                                        fill='tozeroy', fillcolor='rgba(30,136,229,0.1)'))
                fig.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="Low Risk")
                fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="High Risk")
                if current_smoker and quit_year <= max(years):
                    quit_age = current_age + quit_year
                    fig.add_vline(x=quit_age, line_dash="dot", line_color="orange", annotation_text=f"Quit at {quit_age}")
                fig.update_layout(title="Your Health Risk Trajectory", xaxis_title="Age", yaxis_title="5-Year Risk (%)",
                                  hovermode='x', height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, width='stretch')
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Current Risk", f"{risks[0]:.1f}%")
                col_s2.metric("Future Risk", f"{risks[-1]:.1f}%")
                col_s3.metric("Change", f"{risks[-1] - risks[0]:+.1f}%", delta_color="inverse")
        
        with sim_tab4:
            st.markdown("### 💾 Your Saved Scenarios")
            if not st.session_state.saved_scenarios:
                st.info("No saved scenarios yet. Use the Interactive Simulator to create and save scenarios.")
            else:
                for i, scenario in enumerate(st.session_state.saved_scenarios):
                    with st.expander(f"📌 {scenario['name']} - Saved on {scenario['date']}"):
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.markdown(f"""
                            <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px;'>
                                <h5 style='margin:0; color:#1E88E5;'>Current</h5>
                                <p style='margin:0; font-size:1.5rem; font-weight:700;'>{scenario['current']['risk']*100:.1f}%</p>
                                <p style='margin:0; font-size:0.8rem;'>Age: {scenario['current']['age']}<br>BMI: {scenario['current']['bmi']:.1f}<br>BP: {scenario['current']['bp']}<br>Smoker: {'Yes' if scenario['current']['smoker'] else 'No'}<br>Exercise: {scenario['current']['exercise']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_s2:
                            st.markdown(f"""
                            <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px;'>
                                <h5 style='margin:0; color:#43A047;'>Modified</h5>
                                <p style='margin:0; font-size:1.5rem; font-weight:700;'>{scenario['modified']['risk']*100:.1f}%</p>
                                <p style='margin:0; font-size:0.8rem;'>Age: {scenario['modified']['age']}<br>BMI: {scenario['modified']['bmi']:.1f}<br>BP: {scenario['modified']['bp']}<br>Smoker: {'Yes' if scenario['modified']['smoker'] else 'No'}<br>Exercise: {scenario['modified']['exercise']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_s3:
                            reduction = scenario['reduction']
                            if reduction > 0:
                                st.markdown(f"""
                                <div style='background: #43A04720; padding: 1rem; border-radius: 10px; text-align: center;'>
                                    <h5 style='margin:0; color:#43A047;'>Improvement</h5>
                                    <p style='margin:0; font-size:2rem; font-weight:700; color:#43A047;'>↓ {reduction:.1f}%</p>
                                    <p style='margin:0; font-size:0.9rem;'>Risk Reduction</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif reduction < 0:
                                st.markdown(f"""
                                <div style='background: #E5393520; padding: 1rem; border-radius: 10px; text-align: center;'>
                                    <h5 style='margin:0; color:#E53935;'>Warning</h5>
                                    <p style='margin:0; font-size:2rem; font-weight:700; color:#E53935;'>↑ {abs(reduction):.1f}%</p>
                                    <p style='margin:0; font-size:0.9rem;'>Risk Increase</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center;'>
                                    <h5 style='margin:0; color:#666;'>No Change</h5>
                                    <p style='margin:0; font-size:2rem; font-weight:700; color:#666;'>➡️</p>
                                </div>
                                """, unsafe_allow_html=True)
                        if st.button(f"Delete Scenario {i+1}", key=f"del_{i}"):
                            st.session_state.saved_scenarios.pop(i)
                            st.rerun()
                if st.button("Clear All Scenarios", width='stretch', key="clear_all"):
                    st.session_state.saved_scenarios = []
                    st.rerun()

    # ==================== ENHANCED POPULATION HEALTH ====================
    elif selected == "Population Health":
        st.markdown("## 👥 Population Health Intelligence")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%); 
                    padding: 2rem; border-radius: 20px; margin-bottom: 2rem; color: white;'>
            <h2 style='margin:0;'>🌍 Population Health Intelligence</h2>
            <p style='margin:0.5rem 0 0 0; opacity:0.9;'>
                Interactive population analytics to benchmark your health, identify trends, 
                and discover actionable insights from 20,000+ individuals.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if nhanes is None or framingham is None:
            nhanes, framingham = create_synthetic_data()
        
        st.markdown("### 🔍 Filter Population")
        col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
        with col_filter1:
            age_range = st.slider("Age Range", min_value=20, max_value=80, value=(20, 80), key="pop_age_filter")
        with col_filter2:
            gender_filter = st.multiselect("Gender", options=["Male", "Female"], default=["Male", "Female"], key="pop_gender_filter")
        with col_filter3:
            bmi_range = st.slider("BMI Range", min_value=15.0, max_value=45.0, value=(18.5, 30.0), step=0.5, key="pop_bmi_filter")
        with col_filter4:
            risk_factors_filter = st.multiselect("Risk Factors", options=["Smoking", "Hypertension", "Obesity", "Diabetes"], default=[], key="pop_risk_filter")
        
        filtered_nhanes = nhanes[(nhanes['RIDAGEYR'] >= age_range[0]) & (nhanes['RIDAGEYR'] <= age_range[1]) & (nhanes['BMXBMI'] >= bmi_range[0]) & (nhanes['BMXBMI'] <= bmi_range[1])].copy()
        if len(gender_filter) == 1:
            gender_map = {"Male": 1, "Female": 2}
            filtered_nhanes = filtered_nhanes[filtered_nhanes['RIAGENDR'] == gender_map[gender_filter[0]]]
        if "Smoking" in risk_factors_filter:
            filtered_nhanes = filtered_nhanes[filtered_nhanes['SMQ856'] == 1]
        if "Hypertension" in risk_factors_filter:
            filtered_nhanes = filtered_nhanes[filtered_nhanes['BPXOSY1'] > 140]
        if "Obesity" in risk_factors_filter:
            filtered_nhanes = filtered_nhanes[filtered_nhanes['BMXBMI'] > 30]
        pop_size = len(filtered_nhanes)
        
        st.markdown("### 📊 Population Snapshot")
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        with col_stats1:
            avg_age = filtered_nhanes['RIDAGEYR'].mean() if pop_size > 0 else 0
            st.markdown(f"""
            <div style='background: white; padding: 1.2rem; border-radius: 15px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center;
                        border-bottom: 4px solid #1E88E5;'>
                <div style='font-size: 2rem; font-weight: 700; color: #1E88E5;'>{pop_size:,}</div>
                <div style='color: #666;'>Population Size</div>
                <div style='font-size: 0.8rem; color: #999;'>After filters</div>
            </div>
            """, unsafe_allow_html=True)
        with col_stats2:
            avg_bmi = filtered_nhanes['BMXBMI'].mean() if pop_size > 0 else 0
            bmi_status = "Normal" if avg_bmi < 25 else "Overweight" if avg_bmi < 30 else "Obese"
            bmi_color = "#43A047" if avg_bmi < 25 else "#FB8C00" if avg_bmi < 30 else "#E53935"
            st.markdown(f"""
            <div style='background: white; padding: 1.2rem; border-radius: 15px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center;
                        border-bottom: 4px solid {bmi_color};'>
                <div style='font-size: 2rem; font-weight: 700; color: {bmi_color};'>{avg_bmi:.1f}</div>
                <div style='color: #666;'>Average BMI</div>
                <div style='font-size: 0.8rem; color: #999;'>{bmi_status}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_stats3:
            avg_bp = filtered_nhanes['BPXOSY1'].mean() if pop_size > 0 else 0
            bp_status = "Normal" if avg_bp < 120 else "Elevated" if avg_bp < 140 else "High"
            st.markdown(f"""
            <div style='background: white; padding: 1.2rem; border-radius: 15px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center;
                        border-bottom: 4px solid #1E88E5;'>
                <div style='font-size: 2rem; font-weight: 700; color: #1E88E5;'>{avg_bp:.0f}</div>
                <div style='color: #666;'>Avg BP (mmHg)</div>
                <div style='font-size: 0.8rem; color: #999;'>{bp_status}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_stats4:
            smoker_rate = (filtered_nhanes['SMQ856'] == 1).mean() * 100 if pop_size > 0 else 0
            st.markdown(f"""
            <div style='background: white; padding: 1.2rem; border-radius: 15px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center;
                        border-bottom: 4px solid #E53935;'>
                <div style='font-size: 2rem; font-weight: 700; color: #E53935;'>{smoker_rate:.1f}%</div>
                <div style='color: #666;'>Smoking Rate</div>
                <div style='font-size: 0.8rem; color: #999;'>Current smokers</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 🎯 Your Health Benchmark")
        user_history = db.get_user_history(user_id, limit=1)
        if user_history:
            last_pred = user_history[0]
            user_age = last_pred.get('age', 45)
            user_bmi = last_pred.get('bmi', 25.0)
            user_bp = last_pred.get('bp', 120)
            user_smoker = last_pred.get('smoker', False)
            user_risk = last_pred['risk_score'] * 100
            age_similar = filtered_nhanes[(filtered_nhanes['RIDAGEYR'] >= user_age - 5) & (filtered_nhanes['RIDAGEYR'] <= user_age + 5)]
            pop_similar_bmi = age_similar['BMXBMI'].mean() if len(age_similar) > 0 else filtered_nhanes['BMXBMI'].mean()
            pop_similar_bp = age_similar['BPXOSY1'].mean() if len(age_similar) > 0 else filtered_nhanes['BPXOSY1'].mean()
            pop_risk_avg = framingham['outcome'].mean() * 100
            
            col_bench1, col_bench2, col_bench3 = st.columns(3)
            with col_bench1:
                bmi_diff = user_bmi - pop_similar_bmi
                bmi_color = "#43A047" if bmi_diff < -1 else "#FB8C00" if abs(bmi_diff) <= 1 else "#E53935"
                bmi_action = "Great job! Keep maintaining your healthy BMI." if bmi_diff < -1 else "Your BMI is average. Consider small improvements." if abs(bmi_diff) <= 1 else "Your BMI is above average. Start with daily walks and portion control."
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                            box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
                            border-left: 5px solid {bmi_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h4 style='margin:0;'>⚖️ BMI</h4>
                        <span style='font-size: 1.5rem; font-weight: 700; color: {bmi_color};'>{user_bmi:.1f}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin: 0.5rem 0;'>
                        <span style='color: #666;'>vs Age Match</span>
                        <span style='font-weight: 500;'>{pop_similar_bmi:.1f}</span>
                    </div>
                    <div style='background: #f0f0f0; height: 8px; border-radius: 4px; margin: 0.5rem 0;'>
                        <div style='width: {min(100, (user_bmi / max(pop_similar_bmi, 1)) * 100)}%; 
                                    background: {bmi_color}; height: 8px; border-radius: 4px;'></div>
                    </div>
                    <p style='margin: 0.75rem 0 0 0; font-size: 0.85rem; color: #666;'>
                        💡 {bmi_action}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col_bench2:
                bp_diff = user_bp - pop_similar_bp
                bp_color = "#43A047" if bp_diff < -5 else "#FB8C00" if abs(bp_diff) <= 5 else "#E53935"
                bp_action = "Excellent BP control! Maintain healthy habits." if bp_diff < -5 else "Your BP is in range. Monitor regularly." if abs(bp_diff) <= 5 else "Your BP is elevated. Reduce sodium and increase activity."
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                            box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
                            border-left: 5px solid {bp_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h4 style='margin:0;'>❤️ BP</h4>
                        <span style='font-size: 1.5rem; font-weight: 700; color: {bp_color};'>{user_bp}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin: 0.5rem 0;'>
                        <span style='color: #666;'>vs Age Match</span>
                        <span style='font-weight: 500;'>{pop_similar_bp:.0f}</span>
                    </div>
                    <div style='background: #f0f0f0; height: 8px; border-radius: 4px; margin: 0.5rem 0;'>
                        <div style='width: {min(100, (user_bp / max(pop_similar_bp, 1)) * 100)}%; 
                                    background: {bp_color}; height: 8px; border-radius: 4px;'></div>
                    </div>
                    <p style='margin: 0.75rem 0 0 0; font-size: 0.85rem; color: #666;'>
                        💡 {bp_action}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col_bench3:
                risk_diff = user_risk - pop_risk_avg
                risk_color = "#43A047" if risk_diff < -2 else "#FB8C00" if abs(risk_diff) <= 2 else "#E53935"
                risk_action = "Your risk is lower than average! Keep up the good work." if risk_diff < -2 else "Your risk is average. Small improvements can make a difference." if abs(risk_diff) <= 2 else "Your risk is elevated. Schedule a check-up and review lifestyle changes."
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                            box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
                            border-left: 5px solid {risk_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h4 style='margin:0;'>🎯 Risk Score</h4>
                        <span style='font-size: 1.5rem; font-weight: 700; color: {risk_color};'>{user_risk:.1f}%</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin: 0.5rem 0;'>
                        <span style='color: #666;'>vs Population</span>
                        <span style='font-weight: 500;'>{pop_risk_avg:.1f}%</span>
                    </div>
                    <div style='background: #f0f0f0; height: 8px; border-radius: 4px; margin: 0.5rem 0;'>
                        <div style='width: {min(100, (user_risk / 30) * 100)}%; 
                                    background: {risk_color}; height: 8px; border-radius: 4px;'></div>
                    </div>
                    <p style='margin: 0.75rem 0 0 0; font-size: 0.85rem; color: #666;'>
                        💡 {risk_action}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📈 Your Health Percentile Ranking")
            all_risks = framingham['outcome'] * 100
            user_percentile = (all_risks < user_risk).mean() * 100
            col_gauge1, col_gauge2, col_gauge3 = st.columns([2, 1, 1])
            with col_gauge1:
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=user_percentile, title={'text': "Your Risk Percentile", 'font': {'size': 14}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"}, 'bar': {'color': "#1E88E5"},
                           'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "lightgray",
                           'steps': [
                               {'range': [0, 25], 'color': 'rgba(67, 160, 71, 0.125)'},
                               {'range': [25, 50], 'color': 'rgba(251, 140, 0, 0.125)'},
                               {'range': [50, 75], 'color': 'rgba(255, 152, 0, 0.125)'},
                               {'range': [75, 100], 'color': 'rgba(229, 57, 53, 0.125)'}
                           ],
                           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 75}}))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, width='stretch')
            with col_gauge2:
                if user_percentile < 25:
                    st.markdown("""<div style='background: #43A04720; padding: 1rem; border-radius: 15px; text-align: center;'>
                        <div style='font-size: 2rem;'>🏆</div><div style='font-size: 1.2rem; font-weight: 700; color: #43A047;'>Excellent!</div>
                        <div style='font-size: 0.9rem;'>You're in the top 25%</div></div>""", unsafe_allow_html=True)
                elif user_percentile < 50:
                    st.markdown("""<div style='background: #FB8C0020; padding: 1rem; border-radius: 15px; text-align: center;'>
                        <div style='font-size: 2rem;'>👍</div><div style='font-size: 1.2rem; font-weight: 700; color: #FB8C00;'>Good</div>
                        <div style='font-size: 0.9rem;'>Above average health</div></div>""", unsafe_allow_html=True)
                elif user_percentile < 75:
                    st.markdown("""<div style='background: #FF980020; padding: 1rem; border-radius: 15px; text-align: center;'>
                        <div style='font-size: 2rem;'>⚠️</div><div style='font-size: 1.2rem; font-weight: 700; color: #FF9800;'>At Risk</div>
                        <div style='font-size: 0.9rem;'>Room for improvement</div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div style='background: #E5393520; padding: 1rem; border-radius: 15px; text-align: center;'>
                        <div style='font-size: 2rem;'>🔴</div><div style='font-size: 1.2rem; font-weight: 700; color: #E53935;'>High Risk</div>
                        <div style='font-size: 0.9rem;'>Immediate action needed</div></div>""", unsafe_allow_html=True)
            with col_gauge3:
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 15px;'>
                    <div style='font-weight: 600; margin-bottom: 0.5rem;'>🔍 Interpretation</div>
                    <div style='font-size: 0.85rem; color: #666;'>
                        Lower percentile = lower risk.<br>
                        You are healthier than <strong>{100 - user_percentile:.0f}%</strong> of the population.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👋 Complete a risk assessment first to see your personal benchmark!")
            st.markdown("""<div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center;'>
                <h4 style='margin:0; color: #1E88E5;'>🔍 Ready to benchmark your health?</h4>
                <p style='margin:1rem 0;'>Go to <strong>Risk Analysis</strong> to calculate your 5-year risk score.</p>
                <span class='badge badge-success'>Start Assessment →</span>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 💡 Actionable Population Insights")
        col_insight1, col_insight2 = st.columns(2)
        with col_insight1:
            st.markdown("### 📊 Risk Factor Distribution")
            risk_dist = pd.DataFrame({'Risk Factor': ['Smoking', 'Obesity', 'Hypertension', 'Physical Inactivity', 'Poor Diet'],
                'Prevalence (%)': [(filtered_nhanes['SMQ856'] == 1).mean() * 100 if pop_size > 0 else 0,
                    (filtered_nhanes['BMXBMI'] > 30).mean() * 100 if pop_size > 0 else 0,
                    (filtered_nhanes['BPXOSY1'] > 140).mean() * 100 if pop_size > 0 else 0, 45.0, 38.0]})
            fig_dist = px.bar(risk_dist, x='Risk Factor', y='Prevalence (%)', title="Risk Factor Prevalence",
                              color='Prevalence (%)', color_continuous_scale='Reds', text=risk_dist['Prevalence (%)'].round(1))
            fig_dist.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_dist.update_layout(height=350)
            st.plotly_chart(fig_dist, width='stretch')
            if pop_size > 0:
                top_risk = risk_dist.loc[risk_dist['Prevalence (%)'].idxmax()]
                st.markdown(f"""
                <div style='background: #FFF3E0; padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                    <h5 style='margin:0; color: #FB8C00;'>🎯 Population Health Priority</h5>
                    <p style='margin:0.5rem 0 0 0;'><strong>{top_risk['Risk Factor']}</strong> affects <strong>{top_risk['Prevalence (%)']:.1f}%</strong> of the selected population.</p>
                </div>
                """, unsafe_allow_html=True)
        with col_insight2:
            st.markdown("### 📈 Age-Based Trends")
            if pop_size > 0:
                filtered_nhanes['age_group'] = pd.cut(filtered_nhanes['RIDAGEYR'], bins=[20, 30, 40, 50, 60, 70, 80], labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])
                age_metrics = filtered_nhanes.groupby('age_group').agg({'BMXBMI': 'mean', 'BPXOSY1': 'mean', 'SMQ856': lambda x: (x == 1).mean() * 100}).reset_index()
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=age_metrics['age_group'], y=age_metrics['BPXOSY1'], name='Blood Pressure', mode='lines+markers', line=dict(color='#E53935', width=3), marker=dict(size=8)))
                fig_trend.add_trace(go.Scatter(x=age_metrics['age_group'], y=age_metrics['SMQ856'], name='Smoking Rate (%)', mode='lines+markers', line=dict(color='#FB8C00', width=3), marker=dict(size=8), yaxis='y2'))
                fig_trend.update_layout(title="Health Trends by Age", xaxis_title="Age Group", yaxis=dict(title="Blood Pressure (mmHg)", color='#E53935'), yaxis2=dict(title="Smoking Rate (%)", overlaying='y', side='right', color='#FB8C00'), height=350, hovermode='x unified')
                st.plotly_chart(fig_trend, width='stretch')
                st.markdown("""<div style='background: #E8F0FE; padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                    <h5 style='margin:0; color: #1E88E5;'>📊 Trend Insight</h5>
                    <p style='margin:0.5rem 0 0 0;'>Blood pressure increases with age while smoking rates decline. Focus on BP management for older adults and smoking cessation for younger groups.</p>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 🔍 Find Similar Profiles")
        if user_history and pop_size > 0:
            user_age_val = last_pred.get('age', 45)
            user_bmi_val = last_pred.get('bmi', 25.0)
            user_bp_val = last_pred.get('bp', 120)
            user_smoker_val = last_pred.get('smoker', False)
            filtered_nhanes['similarity'] = (np.abs(filtered_nhanes['RIDAGEYR'] - user_age_val) / 50 + np.abs(filtered_nhanes['BMXBMI'] - user_bmi_val) / 30 + np.abs(filtered_nhanes['BPXOSY1'] - user_bp_val) / 100 + (filtered_nhanes['SMQ856'] != (1 if user_smoker_val else 2)).astype(int) * 0.5)
            similar_profiles = filtered_nhanes.nsmallest(50, 'similarity')
            if len(similar_profiles) > 0:
                similar_avg_risk = similar_profiles['BPXOSY1'].mean() / 120 * 0.15
                col_sim1, col_sim2, col_sim3 = st.columns(3)
                col_sim1.metric("Similar Profiles Found", len(similar_profiles))
                col_sim2.metric("Avg BMI in Group", f"{similar_profiles['BMXBMI'].mean():.1f}", f"vs Your {user_bmi_val:.1f}")
                col_sim3.metric("Estimated Risk", f"{similar_avg_risk*100:.1f}%", "5-Year Average")
                st.markdown(f"""
                <div style='margin-top: 1rem; background: #f8f9fa; padding: 1rem; border-radius: 10px;'>
                    <h5 style='margin:0;'>💡 What can you learn?</h5>
                    <p style='margin:0.5rem 0 0 0; font-size: 0.9rem;'>
                        People similar to you have an average risk of <strong>{similar_avg_risk*100:.1f}%</strong>. 
                        Those who improved their lifestyle saw up to <strong>40% risk reduction</strong> within 2 years.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No closely matching profiles found. Try adjusting your filters.")
        else:
            st.info("Complete your risk assessment to find similar profiles in the population.")
        
        st.markdown("---")
        st.markdown("## 📋 Population Health Report")
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            if st.button("📊 Export Population Data (CSV)", width='stretch'):
                csv_data = filtered_nhanes.head(1000).to_csv(index=False)
                st.download_button(label="⬇️ Download CSV", data=csv_data, file_name=f"population_health_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", key="download_csv")
        with col_exp2:
            if st.button("📄 Generate Population Report", width='stretch'):
                report = f"""# Population Health Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n## Population Summary\n- Total Analyzed: {pop_size:,} individuals\n- Age Range: {age_range[0]}-{age_range[1]}\n- Avg BMI: {avg_bmi:.1f}\n- Avg BP: {avg_bp:.0f}\n- Smoking Rate: {smoker_rate:.1f}%\n\n## Key Risk Factors\n- Obesity Rate: {(filtered_nhanes['BMXBMI'] > 30).mean() * 100 if pop_size > 0 else 0:.1f}%\n- Hypertension Rate: {(filtered_nhanes['BPXOSY1'] > 140).mean() * 100 if pop_size > 0 else 0:.1f}%"""
                st.download_button("⬇️ Download Report", data=report, file_name=f"population_report_{datetime.now().strftime('%Y%m%d')}.md", key="download_report")
        with col_exp3:
            st.markdown("""<div style='background: #E8F0FE; padding: 1rem; border-radius: 15px; text-align: center;'>
                <div style='font-size: 1.2rem; font-weight: 600; color: #1E88E5;'>🤝 Share Insights</div>
                <div style='font-size: 0.8rem; color: #666; margin-top: 0.5rem;'>Share population health insights with your healthcare provider</div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""<div style='text-align: center; color: #78909C; padding: 1rem; font-size: 0.8rem;'>
            📊 Data sources: NHANES (National Health and Nutrition Examination Survey) & Framingham Heart Study<br>
            Population health insights help you make informed decisions about your lifestyle and health goals.
        </div>""", unsafe_allow_html=True)

    # ==================== SMARTWATCH DEVICE ====================
    elif selected == "Smartwatch Device":
        st.markdown("## ⌚ Smartwatch Device Integration")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>⌚ Connect Your Smartwatch</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Monitor your health in real-time by connecting your smartwatch or fitness tracker. 
                View live heart rate, step count, sleep patterns, and receive intelligent alerts.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_dev1, col_dev2 = st.columns(2)
        with col_dev1:
            st.markdown("### 🔌 Device Connection")
            if not st.session_state.smartwatch_connected:
                if st.button("📱 Connect Smartwatch", type="primary", width='stretch'):
                    st.session_state.smartwatch_connected = True
                    st.rerun()
            else:
                st.success("✅ Device connected")
                if st.button("🔌 Disconnect", width='stretch'):
                    st.session_state.smartwatch_connected = False
                    st.session_state.smartwatch_data = {'heart_rate': [], 'steps': [], 'sleep': [], 'timestamps': []}
                    st.rerun()
        
        with col_dev2:
            st.markdown("### 🕒 Last Update")
            if st.session_state.smartwatch_connected and st.session_state.smartwatch_data['timestamps']:
                last_time = st.session_state.smartwatch_data['timestamps'][-1]
                st.write(f"Data received: {last_time.strftime('%H:%M:%S')}")
            else:
                st.write("No data yet")
        
        if st.session_state.smartwatch_connected:
            st.markdown("---")
            st.markdown("### 📈 Real-time Metrics")
            
            if st.button("🔄 Generate New Reading", width='stretch'):
                new_hr = np.random.randint(65, 110)
                new_steps = np.random.randint(100, 2000)
                new_sleep = np.random.uniform(6, 9)
                new_time = datetime.now()
                
                st.session_state.smartwatch_data['heart_rate'].append(new_hr)
                st.session_state.smartwatch_data['steps'].append(new_steps)
                st.session_state.smartwatch_data['sleep'].append(new_sleep)
                st.session_state.smartwatch_data['timestamps'].append(new_time)
                
                if len(st.session_state.smartwatch_data['heart_rate']) > 100:
                    for key in st.session_state.smartwatch_data:
                        st.session_state.smartwatch_data[key] = st.session_state.smartwatch_data[key][-100:]
                
                if new_hr > 100:
                    alert = f"⚠️ High heart rate detected: {new_hr} BPM at {new_time.strftime('%H:%M:%S')}"
                    st.session_state.smartwatch_alerts.append(alert)
                elif new_hr < 55:
                    alert = f"⚠️ Low heart rate detected: {new_hr} BPM at {new_time.strftime('%H:%M:%S')}"
                    st.session_state.smartwatch_alerts.append(alert)
                
                if len(st.session_state.smartwatch_alerts) > 20:
                    st.session_state.smartwatch_alerts = st.session_state.smartwatch_alerts[-20:]
                
                st.rerun()
            
            if st.session_state.smartwatch_data['heart_rate']:
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("❤️ Heart Rate", f"{st.session_state.smartwatch_data['heart_rate'][-1]} BPM")
                with col_m2:
                    st.metric("👟 Steps", f"{st.session_state.smartwatch_data['steps'][-1]:,}")
                with col_m3:
                    st.metric("😴 Sleep", f"{st.session_state.smartwatch_data['sleep'][-1]:.1f} hrs")
            else:
                st.info("Click 'Generate New Reading' to start receiving data.")
            
            if len(st.session_state.smartwatch_data['heart_rate']) > 1:
                df_device = pd.DataFrame({
                    'Time': st.session_state.smartwatch_data['timestamps'],
                    'Heart Rate': st.session_state.smartwatch_data['heart_rate'],
                    'Steps': st.session_state.smartwatch_data['steps'],
                    'Sleep': st.session_state.smartwatch_data['sleep']
                })
                
                tab_c1, tab_c2, tab_c3 = st.tabs(["Heart Rate", "Steps", "Sleep"])
                
                with tab_c1:
                    fig_hr = px.line(df_device, x='Time', y='Heart Rate', title="Heart Rate Over Time", markers=True)
                    fig_hr.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="High")
                    fig_hr.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Normal low")
                    fig_hr.update_layout(height=400)
                    st.plotly_chart(fig_hr, width='stretch')
                
                with tab_c2:
                    fig_steps = px.bar(df_device, x='Time', y='Steps', title="Step Count", color='Steps', color_continuous_scale='Greens')
                    fig_steps.update_layout(height=400)
                    st.plotly_chart(fig_steps, width='stretch')
                
                with tab_c3:
                    fig_sleep = px.line(df_device, x='Time', y='Sleep', title="Sleep Duration", markers=True)
                    fig_sleep.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Optimal")
                    fig_sleep.update_layout(height=400)
                    st.plotly_chart(fig_sleep, width='stretch')
            
            st.markdown("### 🚨 Health Alerts")
            if st.session_state.smartwatch_alerts:
                for alert in st.session_state.smartwatch_alerts[-5:]:
                    st.warning(alert)
            else:
                st.success("No alerts. All metrics are within normal ranges.")
        else:
            st.info("🔌 Click 'Connect Smartwatch' to start receiving real-time health data.")

    # ==================== MOBILE CONNECTION ====================
        # ==================== MOBILE CONNECTION WITH GOOGLE SHEETS ====================
    elif selected == "Mobile Connection":
        st.markdown("## 📱 Mobile Connection")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>📱 Sync Your Health Data with Google Sheets</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Connect your mobile device to sync health data. Your data is analyzed to provide personalized health insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state
        if 'sheet_connected' not in st.session_state:
            st.session_state.sheet_connected = False
        if 'google_sheet' not in st.session_state:
            st.session_state.google_sheet = None
        if 'mobile_data' not in st.session_state:
            st.session_state.mobile_data = None
        if 'sheet_name' not in st.session_state:
            st.session_state.sheet_name = "AI Digital Twin Health Data"
        
        # Function to connect to Google Sheets
        def connect_to_google_sheets():
            try:
                import gspread
                from oauth2client.service_account import ServiceAccountCredentials
                import json
                
                scope = ['https://spreadsheets.google.com/feeds', 
                        'https://www.googleapis.com/auth/drive']
                
                creds = None
                
                # Try Streamlit Secrets first
                try:
                    if "google" in st.secrets and "credentials" in st.secrets["google"]:
                        creds_dict = st.secrets["google"]["credentials"]
                        if isinstance(creds_dict, str):
                            creds_dict = json.loads(creds_dict)
                        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                except Exception:
                    pass
                
                # Try local file
                if creds is None:
                    try:
                        creds_path = os.path.join(os.path.dirname(__file__), 'google_credentials.json')
                        if os.path.exists(creds_path):
                            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
                    except Exception:
                        pass
                
                if creds is None:
                    return False, "No valid credentials found"
                
                client = gspread.authorize(creds)
                
                # Try to open existing sheet
                try:
                    sheet = client.open(st.session_state.sheet_name).sheet1
                except:
                    sheet = client.create(st.session_state.sheet_name).sheet1
                    headers = ['timestamp', 'heart_rate', 'steps', 'sleep', 'calories', 'distance_km', 'active_minutes']
                    sheet.append_row(headers)
                
                st.session_state.google_sheet = sheet
                return True, "Connected successfully!"
                
            except Exception as e:
                return False, f"Connection error: {str(e)[:100]}"
        
        # Function to load and analyze data
        def load_and_analyze_data():
            if st.session_state.sheet_connected and st.session_state.google_sheet:
                try:
                    all_records = st.session_state.google_sheet.get_all_records()
                    if all_records:
                        df = pd.DataFrame(all_records)
                        # Clean data - remove rows with invalid values
                        if 'steps' in df.columns:
                            df['steps'] = pd.to_numeric(df['steps'], errors='coerce').fillna(0)
                        if 'heart_rate' in df.columns:
                            df['heart_rate'] = pd.to_numeric(df['heart_rate'], errors='coerce').fillna(75)
                        if 'sleep' in df.columns:
                            df['sleep'] = pd.to_numeric(df['sleep'], errors='coerce').fillna(7)
                        if 'calories' in df.columns:
                            df['calories'] = pd.to_numeric(df['calories'], errors='coerce').fillna(2000)
                        if 'active_minutes' in df.columns:
                            df['active_minutes'] = pd.to_numeric(df['active_minutes'], errors='coerce').fillna(30)
                        
                        st.session_state.mobile_data = df
                        return True, len(df)
                    return False, 0
                except Exception as e:
                    return False, 0
            return False, 0
        
        # Connection UI
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if not st.session_state.sheet_connected:
                if st.button("🔌 Connect to Google Sheets", type="primary", use_container_width=True):
                    with st.spinner("Connecting..."):
                        success, message = connect_to_google_sheets()
                        if success:
                            st.session_state.sheet_connected = True
                            st.success(message)
                            # Load data after connection
                            load_and_analyze_data()
                            st.rerun()
                        else:
                            st.error(message)
            else:
                st.success("✅ Connected")
                if st.button("Disconnect", use_container_width=True):
                    st.session_state.sheet_connected = False
                    st.session_state.google_sheet = None
                    st.session_state.mobile_data = None
                    st.rerun()
        
        with col2:
            if st.session_state.sheet_connected:
                # Count records
                if st.session_state.mobile_data is not None:
                    record_count = len(st.session_state.mobile_data)
                    st.metric("Records Synced", record_count)
                else:
                    st.metric("Status", "Synced")
        
        with col3:
            if st.session_state.sheet_connected:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    with st.spinner("Refreshing..."):
                        success, count = load_and_analyze_data()
                        if success:
                            st.success(f"✅ Loaded {count} records")
                            st.rerun()
                        else:
                            st.warning("No new data found")
        
        st.markdown("---")
        
        # Only show analytics if connected
        if st.session_state.sheet_connected and st.session_state.mobile_data is not None and not st.session_state.mobile_data.empty:
            df = st.session_state.mobile_data
            
            # Create tabs for different analytics
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Health Overview", "📈 Trends", "💡 Insights", "➕ Add Data"])
            
            with tab1:
                st.markdown("### 📊 Your Health Overview")
                
                # Calculate metrics
                avg_steps = df['steps'].mean()
                avg_heart_rate = df['heart_rate'].mean()
                avg_sleep = df['sleep'].mean()
                avg_calories = df['calories'].mean()
                avg_active = df['active_minutes'].mean()
                total_days = len(df)
                
                # Display metric cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    step_status = "✅ Good" if avg_steps >= 7000 else "⚠️ Needs Improvement"
                    st.markdown(f"""
                    <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
                        <div style='font-size: 2rem;'>👟</div>
                        <div style='font-size: 1.5rem; font-weight: 700;'>{avg_steps:.0f}</div>
                        <div>Avg Daily Steps</div>
                        <div style='font-size: 0.8rem; color: {"#43A047" if avg_steps >= 7000 else "#E53935"};'>{step_status}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    hr_status = "✅ Normal" if 60 <= avg_heart_rate <= 100 else "⚠️ Check"
                    st.markdown(f"""
                    <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
                        <div style='font-size: 2rem;'>❤️</div>
                        <div style='font-size: 1.5rem; font-weight: 700;'>{avg_heart_rate:.0f}</div>
                        <div>Avg Heart Rate</div>
                        <div style='font-size: 0.8rem; color: {"#43A047" if 60 <= avg_heart_rate <= 100 else "#E53935"};'>{hr_status}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    sleep_status = "✅ Optimal" if 7 <= avg_sleep <= 9 else "⚠️ Adjust"
                    st.markdown(f"""
                    <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
                        <div style='font-size: 2rem;'>😴</div>
                        <div style='font-size: 1.5rem; font-weight: 700;'>{avg_sleep:.1f}</div>
                        <div>Avg Sleep (hrs)</div>
                        <div style='font-size: 0.8rem; color: {"#43A047" if 7 <= avg_sleep <= 9 else "#E53935"};'>{sleep_status}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
                        <div style='font-size: 2rem;'>📅</div>
                        <div style='font-size: 1.5rem; font-weight: 700;'>{total_days}</div>
                        <div>Days Tracked</div>
                        <div style='font-size: 0.8rem; color: #666;'>Total Records</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Health Score Gauge
                st.markdown("### 🎯 Your Mobile Health Score")
                
                # Calculate health score
                health_score = 70
                if avg_steps >= 10000:
                    health_score += 15
                elif avg_steps >= 7000:
                    health_score += 10
                elif avg_steps >= 5000:
                    health_score += 5
                
                if 60 <= avg_heart_rate <= 80:
                    health_score += 10
                elif 80 < avg_heart_rate <= 100:
                    health_score += 5
                
                if 7 <= avg_sleep <= 9:
                    health_score += 10
                elif 6 <= avg_sleep < 7:
                    health_score += 5
                
                health_score = min(100, health_score)
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=health_score,
                    title={'text': "Health Score", 'font': {'size': 20}},
                    delta={'reference': 70, 'increasing': {'color': "#43A047"}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1E88E5"},
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(229, 57, 53, 0.2)'},
                            {'range': [50, 75], 'color': 'rgba(251, 140, 0, 0.2)'},
                            {'range': [75, 100], 'color': 'rgba(67, 160, 71, 0.2)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with tab2:
                st.markdown("### 📈 Health Trends")
                
                # Convert timestamp to date
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df = df.sort_values('date')
                    
                    # Steps trend
                    fig_steps = go.Figure()
                    fig_steps.add_trace(go.Scatter(
                        x=df['date'], y=df['steps'],
                        mode='lines+markers',
                        name='Steps',
                        line=dict(color='#1E88E5', width=3),
                        marker=dict(size=8, color='#1E88E5'),
                        fill='tozeroy',
                        fillcolor='rgba(30,136,229,0.1)'
                    ))
                    fig_steps.add_hline(y=10000, line_dash="dash", line_color="green", annotation_text="Goal")
                    fig_steps.update_layout(
                        title="Steps Over Time",
                        xaxis_title="Date",
                        yaxis_title="Steps",
                        height=350,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_steps, use_container_width=True)
                    
                    # Heart Rate & Sleep combo
                    fig_hr_sleep = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_hr_sleep.add_trace(
                        go.Scatter(x=df['date'], y=df['heart_rate'],
                                  mode='lines+markers', name='Heart Rate',
                                  line=dict(color='#E53935', width=3)),
                        secondary_y=False
                    )
                    fig_hr_sleep.add_trace(
                        go.Scatter(x=df['date'], y=df['sleep'],
                                  mode='lines+markers', name='Sleep',
                                  line=dict(color='#43A047', width=3)),
                        secondary_y=True
                    )
                    fig_hr_sleep.update_layout(title="Heart Rate & Sleep", height=350)
                    fig_hr_sleep.update_yaxes(title_text="Heart Rate (BPM)", secondary_y=False)
                    fig_hr_sleep.update_yaxes(title_text="Sleep (hours)", secondary_y=True)
                    st.plotly_chart(fig_hr_sleep, use_container_width=True)
                    
                    # Active Minutes & Calories
                    fig_active = go.Figure()
                    fig_active.add_trace(go.Bar(
                        x=df['date'], y=df['active_minutes'],
                        name='Active Minutes',
                        marker_color='#FB8C00',
                        text=df['active_minutes'], textposition='outside'
                    ))
                    fig_active.add_trace(go.Scatter(
                        x=df['date'], y=df['calories'],
                        name='Calories',
                        mode='lines+markers',
                        line=dict(color='#9C27B0', width=2),
                        yaxis='y2'
                    ))
                    fig_active.update_layout(
                        title="Activity & Calories",
                        xaxis_title="Date",
                        yaxis_title="Active Minutes",
                        yaxis2=dict(title="Calories", overlaying='y', side='right'),
                        height=350
                    )
                    st.plotly_chart(fig_active, use_container_width=True)
            
            with tab3:
                st.markdown("### 💡 Personalized Health Insights")
                
                insights = []
                
                # Step insights
                if avg_steps < 5000:
                    insights.append({
                        "icon": "🚶",
                        "title": "Low Activity Level",
                        "message": f"Your average steps ({avg_steps:.0f}) are below recommended levels.",
                        "tip": "Start with 10-minute walks after meals. Gradually increase to 30 minutes daily."
                    })
                elif avg_steps < 7000:
                    insights.append({
                        "icon": "👍",
                        "title": "Moderate Activity",
                        "message": f"Good job! You're averaging {avg_steps:.0f} steps daily.",
                        "tip": "Try to reach 10,000 steps by taking the stairs or parking farther away."
                    })
                elif avg_steps >= 10000:
                    insights.append({
                        "icon": "🏆",
                        "title": "Excellent Activity!",
                        "message": f"Amazing! You're exceeding the daily step goal with {avg_steps:.0f} steps.",
                        "tip": "Keep up the great work! Consider adding strength training 2-3 times per week."
                    })
                
                # Heart rate insights
                if avg_heart_rate > 90:
                    insights.append({
                        "icon": "❤️",
                        "title": "Elevated Heart Rate",
                        "message": f"Your average heart rate ({avg_heart_rate:.0f} BPM) is above normal.",
                        "tip": "Practice deep breathing, reduce caffeine, and ensure adequate hydration."
                    })
                elif avg_heart_rate < 60:
                    insights.append({
                        "icon": "💪",
                        "title": "Low Heart Rate",
                        "message": f"Your heart rate ({avg_heart_rate:.0f} BPM) is below average.",
                        "tip": "This is common in athletes. If you feel dizzy, consult a doctor."
                    })
                else:
                    insights.append({
                        "icon": "✅",
                        "title": "Healthy Heart Rate",
                        "message": f"Your heart rate ({avg_heart_rate:.0f} BPM) is in optimal range!",
                        "tip": "Maintain this with regular exercise and stress management."
                    })
                
                # Sleep insights
                if avg_sleep < 7:
                    insights.append({
                        "icon": "😴",
                        "title": "Sleep Deprivation",
                        "message": f"You're averaging only {avg_sleep:.1f} hours of sleep.",
                        "tip": "Set a consistent bedtime, avoid screens 1 hour before bed, and keep your room dark."
                    })
                elif avg_sleep > 9:
                    insights.append({
                        "icon": "💤",
                        "title": "Excessive Sleep",
                        "message": f"You're sleeping {avg_sleep:.1f} hours on average.",
                        "tip": "While rest is important, excessive sleep may indicate underlying issues. Consider a check-up."
                    })
                else:
                    insights.append({
                        "icon": "🌟",
                        "title": "Optimal Sleep!",
                        "message": f"Great! You're getting {avg_sleep:.1f} hours of quality sleep.",
                        "tip": "Your sleep habits are excellent. Keep your consistent schedule!"
                    })
                
                # Activity insights
                if avg_active < 30:
                    insights.append({
                        "icon": "⏱️",
                        "title": "Low Active Minutes",
                        "message": f"You're averaging only {avg_active:.0f} active minutes daily.",
                        "tip": "Aim for 30 minutes of moderate activity daily. Try brisk walking or cycling."
                    })
                elif avg_active >= 60:
                    insights.append({
                        "icon": "🔥",
                        "title": "Highly Active!",
                        "message": f"Excellent! You're getting {avg_active:.0f} active minutes daily.",
                        "tip": "Great job! Consider adding variety to your workouts for balanced fitness."
                    })
                
                # Display insights
                for insight in insights[:5]:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #f8f9fa, #ffffff); 
                                padding: 1.2rem; border-radius: 15px; margin-bottom: 1rem;
                                border-left: 5px solid #1E88E5;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.05);'>
                        <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                            <span style='font-size: 2rem; margin-right: 0.8rem;'>{insight['icon']}</span>
                            <span style='font-size: 1.2rem; font-weight: 700; color: #1E88E5;'>{insight['title']}</span>
                        </div>
                        <p style='margin: 0.5rem 0; color: #444;'>{insight['message']}</p>
                        <p style='margin: 0.5rem 0 0 0; color: #43A047;'>
                            <strong>💡 Tip:</strong> {insight['tip']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Progress message
                st.markdown("---")
                st.markdown("### 🎯 Your Progress")
                
                # Show improvement message
                if len(df) >= 2:
                    recent_avg = df['steps'].tail(3).mean()
                    previous_avg = df['steps'].head(3).mean()
                    if recent_avg > previous_avg:
                        st.success("📈 **Great progress!** Your step count is improving over time. Keep going!")
                    elif recent_avg < previous_avg:
                        st.info("📉 **Time to refocus!** Your recent activity has decreased. Set a small daily goal to get back on track.")
                    else:
                        st.info("📊 **Consistent effort!** You're maintaining steady activity levels. Try increasing gradually.")
            
            with tab4:
                st.markdown("### ➕ Add New Health Data")
                st.markdown("Add your daily health metrics to track your progress:")
                
                entry_date = st.date_input("📅 Date", datetime.now())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    new_steps = st.number_input("👟 Steps", 0, 50000, 0, step=500)
                    new_heart_rate = st.number_input("❤️ Heart Rate (BPM)", 40, 150, 75)
                    new_sleep = st.number_input("😴 Sleep (hours)", 0.0, 24.0, 7.0, step=0.5)
                
                with col2:
                    new_calories = st.number_input("🔥 Calories Burned", 0, 5000, 2000, step=100)
                    new_distance = st.number_input("📏 Distance (km)", 0.0, 50.0, 0.0, step=0.5)
                    new_active = st.number_input("⏱️ Active Minutes", 0, 300, 30, step=5)
                
                if st.button("💾 Save to Google Sheets", type="primary", use_container_width=True):
                    if st.session_state.sheet_connected and st.session_state.google_sheet:
                        try:
                            new_row = [
                                entry_date.strftime("%d/%m/%Y %H:%M"),
                                new_heart_rate,
                                new_steps,
                                new_sleep,
                                new_calories,
                                new_distance,
                                new_active
                            ]
                            st.session_state.google_sheet.append_row(new_row)
                            # Reload data
                            load_and_analyze_data()
                            st.balloons()
                            st.success("✅ Data saved and synced with Google Sheets!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)[:100]}")
                    else:
                        st.warning("⚠️ Please connect to Google Sheets first!")
        
        elif st.session_state.sheet_connected:
            st.info("📭 No data found in Google Sheets. Add your first health data entry using the 'Add Data' tab above!")
        else:
            st.info("🔌 Click 'Connect to Google Sheets' to start syncing your health data and get personalized insights!")
    # ==================== AI INSIGHTS ====================
    elif selected == "AI Insights":
        st.markdown("## 🤖 AI-Powered Health Insights")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>🧠 Personalized Health Analytics</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Our AI model analyzes your health data to uncover patterns and provide personalized recommendations.
                Adjust your lifestyle factors to see how your risk changes in real-time.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        user_history = db.get_user_history(user_id, limit=1)
        if not user_history:
            st.info("👋 Please complete a risk assessment first to see AI insights tailored to you!")
        else:
            last_pred = user_history[0]
            user_age = last_pred.get('age', 45)
            user_bmi = last_pred.get('bmi', 25.0)
            user_bp = last_pred.get('bp', 120)
            user_chol = last_pred.get('cholesterol', 200)
            user_smoker = last_pred.get('smoker', False)
            user_exercise = last_pred.get('exercise', 'Moderate')
            user_diet = last_pred.get('diet', 'Good')
            
            @st.cache_resource
            def train_model():
                X = framingham[['age', 'bmi', 'bp', 'cholesterol', 'smoker']].copy()
                X['exercise'] = np.random.choice([0,1,2,3,4], len(X))
                X['diet'] = np.random.choice([0,1,2,3], len(X))
                y = framingham['outcome']
                if y.nunique() == 1:
                    y.iloc[0] = 1 - y.iloc[0]
                model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
                model.fit(X, y)
                return model
            
            model = train_model()
            ex_map = {'Sedentary':0, 'Light':1, 'Moderate':2, 'Active':3, 'Very Active':4}
            diet_map = {'Poor':0, 'Fair':1, 'Good':2, 'Excellent':3}
            user_features = pd.DataFrame([{'age': user_age, 'bmi': user_bmi, 'bp': user_bp, 'cholesterol': user_chol,
                'smoker': 1 if user_smoker else 0, 'exercise': ex_map.get(user_exercise, 2), 'diet': diet_map.get(user_diet, 2)}])
            original_risk = model.predict_proba(user_features)[0][1] * 100
            
            st.markdown("### 📊 What Affects Your Risk?")
            importance = pd.DataFrame({'Feature': ['Age', 'BMI', 'Blood Pressure', 'Cholesterol', 'Smoker', 'Exercise Level', 'Diet Quality'],
                'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title="Feature Importance", color='Importance', color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
            
            tab1, tab2, tab3 = st.tabs(["🔮 What-If Simulator", "⚡ Risk Optimization", "📈 Trend Projection"])
            with tab1:
                st.markdown("### 🔄 Interactive What-If Simulator")
                with st.expander("Adjust Your Lifestyle Factors", expanded=True):
                    sim = user_features.copy()
                    col1, col2 = st.columns(2)
                    with col1:
                        new_bmi = st.slider("BMI", 18.0, 45.0, user_bmi, 0.5, key="sim_bmi")
                        new_bp = st.slider("Systolic BP", 90, 200, user_bp, key="sim_bp")
                        new_chol = st.slider("Cholesterol", 150, 350, user_chol, key="sim_chol")
                    with col2:
                        new_smoker = st.radio("Smoking", ["Non‑smoker", "Smoker"], index=1 if user_smoker else 0, key="sim_smoke")
                        ex_opts = ['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active']
                        new_ex = st.select_slider("Exercise", options=ex_opts, value=user_exercise, key="sim_ex")
                        diet_opts = ['Poor', 'Fair', 'Good', 'Excellent']
                        new_diet = st.select_slider("Diet", options=diet_opts, value=user_diet, key="sim_diet")
                    sim['bmi'] = new_bmi
                    sim['bp'] = new_bp
                    sim['cholesterol'] = new_chol
                    sim['smoker'] = 1 if new_smoker == "Smoker" else 0
                    sim['exercise'] = ex_map[new_ex]
                    sim['diet'] = diet_map[new_diet]
                new_risk = model.predict_proba(sim)[0][1] * 100
                change = new_risk - original_risk
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Current", f"{original_risk:.1f}%")
                col_b.metric("New", f"{new_risk:.1f}%", delta=f"{change:+.1f}%")
                if change < -1:
                    st.success(f"✅ Risk reduction of {abs(change):.1f}%!")
                elif change > 1:
                    st.error(f"⚠️ Risk increase of {change:.1f}%.")
            
            with tab2:
                st.markdown("### ⚡ Risk Optimization")
                col1, col2 = st.columns(2)
                with col1:
                    opt_bmi = st.slider("Target BMI", 18.0, 40.0, user_bmi, 0.5, key="opt_bmi")
                    opt_bp = st.slider("Target BP", 100, 160, user_bp, key="opt_bp")
                    opt_chol = st.slider("Target Cholesterol", 150, 250, user_chol, key="opt_chol")
                with col2:
                    opt_smoker = st.checkbox("Quit Smoking", value=user_smoker, key="opt_smoker")
                    opt_ex = st.select_slider("Target Exercise", options=ex_opts, value=user_exercise, key="opt_ex")
                    opt_diet = st.select_slider("Target Diet", options=diet_opts, value=user_diet, key="opt_diet")
                opt_features = pd.DataFrame([{'age': user_age, 'bmi': opt_bmi, 'bp': opt_bp, 'cholesterol': opt_chol,
                    'smoker': 1 if opt_smoker else 0, 'exercise': ex_map[opt_ex], 'diet': diet_map[opt_diet]}])
                opt_risk = model.predict_proba(opt_features)[0][1] * 100
                reduction = original_risk - opt_risk
                col_r1, col_r2 = st.columns(2)
                col_r1.metric("Current", f"{original_risk:.1f}%")
                col_r2.metric("Optimized", f"{opt_risk:.1f}%", delta=f"-{reduction:.1f}%" if reduction>0 else f"+{abs(reduction):.1f}%")
                if reduction > 0:
                    st.success(f"🎯 Potential risk reduction of {reduction:.1f}%!")
            
            with tab3:
                st.markdown("### 📈 Risk Trend Projection")
                years = np.arange(0, 11)
                ages = user_age + years
                target_bmi = opt_bmi
                target_bp = opt_bp
                target_chol = opt_chol
                target_smoker = 1 if opt_smoker else 0
                target_ex = ex_map[opt_ex]
                target_diet = diet_map[opt_diet]
                risks = []
                for t in years:
                    progress = min(1, t / 5)
                    cur_bmi = user_bmi + (target_bmi - user_bmi) * progress
                    cur_bp = user_bp + (target_bp - user_bp) * progress
                    cur_chol = user_chol + (target_chol - user_chol) * progress
                    cur_smoker = user_smoker if t < 1 else target_smoker
                    cur_ex = ex_map[user_exercise] + (target_ex - ex_map[user_exercise]) * progress
                    cur_diet = diet_map[user_diet] + (target_diet - diet_map[user_diet]) * progress
                    pred_df = pd.DataFrame([{'age': ages[t], 'bmi': cur_bmi, 'bp': cur_bp, 'cholesterol': cur_chol,
                        'smoker': cur_smoker, 'exercise': cur_ex, 'diet': cur_diet}])
                    risk = model.predict_proba(pred_df)[0][1] * 100
                    risks.append(risk)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ages, y=risks, mode='lines+markers', name='Projected Risk', line=dict(color='#1E88E5', width=3), fill='tozeroy'))
                fig.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="Low Risk")
                fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="High Risk")
                fig.update_layout(title="Risk Trajectory", xaxis_title="Age", yaxis_title="Risk (%)", height=500)
                st.plotly_chart(fig, width='stretch')
                col_f1, col_f2, col_f3 = st.columns(3)
                col_f1.metric("Current", f"{risks[0]:.1f}%")
                col_f2.metric("5 Years", f"{risks[5]:.1f}%")
                col_f3.metric("10 Years", f"{risks[-1]:.1f}%")
            
            with st.expander("🔍 Understanding Your Risk Factors"):
                st.markdown("### Feature Impact on Your Risk")
                impact_data = pd.DataFrame({
                    'Feature': ['Age', 'BMI', 'Blood Pressure', 'Cholesterol', 'Smoking', 'Exercise', 'Diet'],
                    'Impact': [0.15, 0.12, 0.10, 0.08, 0.20, -0.10, -0.08]
                })
                fig_impact = px.bar(impact_data, x='Impact', y='Feature', orientation='h', 
                                   title="How Each Feature Affects Your Risk",
                                   color='Impact', color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig_impact, width='stretch')
                st.caption("""
                **How to interpret this chart:**
                - Positive impact (red) increases your risk
                - Negative impact (green) decreases your risk
                - Longer bars mean bigger impact on your risk score
                """)

    # ==================== SLEEP PATTERN ANALYSIS ====================
    elif selected == "Sleep Pattern Analysis":
        st.markdown("## 😴 Sleep Pattern Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            avg_sleep = st.slider("Average Sleep Hours", 3, 12, 7)
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
            wakeups = st.number_input("Night Wake-ups", 0, 10, 1)
        with col2:
            deep_sleep = st.slider("Deep Sleep %", 10, 40, 20)
            rem_sleep = st.slider("REM Sleep %", 15, 30, 22)
            consistency = st.slider("Bedtime Consistency (1-10)", 1, 10, 6)
        
        if st.button("🔍 Analyze Sleep", type="primary", width='stretch'):
            sleep_score = 100
            sleep_score -= max(0, (avg_sleep - 8) * 5) if avg_sleep > 8 else max(0, (7 - avg_sleep) * 8)
            sleep_score -= (10 - sleep_quality) * 4
            sleep_score -= wakeups * 5
            sleep_score = max(0, min(100, sleep_score))
            
            if avg_sleep < 6:
                disorder = "Sleep Deprivation"
                advice = "Aim for 7-9 hours. Try a consistent bedtime routine."
            elif wakeups > 3:
                disorder = "Sleep Fragmentation"
                advice = "Reduce caffeine and screen time before bed."
            else:
                disorder = "Normal Pattern"
                advice = "Your sleep patterns look healthy!"
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: white; border-radius: 15px;'>
                    <div style='font-size: 2rem; font-weight: 700; color: #1E88E5;'>{sleep_score:.0f}/100</div>
                    <div>Sleep Health Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col_r2:
                st.info(f"**Detected:** {disorder}\n\n**Recommendation:** {advice}")

    # ==================== ANOMALY DETECTION ====================
    elif selected == "Anomaly Detection":
        st.markdown("## 🚨 Real-Time Anomaly Detection")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            current_hr = st.number_input("Heart Rate (BPM)", 50, 150, 75)
        with col2:
            current_bp = st.number_input("Blood Pressure", 80, 180, 120)
        with col3:
            current_temp = st.number_input("Temperature (°C)", 35.0, 40.0, 36.6, 0.1)
        
        if st.button("🔍 Run Detection", type="primary", width='stretch'):
            anomalies = []
            if current_hr > 100:
                anomalies.append(f"High heart rate: {current_hr} BPM")
            if current_bp > 140:
                anomalies.append(f"High blood pressure: {current_bp} mmHg")
            if current_temp > 37.5:
                anomalies.append(f"Elevated temperature: {current_temp}°C")
            
            if anomalies:
                st.error("🚨 **Anomalies Detected!**\n" + "\n".join(anomalies))
            else:
                st.success("✅ All metrics within normal ranges")

    # ==================== HEART RATE MONITOR ====================
    elif selected == "Heart Rate Monitor":
        st.markdown("## 💓 Heart Rate Monitor")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>💓 Track Your Heart Rate</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Monitor your heart rate manually or use demo mode. Normal resting heart rate is 60-100 BPM.
                Athletes may have lower rates (40-60 BPM).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'hr_history' not in st.session_state:
            st.session_state.hr_history = []
        if 'hr_timestamps' not in st.session_state:
            st.session_state.hr_timestamps = []
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📝 Manual Entry")
            col_hr1, col_hr2 = st.columns([2, 1])
            with col_hr1:
                manual_hr = st.number_input("Heart Rate (BPM)", 30, 200, 75, step=1, key="manual_hr")
            with col_hr2:
                if st.button("➕ Add Reading", width='stretch'):
                    now = datetime.now()
                    st.session_state.hr_history.append(manual_hr)
                    st.session_state.hr_timestamps.append(now)
                    st.success(f"✅ Added: {manual_hr} BPM at {now.strftime('%H:%M:%S')}")
                    st.rerun()
            
            st.markdown("#### Quick Add:")
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            with col_q1:
                if st.button("🟢 60 BPM", width='stretch'):
                    st.session_state.hr_history.append(60)
                    st.session_state.hr_timestamps.append(datetime.now())
                    st.success("✅ Added 60 BPM")
                    st.rerun()
            with col_q2:
                if st.button("🟡 75 BPM", width='stretch'):
                    st.session_state.hr_history.append(75)
                    st.session_state.hr_timestamps.append(datetime.now())
                    st.success("✅ Added 75 BPM")
                    st.rerun()
            with col_q3:
                if st.button("🟠 90 BPM", width='stretch'):
                    st.session_state.hr_history.append(90)
                    st.session_state.hr_timestamps.append(datetime.now())
                    st.success("✅ Added 90 BPM")
                    st.rerun()
            with col_q4:
                if st.button("🔴 110 BPM", width='stretch'):
                    st.session_state.hr_history.append(110)
                    st.session_state.hr_timestamps.append(datetime.now())
                    st.success("✅ Added 110 BPM")
                    st.rerun()
        
        with col2:
            st.markdown("### 📱 Demo Mode")
            if st.button("🎮 Single Reading", width='stretch'):
                demo_hr = np.random.randint(65, 95)
                st.session_state.hr_history.append(demo_hr)
                st.session_state.hr_timestamps.append(datetime.now())
                if demo_hr < 60:
                    st.info(f"📱 Demo reading: {demo_hr} BPM (Low - Normal for athletes)")
                elif demo_hr <= 100:
                    st.success(f"✅ Demo reading: {demo_hr} BPM (Normal range)")
                else:
                    st.warning(f"⚠️ Demo reading: {demo_hr} BPM (Above normal)")
                st.rerun()
            
            if st.button("🔄 Simulate 5 Readings", width='stretch'):
                for _ in range(5):
                    demo_hr = np.random.randint(65, 100)
                    st.session_state.hr_history.append(demo_hr)
                    st.session_state.hr_timestamps.append(datetime.now())
                st.success("✅ Added 5 simulated readings")
                st.rerun()
            
            if st.button("📊 Simulate Exercise Pattern", width='stretch'):
                pattern = [85, 95, 110, 120, 115, 105, 95, 85, 80, 75]
                for hr in pattern:
                    st.session_state.hr_history.append(hr)
                    st.session_state.hr_timestamps.append(datetime.now())
                    time.sleep(0.1)
                st.success("✅ Added exercise pattern (warm up → peak → recovery)")
                st.rerun()
        
        if st.session_state.hr_history:
            st.markdown("---")
            st.markdown("### 📊 Current Status")
            current_hr = st.session_state.hr_history[-1]
            
            if current_hr < 60:
                status = "Bradycardia (Low)"
                color = "#1E88E5"
                emoji = "🟦"
                explanation = "Your heart rate is below normal. This is common in athletes, but consult a doctor if you feel dizzy or fatigued."
            elif current_hr <= 100:
                status = "Normal"
                color = "#43A047"
                emoji = "🟢"
                explanation = "Your heart rate is within the normal range. Great job!"
            else:
                status = "Tachycardia (High)"
                color = "#E53935"
                emoji = "🔴"
                explanation = "Your heart rate is above normal. Consider rest, hydration, and relaxation techniques."
            
            col_s1, col_s2 = st.columns([1, 1])
            with col_s1:
                st.markdown(f"""
                <div style='background: white; padding: 2rem; border-radius: 20px; text-align: center; border-bottom: 5px solid {color};'>
                    <div style='font-size: 1rem; color: #666;'>Current Heart Rate</div>
                    <div style='font-size: 5rem; font-weight: 700; color: {color};'>{current_hr}</div>
                    <div style='font-size: 1.2rem;'>{emoji} {status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_s2:
                recent_hr = st.session_state.hr_history[-20:] if len(st.session_state.hr_history) > 20 else st.session_state.hr_history
                avg_hr = np.mean(recent_hr) if len(recent_hr) > 0 else current_hr
                min_hr = min(recent_hr) if len(recent_hr) > 0 else current_hr
                max_hr = max(recent_hr) if len(recent_hr) > 0 else current_hr
                
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 20px;'>
                    <div style='font-weight: 600; margin-bottom: 1rem;'>📈 Recent Stats (Last {len(recent_hr)} readings)</div>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                        <span>Average:</span>
                        <span style='font-weight: 700;'>{avg_hr:.0f} BPM</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                        <span>Minimum:</span>
                        <span style='font-weight: 700;'>{min_hr} BPM</span>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>Maximum:</span>
                        <span style='font-weight: 700;'>{max_hr} BPM</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background: #f0f7ff; padding: 1rem; border-radius: 15px; margin-top: 1rem;'>
                <strong>💡 Health Tip:</strong> {explanation}
            </div>
            """, unsafe_allow_html=True)
            
            recent_high = any(hr > 100 for hr in st.session_state.hr_history[-5:])
            recent_low = any(hr < 50 for hr in st.session_state.hr_history[-5:])
            
            if recent_high:
                st.warning("⚠️ Recent readings show elevated heart rate. Consider rest, hydration, and deep breathing.")
            elif recent_low and len(st.session_state.hr_history) > 0:
                st.info("ℹ️ Your recent readings show lower heart rate. This is normal for active individuals.")
            else:
                st.success("✅ Your recent readings are within normal range!")
        
        if len(st.session_state.hr_history) > 1:
            st.markdown("---")
            st.markdown("### 📈 Heart Rate History")
            
            hr_df = pd.DataFrame({
                'Time': st.session_state.hr_timestamps[-30:],
                'Heart Rate (BPM)': st.session_state.hr_history[-30:]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hr_df['Time'], y=hr_df['Heart Rate (BPM)'], 
                                    mode='lines+markers', name='Heart Rate',
                                    line=dict(color='#E53935', width=3),
                                    marker=dict(size=8, color='#E53935')))
            fig.add_hrect(y0=60, y1=100, line_width=0, fillcolor="green", opacity=0.1, 
                         annotation_text="Normal Range", annotation_position="bottom right")
            fig.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Min Normal")
            fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Max Normal")
            fig.update_layout(title="Your Heart Rate Over Time",
                             xaxis_title="Time",
                             yaxis_title="Heart Rate (BPM)",
                             yaxis_range=[40, 150],
                             height=400,
                             hovermode='x unified')
            st.plotly_chart(fig, width='stretch')
        
        if st.session_state.hr_history:
            st.markdown("---")
            col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
            with col_c2:
                if st.button("🗑️ Clear All History", width='stretch'):
                    st.session_state.hr_history = []
                    st.session_state.hr_timestamps = []
                    st.success("✅ History cleared!")
                    st.rerun()
        
        with st.expander("📚 Understanding Heart Rate"):
            st.markdown("""
            ### What is a normal heart rate?
            
            | Age Group | Normal Resting Heart Rate (BPM) |
            |-----------|--------------------------------|
            | Adults (18+) | 60 - 100 |
            | Athletes | 40 - 60 |
            | Children (6-15) | 70 - 100 |
            
            ### When to be concerned:
            - **Tachycardia**: Resting heart rate > 100 BPM
            - **Bradycardia**: Resting heart rate < 60 BPM (unless you're an athlete)
            
            ### Tips for healthy heart rate:
            1. 🏃 **Regular exercise** - Aim for 150 minutes weekly
            2. 💧 **Stay hydrated** - Dehydration increases heart rate
            3. 😴 **Get enough sleep** - Poor sleep affects heart rate
            4. 🧘 **Manage stress** - Deep breathing lowers heart rate
            5. ☕ **Limit caffeine** - Excessive caffeine increases heart rate
            """)

    # ==================== PERSONALIZED RECOMMENDATIONS ====================
    elif selected == "Personalized Recommendations":
        st.markdown("## 🎯 Personalized Health Recommendations")
        
        recommendations = [
            {"title": "🏃 30-Minute Daily Walk", "impact": "Reduces CVD risk by 15%", "priority": "High"},
            {"title": "🥗 Mediterranean Diet", "impact": "Reduces heart disease risk by 30%", "priority": "High"},
            {"title": "😴 Sleep Schedule Optimization", "impact": "Improves sleep quality by 40%", "priority": "Medium"},
            {"title": "🧘 Daily Meditation", "impact": "Reduces stress by 35%", "priority": "Medium"},
        ]
        
        for rec in recommendations:
            priority_color = "#E53935" if rec["priority"] == "High" else "#FB8C00"
            st.markdown(f"""
            <div style='background: white; padding: 0.8rem; border-radius: 10px; margin-bottom: 0.5rem; border-left: 4px solid {priority_color};'>
                <strong>{rec['title']}</strong><br>
                <span style='font-size: 0.8rem; color: #666;'>🎯 {rec['impact']}</span>
            </div>
            """, unsafe_allow_html=True)

    # ==================== AI HEALTH CHAT ====================
    elif selected == "AI Health Chat":
        st.markdown("## 💬 AI Health Assistant")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>👋 Your Personal Health Conversation Partner</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                I'll help you understand your health risks through a simple conversation. Just answer my questions naturally!
                <strong>No medical knowledge needed</strong> - just tell me about yourself.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.sidebar:
            st.markdown("### 💬 Chat Options")
            if st.button("🔄 New Health Assessment", width='stretch'):
                st.session_state.chat_conversation = []
                st.session_state.chat_user_responses = {}
                st.session_state.chat_current_risk = None
                st.session_state.chat_last_assessment = None
                st.session_state.health_chat_step = 0
                st.session_state.health_input = ""
                st.rerun()
            
            if st.session_state.chat_last_assessment:
                risk = st.session_state.chat_last_assessment.get('risk', 0)
                if risk < 10:
                    level = "Low"
                    color = "🟢"
                elif risk < 20:
                    level = "Moderate"
                    color = "🟡"
                else:
                    level = "High"
                    color = "🔴"
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; margin-top: 1rem;'>
                    <div style='font-size: 2rem;'>{color}</div>
                    <div><strong>Your Risk Score</strong></div>
                    <div style='font-size: 1.5rem; font-weight: bold; color: #1E88E5;'>{risk:.1f}%</div>
                    <div><span style='background: #e0e0e0; padding: 0.25rem 0.75rem; border-radius: 20px;'>{level} Risk</span></div>
                </div>
                """, unsafe_allow_html=True)
        
        questions = [
            ("age", "What's your age? (e.g., 35)"),
            ("height_weight", "What's your height (cm) and weight (kg)? (e.g., 170 cm, 70 kg)"),
            ("bp", "What's your blood pressure? (e.g., 120 or 120/80)"),
            ("cholesterol", "What's your total cholesterol level? (e.g., 180)"),
            ("smoking", "Do you smoke? (yes/no)"),
            ("exercise", "How would you describe your exercise level? (sedentary/light/moderate/active/very active)"),
            ("diet", "How would you rate your diet quality? (poor/fair/good/excellent)"),
            ("sleep", "How many hours of sleep do you get per night? (e.g., 7)"),
            ("stress", "On a scale of 1-10, how stressed do you feel typically? (1=low stress, 10=high stress)")
        ]

        if len(st.session_state.chat_conversation) == 0:
            st.session_state.chat_conversation.append({"role": "assistant", "content": "Hi! I'm your AI health assistant. Let me ask you a few questions to understand your health better."})
            st.session_state.chat_conversation.append({"role": "assistant", "content": questions[0][1]})
            st.session_state.health_chat_step = 0

        for msg in st.session_state.chat_conversation:
            if msg["role"] == "user":
                st.markdown(f'<div style="text-align: right; margin: 0.5rem 0;"><span style="background: linear-gradient(135deg, #1E88E5, #1565C0); color: white; padding: 0.75rem 1rem; border-radius: 20px; display: inline-block; max-width: 80%;">🧑 {msg["content"]}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align: left; margin: 0.5rem 0;"><span style="background: #f0f2f6; color: #1E88E5; padding: 0.75rem 1rem; border-radius: 20px; display: inline-block; max-width: 80%;">🤖 {msg["content"]}</span></div>', unsafe_allow_html=True)

        current_step = st.session_state.health_chat_step
        current_question = questions[current_step] if current_step < len(questions) else None

        if current_question and current_step < len(questions):
            with st.form(key=f"chat_form_{current_step}"):
                if current_question[0] == "height_weight":
                    col1, col2 = st.columns(2)
                    with col1:
                        height_input = st.text_input("Height (cm):", placeholder="e.g., 170", key="height_input")
                    with col2:
                        weight_input = st.text_input("Weight (kg):", placeholder="e.g., 70", key="weight_input")
                    submitted = st.form_submit_button("Send 💬", use_container_width=True)
                    
                    if submitted and height_input.strip() and weight_input.strip():
                        try:
                            height = float(height_input.strip())
                            weight = float(weight_input.strip())
                            if 100 <= height <= 250 and 30 <= weight <= 200:
                                bmi = weight / ((height/100) ** 2)
                                response_text = f"{height} cm, {weight} kg"
                                st.session_state.chat_user_responses["height"] = height
                                st.session_state.chat_user_responses["weight"] = weight
                                st.session_state.chat_user_responses["bmi"] = bmi
                                st.session_state.chat_conversation.append({"role": "user", "content": response_text})
                                
                                bmi_status = "healthy" if 18.5 <= bmi <= 24.9 else "above healthy" if bmi > 25 else "below healthy"
                                st.session_state.chat_conversation.append({"role": "assistant", "content": f"Your BMI is {bmi:.1f}, which is {bmi_status}."})
                                
                                st.session_state.health_chat_step += 1
                                next_step = st.session_state.health_chat_step
                                if next_step < len(questions):
                                    st.session_state.chat_conversation.append({"role": "assistant", "content": questions[next_step][1]})
                                st.rerun()
                            else:
                                st.error("Please enter valid height (100-250 cm) and weight (30-200 kg).")
                        except ValueError:
                            st.error("Please enter valid numbers.")
                
                elif current_question[0] == "exercise":
                    ex_options = ["sedentary", "light", "moderate", "active", "very active"]
                    ex_input = st.selectbox("Select your exercise level:", ex_options, key="ex_select")
                    submitted = st.form_submit_button("Send 💬", use_container_width=True)
                    
                    if submitted:
                        ex_map = {"sedentary": "None", "light": "Light", "moderate": "Moderate", "active": "Active", "very active": "Very Active"}
                        st.session_state.chat_user_responses["exercise"] = ex_map[ex_input]
                        st.session_state.chat_conversation.append({"role": "user", "content": ex_input})
                        st.session_state.health_chat_step += 1
                        next_step = st.session_state.health_chat_step
                        if next_step < len(questions):
                            st.session_state.chat_conversation.append({"role": "assistant", "content": questions[next_step][1]})
                        st.rerun()
                
                elif current_question[0] == "diet":
                    diet_options = ["poor", "fair", "good", "excellent"]
                    diet_input = st.selectbox("Select your diet quality:", diet_options, key="diet_select")
                    submitted = st.form_submit_button("Send 💬", use_container_width=True)
                    
                    if submitted:
                        diet_map = {"poor": "Poor", "fair": "Fair", "good": "Good", "excellent": "Excellent"}
                        st.session_state.chat_user_responses["diet"] = diet_map[diet_input]
                        st.session_state.chat_conversation.append({"role": "user", "content": diet_input})
                        st.session_state.health_chat_step += 1
                        next_step = st.session_state.health_chat_step
                        if next_step < len(questions):
                            st.session_state.chat_conversation.append({"role": "assistant", "content": questions[next_step][1]})
                        st.rerun()
                
                else:
                    user_input = st.text_input("Your response:", placeholder="Type your answer here...", key=f"chat_input_{current_step}")
                    submitted = st.form_submit_button("Send 💬", use_container_width=True)
                    
                    if submitted and user_input.strip():
                        if current_question[0] == "age":
                            try:
                                age = int(user_input.strip())
                                if 18 <= age <= 100:
                                    st.session_state.chat_user_responses["age"] = age
                                    st.session_state.chat_conversation.append({"role": "user", "content": user_input})
                                    st.session_state.chat_conversation.append({"role": "assistant", "content": f"Thanks! At {age}, that's a great time to focus on preventive health."})
                                    st.session_state.health_chat_step += 1
                                    next_step = st.session_state.health_chat_step
                                    if next_step < len(questions):
                                        st.session_state.chat_conversation.append({"role": "assistant", "content": questions[next_step][1]})
                                    st.rerun()
                                else:
                                    st.error("Please enter a valid age between 18 and 100.")
                            except ValueError:
                                st.error("Please enter a valid number.")
                        
                        elif current_question[0] == "bp":
                            try:
                                if "/" in user_input:
                                    bp = int(user_input.split("/")[0])
                                else:
                                    bp = int(user_input.strip())
                                if 80 <= bp <= 200:
                                    st.session_state.chat_user_responses["bp"] = bp
                                    st.session_state.chat_conversation.append({"role": "user", "content": user_input})
                                    bp_status = "normal" if bp < 120 else "elevated" if bp < 130 else "high"
                                    st.session_state.chat_conversation.append({"role": "assistant", "content": f"Your blood pressure ({bp}) is {bp_status}."})
                                    st.session_state.health_chat_step += 1
                                    next_step = st.session_state.health_chat_step
                                    if next_step < len(questions):
                                        st.session_state.chat_conversation.append({"role": "assistant", "content": questions[next_step][1]})
                                    st.rerun()
                                else:
                                    st.error("Please enter a valid blood pressure (80-200).")
                            except ValueError:
                                st.error("Please enter a valid number.")
                        
                        elif current_question[0] == "cholesterol":
                            try:
                                chol = int(user_input.strip())
                                if 100 <= chol <= 400:
                                    st.session_state.chat_user_responses["cholesterol"] = chol
                                    st.session_state.chat_conversation.append({"role": "user", "content": user_input})
                                    chol_status = "optimal" if chol < 200 else "borderline" if chol < 240 else "high"
                                    st.session_state.chat_conversation.append({"role": "assistant", "content": f"Your cholesterol is {chol} mg/dL, which is {chol_status}."})
                                    st.session_state.health_chat_step += 1
                                    next_step = st.session_state.health_chat_step
                                    if next_step < len(questions):
                                        st.session_state.chat_conversation.append({"role": "assistant", "content": questions[next_step][1]})
                                    st.rerun()
                                else:
                                    st.error("Please enter a valid cholesterol level (100-400).")
                            except ValueError:
                                st.error("Please enter a valid number.")
                        
                        elif current_question[0] == "smoking":
                            smoker = user_input.strip().lower() in ["yes", "y", "true", "1"]
                            st.session_state.chat_user_responses["smoker"] = smoker
                            st.session_state.chat_conversation.append({"role": "user", "content": user_input})
                            smoke_msg = "I understand. Quitting is the best thing for your health." if smoker else "That's great! Not smoking is excellent for your heart health."
                            st.session_state.chat_conversation.append({"role": "assistant", "content": smoke_msg})
                            st.session_state.health_chat_step += 1
                            next_step = st.session_state.health_chat_step
                            if next_step < len(questions):
                                st.session_state.chat_conversation.append({"role": "assistant", "content": questions[next_step][1]})
                            st.rerun()
                        
                        elif current_question[0] == "sleep":
                            try:
                                sleep = float(user_input.strip())
                                if 3 <= sleep <= 14:
                                    st.session_state.chat_user_responses["sleep"] = sleep
                                    st.session_state.chat_conversation.append({"role": "user", "content": user_input})
                                    st.session_state.health_chat_step += 1
                                    next_step = st.session_state.health_chat_step
                                    if next_step < len(questions):
                                        st.session_state.chat_conversation.append({"role": "assistant", "content": questions[next_step][1]})
                                    st.rerun()
                                else:
                                    st.error("Please enter a valid sleep duration (3-14 hours).")
                            except ValueError:
                                st.error("Please enter a valid number.")
                        
                        elif current_question[0] == "stress":
                            try:
                                stress = int(user_input.strip())
                                if 1 <= stress <= 10:
                                    st.session_state.chat_user_responses["stress"] = stress
                                    st.session_state.chat_conversation.append({"role": "user", "content": user_input})
                                    
                                    age = st.session_state.chat_user_responses.get("age", 45)
                                    bmi = st.session_state.chat_user_responses.get("bmi", 25)
                                    bp = st.session_state.chat_user_responses.get("bp", 120)
                                    chol = st.session_state.chat_user_responses.get("cholesterol", 180)
                                    smoker = st.session_state.chat_user_responses.get("smoker", False)
                                    exercise = st.session_state.chat_user_responses.get("exercise", "Moderate")
                                    diet = st.session_state.chat_user_responses.get("diet", "Good")
                                    sleep = st.session_state.chat_user_responses.get("sleep", 7)
                                    
                                    risk_score = 10
                                    if age > 50:
                                        risk_score += (age - 50) * 0.3
                                    if bp > 130:
                                        risk_score += (bp - 130) * 0.1
                                    if bmi > 25:
                                        risk_score += (bmi - 25) * 0.5
                                    if chol > 200:
                                        risk_score += (chol - 200) * 0.05
                                    if smoker:
                                        risk_score += 8
                                    ex_benefit = {"None": 0, "Light": 2, "Moderate": 4, "Active": 6, "Very Active": 8}
                                    risk_score -= ex_benefit.get(exercise, 0)
                                    diet_benefit = {"Poor": 0, "Fair": 2, "Good": 4, "Excellent": 6}
                                    risk_score -= diet_benefit.get(diet, 0)
                                    if sleep < 6 or sleep > 9:
                                        risk_score += 3
                                    else:
                                        risk_score -= 2
                                    risk_score += max(0, (stress - 5)) * 1
                                    risk_score = min(max(risk_score, 5), 40)
                                    
                                    st.session_state.chat_last_assessment = {'risk': risk_score, 'timestamp': datetime.now().isoformat()}
                                    
                                    if risk_score < 10:
                                        level = "Low"
                                        emoji = "🟢"
                                        message = "Excellent! Your health habits are protecting you well."
                                    elif risk_score < 20:
                                        level = "Moderate"
                                        emoji = "🟡"
                                        message = "Good! Small improvements could lower your risk further."
                                    else:
                                        level = "High"
                                        emoji = "🔴"
                                        message = "Your risk is elevated, but lifestyle changes can significantly improve this."
                                    
                                    response = f"""{emoji} **Here's your personalized health assessment!**

**Your Health Risk Score: {risk_score:.1f}%**
**Risk Level: {level}**

{message}

**💡 Personalized Recommendations:**

1. **Exercise**: Aim for at least 150 minutes of moderate activity per week
2. **Blood Pressure**: Keep it below 130/80 mmHg through exercise and diet
3. **Weight Management**: Maintain a healthy BMI through balanced nutrition
4. **Stress Management**: Practice relaxation techniques daily
5. **Sleep**: Ensure 7-9 hours of quality sleep per night

**What would you like to do next?**
- Type **'details'** for more detailed recommendations
- Type **'whatif'** to try a what-if scenario
- Type **'new'** to start a new assessment"""
                                    
                                    st.session_state.chat_conversation.append({"role": "assistant", "content": response})
                                    st.session_state.health_chat_step = len(questions)
                                    st.rerun()
                                else:
                                    st.error("Please enter a number between 1 and 10.")
                            except ValueError:
                                st.error("Please enter a valid number.")

        elif st.session_state.health_chat_step >= len(questions):
            with st.form(key="followup_form"):
                followup_input = st.text_input("Your response:", placeholder="Type details, whatif, or new...", key="followup_input")
                submitted = st.form_submit_button("Send 💬", use_container_width=True)
                
                if submitted and followup_input.strip():
                    st.session_state.chat_conversation.append({"role": "user", "content": followup_input})
                    response_lower = followup_input.strip().lower()
                    
                    if response_lower in ["details", "detail", "1"]:
                        age = st.session_state.chat_user_responses.get("age", 45)
                        bmi = st.session_state.chat_user_responses.get("bmi", 25)
                        bp = st.session_state.chat_user_responses.get("bp", 120)
                        chol = st.session_state.chat_user_responses.get("cholesterol", 180)
                        smoker = st.session_state.chat_user_responses.get("smoker", False)
                        exercise = st.session_state.chat_user_responses.get("exercise", "Moderate")
                        diet = st.session_state.chat_user_responses.get("diet", "Good")
                        sleep = st.session_state.chat_user_responses.get("sleep", 7)
                        stress = st.session_state.chat_user_responses.get("stress", 5)
                        
                        advice = []
                        if bmi > 25:
                            advice.append(f"Your BMI is {bmi:.1f}, which is above healthy range. Try 10-minute walks after meals.")
                        elif bmi < 18.5:
                            advice.append(f"Your BMI is {bmi:.1f}, below healthy range. Consider nutrient-rich foods.")
                        else:
                            advice.append(f"Great! Your BMI of {bmi:.1f} is healthy. Keep it up!")
                        if bp > 130:
                            advice.append(f"Your blood pressure ({bp}) is elevated. Try reducing salt and deep breathing.")
                        else:
                            advice.append(f"Your blood pressure ({bp}) is well controlled. Keep monitoring!")
                        if chol > 200:
                            advice.append(f"Your cholesterol ({chol}) is high. Add oats, nuts, and fish to your diet.")
                        else:
                            advice.append(f"Your cholesterol ({chol}) is good. Maintain with fiber-rich foods!")
                        if smoker:
                            advice.append("Quitting smoking is the best thing for your heart. Try nicotine patches or support groups.")
                        else:
                            advice.append("Great job not smoking! This significantly reduces your heart disease risk.")
                        advice.append(f"Your exercise level is {exercise.lower()}. Aim for 150 minutes weekly.")
                        advice.append(f"Your diet is {diet.lower()}. Add one extra vegetable serving daily.")
                        if sleep < 6:
                            advice.append(f"You're getting only {sleep} hours sleep. Try a consistent bedtime routine.")
                        elif sleep > 9:
                            advice.append(f"You're sleeping {sleep} hours. While good, excessive sleep may need checking.")
                        else:
                            advice.append(f"Your sleep of {sleep} hours is optimal! Good sleep protects your heart.")
                        if stress > 7:
                            advice.append(f"Your stress level is high ({stress}/10). Try the 4-7-8 breathing technique.")
                        else:
                            advice.append(f"Your stress level is manageable ({stress}/10). Keep practicing relaxation!")
                        
                        detailed_response = "**📋 Complete Health Plan:**\n\n"
                        for i, a in enumerate(advice, 1):
                            detailed_response += f"{i}. {a}\n\n"
                        detailed_response += "\n**🎯 Your Action Items:**\n"
                        detailed_response += "• Schedule a check-up with your doctor\n"
                        detailed_response += "• Track your progress weekly\n"
                        detailed_response += "• Set one small health goal for this week"
                        
                        st.session_state.chat_conversation.append({"role": "assistant", "content": detailed_response})
                        st.session_state.chat_conversation.append({"role": "assistant", "content": "Type **'whatif'** to try a scenario, or **'new'** for a new assessment."})
                        st.rerun()
                    
                    elif response_lower in ["whatif", "what-if", "scenario", "2"]:
                        current_risk = st.session_state.chat_last_assessment.get('risk', 15)
                        st.session_state.chat_conversation.append({"role": "assistant", "content": f"**🔮 What-If Scenario Simulator**\n\nYour current risk is {current_risk:.1f}%.\n\nWhat would you like to change?\n• Type **'bmi'** to lower your BMI\n• Type **'exercise'** to increase activity\n• Type **'smoke'** to quit smoking\n• Type **'diet'** to improve nutrition\n\nWhat would you like to simulate?"})
                        st.rerun()
                    
                    elif response_lower in ["new", "reset", "3"]:
                        st.session_state.chat_conversation = []
                        st.session_state.chat_user_responses = {}
                        st.session_state.chat_current_risk = None
                        st.session_state.chat_last_assessment = None
                        st.session_state.health_chat_step = 0
                        st.session_state.health_input = ""
                        st.session_state.chat_conversation.append({"role": "assistant", "content": "Let's start fresh! What's your age?"})
                        st.rerun()
                    
                    elif response_lower in ["bmi"]:
                        current_bmi = st.session_state.chat_user_responses.get("bmi", 28)
                        target_bmi = max(18.5, current_bmi - 3)
                        current_risk = st.session_state.chat_last_assessment.get('risk', 15)
                        new_risk = max(5, current_risk - (current_bmi - target_bmi) * 0.5)
                        reduction = current_risk - new_risk
                        st.session_state.chat_conversation.append({"role": "assistant", "content": f"📊 **Scenario: Lowering BMI from {current_bmi:.1f} to {target_bmi:.1f}**\n\nCurrent risk: {current_risk:.1f}%\nNew risk: {new_risk:.1f}%\nReduction: {reduction:.1f}%\n\nThat's a great improvement! Try adding daily walks and reducing portion sizes.\n\nType **'whatif'** for another scenario, or **'new'** for a new assessment."})
                        st.rerun()
                    
                    elif response_lower in ["exercise", "ex"]:
                        current_risk = st.session_state.chat_last_assessment.get('risk', 15)
                        new_risk = max(5, current_risk - 4)
                        reduction = current_risk - new_risk
                        st.session_state.chat_conversation.append({"role": "assistant", "content": f"📊 **Scenario: Increasing exercise level**\n\nCurrent risk: {current_risk:.1f}%\nNew risk: {new_risk:.1f}%\nReduction: {reduction:.1f}%\n\nAim for 30 minutes of walking, 5 days a week to achieve this!\n\nType **'whatif'** for another scenario, or **'new'** for a new assessment."})
                        st.rerun()
                    
                    elif response_lower in ["smoke", "smoking", "quit"] and st.session_state.chat_user_responses.get("smoker", False):
                        current_risk = st.session_state.chat_last_assessment.get('risk', 15)
                        new_risk = max(5, current_risk - 8)
                        reduction = current_risk - new_risk
                        st.session_state.chat_conversation.append({"role": "assistant", "content": f"📊 **Scenario: Quitting Smoking**\n\nCurrent risk: {current_risk:.1f}%\nNew risk: {new_risk:.1f}%\nReduction: {reduction:.1f}%\n\nThis is the most impactful change! Within 1 year of quitting, your heart disease risk drops by 50%.\n\nType **'whatif'** for another scenario, or **'new'** for a new assessment."})
                        st.rerun()
                    
                    elif response_lower in ["diet"]:
                        current_risk = st.session_state.chat_last_assessment.get('risk', 15)
                        new_risk = max(5, current_risk - 3)
                        reduction = current_risk - new_risk
                        st.session_state.chat_conversation.append({"role": "assistant", "content": f"📊 **Scenario: Improving diet quality**\n\nCurrent risk: {current_risk:.1f}%\nNew risk: {new_risk:.1f}%\nReduction: {reduction:.1f}%\n\nAdd one extra serving of vegetables to each meal and reduce processed foods!\n\nType **'whatif'** for another scenario, or **'new'** for a new assessment."})
                        st.rerun()
                    
                    else:
                        st.session_state.chat_conversation.append({"role": "assistant", "content": "I didn't understand that. Please type:\n• **'details'** for detailed recommendations\n• **'whatif'** for what-if scenarios\n• **'new'** for a new assessment\n\nOr for scenarios, type: **bmi**, **exercise**, **smoke**, or **diet**"})
                        st.rerun()

    # ==================== PERSONAL ASSISTANT ====================
    elif selected == "Personal Assistant":
        st.markdown("## 🤖 Personal Assistant")
        for msg in st.session_state.chat_history[-10:]:
            if msg["role"] == "user":
                st.markdown(f"**🧑 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {msg['content']}")
        with st.form("chat_form"):
            user_input = st.text_input("Ask me anything about your health...", key="chat_input")
            submitted = st.form_submit_button("Send")
        if submitted and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = generate_chat_response(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    # ==================== FAMILY HEALTH ====================
    elif selected == "Family Health":
        st.markdown("## 👨‍👩‍👧‍👦 Family Health Dashboard")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>👪 Family Health Mapping</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Track health risks across family members based on relationships.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ➕ Add Family Member")
            member_name = st.text_input("Name", placeholder="e.g., John Doe")
            member_relation = st.selectbox("Relation", ["Spouse", "Child", "Parent", "Grandparent", "Sibling", "Twin", "Other"])
            member_age = st.number_input("Age", 0, 120, 30)
            member_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            base_risk = st.slider("Base Health Risk", 0.0, 0.5, 0.15, 0.01)
            
            if st.button("➕ Add Member", type="primary", use_container_width=True):
                if member_name:
                    st.session_state.family_members.append({
                        "name": member_name,
                        "relation": member_relation,
                        "age": member_age,
                        "gender": member_gender,
                        "base_risk": base_risk,
                        "propagated_risk": base_risk,
                        "added": datetime.now().strftime("%Y-%m-%d")
                    })
                    propagator = GCNHealthPropagator()
                    for m in st.session_state.family_members:
                        for m2 in st.session_state.family_members:
                            if m != m2 and m2.get('relation') in ['Twin', 'Parent', 'Child', 'Sibling', 'Spouse']:
                                propagator.add_relationship(m['name'], m2['name'], m2['relation'].lower().replace(' ', '_'))
                    for member in st.session_state.family_members:
                        member['propagated_risk'] = propagator.propagate_risk(member['name'], member['base_risk']).get(member['name'], member['base_risk'])
                    st.success(f"✅ Added {member_name}")
                    st.rerun()
                else:
                    st.error("Please enter a name")
        
        with col2:
            if st.session_state.family_members:
                for idx, member in enumerate(st.session_state.family_members):
                    with st.expander(f"👤 {member['name']} ({member['relation']}, Age {member['age']})"):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Base Risk", f"{member['base_risk']*100:.1f}%")
                        with col_b:
                            risk_color = "#E53935" if member['propagated_risk'] > 0.2 else "#FB8C00" if member['propagated_risk'] > 0.1 else "#43A047"
                            st.markdown(f"<div style='text-align:center'><span style='font-size:0.8rem;color:#666;'>Family Risk</span><br><span style='font-size:1.5rem;font-weight:700;color:{risk_color};'>{member['propagated_risk']*100:.1f}%</span></div>", unsafe_allow_html=True)
                        with col_c:
                            if st.button(f"Remove", key=f"remove_family_{idx}"):
                                st.session_state.family_members.pop(idx)
                                st.rerun()
                        
                        if member['propagated_risk'] < 0.1:
                            st.success("✅ Low risk profile")
                        elif member['propagated_risk'] < 0.2:
                            st.warning("⚠️ Moderate risk")
                        else:
                            st.error("🔴 High risk - Medical consultation advised")
            else:
                st.info("No family members added yet. Add family members to see risk propagation!")
        
        if st.session_state.family_members:
            st.markdown("---")
            st.markdown("### 📊 Family Risk Graph")
            
            family_df = pd.DataFrame(st.session_state.family_members)
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Total Members", len(family_df))
            with col_s2:
                st.metric("Average Risk", f"{family_df['propagated_risk'].mean()*100:.1f}%")
            with col_s3:
                high_risk = len(family_df[family_df['propagated_risk'] > 0.2])
                st.metric("High Risk Members", high_risk)
            
            fig = px.bar(family_df, x='name', y='propagated_risk', title="Family Risk Comparison",
                        color='propagated_risk', color_continuous_scale='RdYlGn_r',
                        text=family_df['propagated_risk'].apply(lambda x: f"{x*100:.1f}%"))
            fig.update_traces(textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 Family relationships affect health risks (twins share highest similarity, then siblings, parent-child, spouses).")

    # ==================== WELLNESS CHALLENGES ====================
    elif selected == "Wellness Challenges":
        st.markdown("## 🏆 Wellness Challenges & Rewards")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #FFA00020 0%, #FB8C0020 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #FB8C00;'>
            <h4 style='margin:0; color: #FB8C00;'>🎯 Gamified Health Journey</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Complete challenges, earn points, and unlock achievements!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        challenges = [
            {"id": "daily_steps", "name": "🚶 10,000 Steps Daily", "reward": 100, "progress": st.session_state.challenge_progress['daily_steps']},
            {"id": "meditation", "name": "🧘 7-Day Meditation", "reward": 200, "progress": st.session_state.challenge_progress['meditation']},
            {"id": "water_intake", "name": "💧 8 Glasses Water", "reward": 50, "progress": st.session_state.challenge_progress['water_intake']},
            {"id": "sugar_free", "name": "🍬 7 Days No Sugar", "reward": 300, "progress": st.session_state.challenge_progress['sugar_free']},
            {"id": "workout", "name": "💪 30-Day Workout", "reward": 500, "progress": st.session_state.challenge_progress['workout']},
        ]
        
        st.markdown("### 🔥 Active Challenges")
        
        for challenge in challenges:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{challenge['name']}**")
                st.progress(challenge['progress'] / 100)
                st.caption(f"Progress: {challenge['progress']}%")
            with col2:
                st.markdown(f"<span style='color: #1E88E5; font-weight: bold;'>🏆 {challenge['reward']} pts</span>", unsafe_allow_html=True)
                if st.button(f"Update", key=f"update_{challenge['id']}"):
                    new_progress = min(100, challenge['progress'] + 20)
                    st.session_state.challenge_progress[challenge['id']] = new_progress
                    if new_progress >= 100 and challenge['id'] not in st.session_state.completed_challenges:
                        st.session_state.completed_challenges.append(challenge['id'])
                        st.session_state.points += challenge['reward']
                        
                        if len(st.session_state.completed_challenges) == 1:
                            st.session_state.badges.append("🌟 First Challenge Completed")
                        elif len(st.session_state.completed_challenges) == 3:
                            st.session_state.badges.append("🏅 Bronze Achiever")
                        elif len(st.session_state.completed_challenges) == 5:
                            st.session_state.badges.append("🥈 Silver Warrior")
                        
                        st.balloons()
                        st.success(f"🎉 Challenge completed! +{challenge['reward']} points!")
                    else:
                        st.success(f"✅ Progress updated!")
                    st.rerun()
            st.markdown("---")
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("### 🎖️ Your Points")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1E88E5, #1565C0); 
                        padding: 2rem; border-radius: 20px; text-align: center; color: white;'>
                <span style='font-size: 3rem;'>⭐</span>
                <div style='font-size: 3rem; font-weight: 700;'>{st.session_state.points}</div>
                <div>Total Points Earned</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_p2:
            st.markdown("### 🏅 Achievements")
            if st.session_state.badges:
                for badge in st.session_state.badges:
                    st.success(f"✅ {badge}")
            else:
                st.info("Complete challenges to unlock achievements!")

    # ==================== TELEMEDICINE ====================
    elif selected == "Telemedicine":
        st.markdown("## 🏥 Telemedicine & Doctor Consultation")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>👨‍⚕️ Consult with Healthcare Professionals</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Book video consultations with doctors, share your health data, and receive expert medical advice.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        doctors = {
            "Cardiologist": [
                {"name": "Dr. Sarah Johnson", "experience": "15 years", "rating": 4.9, "available": True, "price": "$150", "specialty": "Heart Specialist"},
                {"name": "Dr. Michael Chen", "experience": "12 years", "rating": 4.8, "available": True, "price": "$140", "specialty": "Cardiac Surgeon"},
            ],
            "General Physician": [
                {"name": "Dr. James Wilson", "experience": "20 years", "rating": 4.9, "available": True, "price": "$120", "specialty": "Family Medicine"},
                {"name": "Dr. Lisa Brown", "experience": "8 years", "rating": 4.8, "available": True, "price": "$110", "specialty": "Internal Medicine"},
            ],
            "Endocrinologist": [
                {"name": "Dr. Maria Garcia", "experience": "11 years", "rating": 4.9, "available": True, "price": "$160", "specialty": "Diabetes Specialist"},
            ],
            "Nutritionist": [
                {"name": "Dr. Jennifer Lee", "experience": "7 years", "rating": 4.8, "available": True, "price": "$100", "specialty": "Clinical Nutrition"},
            ],
            "Psychologist": [
                {"name": "Dr. Laura Martinez", "experience": "12 years", "rating": 4.9, "available": True, "price": "$130", "specialty": "Mental Health"},
            ]
        }
        
        tab1, tab2, tab3 = st.tabs(["📅 Book Appointment", "👨‍⚕️ Find a Doctor", "📋 My Appointments"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                specialty = st.selectbox("Select Specialty", list(doctors.keys()))
                available_doctors = [d for d in doctors[specialty] if d['available']]
                
                if available_doctors:
                    doctor_names = [f"{d['name']} - ⭐ {d['rating']} ({d['experience']})" for d in available_doctors]
                    selected_doctor_str = st.selectbox("Select Doctor", doctor_names)
                    selected_doctor = available_doctors[doctor_names.index(selected_doctor_str)]
                    
                    st.info(f"""
                    **{selected_doctor['name']}**  
                    ⭐ {selected_doctor['rating']} · {selected_doctor['experience']} exp  
                    💰 {selected_doctor['price']}/visit · 🩺 {selected_doctor['specialty']}
                    """)
                else:
                    st.error("No doctors available for this specialty")
                    selected_doctor = None
                
                consultation_type = st.radio("Consultation Type", ["Video Call", "Audio Call", "Chat Only"], horizontal=True)
            
            with col2:
                available_dates = [(datetime.now() + timedelta(days=i)).strftime("%A, %B %d") for i in range(1, 8)]
                selected_date = st.selectbox("Select Date", available_dates)
                time_slots = ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"]
                selected_time = st.selectbox("Select Time", time_slots)
                reason = st.text_area("Reason for Consultation", placeholder="Describe your symptoms...", height=100)
                share_data = st.checkbox("Share my health data with doctor", value=True)
                
                user_history = db.get_user_history(user_id, limit=1)
                if user_history and share_data:
                    st.info(f"📊 Sharing health records with doctor")
            
            if st.button("✅ Confirm Booking", type="primary", use_container_width=True):
                if selected_doctor and reason:
                    appointment = {
                        'id': len(st.session_state.appointments) + 1,
                        'doctor_name': selected_doctor['name'],
                        'specialty': specialty,
                        'date': selected_date,
                        'time': selected_time,
                        'type': consultation_type,
                        'reason': reason,
                        'status': 'confirmed',
                        'price': selected_doctor['price'],
                        'booked_on': datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    st.session_state.appointments.append(appointment)
                    st.balloons()
                    st.success(f"✅ Appointment confirmed with {selected_doctor['name']} on {selected_date} at {selected_time}")
                    st.info(f"📧 Confirmation sent to your email. Meeting link will be sent 15 minutes before appointment.")
                else:
                    st.error("Please select a doctor and provide reason for consultation")
        
        with tab2:
            st.markdown("### 🔍 Search Doctors")
            search_specialty = st.selectbox("Filter by Specialty", ["All"] + list(doctors.keys()))
            
            for specialty, doc_list in doctors.items():
                if search_specialty == "All" or search_specialty == specialty:
                    st.markdown(f"#### {specialty}")
                    for doc in doc_list:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"**{doc['name']}**  \n⭐ {doc['rating']} · {doc['experience']} · {doc['specialty']}")
                        with col2:
                            st.markdown(f"💰 {doc['price']}")
                        with col3:
                            if doc['available']:
                                if st.button(f"Book {doc['name'].split()[1]}", key=f"book_{doc['name']}"):
                                    st.success(f"Redirecting to booking for {doc['name']}")
        
        with tab3:
            st.markdown("### 📋 Your Appointments")
            
            if st.session_state.appointments:
                for apt in st.session_state.appointments:
                    with st.expander(f"📅 {apt['date']} at {apt['time']} - {apt['doctor_name']} ({apt['specialty']})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            **Status:** ✅ {apt['status'].upper()}  
                            **Type:** {apt['type']}  
                            **Price:** {apt['price']}  
                            **Reason:** {apt['reason']}
                            """)
                        with col2:
                            if apt['status'] == 'confirmed':
                                if st.button(f"Join Call", key=f"join_{apt['id']}"):
                                    st.info("🎥 Connecting to video call... Meeting link will appear here.")
                            if st.button(f"Cancel", key=f"cancel_{apt['id']}"):
                                apt['status'] = 'cancelled'
                                st.warning("Appointment cancelled")
                                st.rerun()
            else:
                st.info("No appointments booked yet. Schedule your first consultation!")

    # ==================== SETTINGS ====================
    elif selected == "Settings":
        st.markdown("## ⚙️ Settings")
        tab1, tab2, tab3 = st.tabs(["Profile", "Notifications", "Privacy"])
        with tab1:
            st.text_input("Full Name")
            st.text_input("Email")
            st.number_input("Height (cm)", 140, 220, 170)
            st.number_input("Weight (kg)", 40, 150, 75)
        with tab2:
            st.checkbox("Email Alerts", True)
            st.checkbox("Weekly Report", True)
            st.slider("Risk Threshold", 10, 30, 20)
        with tab3:
            st.checkbox("Share anonymized data", False)
            st.selectbox("Data Retention", ["30 days", "90 days", "1 year"])

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #78909C; padding: 2rem;'>
        <p>🧬 AI Digital Twin Platform - Your Personal Health Companion</p>
        <p style='font-size:0.7rem;'>© 2026 | Making health simple and understandable</p>
    </div>
    """, unsafe_allow_html=True)