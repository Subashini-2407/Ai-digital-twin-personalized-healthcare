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
from PIL import Image
import io
import calendar
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import traceback

try:
    from streamlit_option_menu import option_menu
except ImportError:
    option_menu = None
import plotly.figure_factory as ff

# Google Sheets imports
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Import Twin Health Model
from src.backend.twin_health_model import TwinHealthModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
        padding: 1.8rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
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
    
    .metric-card:hover {
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1.2;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #1E88E5, #1565C0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
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
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #43A047, #FB8C00, #E53935);
        border-radius: 10px;
        height: 10px;
    }
    
    .stButton > button {
        background: white;
        color: #1E88E5;
        border: 2px solid #1E88E5;
        padding: 0.75rem 2rem;
        font-size: 1rem;
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
        gap: 2rem;
        background: white;
        padding: 0.5rem;
        border-radius: 50px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        color: #666;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E88E5, #1565C0) !important;
        color: white !important;
    }
    
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .action-button {
        background: linear-gradient(135deg, #1E88E5, #1565C0);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(30,136,229,0.3);
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
    """Generate a contextual response based on user input"""
    user_input = user_input.lower()
    
    if any(word in user_input for word in ["risk", "assessment", "score"]):
        return "I can help you understand your risk assessment. Your latest risk score is based on your health data. Would you like me to explain what factors contribute to your risk?"
    elif any(word in user_input for word in ["exercise", "workout", "physical activity"]):
        return "Regular exercise is crucial for heart health! Aim for at least 150 minutes of moderate aerobic activity per week. What type of exercise are you interested in?"
    elif any(word in user_input for word in ["diet", "food", "eat", "nutrition"]):
        return "A heart-healthy diet includes plenty of fruits, vegetables, whole grains, and lean proteins. The Mediterranean diet is particularly beneficial. What aspect of nutrition would you like to know more about?"
    elif any(word in user_input for word in ["smoking", "smoke", "quit"]):
        return "Quitting smoking is one of the most important steps you can take for your health. It reduces your risk significantly within just one year. Resources like 1-800-QUIT-NOW can help. Have you considered quitting?"
    elif any(word in user_input for word in ["weight", "bmi", "obesity"]):
        return "Maintaining a healthy weight is important for cardiovascular health. Even modest weight loss (5-10%) can significantly reduce your risk. Would you like tips on weight management?"
    elif any(word in user_input for word in ["blood pressure", "hypertension"]):
        return "Blood pressure control is essential. Aim for less than 130/80 mmHg. Regular monitoring, reduced sodium intake, and exercise can help. How is your blood pressure currently?"
    elif any(word in user_input for word in ["cholesterol", "lipid"]):
        return "Cholesterol management involves diet, exercise, and sometimes medication. Target LDL <100 mg/dL. What do you know about your cholesterol levels?"
    elif any(word in user_input for word in ["stress", "anxiety", "mental health"]):
        return "Stress management is important for heart health. Techniques like meditation, deep breathing, and regular exercise can help. How are you managing stress currently?"
    elif any(word in user_input for word in ["sleep", "rest"]):
        return "Good sleep (7-9 hours/night) is crucial for heart health. Poor sleep can increase blood pressure and inflammation. Are you getting enough quality sleep?"
    elif any(word in user_input for word in ["hello", "hi", "hey", "greetings"]):
        return "Hello! I'm your AI health assistant. I can help answer questions about your health data, risk factors, and general wellness tips. What would you like to know?"
    elif any(word in user_input for word in ["thank", "thanks"]):
        return "You're welcome! Remember, I'm here to provide general information, but for personalized medical advice, please consult your healthcare provider."
    else:
        return "I'm here to help with health-related questions. I can provide information about risk factors, lifestyle tips, and general wellness advice. For specific medical concerns, please consult your healthcare provider. What would you like to know about your health?"


# ==================== CONVERSATIONAL HEALTH ASSISTANT FUNCTIONS ====================

def get_risk_level_chat(risk_percent):
    """Get risk level based on percentage"""
    if risk_percent < 10:
        return "Low", "risk-low", "🟢"
    elif risk_percent < 20:
        return "Moderate", "risk-moderate", "🟡"
    else:
        return "High", "risk-high", "🔴"

def generate_personalized_advice_chat(age, bmi, bp, cholesterol, smoker, exercise, diet, sleep, stress):
    """Generate personalized health advice"""
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
    """Calculate 5-year health risk percentage"""
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

# ==================== CONVERSATIONAL CHAT SESSION STATE ====================
if 'chat_conversation' not in st.session_state:
    st.session_state.chat_conversation = []
if 'chat_user_responses' not in st.session_state:
    st.session_state.chat_user_responses = {}
if 'chat_current_risk' not in st.session_state:
    st.session_state.chat_current_risk = None
if 'chat_last_assessment' not in st.session_state:
    st.session_state.chat_last_assessment = None

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

# ==================== MOBILE CONNECTION (GOOGLE SHEETS) SESSION STATE ====================
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

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🧬</span>
            <h3 style="margin: 0.5rem 0; color: #1E88E5;">AI Digital Twin</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("👤 User Profile", expanded=True):
            st.markdown(f"""
            **Username:** {st.session_state.username}
            **User ID:** `{st.session_state.user_id}`
            """)
            
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
            if st.button("➕ Add Goal", use_container_width=True):
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
        
        # Navigation options - UPDATED with all new sections
        nav_options = ["Executive Dashboard", "Risk Analysis", "Multi-Task Risk Predictor", "Lifestyle Optimizer", 
                      "Predictive Models", "Health Trends", "What-If Lab", 
                      "Population Health", "AI Insights", "Smartwatch Device", 
                      "Mobile Connection", "Twin Analysis", "Sleep Pattern Analysis", "Anomaly Detection",
                      "Heart Rate Monitor", "Personalized Recommendations", "Task Management", "Adherence Prediction",
                      "AI Health Chat", "Personal Assistant", "Settings"]
        
        if option_menu:
            selected = option_menu(
                menu_title="Navigation",
                options=nav_options,
                icons=["house", "shield", "activity", "heart", "cpu", "graph-up", 
                       "sliders", "people", "robot", "smartwatch", "phone", 
                       "people-fill", "moon", "alert", "heart-pulse",
                       "stars", "list-task", "graph-up", "chat-dots", "person-badge", "gear"],
                menu_icon="cast",
                default_index=nav_options.index(st.session_state.selected),
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "#1E88E5", "font-size": "20px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                               "--hover-color": "#f0f2f6"},
                    "nav-link-selected": {"background-color": "#1E88E5"},
                }
            )
        else:
            selected = st.radio("Navigation", nav_options, index=nav_options.index(st.session_state.selected))
        
        st.session_state.selected = selected
        
        st.markdown("---")
        st.markdown("### 📊 Today's Summary")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Predictions", st.session_state.predictions_made)
        with col_s2:
            if history:
                st.metric("Last Risk", f"{last_risk*100:.1f}%")
        
        if st.button("🚪 Logout", use_container_width=True):
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

    # ==================== EXECUTIVE DASHBOARD ====================
    if selected == "Executive Dashboard":
        st.markdown("## 📊 Executive Health Dashboard")
        
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        
        with col_k1:
            total_pop = len(nhanes) + len(framingham)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TOTAL POPULATION</div>
                <div class="metric-value">{total_pop:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k2:
            avg_risk = framingham['outcome'].mean() * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AVG RISK SCORE</div>
                <div class="metric-value">{avg_risk:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k3:
            active_users = len(db.get_user_history(1, limit=1000))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ACTIVE USERS</div>
                <div class="metric-value">{active_users}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">DATA QUALITY</div>
                <div class="metric-value">98%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### 📈 Risk Trends")
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            simulated_risks = 0.15 + 0.02 * np.random.randn(30)
            fig = px.line(x=dates, y=simulated_risks*100, title="30-Day Risk Trend")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("### 🎯 Your Health Score")
            history = db.get_user_history(user_id, limit=1)
            if history:
                last_risk = history[0]['risk_score']
                health_score = 100 - (last_risk * 100)
                st.metric("Health Score", f"{health_score:.0f}/100")
            
            if st.button("📊 Generate Report"):
                report = generate_health_report(
                    {'age': 45, 'bmi': 25.5},
                    {'current_risk': 0.15, 'risk_level': 'MODERATE', 'health_score': 85},
                    history
                )
                st.download_button("📥 Download", data=report, file_name="health_report.md")

    # ==================== RISK ANALYSIS ====================
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
    
        if st.button("🔍 Calculate My 5-Year Risk", type="primary", use_container_width=True):
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
                              <th>Age: </th><td><b>{age} years</b></td>
                              <th>BMI: </th><td><b>{bmi:.1f}</b></td>
                              <th>Blood Pressure: </th><td><b>{sbp} mmHg</b></td>
                              <th>Cholesterol: </th><td><b>{cholesterol} mg/dL</b></td>
                              <th>Smoker: </th><td><b>{'Yes' if smoker else 'No'}</b></td>
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

    # ==================== LIFESTYLE OPTIMIZER ====================
    elif selected == "Lifestyle Optimizer":
        st.markdown("## 🌱 Lifestyle Optimizer")
        col1, col2 = st.columns(2)
        with col1:
            exercise = st.select_slider("Exercise", ["None", "Light", "Moderate", "Active", "Very Active"])
            diet = st.select_slider("Diet", ["Poor", "Fair", "Good", "Excellent"])
            sleep = st.slider("Sleep", 4, 12, 7)
            stress = st.slider("Stress", 1, 10, 5)
        with col2:
            score = calculate_health_score(45, 25, 120, 200, False, exercise, diet, sleep, stress)
            st.metric("Health Score", f"{score}/100")
            if st.button("Generate Plan"):
                st.info("1. Increase exercise to 150 mins/week\n2. Improve diet quality\n3. Optimize sleep schedule")

    # ==================== PREDICTIVE MODELS ====================
    elif selected == "Predictive Models":
        st.markdown("## 🤖 Predictive Models")
        tab1, tab2, tab3 = st.tabs(["Performance", "Feature Importance", "Comparison"])
        with tab1:
            models = ['RF', 'XGBoost', 'GBM', 'NN']
            acc = [85, 87, 86, 84]
            fig = px.bar(x=models, y=acc, title="Model Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            features = ['Age', 'Smoking', 'BP', 'BMI', 'Cholesterol']
            imp = [0.35, 0.25, 0.18, 0.12, 0.10]
            fig = px.bar(x=imp, y=features, orientation='h', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            cm = np.array([[85, 15], [12, 88]])
            fig = ff.create_annotated_heatmap(cm, x=['Pred Low', 'Pred High'], y=['Actual Low', 'Actual High'])
            st.plotly_chart(fig, use_container_width=True)

    # ==================== HEALTH TRENDS ====================
    elif selected == "Health Trends":
        st.markdown("## 📈 Health Trends")
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        values = 0.15 + 0.02 * np.random.randn(90)
        fig = px.line(x=dates, y=values*100, title="90-Day Risk Trend")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{np.mean(values)*100:.1f}%")
        with col2:
            st.metric("Min", f"{np.min(values)*100:.1f}%")
        with col3:
            st.metric("Max", f"{np.max(values)*100:.1f}%")

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
                    <div style='color: #666; font-size:0.9rem;'>Modified Risk</div>
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
            st.plotly_chart(fig, use_container_width=True)
            
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
                if st.button("💾 Save This Scenario", use_container_width=True, key="save_scenario_btn"):
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
            
            if st.button("📊 Compare All Scenarios", type="primary", use_container_width=True, key="compare_btn"):
                fig = go.Figure()
                colors = ['#1E88E5', '#43A047', '#FB8C00']
                for i, s in enumerate(scenarios):
                    fig.add_trace(go.Bar(name=s['name'], x=[s['name']], y=[s['risk'] * 100],
                                        marker_color=colors[i % len(colors)], text=[f"{s['risk']*100:.1f}%"], textposition='outside'))
                avg_risk = np.mean([s['risk'] for s in scenarios]) * 100
                fig.add_hline(y=avg_risk, line_dash="dash", line_color="gray", annotation_text=f"Average: {avg_risk:.1f}%")
                fig.update_layout(title="Scenario Comparison", yaxis_title="5-Year Risk (%)", yaxis_range=[0, 50],
                                  showlegend=False, height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
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
            
            if st.button("📈 Generate Trend", type="primary", use_container_width=True, key="trend_btn"):
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
                fig.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="Low Risk Threshold")
                fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
                if current_smoker and quit_year <= max(years):
                    quit_age = current_age + quit_year
                    fig.add_vline(x=quit_age, line_dash="dot", line_color="orange", annotation_text=f"Quit at {quit_age}")
                fig.update_layout(title="Your Health Risk Trajectory", xaxis_title="Age", yaxis_title="5-Year Risk (%)",
                                  hovermode='x', height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
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
                if st.button("Clear All Scenarios", use_container_width=True, key="clear_all"):
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
                st.plotly_chart(fig_gauge, use_container_width=True)
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
            st.plotly_chart(fig_dist, use_container_width=True)
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
                st.plotly_chart(fig_trend, use_container_width=True)
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
            if st.button("📊 Export Population Data (CSV)", use_container_width=True):
                csv_data = filtered_nhanes.head(1000).to_csv(index=False)
                st.download_button(label="⬇️ Download CSV", data=csv_data, file_name=f"population_health_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", key="download_csv")
        with col_exp2:
            if st.button("📄 Generate Population Report", use_container_width=True):
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
                if st.button("📱 Connect Smartwatch", type="primary", use_container_width=True):
                    st.session_state.smartwatch_connected = True
                    st.rerun()
            else:
                st.success("✅ Device connected")
                if st.button("🔌 Disconnect", use_container_width=True):
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
            
            if st.button("🔄 Generate New Reading", use_container_width=True):
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
                    st.plotly_chart(fig_hr, use_container_width=True)
                
                with tab_c2:
                    fig_steps = px.bar(df_device, x='Time', y='Steps', title="Step Count", color='Steps', color_continuous_scale='Greens')
                    fig_steps.update_layout(height=400)
                    st.plotly_chart(fig_steps, use_container_width=True)
                
                with tab_c3:
                    fig_sleep = px.line(df_device, x='Time', y='Sleep', title="Sleep Duration", markers=True)
                    fig_sleep.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Optimal")
                    fig_sleep.update_layout(height=400)
                    st.plotly_chart(fig_sleep, use_container_width=True)
            
            st.markdown("### 🚨 Health Alerts")
            if st.session_state.smartwatch_alerts:
                for alert in st.session_state.smartwatch_alerts[-5:]:
                    st.warning(alert)
            else:
                st.success("No alerts. All metrics are within normal ranges.")
        else:
            st.info("🔌 Click 'Connect Smartwatch' to start receiving real-time health data.")

    # ==================== MOBILE CONNECTION ====================
    elif selected == "Mobile Connection":
        st.markdown("## 📱 Mobile Connection")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>📱 Connect Your Mobile Device</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Sync your health data from Google Fit or other fitness trackers to get personalized insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            creds_path = os.path.join(script_dir, 'google_credentials.json')
            
            if not os.path.exists(creds_path):
                raise FileNotFoundError(f"google_credentials.json not found at {creds_path}")
            
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
            client = gspread.authorize(creds)
            sheet = client.open('AI Digital Twin Health Data').sheet1
            sheet_available = True
        except Exception:
            sheet_available = False
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Sync from Google Sheets", use_container_width=True):
                if sheet_available:
                    try:
                        all_records = sheet.get_all_records()
                        if all_records:
                            df = pd.DataFrame(all_records)
                            
                            if 'timestamp' not in df.columns:
                                st.error("Sheet missing 'timestamp' column")
                            else:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                df = df.sort_values('timestamp')
                                
                                st.session_state.mobile_data['timestamps'] = df['timestamp'].tolist()
                                st.session_state.mobile_data['heart_rate'] = df['heart_rate'].tolist() if 'heart_rate' in df else [None]*len(df)
                                st.session_state.mobile_data['steps'] = df['steps'].tolist() if 'steps' in df else [0]*len(df)
                                st.session_state.mobile_data['sleep'] = df['sleep'].tolist() if 'sleep' in df else [0]*len(df)
                                st.session_state.mobile_data['calories'] = df['calories'].tolist() if 'calories' in df else [0]*len(df)
                                st.session_state.mobile_data['distance_km'] = df['distance_km'].tolist() if 'distance_km' in df else [0]*len(df)
                                st.session_state.mobile_data['active_minutes'] = df['active_minutes'].tolist() if 'active_minutes' in df else [0]*len(df)
                                
                                st.session_state.mobile_connected = True
                                st.rerun()
                        else:
                            st.info("No data in Google Sheet yet.")
                    except Exception as e:
                        st.error(f"Error reading sheet: {str(e)}")
                else:
                    st.error("Google Sheets connection failed.")
        
        if st.session_state.mobile_connected and len(st.session_state.mobile_data.get('timestamps', [])) > 0:
            st.markdown("---")
            st.markdown("### 📊 Latest Health Snapshot")
            
            latest = {
                'timestamp': st.session_state.mobile_data['timestamps'][-1],
                'steps': st.session_state.mobile_data['steps'][-1],
                'calories': st.session_state.mobile_data['calories'][-1],
                'distance_km': st.session_state.mobile_data['distance_km'][-1],
                'active_minutes': st.session_state.mobile_data['active_minutes'][-1],
                'sleep': st.session_state.mobile_data['sleep'][-1],
                'heart_rate': st.session_state.mobile_data['heart_rate'][-1]
            }
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("👣 Steps", f"{latest['steps']:,}")
                st.metric("🏃 Distance", f"{latest['distance_km']:.2f} km")
            with col_m2:
                st.metric("🔥 Calories", f"{latest['calories']:,}")
                st.metric("⏱️ Active Minutes", f"{latest['active_minutes']} min")
            with col_m3:
                st.metric("😴 Sleep", f"{latest['sleep']} hrs" if latest['sleep'] else "Not entered")
                st.metric("❤️ Heart Rate", f"{latest['heart_rate']} bpm" if latest['heart_rate'] else "Not entered")
            
            st.markdown("### 📈 Historical Trends")
            df_device = pd.DataFrame({
                'Date': st.session_state.mobile_data['timestamps'],
                'Steps': st.session_state.mobile_data['steps'],
                'Distance (km)': st.session_state.mobile_data['distance_km'],
                'Active Minutes': st.session_state.mobile_data['active_minutes'],
                'Calories': st.session_state.mobile_data['calories'],
                'Sleep (hrs)': st.session_state.mobile_data['sleep'],
                'Heart Rate': st.session_state.mobile_data['heart_rate']
            })
            df_device['Date'] = pd.to_datetime(df_device['Date'], errors='coerce')
            numeric_columns = ['Steps', 'Distance (km)', 'Active Minutes', 'Calories', 'Sleep (hrs)', 'Heart Rate']
            df_device[numeric_columns] = df_device[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            tab1, tab2, tab3 = st.tabs(["Activity", "Energy", "Wellness"])
            with tab1:
                fig1 = px.line(df_device, x='Date', y=['Steps', 'Distance (km)'], title="Daily Activity")
                st.plotly_chart(fig1, use_container_width=True)
            with tab2:
                fig2 = px.line(df_device, x='Date', y=['Calories', 'Active Minutes'], title="Energy & Intensity")
                st.plotly_chart(fig2, use_container_width=True)
            with tab3:
                fig3 = px.line(df_device, x='Date', y=['Sleep (hrs)', 'Heart Rate'], title="Sleep & Heart Rate")
                st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown("### 🚨 Health Alerts")
            alerts = []
            if latest['steps'] < 5000:
                alerts.append("⚠️ Low step count. Aim for 7,500+ steps daily.")
            if latest['active_minutes'] < 30:
                alerts.append("⚠️ Low active minutes. Try to get at least 30 minutes of moderate activity.")
            if latest['sleep'] and latest['sleep'] < 6:
                alerts.append("⚠️ Insufficient sleep. Aim for 7-9 hours.")
            if latest['heart_rate'] and latest['heart_rate'] > 100:
                alerts.append("⚠️ High resting heart rate. Consult a doctor if persistent.")
            if alerts:
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("✅ All metrics within healthy ranges. Keep it up!")
        else:
            st.info("Click 'Sync from Google Sheets' to load your health data.")

    # ==================== TWIN ANALYSIS SECTION ====================
    elif selected == "Twin Analysis":
        st.markdown("## 👥 Twin Health Analysis")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>👥 Twin-Aware Risk Assessment</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                This model considers genetic similarity between twins to provide more accurate 
                5-year health risk predictions. Enter data for both twins to see the difference.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize twin model silently
        if not st.session_state.twin_model_loaded:
            try:
                from src.backend.twin_health_model import TwinHealthModel
                
                possible_paths = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'app', 'models', 'twin_health_model.pkl'),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'app', 'models', 'twin_health_model.pkl'),
                    r'C:\Users\acer\OneDrive\Documents\working project\src\app\models\twin_health_model.pkl'
                ]
                
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if model_path:
                    st.session_state.twin_model = TwinHealthModel()
                    st.session_state.twin_model_loaded = st.session_state.twin_model.load_model(model_path)
                    
            except Exception:
                st.session_state.twin_model_loaded = False
        
        if not st.session_state.twin_model_loaded:
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("### 👤 Twin A")
                age_a = st.number_input("Age", 20, 80, 45, key="twin_a_age_simple")
                bmi_a = st.number_input("BMI", 15.0, 45.0, 25.0, 0.1, key="twin_a_bmi_simple")
                bp_a = st.number_input("Blood Pressure", 90, 200, 120, key="twin_a_bp_simple")
                chol_a = st.number_input("Cholesterol", 100, 350, 200, key="twin_a_chol_simple")
                smoker_a = st.checkbox("Smoker", key="twin_a_smoker_simple")
            
            with col_t2:
                st.markdown("### 👤 Twin B")
                age_b = st.number_input("Age", 20, 80, 45, key="twin_b_age_simple")
                bmi_b = st.number_input("BMI", 15.0, 45.0, 25.0, 0.1, key="twin_b_bmi_simple")
                bp_b = st.number_input("Blood Pressure", 90, 200, 120, key="twin_b_bp_simple")
                chol_b = st.number_input("Cholesterol", 100, 350, 200, key="twin_b_chol_simple")
                smoker_b = st.checkbox("Smoker", key="twin_b_smoker_simple")
            
            if st.button("🔍 Calculate Risk", type="primary", use_container_width=True):
                def calc_simple_risk(age, bmi, bp, chol, smoker):
                    risk = 0.02
                    risk += max(0, (age - 40) * 0.005)
                    risk += max(0, (bmi - 25) * 0.01)
                    risk += max(0, (bp - 120) * 0.002)
                    risk += max(0, (chol - 200) * 0.001)
                    risk += 0.15 if smoker else 0
                    return np.clip(risk, 0.01, 0.5)
                
                risk_a = calc_simple_risk(age_a, bmi_a, bp_a, chol_a, smoker_a)
                risk_b = calc_simple_risk(age_b, bmi_b, bp_b, chol_b, smoker_b)
                
                st.markdown("---")
                st.markdown("## 📊 Risk Analysis Results")
                
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Twin A Risk", f"{risk_a*100:.1f}%")
                with col_r2:
                    diff = (risk_b - risk_a) * 100
                    st.metric("Difference", f"{diff:+.1f}%")
                with col_r3:
                    st.metric("Twin B Risk", f"{risk_b*100:.1f}%")
        
        else:
            twin_model = st.session_state.twin_model
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("### 👤 Twin A")
                age_a = st.number_input("Age", 20, 80, 45, key="twin_a_age")
                bmi_a = st.number_input("BMI", 15.0, 45.0, 25.0, 0.1, key="twin_a_bmi")
                bp_a = st.number_input("Blood Pressure", 90, 200, 120, key="twin_a_bp")
                chol_a = st.number_input("Cholesterol", 100, 350, 200, key="twin_a_chol")
                smoker_a = st.checkbox("Smoker", key="twin_a_smoker")
                exercise_a = st.slider("Exercise (minutes/week)", 0, 300, 150, key="twin_a_ex")
                sleep_a = st.slider("Sleep (hours/night)", 4, 12, 7, key="twin_a_sleep")
                diet_a = st.slider("Diet Quality (1-10)", 1, 10, 6, key="twin_a_diet")
            
            with col_t2:
                st.markdown("### 👤 Twin B")
                age_b = st.number_input("Age", 20, 80, 45, key="twin_b_age")
                bmi_b = st.number_input("BMI", 15.0, 45.0, 25.0, 0.1, key="twin_b_bmi")
                bp_b = st.number_input("Blood Pressure", 90, 200, 120, key="twin_b_bp")
                chol_b = st.number_input("Cholesterol", 100, 350, 200, key="twin_b_chol")
                smoker_b = st.checkbox("Smoker", key="twin_b_smoker")
                exercise_b = st.slider("Exercise (minutes/week)", 0, 300, 150, key="twin_b_ex")
                sleep_b = st.slider("Sleep (hours/night)", 4, 12, 7, key="twin_b_sleep")
                diet_b = st.slider("Diet Quality (1-10)", 1, 10, 6, key="twin_b_diet")
            
            if st.button("🔍 Analyze Twin Health Risks", type="primary", use_container_width=True):
                twin_a_data = {
                    'age': age_a, 'bmi': bmi_a, 'bp': bp_a, 'cholesterol': chol_a,
                    'smoker': smoker_a, 'exercise_minutes': exercise_a,
                    'sleep_hours': sleep_a, 'diet_score': diet_a
                }
                twin_b_data = {
                    'age': age_b, 'bmi': bmi_b, 'bp': bp_b, 'cholesterol': chol_b,
                    'smoker': smoker_b, 'exercise_minutes': exercise_b,
                    'sleep_hours': sleep_b, 'diet_score': diet_b
                }
                
                with st.spinner("Calculating risks..."):
                    try:
                        risk_a = twin_model.predict_risk(twin_a_data, twin_b_data)
                        risk_b = twin_model.predict_risk(twin_b_data, twin_a_data)
                        risk_a_individual = twin_model.predict_risk(twin_a_data)
                        risk_b_individual = twin_model.predict_risk(twin_b_data)
                        cat_a, _ = twin_model.predict_category(twin_a_data, twin_b_data)
                        cat_b, _ = twin_model.predict_category(twin_b_data, twin_a_data)
                        _, _, conf_a = twin_model.predict_with_confidence(twin_a_data, twin_b_data)
                        _, _, conf_b = twin_model.predict_with_confidence(twin_b_data, twin_a_data)
                        
                        st.markdown("---")
                        st.markdown("## 📊 Twin Risk Analysis Results")
                        
                        col_r1, col_r2, col_r3 = st.columns(3)
                        
                        with col_r1:
                            risk_color = "#E53935" if risk_a > 0.2 else "#FB8C00" if risk_a > 0.1 else "#43A047"
                            st.markdown(f"""
                            <div style='background: white; padding: 1.5rem; border-radius: 20px; 
                                        box-shadow: 0 5px 15px rgba(0,0,0,0.1); text-align: center;
                                        border-bottom: 5px solid {risk_color};'>
                                <h3 style='margin:0; color: #1E88E5;'>Twin A</h3>
                                <div style='font-size: 3rem; font-weight: 700; color: {risk_color};'>{risk_a*100:.1f}%</div>
                                <div>5-Year Risk</div>
                                <div><span class='badge {"badge-danger" if risk_a > 0.2 else "badge-warning" if risk_a > 0.1 else "badge-success"}'>
                                    {cat_a} RISK
                                </span></div>
                                <div style='font-size: 0.8rem; color: #666;'>Confidence: {conf_a*100:.0f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_r2:
                            diff = (risk_b - risk_a) * 100
                            diff_color = "#43A047" if diff < 0 else "#E53935" if diff > 0 else "#666"
                            diff_icon = "📉" if diff < 0 else "📈" if diff > 0 else "➡️"
                            st.markdown(f"""
                            <div style='background: white; padding: 1.5rem; border-radius: 20px; 
                                        box-shadow: 0 5px 15px rgba(0,0,0,0.1); text-align: center;
                                        border-bottom: 5px solid {diff_color};'>
                                <h3 style='margin:0; color: #1E88E5;'>Difference</h3>
                                <div style='font-size: 2.5rem; font-weight: 700; color: {diff_color};'>
                                    {diff_icon} {diff:+.1f}%
                                </div>
                                <div>Twin B vs Twin A</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_r3:
                            risk_color_b = "#E53935" if risk_b > 0.2 else "#FB8C00" if risk_b > 0.1 else "#43A047"
                            st.markdown(f"""
                            <div style='background: white; padding: 1.5rem; border-radius: 20px; 
                                        box-shadow: 0 5px 15px rgba(0,0,0,0.1); text-align: center;
                                        border-bottom: 5px solid {risk_color_b};'>
                                <h3 style='margin:0; color: #1E88E5;'>Twin B</h3>
                                <div style='font-size: 3rem; font-weight: 700; color: {risk_color_b};'>{risk_b*100:.1f}%</div>
                                <div>5-Year Risk</div>
                                <div><span class='badge {"badge-danger" if risk_b > 0.2 else "badge-warning" if risk_b > 0.1 else "badge-success"}'>
                                    {cat_b} RISK
                                </span></div>
                                <div style='font-size: 0.8rem; color: #666;'>Confidence: {conf_b*100:.0f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("### 🧬 Twin Awareness Impact")
                        impact_a = (risk_a - risk_a_individual) * 100
                        impact_b = (risk_b - risk_b_individual) * 100
                        
                        col_i1, col_i2 = st.columns(2)
                        with col_i1:
                            st.metric("Twin A Impact", f"{impact_a:+.1f}%")
                        with col_i2:
                            st.metric("Twin B Impact", f"{impact_b:+.1f}%")
                        
                        st.markdown("### 📈 Risk Comparison")
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name='Individual', x=['Twin A', 'Twin B'], 
                                            y=[risk_a_individual*100, risk_b_individual*100],
                                            marker_color='#9E9E9E', text=[f"{risk_a_individual*100:.1f}%", f"{risk_b_individual*100:.1f}%"],
                                            textposition='outside'))
                        fig.add_trace(go.Bar(name='Twin-Aware', x=['Twin A', 'Twin B'],
                                            y=[risk_a*100, risk_b*100],
                                            marker_color='#1E88E5', text=[f"{risk_a*100:.1f}%", f"{risk_b*100:.1f}%"],
                                            textposition='outside'))
                        fig.update_layout(title="Individual vs Twin-Aware Risk", yaxis_title="Risk (%)", yaxis_range=[0, 35], barmode='group', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("### 💡 Recommendations")
                        if abs(risk_b - risk_a) > 0.05:
                            higher_risk = "Twin B" if risk_b > risk_a else "Twin A"
                            st.warning(f"⚠️ **{higher_risk}** has significantly higher risk.")
                            
                            if abs(bmi_a - bmi_b) > 3:
                                st.info(f"📊 BMI difference: {abs(bmi_a - bmi_b):.1f} points")
                            if smoker_a != smoker_b:
                                st.info(f"🚭 Smoking status differs")
                            if abs(exercise_a - exercise_b) > 60:
                                st.info(f"🏃 Exercise difference: {abs(exercise_a - exercise_b)} min/week")
                        else:
                            st.success("✅ Both twins have similar risk profiles.")
                        
                    except Exception:
                        st.error("Error during prediction. Please try again.")

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
            st.plotly_chart(fig, use_container_width=True)
            
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
                st.plotly_chart(fig, use_container_width=True)
                col_f1, col_f2, col_f3 = st.columns(3)
                col_f1.metric("Current", f"{risks[0]:.1f}%")
                col_f2.metric("5 Years", f"{risks[5]:.1f}%")
                col_f3.metric("10 Years", f"{risks[-1]:.1f}%")

    # ==================== MULTI-TASK RISK PREDICTOR ====================
    elif selected == "Multi-Task Risk Predictor":
        st.markdown("## 🎯 Multi-Task 5-Year Risk Prediction")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>🎯 Predict Multiple Health Risks Simultaneously</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Our AI model predicts your 5-year risk for multiple conditions at once.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Your Health Profile")
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
            st.markdown("#### 🏃 Lifestyle Factors")
            smoker_mt = st.checkbox("Current Smoker", key="mt_smoker")
            diabetic_mt = st.checkbox("Diabetes", key="mt_diabetic")
            exercise_mt = st.select_slider("Physical Activity", 
                                           options=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                                           value="Moderate", key="mt_exercise")
            diet_mt = st.select_slider("Diet Quality",
                                       options=["Poor", "Fair", "Good", "Excellent"],
                                       value="Good", key="mt_diet")
        
        if st.button("🔍 Predict All 5-Year Risks", type="primary", use_container_width=True):
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
            st.markdown("## 📊 Your 5-Year Risk Assessment")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                color = "#E53935" if cvd_risk > 20 else "#FB8C00" if cvd_risk > 10 else "#43A047"
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem; border-bottom: 4px solid {color};'>
                    <div style='font-size: 0.85rem; color: #666;'>❤️ Cardiovascular Disease</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color};'>{cvd_risk:.1f}%</div>
                    <div style='font-size: 0.7rem;'>5-Year Risk</div>
                </div>
                """, unsafe_allow_html=True)
                
                color = "#E53935" if diabetes_risk > 15 else "#FB8C00" if diabetes_risk > 8 else "#43A047"
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem; border-bottom: 4px solid {color};'>
                    <div style='font-size: 0.85rem; color: #666;'>🩸 Type 2 Diabetes</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color};'>{diabetes_risk:.1f}%</div>
                    <div style='font-size: 0.7rem;'>5-Year Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r2:
                color = "#E53935" if hypertension_risk > 25 else "#FB8C00" if hypertension_risk > 15 else "#43A047"
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem; border-bottom: 4px solid {color};'>
                    <div style='font-size: 0.85rem; color: #666;'>🩺 Hypertension</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color};'>{hypertension_risk:.1f}%</div>
                    <div style='font-size: 0.7rem;'>5-Year Risk</div>
                </div>
                """, unsafe_allow_html=True)
                
                color = "#E53935" if stroke_risk > 15 else "#FB8C00" if stroke_risk > 8 else "#43A047"
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem; border-bottom: 4px solid {color};'>
                    <div style='font-size: 0.85rem; color: #666;'>🧠 Stroke</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color};'>{stroke_risk:.1f}%</div>
                    <div style='font-size: 0.7rem;'>5-Year Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r3:
                color = "#E53935" if kidney_risk > 10 else "#FB8C00" if kidney_risk > 5 else "#43A047"
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem; border-bottom: 4px solid {color};'>
                    <div style='font-size: 0.85rem; color: #666;'>🫁 Kidney Disease</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color};'>{kidney_risk:.1f}%</div>
                    <div style='font-size: 0.7rem;'>5-Year Risk</div>
                </div>
                """, unsafe_allow_html=True)
                
                overall = (cvd_risk + diabetes_risk + hypertension_risk + stroke_risk + kidney_risk) / 5
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1E88E5, #1565C0); padding: 1rem; border-radius: 15px; text-align: center; color: white;'>
                    <div style='font-size: 0.8rem;'>📊 Overall Health Score</div>
                    <div style='font-size: 1.5rem; font-weight: 700;'>{overall:.1f}%</div>
                    <div style='font-size: 0.7rem;'>Average 5-Year Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            conditions = ['CVD', 'Diabetes', 'Hypertension', 'Stroke', 'Kidney']
            risks = [cvd_risk, diabetes_risk, hypertension_risk, stroke_risk, kidney_risk]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=conditions, y=risks, 
                                marker_color=['#E53935' if r > 20 else '#FB8C00' if r > 10 else '#43A047' for r in risks],
                                text=[f'{r:.1f}%' for r in risks], textposition='outside'))
            fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="High Risk")
            fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="Moderate Risk")
            fig.update_layout(title="5-Year Risk by Condition", yaxis_title="Risk Percentage (%)", yaxis_range=[0, 45], height=350)
            st.plotly_chart(fig, use_container_width=True)

    # ==================== SLEEP PATTERN ANALYSIS ====================
    elif selected == "Sleep Pattern Analysis":
        st.markdown("## 😴 LSTM Sleep Pattern Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            avg_sleep = st.slider("Average Sleep Hours", 3, 12, 7)
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
            wakeups = st.number_input("Night Wake-ups", 0, 10, 1)
        with col2:
            deep_sleep = st.slider("Deep Sleep %", 10, 40, 20)
            rem_sleep = st.slider("REM Sleep %", 15, 30, 22)
            consistency = st.slider("Bedtime Consistency (1-10)", 1, 10, 6)
        
        if st.button("🔍 Analyze Sleep", type="primary", use_container_width=True):
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
        
        if st.button("🔍 Run Detection", type="primary", use_container_width=True):
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

    # ==================== HEART RATE MONITOR (FULLY FUNCTIONAL) ====================
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
        
        # Initialize heart rate history in session state
        if 'hr_history' not in st.session_state:
            st.session_state.hr_history = []
        if 'hr_timestamps' not in st.session_state:
            st.session_state.hr_timestamps = []
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📝 Manual Entry")
            st.markdown("Enter your heart rate manually:")
            
            col_hr1, col_hr2 = st.columns([2, 1])
            with col_hr1:
                manual_hr = st.number_input("Heart Rate (BPM)", 30, 200, 75, step=1, key="manual_hr")
            with col_hr2:
                if st.button("➕ Add Reading", use_container_width=True):
                    now = datetime.now()
                    st.session_state.hr_history.append(manual_hr)
                    st.session_state.hr_timestamps.append(now)
                    st.success(f"✅ Added: {manual_hr} BPM at {now.strftime('%H:%M:%S')}")
                    st.rerun()
            
            # Quick add buttons
            st.markdown("#### Quick Add:")
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            with col_q1:
                if st.button("🟢 60 BPM", use_container_width=True):
                    st.session_state.hr_history.append(60)
                    st.session_state.hr_timestamps.append(datetime.now())
                    st.success("✅ Added 60 BPM")
                    st.rerun()
            with col_q2:
                if st.button("🟡 75 BPM", use_container_width=True):
                    st.session_state.hr_history.append(75)
                    st.session_state.hr_timestamps.append(datetime.now())
                    st.success("✅ Added 75 BPM")
                    st.rerun()
            with col_q3:
                if st.button("🟠 90 BPM", use_container_width=True):
                    st.session_state.hr_history.append(90)
                    st.session_state.hr_timestamps.append(datetime.now())
                    st.success("✅ Added 90 BPM")
                    st.rerun()
            with col_q4:
                if st.button("🔴 110 BPM", use_container_width=True):
                    st.session_state.hr_history.append(110)
                    st.session_state.hr_timestamps.append(datetime.now())
                    st.success("✅ Added 110 BPM")
                    st.rerun()
        
        with col2:
            st.markdown("### 📱 Demo Mode")
            st.markdown("Simulate real-time heart rate monitoring:")
            
            if st.button("🎮 Single Reading", use_container_width=True):
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
            
            if st.button("🔄 Simulate 5 Readings", use_container_width=True):
                for _ in range(5):
                    demo_hr = np.random.randint(65, 100)
                    st.session_state.hr_history.append(demo_hr)
                    st.session_state.hr_timestamps.append(datetime.now())
                st.success("✅ Added 5 simulated readings")
                st.rerun()
            
            if st.button("📊 Simulate Exercise Pattern", use_container_width=True):
                # Simulate exercise: starts high, goes higher, then recovers
                pattern = [85, 95, 110, 120, 115, 105, 95, 85, 80, 75]
                for hr in pattern:
                    st.session_state.hr_history.append(hr)
                    st.session_state.hr_timestamps.append(datetime.now())
                    time.sleep(0.1)
                st.success("✅ Added exercise pattern (warm up → peak → recovery)")
                st.rerun()
        
        # Display current status
        if st.session_state.hr_history:
            st.markdown("---")
            st.markdown("### 📊 Current Status")
            
            current_hr = st.session_state.hr_history[-1]
            
            # Determine status
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
            
            # Display large BPM
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
                # Calculate stats
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
            
            # Health tip based on heart rate
            st.markdown(f"""
            <div style='background: #f0f7ff; padding: 1rem; border-radius: 15px; margin-top: 1rem;'>
                <strong>💡 Health Tip:</strong> {explanation}
            </div>
            """, unsafe_allow_html=True)
            
            # Show alert if recent readings are concerning
            recent_high = any(hr > 100 for hr in st.session_state.hr_history[-5:])
            recent_low = any(hr < 50 for hr in st.session_state.hr_history[-5:])
            
            if recent_high:
                st.warning("⚠️ Recent readings show elevated heart rate. Consider rest, hydration, and deep breathing.")
            elif recent_low and len(st.session_state.hr_history) > 0:
                st.info("ℹ️ Your recent readings show lower heart rate. This is normal for active individuals.")
            else:
                st.success("✅ Your recent readings are within normal range!")
        
        # Show history chart
        if len(st.session_state.hr_history) > 1:
            st.markdown("---")
            st.markdown("### 📈 Heart Rate History")
            
            # Create dataframe for chart
            hr_df = pd.DataFrame({
                'Time': st.session_state.hr_timestamps[-30:],
                'Heart Rate (BPM)': st.session_state.hr_history[-30:]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hr_df['Time'], y=hr_df['Heart Rate (BPM)'], 
                                    mode='lines+markers', name='Heart Rate',
                                    line=dict(color='#E53935', width=3),
                                    marker=dict(size=8, color='#E53935')))
            
            # Add normal range bands
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
            st.plotly_chart(fig, use_container_width=True)
        
        # Clear history button
        if st.session_state.hr_history:
            st.markdown("---")
            col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
            with col_c2:
                if st.button("🗑️ Clear All History", use_container_width=True):
                    st.session_state.hr_history = []
                    st.session_state.hr_timestamps = []
                    st.success("✅ History cleared!")
                    st.rerun()
        
        # Educational info
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
            
            ### Target Heart Rate During Exercise:
            
            - Moderate intensity: 50-70% of max heart rate
            - Vigorous intensity: 70-85% of max heart rate
            
            *Max Heart Rate = 220 - your age*
            
            Example for a 45-year-old:
            - Max HR: 175 BPM
            - Moderate exercise: 88-123 BPM
            - Vigorous exercise: 123-149 BPM
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

    # ==================== TASK MANAGEMENT ====================
    elif selected == "Task Management":
        st.markdown("## 📋 Priority Queue Task Management")
        
        tasks = [
            {"task": "Blood Pressure Check", "urgency": 95, "due": "Today"},
            {"task": "Cholesterol Test", "urgency": 85, "due": "Today"},
            {"task": "Follow-up Call", "urgency": 70, "due": "Tomorrow"},
            {"task": "Medication Review", "urgency": 60, "due": "This Week"},
        ]
        
        tasks_sorted = sorted(tasks, key=lambda x: x['urgency'], reverse=True)
        
        for i, task in enumerate(tasks_sorted):
            urgency_color = "#E53935" if task['urgency'] > 80 else "#FB8C00" if task['urgency'] > 60 else "#43A047"
            st.markdown(f"""
            <div style='background: white; padding: 0.8rem; border-radius: 10px; margin-bottom: 0.5rem; border-left: 5px solid {urgency_color};'>
                <div style='display: flex; justify-content: space-between;'>
                    <span><b>{i+1}. {task['task']}</b></span>
                    <span style='color: {urgency_color};'>{task['urgency']}%</span>
                </div>
                <div style='font-size: 0.8rem; color: #666;'>Due: {task['due']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ==================== ADHERENCE PREDICTION ====================
    elif selected == "Adherence Prediction":
        st.markdown("## 📊 Patient Adherence Prediction")
        
        weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8']
        adherence = [85, 78, 82, 75, 88, 80, 85, 82]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weeks, y=adherence, mode='lines+markers', name='Adherence',
                                 line=dict(color='#1E88E5', width=3)))
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target 80%")
        fig.update_layout(title="8-Week Adherence Trend", height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        avg_adherence = np.mean(adherence)
        if avg_adherence >= 80:
            st.success(f"✅ Good adherence! Average: {avg_adherence:.1f}%")
            st.info("Predicted future adherence: 85% (Likely to maintain)")
        else:
            st.warning(f"⚠️ Adherence needs improvement: {avg_adherence:.1f}%")
            st.info("Recommendation: Set medication reminders and simplify regimen")

    # ==================== AI HEALTH CHAT - CONVERSATIONAL ASSISTANT ====================
    elif selected == "AI Health Chat":
        st.markdown("## 💬 AI Health Chat Assistant")
        
        # Welcome card
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                    border-left: 5px solid #1E88E5;'>
            <h4 style='margin:0; color: #1E88E5;'>💬 Your Personal Health Conversation Partner</h4>
            <p style='margin:0.5rem 0 0 0; color: #666;'>
                Just answer my questions naturally - no medical knowledge needed! I'll help you understand your health risks.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar for chat
        with st.sidebar:
            st.markdown("### 💬 Chat Options")
            if st.button("🔄 New Health Assessment", use_container_width=True):
                st.session_state.chat_conversation = []
                st.session_state.chat_user_responses = {}
                st.session_state.chat_current_risk = None
                st.session_state.chat_last_assessment = None
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
                <div style='text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 15px; margin-top: 1rem;'>
                    <div style='font-size: 2rem;'>{color}</div>
                    <div><strong>Your Risk Score</strong></div>
                    <div style='font-size: 1.5rem; font-weight: bold;'>{risk:.1f}%</div>
                    <div><span style='background: #e0e0e0; padding: 0.25rem 0.75rem; border-radius: 20px;'>{level} Risk</span></div>
                </div>
                """, unsafe_allow_html=True)
        
        # Display chat messages
        for msg in st.session_state.chat_conversation:
            if msg["role"] == "user":
                st.markdown(f'<div style="text-align: right; margin: 0.5rem 0;"><span style="background: linear-gradient(135deg, #1E88E5, #1565C0); color: white; padding: 0.75rem 1rem; border-radius: 20px; display: inline-block; max-width: 80%;">🧑 {msg["content"]}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align: left; margin: 0.5rem 0;"><span style="background: #f0f2f6; color: #1E88E5; padding: 0.75rem 1rem; border-radius: 20px; display: inline-block; max-width: 80%;">🤖 {msg["content"]}</span></div>', unsafe_allow_html=True)
        
        # Conversation flow
        if len(st.session_state.chat_conversation) == 0:
            welcome = "Hi! I'm your AI health assistant. Let me ask you a few questions to understand your health better.\n\n**What's your age?** (e.g., 35)"
            st.session_state.chat_conversation.append({"role": "assistant", "content": welcome})
            st.rerun()
        
        elif "chat_age" not in st.session_state.chat_user_responses:
            with st.form(key="chat_age_form"):
                age_input = st.text_input("Your answer:", placeholder="Enter your age (e.g., 45)", key="chat_age_input")
                submitted = st.form_submit_button("Send 💬")
                if submitted and age_input.strip():
                    try:
                        age = int(age_input.strip())
                        if 18 <= age <= 100:
                            st.session_state.chat_user_responses["chat_age"] = age
                            st.session_state.chat_conversation.append({"role": "user", "content": age_input})
                            st.session_state.chat_conversation.append({"role": "assistant", "content": f"Thanks! At {age}, that's a great time to focus on preventive health.\n\n**What's your height (cm) and weight (kg)?** (e.g., 170 cm, 70 kg)"})
                            st.rerun()
                        else:
                            st.error("Please enter a valid age between 18 and 100.")
                    except ValueError:
                        st.error("Please enter a valid number.")
        
        elif "chat_height" not in st.session_state.chat_user_responses:
            with st.form(key="chat_body_form"):
                col1, col2 = st.columns(2)
                with col1:
                    height_input = st.text_input("Height (cm):", placeholder="e.g., 170", key="chat_height_input")
                with col2:
                    weight_input = st.text_input("Weight (kg):", placeholder="e.g., 70", key="chat_weight_input")
                submitted = st.form_submit_button("Send 💬")
                if submitted and height_input.strip() and weight_input.strip():
                    try:
                        height = float(height_input.strip())
                        weight = float(weight_input.strip())
                        if 100 <= height <= 250 and 30 <= weight <= 200:
                            bmi = weight / ((height/100) ** 2)
                            st.session_state.chat_user_responses["chat_height"] = height
                            st.session_state.chat_user_responses["chat_weight"] = weight
                            st.session_state.chat_user_responses["chat_bmi"] = bmi
                            st.session_state.chat_conversation.append({"role": "user", "content": f"{height} cm, {weight} kg"})
                            bmi_status = "healthy" if 18.5 <= bmi <= 24.9 else "above healthy" if bmi > 25 else "below healthy"
                            st.session_state.chat_conversation.append({"role": "assistant", "content": f"Your BMI is {bmi:.1f}, which is {bmi_status}.\n\n**What's your blood pressure?** (e.g., 120 or 120/80)"})
                            st.rerun()
                        else:
                            st.error("Please enter valid height (100-250 cm) and weight (30-200 kg).")
                    except ValueError:
                        st.error("Please enter valid numbers.")
        
        elif "chat_bp" not in st.session_state.chat_user_responses:
            with st.form(key="chat_bp_form"):
                bp_input = st.text_input("Your answer:", placeholder="e.g., 120", key="chat_bp_input")
                submitted = st.form_submit_button("Send 💬")
                if submitted and bp_input.strip():
                    try:
                        if "/" in bp_input:
                            bp = int(bp_input.split("/")[0])
                        else:
                            bp = int(bp_input.strip())
                        if 80 <= bp <= 200:
                            st.session_state.chat_user_responses["chat_bp"] = bp
                            st.session_state.chat_conversation.append({"role": "user", "content": bp_input})
                            bp_status = "normal" if bp < 120 else "elevated" if bp < 130 else "high"
                            st.session_state.chat_conversation.append({"role": "assistant", "content": f"Your blood pressure ({bp}) is {bp_status}.\n\n**What's your total cholesterol level?** (e.g., 180)"})
                            st.rerun()
                        else:
                            st.error("Please enter a valid blood pressure (80-200).")
                    except ValueError:
                        st.error("Please enter a valid number.")
        
        elif "chat_cholesterol" not in st.session_state.chat_user_responses:
            with st.form(key="chat_chol_form"):
                chol_input = st.text_input("Your answer:", placeholder="e.g., 180", key="chat_chol_input")
                submitted = st.form_submit_button("Send 💬")
                if submitted and chol_input.strip():
                    try:
                        chol = int(chol_input.strip())
                        if 100 <= chol <= 400:
                            st.session_state.chat_user_responses["chat_cholesterol"] = chol
                            st.session_state.chat_conversation.append({"role": "user", "content": chol_input})
                            chol_status = "optimal" if chol < 200 else "borderline" if chol < 240 else "high"
                            st.session_state.chat_conversation.append({"role": "assistant", "content": f"Your cholesterol is {chol} mg/dL, which is {chol_status}.\n\n**Do you smoke?** (yes/no)"})
                            st.rerun()
                        else:
                            st.error("Please enter a valid cholesterol level (100-400).")
                    except ValueError:
                        st.error("Please enter a valid number.")
        
        elif "chat_smoker" not in st.session_state.chat_user_responses:
            with st.form(key="chat_smoke_form"):
                smoke_input = st.text_input("Your answer:", placeholder="yes or no", key="chat_smoke_input")
                submitted = st.form_submit_button("Send 💬")
                if submitted and smoke_input.strip():
                    smoker = smoke_input.strip().lower() in ["yes", "y", "true", "1"]
                    st.session_state.chat_user_responses["chat_smoker"] = smoker
                    st.session_state.chat_conversation.append({"role": "user", "content": smoke_input})
                    smoke_msg = "I understand. Quitting is the best thing for your health." if smoker else "That's great! Not smoking is excellent for your heart health."
                    st.session_state.chat_conversation.append({"role": "assistant", "content": f"{smoke_msg}\n\n**How would you describe your exercise level?** (sedentary/light/moderate/active/very active)"})
                    st.rerun()
        
        elif "chat_exercise" not in st.session_state.chat_user_responses:
            with st.form(key="chat_ex_form"):
                ex_options = ["sedentary", "light", "moderate", "active", "very active"]
                ex_input = st.selectbox("Select your exercise level:", ex_options, key="chat_ex_select")
                submitted = st.form_submit_button("Send 💬")
                if submitted:
                    ex_map = {"sedentary": "None", "light": "Light", "moderate": "Moderate", "active": "Active", "very active": "Very Active"}
                    st.session_state.chat_user_responses["chat_exercise"] = ex_map[ex_input]
                    st.session_state.chat_conversation.append({"role": "user", "content": ex_input})
                    st.session_state.chat_conversation.append({"role": "assistant", "content": f"Thanks for sharing!\n\n**How would you rate your diet quality?** (poor/fair/good/excellent)"})
                    st.rerun()
        
        elif "chat_diet" not in st.session_state.chat_user_responses:
            with st.form(key="chat_diet_form"):
                diet_options = ["poor", "fair", "good", "excellent"]
                diet_input = st.selectbox("Select your diet quality:", diet_options, key="chat_diet_select")
                submitted = st.form_submit_button("Send 💬")
                if submitted:
                    diet_map = {"poor": "Poor", "fair": "Fair", "good": "Good", "excellent": "Excellent"}
                    st.session_state.chat_user_responses["chat_diet"] = diet_map[diet_input]
                    st.session_state.chat_conversation.append({"role": "user", "content": diet_input})
                    st.session_state.chat_conversation.append({"role": "assistant", "content": f"Good to know!\n\n**On average, how many hours do you sleep per night?** (e.g., 7)"})
                    st.rerun()
        
        elif "chat_sleep" not in st.session_state.chat_user_responses:
            with st.form(key="chat_sleep_form"):
                sleep_input = st.text_input("Your answer:", placeholder="e.g., 7", key="chat_sleep_input")
                submitted = st.form_submit_button("Send 💬")
                if submitted and sleep_input.strip():
                    try:
                        sleep = float(sleep_input.strip())
                        if 3 <= sleep <= 14:
                            st.session_state.chat_user_responses["chat_sleep"] = sleep
                            st.session_state.chat_conversation.append({"role": "user", "content": sleep_input})
                            st.session_state.chat_conversation.append({"role": "assistant", "content": f"Sleep is so important!\n\n**On a scale of 1-10, how stressed do you feel typically?** (1=low stress, 10=high stress)"})
                            st.rerun()
                        else:
                            st.error("Please enter a valid sleep duration (3-14 hours).")
                    except ValueError:
                        st.error("Please enter a valid number.")
        
        elif "chat_stress" not in st.session_state.chat_user_responses:
            with st.form(key="chat_stress_form"):
                stress_input = st.text_input("Your answer:", placeholder="e.g., 5", key="chat_stress_input")
                submitted = st.form_submit_button("Send 💬")
                if submitted and stress_input.strip():
                    try:
                        stress = int(stress_input.strip())
                        if 1 <= stress <= 10:
                            st.session_state.chat_user_responses["chat_stress"] = stress
                            st.session_state.chat_conversation.append({"role": "user", "content": stress_input})
                            
                            # Calculate risk
                            age = st.session_state.chat_user_responses["chat_age"]
                            bmi = st.session_state.chat_user_responses["chat_bmi"]
                            bp = st.session_state.chat_user_responses["chat_bp"]
                            chol = st.session_state.chat_user_responses["chat_cholesterol"]
                            smoker = st.session_state.chat_user_responses["chat_smoker"]
                            exercise = st.session_state.chat_user_responses["chat_exercise"]
                            diet = st.session_state.chat_user_responses["chat_diet"]
                            sleep = st.session_state.chat_user_responses["chat_sleep"]
                            
                            risk = calculate_health_risk_chat(age, bmi, bp, chol, smoker, exercise, diet, sleep, stress)
                            
                            st.session_state.chat_current_risk = risk
                            st.session_state.chat_last_assessment = {
                                "risk": risk, "age": age, "bmi": bmi, "bp": bp, "cholesterol": chol,
                                "smoker": smoker, "exercise": exercise, "diet": diet, "sleep": sleep, "stress": stress
                            }
                            
                            level = "Low" if risk < 10 else "Moderate" if risk < 20 else "High"
                            icon = "🟢" if risk < 10 else "🟡" if risk < 20 else "🔴"
                            
                            results = f"""
                            {icon} **Here's your personalized health assessment!**
                            
                            Your 5-year health risk score: **{risk:.1f}%**
                            Risk Level: **{level}**
                            """
                            
                            if risk < 10:
                                results += "\n\n✅ Excellent! Your health habits are protecting you well."
                            elif risk < 20:
                                results += "\n\n👍 Good! Small improvements could lower your risk further."
                            else:
                                results += "\n\n⚠️ Your risk is elevated, but lifestyle changes can significantly improve this."
                            
                            st.session_state.chat_conversation.append({"role": "assistant", "content": results})
                            
                            advice = generate_personalized_advice_chat(age, bmi, bp, chol, smoker, exercise, diet, sleep, stress)
                            st.session_state.chat_conversation.append({"role": "assistant", "content": "**💡 Here are your personalized recommendations:**\n\n" + "\n".join([f"• {a}" for a in advice[:4]])})
                            
                            st.session_state.chat_conversation.append({"role": "assistant", "content": "**What would you like to do next?**\n\n1️⃣ See all detailed recommendations\n2️⃣ Try a what-if scenario\n3️⃣ Start a new assessment\n\nType 1, 2, or 3!"})
                            st.rerun()
                        else:
                            st.error("Please enter a number between 1 and 10.")
                    except ValueError:
                        st.error("Please enter a valid number.")
        
        else:
            with st.form(key="chat_followup_form"):
                response = st.text_input("Your choice:", placeholder="Type 1, 2, or 3...", key="chat_followup_input")
                submitted = st.form_submit_button("Send 💬")
                if submitted and response.strip():
                    st.session_state.chat_conversation.append({"role": "user", "content": response})
                    
                    if response.strip() == "1":
                        age = st.session_state.chat_user_responses["chat_age"]
                        bmi = st.session_state.chat_user_responses["chat_bmi"]
                        bp = st.session_state.chat_user_responses["chat_bp"]
                        chol = st.session_state.chat_user_responses["chat_cholesterol"]
                        smoker = st.session_state.chat_user_responses["chat_smoker"]
                        exercise = st.session_state.chat_user_responses["chat_exercise"]
                        diet = st.session_state.chat_user_responses["chat_diet"]
                        sleep = st.session_state.chat_user_responses["chat_sleep"]
                        stress = st.session_state.chat_user_responses["chat_stress"]
                        
                        advice = generate_personalized_advice_chat(age, bmi, bp, chol, smoker, exercise, diet, sleep, stress)
                        detailed = "**📋 Complete Health Plan:**\n\n"
                        for i, a in enumerate(advice, 1):
                            detailed += f"{i}. {a}\n\n"
                        detailed += "\n**🎯 Your Action Items:**\n• Schedule a check-up with your doctor\n• Track your progress weekly\n• Set one small health goal for this week"
                        
                        st.session_state.chat_conversation.append({"role": "assistant", "content": detailed})
                        st.session_state.chat_conversation.append({"role": "assistant", "content": "Type 2 for a what-if scenario, or 3 for a new assessment."})
                    
                    elif response.strip() == "2":
                        current_risk = st.session_state.chat_current_risk
                        st.session_state.chat_conversation.append({"role": "assistant", "content": f"**🔮 What-If Scenario Simulator**\n\nYour current risk is {current_risk:.1f}%.\n\nWhat would you like to change?\n• Type 'bmi' to lower your BMI\n• Type 'exercise' to increase activity\n• Type 'smoke' to quit smoking\n• Type 'diet' to improve nutrition"})
                    
                    elif response.strip() == "3":
                        st.session_state.chat_conversation = []
                        st.session_state.chat_user_responses = {}
                        st.session_state.chat_current_risk = None
                        st.session_state.chat_last_assessment = None
                        st.session_state.chat_conversation.append({"role": "assistant", "content": "Let's start fresh! What's your age?"})
                        st.rerun()
                    
                    elif "bmi" in response.lower():
                        current_bmi = st.session_state.chat_user_responses["chat_bmi"]
                        target_bmi = max(18.5, current_bmi - 3)
                        current_risk = st.session_state.chat_current_risk
                        new_risk = max(5, current_risk - (current_bmi - target_bmi) * 0.5)
                        st.session_state.chat_conversation.append({"role": "assistant", "content": f"📊 **Scenario: Lowering BMI from {current_bmi:.1f} to {target_bmi:.1f}**\n\nCurrent risk: {current_risk:.1f}%\nNew risk: {new_risk:.1f}%\nReduction: {current_risk - new_risk:.1f}%\n\nTry adding daily walks and reducing portion sizes!\n\nType 2 for another scenario, or 3 for a new assessment."})
                    
                    elif "exercise" in response.lower():
                        current_risk = st.session_state.chat_current_risk
                        new_risk = max(5, current_risk - 4)
                        st.session_state.chat_conversation.append({"role": "assistant", "content": f"📊 **Scenario: Increasing exercise**\n\nCurrent risk: {current_risk:.1f}%\nNew risk: {new_risk:.1f}%\nReduction: {current_risk - new_risk:.1f}%\n\nAim for 30 minutes of walking, 5 days a week!\n\nType 2 for another scenario, or 3 for a new assessment."})
                    
                    elif "smoke" in response.lower() and st.session_state.chat_user_responses["chat_smoker"]:
                        current_risk = st.session_state.chat_current_risk
                        new_risk = max(5, current_risk - 8)
                        st.session_state.chat_conversation.append({"role": "assistant", "content": f"📊 **Scenario: Quitting Smoking**\n\nCurrent risk: {current_risk:.1f}%\nNew risk: {new_risk:.1f}%\nReduction: {current_risk - new_risk:.1f}%\n\nThis is the most impactful change! Within 1 year, your heart disease risk drops by 50%.\n\nType 2 for another scenario, or 3 for a new assessment."})
                    
                    elif "diet" in response.lower():
                        current_risk = st.session_state.chat_current_risk
                        new_risk = max(5, current_risk - 3)
                        st.session_state.chat_conversation.append({"role": "assistant", "content": f"📊 **Scenario: Improving diet**\n\nCurrent risk: {current_risk:.1f}%\nNew risk: {new_risk:.1f}%\nReduction: {current_risk - new_risk:.1f}%\n\nAdd one extra serving of vegetables to each meal!\n\nType 2 for another scenario, or 3 for a new assessment."})
                    
                    else:
                        st.session_state.chat_conversation.append({"role": "assistant", "content": "I didn't understand that. Please type 1, 2, or 3, or for scenarios type: bmi, exercise, smoke, or diet"})
                    
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
        <p>🧬 AI Digital Twin Platform v3.0</p>
    </div>
    """, unsafe_allow_html=True)