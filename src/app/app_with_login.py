import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.auth.login_page import LoginSystem
from backend.database.db_setup import DatabaseManager

# Page config
st.set_page_config(
    page_title="AI Digital Twin Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize login system
login_system = LoginSystem()
db = DatabaseManager()

# Check authentication
authenticated, user_id, username = login_system.login_page()

if authenticated:
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(45deg, #1E88E5, #1565C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
            font-weight: bold;
        }
        .welcome-banner {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Welcome banner
    st.markdown(f"""
    <div class="welcome-banner">
        <h2>🧬 Welcome back, {st.session_state.get('username', 'User')}!</h2>
        <p>Your AI Digital Twin is ready to analyze your health data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load your existing dashboard code here
    # Import and run your main dashboard
    # For now, show a simple dashboard
    
    st.markdown('<p class="main-header">AI Digital Twin Platform</p>', unsafe_allow_html=True)
    
    # Sidebar with user info
    with st.sidebar:
        st.markdown("## 👤 User Profile")
        st.info(f"**User:** {st.session_state.username}")
        if st.session_state.get('user_email'):
            st.info(f"**Email:** {st.session_state.user_email}")
        
        st.markdown("---")
        st.markdown("## 📊 Quick Stats")
        
        # Get user history
        history = db.get_user_history(user_id, limit=5)
        if history:
            st.metric("Total Predictions", len(history))
            last_risk = history[0]['risk_score'] * 100 if history else 0
            st.metric("Last Risk", f"{last_risk:.1f}%")
        
        st.markdown("---")
        
        # Navigation
        st.markdown("## 🧭 Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Home", 
             "🔮 Risk Predictor",
             "📊 My History",
             "📈 Health Tracker",
             "⚙️ Settings"]
        )
    
    # Main content area
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1E88E5, #1565C0); padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
                <h3>Welcome!</h3>
                <p>Use the navigation menu to access all features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #43A047, #2E7D32); padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
                <h3>Quick Actions</h3>
                <p>Predict your risk or track health metrics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FB8C00, #EF6C00); padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
                <h3>History</h3>
                <p>View your past predictions and trends</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity
        st.markdown("---")
        st.subheader("📋 Recent Activity")
        
        history = db.get_user_history(user_id, limit=5)
        if history:
            for item in history:
                risk_color = "🟢" if item['risk_level'] == "LOW" else "🟡" if item['risk_level'] == "MODERATE" else "🔴"
                st.write(f"{risk_color} **{item['date']}** - Risk: {item['risk_score']*100:.1f}% ({item['risk_level']})")
        else:
            st.info("No recent activity. Try making a prediction!")
    
    elif page == "🔮 Risk Predictor":
        st.subheader("🔮 Health Risk Predictor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 20, 90, 45)
            bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
            bp = st.number_input("Blood Pressure", 80, 200, 120)
            
        with col2:
            cholesterol = st.number_input("Cholesterol", 100, 400, 200)
            smoker = st.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x else "No")
            exercise = st.select_slider("Exercise", ["None", "Light", "Moderate", "Active"], value="Moderate")
        
        if st.button("Calculate Risk", type="primary"):
            # Simple risk calculation
            ex_map = {"None": 0, "Light": 0.5, "Moderate": 1, "Active": 1.5}
            
            risk = 0.02
            risk += max(0, (age - 40) * 0.005)
            risk += max(0, (bmi - 25) * 0.01)
            risk += max(0, (bp - 120) * 0.002)
            risk += max(0, (cholesterol - 200) * 0.001)
            risk += 0.15 if smoker else 0
            risk -= ex_map[exercise] * 0.02
            
            risk = np.clip(risk, 0.01, 0.5)
            risk_level = "LOW" if risk < 0.1 else "MODERATE" if risk < 0.2 else "HIGH"
            
            # Save to database
            prediction_data = {
                'age': age, 'bmi': bmi, 'bp': bp, 'cholesterol': cholesterol,
                'smoker': smoker, 'exercise': exercise
            }
            db.save_risk_prediction(user_id, risk, risk_level, prediction_data)
            
            # Display result
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("5-Year Risk", f"{risk*100:.1f}%")
            with col_r2:
                color = "🟢" if risk_level == "LOW" else "🟡" if risk_level == "MODERATE" else "🔴"
                st.metric("Risk Level", f"{color} {risk_level}")
            with col_r3:
                st.progress(float(risk))
            
            st.success("✅ Prediction saved to your history!")
    
    elif page == "📊 My History":
        st.subheader("📊 Your Prediction History")
        
        history = db.get_user_history(user_id, limit=20)
        
        if history:
            # Create dataframe for visualization
            df_history = pd.DataFrame(history)
            df_history['date'] = pd.to_datetime(df_history['date'])
            df_history = df_history.sort_values('date')
            
            # Line chart
            fig = px.line(df_history, x='date', y='risk_score', 
                         title="Risk Score History",
                         markers=True)
            fig.update_layout(yaxis_range=[0, 0.5])
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.dataframe(df_history[['date', 'risk_score', 'risk_level']])
        else:
            st.info("No prediction history yet. Try making a prediction!")
    
    elif page == "📈 Health Tracker":
        st.subheader("📈 Track Your Health Metrics")
        
        with st.form("health_tracker"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                weight = st.number_input("Weight (kg)", 40.0, 150.0, 75.0)
                bp_sys = st.number_input("Systolic BP", 80, 200, 120)
                bp_dia = st.number_input("Diastolic BP", 50, 140, 80)
                
            with col2:
                heart_rate = st.number_input("Heart Rate", 40, 120, 72)
                steps = st.number_input("Daily Steps", 0, 30000, 8000)
                sleep = st.number_input("Sleep (hours)", 4.0, 12.0, 7.0)
                
            with col3:
                stress = st.slider("Stress Level", 1, 10, 5)
                notes = st.text_area("Notes", "")
            
            if st.form_submit_button("Save Metrics"):
                metrics = {
                    'weight': weight,
                    'bp_sys': bp_sys,
                    'bp_dia': bp_dia,
                    'heart_rate': heart_rate,
                    'steps': steps,
                    'sleep': sleep,
                    'stress': stress,
                    'notes': notes
                }
                db.save_health_metrics(user_id, metrics)
                st.success("✅ Health metrics saved!")
    
    elif page == "⚙️ Settings":
        st.subheader("⚙️ Account Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Profile Settings")
            new_name = st.text_input("Full Name", value=st.session_state.get('user_fullname', ''))
            if st.button("Update Profile"):
                st.success("Profile updated! (Demo)")
        
        with col2:
            st.markdown("### Notification Settings")
            email_alerts = st.checkbox("Email Alerts", True)
            risk_threshold = st.slider("Risk Alert Threshold", 10, 30, 20)
            if st.button("Save Settings"):
                st.success("Settings saved!")
        
        # Danger zone
        st.markdown("---")
        st.markdown("### ⚠️ Danger Zone")
        if st.button("Delete Account", type="secondary"):
            st.warning("This action cannot be undone. Contact support.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #78909C; padding: 1rem;'>
        <p>🧬 AI Digital Twin Platform | Secure & Private</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show login page (already handled by login_system.login_page())
    pass