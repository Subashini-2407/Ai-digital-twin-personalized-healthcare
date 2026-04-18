import streamlit as st
import time
import sys
import os

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from backend.database.db_setup import DatabaseManager
except:
    from src.backend.database.db_setup import DatabaseManager

class LoginSystem:
    def __init__(self):
        self.db = DatabaseManager()
        
    def login_page(self):
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.username = None
        
        if st.session_state.authenticated:
            return True, st.session_state.user_id, st.session_state.username
        
        st.markdown("""
        <style>
        .login-box {
            max-width: 400px;
            margin: 50px auto;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown('<div class="login-box">', unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center;'>🧬 AI Digital Twin</h2>", unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                with st.form("login"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    if st.form_submit_button("Login", use_container_width=True):
                        success, result = self.db.authenticate_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.user_id = result['user_id']
                            st.session_state.username = result['username']
                            st.rerun()
                        else:
                            st.error(result)
            
            with tab2:
                with st.form("register"):
                    new_user = st.text_input("Username")
                    new_email = st.text_input("Email")
                    new_pass = st.text_input("Password", type="password")
                    confirm = st.text_input("Confirm", type="password")
                    if st.form_submit_button("Register", use_container_width=True):
                        if new_pass != confirm:
                            st.error("Passwords don't match")
                        else:
                            success, result = self.db.create_user(new_user, new_email, new_pass)
                            if success:
                                st.success("Registered! Please login.")
                            else:
                                st.error(result)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        return False, None, None
    
    def logout(self):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()