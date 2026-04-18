import sqlite3
import hashlib
import os
from datetime import datetime, timedelta
import json

class DatabaseManager:
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT,
                age INTEGER,
                gender TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # User profiles table (health data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE,
                age INTEGER,
                gender TEXT,
                height REAL,
                weight REAL,
                bmi REAL,
                blood_pressure_sys INTEGER,
                blood_pressure_dia INTEGER,
                cholesterol REAL,
                hdl REAL,
                smoker_status TEXT,
                diabetes_status TEXT,
                exercise_level TEXT,
                family_history TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Risk predictions history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                risk_score REAL,
                risk_level TEXT,
                age INTEGER,
                bmi REAL,
                blood_pressure INTEGER,
                cholesterol REAL,
                smoker INTEGER,
                exercise TEXT,
                diabetes INTEGER,
                prediction_data TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Health metrics tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                weight REAL,
                blood_pressure_sys INTEGER,
                blood_pressure_dia INTEGER,
                heart_rate INTEGER,
                steps INTEGER,
                sleep_hours REAL,
                stress_level INTEGER,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                task_name TEXT NOT NULL,
                urgency INTEGER,
                due_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE,
                login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def hash_password(self, password, salt=None):
        """Hash password with salt"""
        if salt is None:
            salt = os.urandom(32).hex()
        # Combine password and salt, then hash
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return password_hash, salt
    
    def create_user(self, username, email, password, full_name="", age=None, gender=""):
        """Create a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if user exists
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                return False, "Username or email already exists"
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Insert user
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt, full_name, age, gender)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, salt, full_name, age, gender))
            
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return True, {"user_id": user_id, "username": username}
        except Exception as e:
            conn.close()
            return False, str(e)
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, password_hash, salt, full_name 
            FROM users WHERE username = ? AND is_active = 1
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            user_id, db_username, email, db_hash, salt, full_name = user
            # Hash input password with stored salt
            input_hash, _ = self.hash_password(password, salt)
            
            if input_hash == db_hash:
                # Update last login
                self.update_last_login(user_id)
                return True, {
                    "user_id": user_id,
                    "username": db_username,
                    "email": email,
                    "full_name": full_name
                }
        
        return False, "Invalid username or password"
    
    def update_last_login(self, user_id):
        """Update user's last login time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
        ''', (user_id,))
        conn.commit()
        conn.close()
    
    def create_session(self, user_id, ip_address="", user_agent=""):
        """Create a new session for user"""
        import secrets
        session_token = secrets.token_urlsafe(32)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_sessions (user_id, session_token, ip_address, user_agent)
            VALUES (?, ?, ?, ?)
        ''', (user_id, session_token, ip_address, user_agent))
        
        conn.commit()
        conn.close()
        return session_token
    
    def validate_session(self, session_token):
        """Validate if session is active"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id FROM user_sessions 
            WHERE session_token = ? AND is_active = 1 
            AND last_activity > datetime('now', '-1 day')
        ''', (session_token,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Update last activity
            self.update_session_activity(session_token)
            return True, result[0]
        return False, None
    
    def update_session_activity(self, session_token):
        """Update session last activity time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE user_sessions SET last_activity = CURRENT_TIMESTAMP 
            WHERE session_token = ?
        ''', (session_token,))
        conn.commit()
        conn.close()
    
    def logout_session(self, session_token):
        """Deactivate session on logout"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE user_sessions SET is_active = 0 WHERE session_token = ?
        ''', (session_token,))
        conn.commit()
        conn.close()
    
    def save_risk_prediction(self, user_id, risk_score, risk_level, prediction_data):
        """Save risk prediction to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO risk_predictions (user_id, risk_score, risk_level, prediction_data)
            VALUES (?, ?, ?, ?)
        ''', (user_id, risk_score, risk_level, json.dumps(prediction_data)))
        
        conn.commit()
        conn.close()
    
    def get_user_history(self, user_id, limit=10):
        """Get user's prediction history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT prediction_date, risk_score, risk_level, prediction_data
            FROM risk_predictions
            WHERE user_id = ?
            ORDER BY prediction_date DESC
            LIMIT ?
        ''', (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        history = []
        for row in results:
            history.append({
                'date': row[0],
                'risk_score': row[1],
                'risk_level': row[2],
                'data': json.loads(row[3]) if row[3] else {}
            })
        return history
    
    def save_health_metrics(self, user_id, metrics):
        """Save daily health metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO health_metrics (
                user_id, weight, blood_pressure_sys, blood_pressure_dia,
                heart_rate, steps, sleep_hours, stress_level, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            metrics.get('weight'),
            metrics.get('bp_sys'),
            metrics.get('bp_dia'),
            metrics.get('heart_rate'),
            metrics.get('steps'),
            metrics.get('sleep'),
            metrics.get('stress'),
            metrics.get('notes', '')
        ))
        
        conn.commit()
        conn.close()
    
    def get_health_trends(self, user_id, days=30):
        """Get user's health trends"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT date, weight, blood_pressure_sys, heart_rate, steps, sleep_hours
            FROM health_metrics
            WHERE user_id = ? AND date > datetime('now', '-? days')
            ORDER BY date ASC
        ''', (user_id, days))
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def add_task(self, user_id, task_name, urgency, due_date):
        """Add a new task for the user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tasks (user_id, task_name, urgency, due_date)
            VALUES (?, ?, ?, ?)
        ''', (user_id, task_name, urgency, due_date))
        
        conn.commit()
        conn.close()
    
    def get_tasks(self, user_id):
        """Get all tasks for the user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT task_name, urgency, due_date, created_at, completed
            FROM tasks
            WHERE user_id = ?
            ORDER BY urgency DESC, due_date ASC
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        tasks = []
        for row in results:
            due_value = row[2]
            if isinstance(due_value, str):
                try:
                    due_date = datetime.strptime(due_value, "%Y-%m-%d").date()
                except ValueError:
                    try:
                        due_date = datetime.fromisoformat(due_value).date()
                    except ValueError:
                        due_date = due_value
            else:
                due_date = due_value
            tasks.append({
                'task': row[0],
                'urgency': row[1],
                'due': due_date,
                'created': row[3],
                'completed': bool(row[4])
            })
        return tasks