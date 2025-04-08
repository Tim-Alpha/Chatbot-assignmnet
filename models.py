import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import httpx
from enum import Enum
import random
import time
import threading

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0
)

# Configuration
class Config:
    DATABASE_NAME = "dscpl_history.db"
    API_BASE_URL = "https://api.socialverseapp.com"
    FLIC_TOKEN = "flic_b1c6b09d98e2d4884f61b9b3131dbb27a6af84788e4a25db067a22008ea9cce5"
    GOOGLE_CREDENTIALS_FILE = "credentials.json"  # Simplified path
    SCOPES = ['https://www.googleapis.com/auth/calendar','https://www.googleapis.com/auth/calendar.events.owned']

# Simplified category enum
class Category(str, Enum):
    DEVOTION = "Daily Devotion"
    PRAYER = "Daily Prayer"
    MEDITATION = "Daily Meditation"
    ACCOUNTABILITY = "Daily Accountability"
    JUST_CHAT = "Just Chat"
    PROGRESS = "View Progress"

class DevotionTopic(str, Enum):
    STRESS = "Dealing with Stress"
    FEAR = "Overcoming Fear"
    DEPRESSION = "Conquering Depression"
    RELATIONSHIPS = "Relationships"
    HEALING = "Healing"
    PURPOSE = "Purpose & Calling"
    ANXIETY = "Anxiety"
    OTHER = "Something else..." 

class PrayerTopic(str, Enum):
    GROWTH = "Personal Growth"
    HEALING = "Healing"
    FAMILY = "Family/Friends"
    FORGIVENESS = "Forgiveness"
    FINANCES = "Finances"
    WORK = "Work/Career"
    OTHER = "Something else..."

class MeditationTopic(str, Enum):
    PEACE = "Peace"
    GODS_PRESENCE = "God's Presence"
    STRENGTH = "Strength"
    WISDOM = "Wisdom"
    FAITH = "Faith"
    OTHER = "Something else..."

class AccountabilityTopic(str, Enum):
    PORNOGRAPHY = "Pornography"
    ALCOHOL = "Alcohol"
    DRUGS = "Drugs"
    SEX = "Sex"
    ADDICTION = "Addiction"
    LAZINESS = "Laziness"
    OTHER = "Something else..."

class ProgramLength(int, Enum):
    TODAY_ONLY = 1
    SEVEN_DAYS = 7
    FOURTEEN_DAYS = 14
    THIRTY_DAYS = 30

class ContentType(str, Enum):
    TEXT = "Text Only"
    VIDEO = "Video Only"
    BOTH = "Both Text and Video"

# Initialize database
def init_db():
    conn = sqlite3.connect(Config.DATABASE_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_sessions (
        session_id TEXT PRIMARY KEY,
        user_id TEXT,
        current_state TEXT,
        selected_category TEXT,
        selected_topic TEXT,
        program_length INTEGER,
        program_start_date TEXT,
        current_day INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS program_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        session_id TEXT,
        category TEXT,
        topic TEXT,
        program_length INTEGER,
        start_date TEXT,
        end_date TEXT,
        completed BOOLEAN DEFAULT 0,
        paused BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        day_number INTEGER,
        completed BOOLEAN DEFAULT 0,
        completed_at TIMESTAMP,
        notes TEXT
    )""")
    
    conn.commit()
    conn.close()

init_db()

# State management
class StateManager:
    @staticmethod
    def create_session(user_id: str) -> str:
        conn = sqlite3.connect(Config.DATABASE_NAME)
        cursor = conn.cursor()
        session_id = f"session_{datetime.now().timestamp()}"
        cursor.execute(
            "INSERT INTO user_sessions (session_id, user_id, current_state) VALUES (?, ?, ?)",
            (session_id, user_id, "initial")
        )
        conn.commit()
        conn.close()
        return session_id

    @staticmethod
    def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(Config.DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "session_id": row[0],
                "user_id": row[1],
                "current_state": row[2],
                "selected_category": row[3],
                "selected_topic": row[4],
                "program_length": row[5],
                "program_start_date": row[6],
                "current_day": row[7]
            }
        return None

    @staticmethod
    def update_session(session_id: str, updates: Dict[str, Any]):
        conn = sqlite3.connect(Config.DATABASE_NAME)
        cursor = conn.cursor()
        current = StateManager.get_session(session_id)
        if not current:
            raise ValueError("Session not found")
        
        merged = {**current, **updates}
        cursor.execute("""
            UPDATE user_sessions 
            SET current_state = ?,
                selected_category = ?,
                selected_topic = ?,
                program_length = ?,
                program_start_date = ?,
                current_day = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?""", (
            merged["current_state"],
            merged["selected_category"],
            merged["selected_topic"],
            merged["program_length"],
            merged["program_start_date"],
            merged["current_day"],
            session_id
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def add_message(session_id: str, role: str, content: str):
        conn = sqlite3.connect(Config.DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversation_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
        conn = sqlite3.connect(Config.DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM conversation_history WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        history = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
        conn.close()
        return history

    @staticmethod
    def add_program_to_history(session_id: str, completed: bool = False):
        """Add a program to the user's history"""
        session = StateManager.get_session(session_id)
        if not session:
            return False
        
        conn = sqlite3.connect(Config.DATABASE_NAME)
        cursor = conn.cursor()
        
        # Calculate end date based on program length and start date
        start_date = datetime.fromisoformat(session["program_start_date"])
        end_date = start_date + timedelta(days=session["program_length"] - 1)
        
        cursor.execute("""
            INSERT INTO program_history 
            (user_id, session_id, category, topic, program_length, start_date, end_date, completed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            session_id,
            session["selected_category"],
            session["selected_topic"],
            session["program_length"],
            session["program_start_date"],
            end_date.isoformat(),
            1 if completed else 0
        ))
        
        conn.commit()
        conn.close()
        return True
    
    @staticmethod
    def mark_day_completed(session_id: str, day_number: int, notes: str = ""):
        """Mark a specific day as completed in the program"""
        conn = sqlite3.connect(Config.DATABASE_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO daily_progress 
            (session_id, day_number, completed, completed_at, notes)
            VALUES (?, ?, 1, CURRENT_TIMESTAMP, ?)
        """, (session_id, day_number, notes))
        
        conn.commit()
        conn.close()
        return True
    
    @staticmethod
    def get_program_history(user_id: str) -> List[Dict[str, Any]]:
        """Get all programs for a user"""
        conn = sqlite3.connect(Config.DATABASE_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, session_id, category, topic, program_length, start_date, end_date, completed, paused
            FROM program_history
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "id": row[0],
                "session_id": row[1],
                "category": row[2],
                "topic": row[3],
                "program_length": row[4],
                "start_date": row[5],
                "end_date": row[6],
                "completed": bool(row[7]),
                "paused": bool(row[8])
            })
        
        conn.close()
        return history
    
    @staticmethod
    def get_program_progress(session_id: str) -> Dict[str, Any]:
        """Get progress for a specific program"""
        session = StateManager.get_session(session_id)
        if not session:
            return {}
        
        conn = sqlite3.connect(Config.DATABASE_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT day_number, completed, completed_at, notes
            FROM daily_progress
            WHERE session_id = ?
            ORDER BY day_number
        """, (session_id,))
        
        progress = {
            "total_days": session["program_length"],
            "current_day": session["current_day"],
            "completed_days": [],
            "remaining_days": []
        }
        
        completed_days = {row[0]: {"completed_at": row[2], "notes": row[3]} for row in cursor.fetchall()}
        
        for day in range(1, session["program_length"] + 1):
            if day in completed_days:
                progress["completed_days"].append({
                    "day": day,
                    "completed_at": completed_days[day]["completed_at"],
                    "notes": completed_days[day]["notes"]
                })
            else:
                progress["remaining_days"].append(day)
        
        conn.close()
        return progress

# API Client
class SocialVerseClient:
    @staticmethod
    def get_videos(topic: Optional[str] = None, max_results: int = 3, bible_verse: Optional[str] = None):
        headers = {"Flic-Token": Config.FLIC_TOKEN}
        url = f"{Config.API_BASE_URL}/posts/summary/get?page=1&page_size=1000"
        
        try:
            response = httpx.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    # Get all posts with video links
                    video_posts = [p for p in data.get("posts", []) if p.get("video_link")]
                    
                    # If a Bible verse is provided, try to find videos related to it
                    if bible_verse and isinstance(bible_verse, str):
                        # Extract key terms from the Bible verse
                        verse_terms_response = llm.invoke(f"Extract 3-5 key terms from this Bible verse that would be useful for finding related videos: {bible_verse}")
                        verse_terms_str = str(verse_terms_response.content)
                        verse_terms = [term.strip().lower() for term in verse_terms_str.split(',')]
                        
                        # Score videos based on relevance to verse terms
                        scored_videos = []
                        for vid in video_posts:
                            # Score based on verse terms
                            verse_score = sum(1 for term in verse_terms if term in vid.get("title", "").lower())
                            
                            # Score based on topic if provided
                            topic_score = 0
                            if topic:
                                topic_lower = topic.lower()
                                topic_score = sum(1 for kw in vid.get("topics", []) if topic_lower in kw)
                            
                            # Combined score
                            total_score = verse_score * 2 + topic_score
                            scored_videos.append((total_score, vid))
                        
                        # Sort by score
                        scored_videos.sort(reverse=True, key=lambda x: x[0])
                        
                        # Return top videos
                        selected = [vid for _, vid in scored_videos[:max_results]]
                        if not selected and video_posts:
                            # Fallback to random videos if no good matches
                            random.shuffle(video_posts)
                            selected = video_posts[:max_results]
                    
                    # If no Bible verse or no matches found, use topic-based selection
                    elif topic:
                        topic_lower = topic.lower()
                        scored_videos = []
                        for vid in video_posts:
                            score = sum(1 for kw in vid.get("topics", []) if topic_lower in kw)
                            scored_videos.append((score, vid))
                        scored_videos.sort(reverse=True, key=lambda x: x[0])
                        selected = [vid for _, vid in scored_videos[:max_results]]
                        if not selected and video_posts:
                            # Fallback to random videos if no good matches
                            random.shuffle(video_posts)
                            selected = video_posts[:max_results]
                    
                    # If no topic or Bible verse, just return random videos
                    else:
                        random.shuffle(video_posts)
                        selected = video_posts[:max_results]
                    
                    # Format results with required fields
                    return [{
                        "title": p.get("title", "Untitled"),
                        "url": p.get("video_link"),
                        "thumbnail": p.get("thumbnail_url"),
                        "topics": [kw["keyword"] for kw in 
                                  p.get("post_summary", {}).get("keywords", [])]
                    } for p in selected]
        
        except Exception as e:
            print(f"API Error: {e}")
        
        return []

# Content generators
def generate_devotion_content(topic: str, day: int, content_type: ContentType):
    if content_type == ContentType.TEXT:
        bible_reading = llm.invoke(f"Generate a 5-minute Bible reading about {topic} for day {day}")
        prayer = llm.invoke(f"Write a short prayer about {topic}")
        declaration = llm.invoke(f"Create a faith declaration about {topic}")
        return {
            "scripture": bible_reading.content,
            "prayer": prayer.content,
            "declaration": declaration.content,
            "video_recommendation": None
        }
    elif content_type == ContentType.VIDEO:
        # First get the Bible verse to find relevant videos
        bible_reading = llm.invoke(f"Generate a Bible verse about {topic} for day {day}")
        bible_verse = str(bible_reading.content)
        
        # Get videos related to the Bible verse
        videos = SocialVerseClient.get_videos(topic=topic, bible_verse=bible_verse)
        if videos:
            return {
                "scripture": bible_verse,
                "prayer": None,
                "declaration": None,
                "video_recommendation": f"{videos[0]['title']} ({videos[0]['url']})"
            }
        return {
            "scripture": bible_verse,
            "prayer": None,
            "declaration": None,
            "video_recommendation": "No videos found for this topic"
        }
    else:  # BOTH
        bible_reading = llm.invoke(f"Generate a 5-minute Bible reading about {topic} for day {day}")
        prayer = llm.invoke(f"Write a short prayer about {topic}")
        declaration = llm.invoke(f"Create a faith declaration about {topic}")
        
        # Extract the main Bible verse from the reading
        bible_verse_prompt = llm.invoke(f"Extract the main Bible verse from this reading: {bible_reading.content}")
        bible_verse = str(bible_verse_prompt.content)
        
        # Get videos related to the Bible verse
        videos = SocialVerseClient.get_videos(topic=topic, bible_verse=bible_verse)
        
        video_rec = "No videos found for this topic"
        if videos:
            video_rec = f"{videos[0]['title']} ({videos[0]['url']})"
        
        return {
            "scripture": bible_reading.content,
            "prayer": prayer.content,
            "declaration": declaration.content,
            "video_recommendation": video_rec
        }

def generate_prayer_content(topic: str):
    prayer = llm.invoke(f"Create an ACTS model prayer about {topic}")
    return prayer.content

def generate_meditation_content(topic: str):
    meditation = llm.invoke(f"""Create a meditation guide about {topic} including:
1. Scripture focus
2. Meditation prompts
3. Breathing guide""")
    return meditation.content

def generate_accountability_content(topic: str):
    content = llm.invoke(f"""Create accountability support for {topic} including:
1. Scripture
2. Truth declarations
3. Action plan""")
    return content.content

def generate_sos_content(topic: str):
    """Generate immediate support content for users in crisis"""
    content = llm.invoke(f"""Create immediate emergency support for someone struggling with {topic} including:
1. Immediate encouragement
2. Relevant scripture for strength
3. Specific action steps to take right now
4. Prayer for immediate relief
5. Contact information for support services""")
    return content.content

# Prompt generators
def get_category_prompt():
    return """Welcome to DSCPL - Your Personal Spiritual Assistant
What do you need today? Please select:

1. Daily Devotion (video or text)
2. Daily Prayer Guidance
3. Daily Meditation Practice 
4. Daily Accountability Support
5. Just Chat
6. View Progress Dashboard

Enter the number of your choice: """

def get_topic_prompt(category: Category):
    if category == Category.DEVOTION:
        options = [f"{i+1}. {topic.value}" for i, topic in enumerate(DevotionTopic)]
    elif category == Category.PRAYER:
        options = [f"{i+1}. {topic.value}" for i, topic in enumerate(PrayerTopic)]
    elif category == Category.MEDITATION:
        options = [f"{i+1}. {topic.value}" for i, topic in enumerate(MeditationTopic)]
    elif category == Category.ACCOUNTABILITY.value:
        options = [f"{i+1}. {topic.value}" for i, topic in enumerate(AccountabilityTopic)]
    else:
        return ""
    return "Select a topic:\n" + "\n".join(options) + "\nEnter the number of your choice: "

def get_program_length_prompt():
    options = [f"{i+1}. {length.value} {'day' if length.value == 1 else 'days'}" for i, length in enumerate(ProgramLength)]
    return "Select program length:\n" + "\n".join(options) + "\nEnter the number of your choice: "

def get_confirmation_prompt():
    return "Would you like to begin this program? (yes/no): "

def get_calendar_prompt():
    return "Would you like to set up daily calendar reminders? (yes/no): "

def get_content_type_prompt():
    options = [f"{i+1}. {type.value}" for i, type in enumerate(ContentType)]
    return "How would you like to receive your devotion content?\n" + "\n".join(options) + "\nEnter the number of your choice: "

# Calendar functions
def create_calendar_events(session_id: str, program_length: int, preferred_time: str = "08:00"):
    """Create daily calendar events for the program"""
    try:
        creds = None
        # First look for existing credentials
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', Config.SCOPES)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(Config.GOOGLE_CREDENTIALS_FILE):
                    raise FileNotFoundError(
                        f"Google credentials file not found at {Config.GOOGLE_CREDENTIALS_FILE}. "
                        "Please download it from Google Cloud Console and place it in the correct location."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    Config.GOOGLE_CREDENTIALS_FILE,
                    Config.SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        service = build('calendar', 'v3', credentials=creds)
        
        # Get user's timezone from calendar settings
        calendar = service.calendars().get(calendarId='primary').execute()
        user_timezone = calendar.get('timeZone', 'UTC')
        
        # Get user's session data
        session = StateManager.get_session(session_id)
        if not session:
            raise ValueError("Session not found. Cannot create calendar events.")
            
        start_date = datetime.fromisoformat(session["program_start_date"])
        
        # Parse preferred time
        hours, minutes = map(int, preferred_time.split(":"))
        
        for day in range(program_length):
            event_date = start_date + timedelta(days=day)
            # Set the event time to the preferred time
            event_datetime = event_date.replace(hour=hours, minute=minutes)
            
            # Calculate event duration based on category
            category = session.get('selected_category')
            if category == "Daily Devotion":
                duration_minutes = 30
            elif category == "Daily Meditation":
                duration_minutes = 20
            elif category == "Daily Prayer":
                duration_minutes = 15
            else:
                duration_minutes = 30  # Default duration
            
            event = {
                'summary': f'DSCPL Program Day {day+1}',
                'description': (
                    f"Your daily {session['selected_category']} program\n"
                    f"Topic: {session.get('selected_topic', 'Not specified')}\n"
                    f"Day {day+1} of {program_length}"
                ),
                'start': {
                    'dateTime': event_datetime.isoformat(),
                    'timeZone': user_timezone,
                },
                'end': {
                    'dateTime': (event_datetime + timedelta(minutes=duration_minutes)).isoformat(),
                    'timeZone': user_timezone,
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'popup', 'minutes': 10},
                        {'method': 'email', 'minutes': 60},
                    ]
                },
            }
            
            try:
                service.events().insert(
                    calendarId='primary',
                    body=event
                ).execute()
                print(f"✅ Calendar event created for Day {day+1} at {preferred_time}")
            except Exception as e:
                print(f"⚠️ Failed to create calendar event for Day {day+1}: {str(e)}")
                # Continue creating other events even if one fails
                continue
    
    except FileNotFoundError as e:
        print(f"\n⚠️ Calendar setup failed: {str(e)}")
        return False
    except Exception as e:
        print(f"\n⚠️ Calendar setup failed: An unexpected error occurred - {str(e)}")
        return False
    
    return True

# Notification system
class NotificationManager:
    _instance = None
    _notification_thread = None
    _running = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = NotificationManager()
        return cls._instance
    
    def __init__(self):
        self.notifications = []
    
    def start(self):
        """Start the notification thread"""
        if self._notification_thread is None or not self._notification_thread.is_alive():
            self._running = True
            self._notification_thread = threading.Thread(target=self._notification_loop)
            self._notification_thread.daemon = True
            self._notification_thread.start()
            print("Notification system started")
    
    def stop(self):
        """Stop the notification thread"""
        self._running = False
        if self._notification_thread and self._notification_thread.is_alive():
            self._notification_thread.join(timeout=1.0)
            print("Notification system stopped")
    
    def _notification_loop(self):
        """Background thread that checks for and sends notifications"""
        while self._running:
            now = datetime.now()
            
            # Check for notifications that need to be sent
            notifications_to_send = []
            for notification in self.notifications:
                if notification["scheduled_time"] <= now and not notification["sent"]:
                    notifications_to_send.append(notification)
            
            # Send notifications
            for notification in notifications_to_send:
                self._send_notification(notification)
                notification["sent"] = True
            
            # Sleep for a minute before checking again
            time.sleep(60)
    
    def _send_notification(self, notification):
        """Send a notification to the user"""
        # In a real app, this would use a proper notification service
        # For this demo, we'll just print to the console
        print("\n" + "="*50)
        print(f"NOTIFICATION: {notification['title']}")
        print(f"{notification['message']}")
        print("="*50 + "\n")
    
    def schedule_notification(self, user_id: str, title: str, message: str, scheduled_time: datetime):
        """Schedule a notification for a specific time"""
        notification = {
            "user_id": user_id,
            "title": title,
            "message": message,
            "scheduled_time": scheduled_time,
            "sent": False
        }
        self.notifications.append(notification)
        return notification
    
    def schedule_daily_notifications(self, session_id: str, program_length: int, start_date: str):
        """Schedule daily notifications for a program"""
        session = StateManager.get_session(session_id)
        if not session:
            return False
        
        user_id = session["user_id"]
        category = session["selected_category"]
        topic = session["selected_topic"]
        
        start_datetime = datetime.fromisoformat(start_date)
        
        for day in range(program_length):
            notification_time = start_datetime + timedelta(days=day)
            
            # Only schedule future notifications
            if notification_time > datetime.now():
                title = f"DSCPL Daily {category}"
                message = f"Time for your daily {category.lower()} on {topic}. Day {day+1} of {program_length}."
                
                self.schedule_notification(user_id, title, message, notification_time)
        
        return True 