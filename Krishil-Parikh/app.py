import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import argparse
import threading
import pywhatkit
import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import icalendar
import pytz
import tempfile
import webbrowser
from urllib.parse import quote
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
import requests
import json
from datetime import datetime, timedelta
import uuid
import pathlib
import time
import pandas as pd
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY", "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")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "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")
hf_token = os.getenv("HF_TOKEN", "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")

llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

st.set_page_config(
    page_title="DSCPL - Your Spiritual Assistant",
    page_icon="✝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'user_id' not in st.session_state:
    # st.session_state.user_id = str(uuid.uuid4())
    st.session_state.user_id = "1bb05647-268f-4877-90ef-d825c7a47668"

USER_DIR = pathlib.Path(f"users/{st.session_state.user_id}")
USER_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = USER_DIR / "conversation_history.json"
REMINDERS_FILE = USER_DIR / "reminders.json"
PROGRESS_FILE = USER_DIR / "progress.json"

if 'chat_history' not in st.session_state:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r') as f:
            st.session_state.chat_history = json.load(f)
    else:
        st.session_state.chat_history = []

if 'reminders' not in st.session_state:
    if REMINDERS_FILE.exists():
        with open(REMINDERS_FILE, 'r') as f:
            st.session_state.reminders = json.load(f)
    else:
        st.session_state.reminders = []

if 'progress' not in st.session_state:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            st.session_state.progress = json.load(f)
    else:
        st.session_state.progress = {
            "completed_devotions": 0,
            "completed_prayers": 0,
            "completed_meditations": 0,
            "accountability_streaks": 0,
            "programs": []
        }

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

if 'conversation' not in st.session_state:
    st.session_state.conversation = None

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def save_chat_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(st.session_state.chat_history, f)

def save_reminders():
    with open(REMINDERS_FILE, 'w') as f:
        json.dump(st.session_state.reminders, f)

def save_progress():
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(st.session_state.progress, f)

def add_to_chat_history(role, content, additional_info=None):
    timestamp = datetime.now().isoformat()
    entry = {"role": role, "content": content, "timestamp": timestamp}
    if additional_info:
        entry.update(additional_info)
    
    st.session_state.chat_history.append(entry)
    save_chat_history()
    
    if 'vector_store' in st.session_state:
        doc_content = f"{role}: {content} (at {timestamp})"
        st.session_state.vector_store.add_texts([doc_content])


def create_reminder(title, date, description, category):
    reminder = {
        "id": str(uuid.uuid4()),
        "title": title,
        "date": date,
        "description": description,
        "category": category,
        "completed": False
    }
    st.session_state.reminders.append(reminder)
    save_reminders()
    return reminder

def setup_google_calendar_oauth():
    """Initiates the Google Calendar OAuth flow"""
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    flow = InstalledAppFlow.from_client_secrets_file(
        'credentials.json',  
        SCOPES,
        redirect_uri='http://localhost:8501/'  
    )
    auth_url, _ = flow.authorization_url(prompt='consent')
    
    webbrowser.open(auth_url)
    
    st.session_state.auth_flow = flow
    return auth_url

def handle_google_auth_callback(code):
    """Handle the callback from Google OAuth"""
    if 'auth_flow' not in st.session_state:
        st.error("Authentication flow not found. Please try again.")
        return None
    
    flow = st.session_state.auth_flow
    try:
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # Save credentials
        with open('token.json', 'w') as token:
            token.write(credentials.to_json())
            
        return build('calendar', 'v3', credentials=credentials)
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return None

def add_to_google_calendar(reminders):
    """Add reminders to Google Calendar"""
    try:
        # Check if we have stored credentials
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_info(json.load(open('token.json')))
            if creds.valid:
                service = build('calendar', 'v3', credentials=creds)
            else:
                # If credentials expired, re-authenticate
                auth_url = setup_google_calendar_oauth()
                st.info(f"Please authenticate with Google: [Click here to authorize]({auth_url})")
                auth_code = st.text_input("Enter the authorization code:")
                if auth_code:
                    service = handle_google_auth_callback(auth_code)
                else:
                    return False
        else:
            # First time authentication
            auth_url = setup_google_calendar_oauth()
            st.info(f"Please authenticate with Google: [Click here to authorize]({auth_url})")
            auth_code = st.text_input("Enter the authorization code:")
            if auth_code:
                service = handle_google_auth_callback(auth_code)
            else:
                return False
        
        if not service:
            return False
            
        for reminder in reminders:
            event = {
                'summary': reminder['title'],
                'description': reminder['description'],
                'start': {
                    'dateTime': reminder['date'],
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': (datetime.fromisoformat(reminder['date']) + timedelta(hours=1)).isoformat(),
                    'timeZone': 'UTC',
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'popup', 'minutes': 30},
                    ],
                },
            }
            
            service.events().insert(calendarId='primary', body=event).execute()
        
        return True
    except Exception as e:
        st.error(f"Failed to add to Google Calendar: {str(e)}")
        return False

def generate_ical_file(reminders):
    """Generate an iCal file for Apple Calendar"""
    cal = icalendar.Calendar()
    cal.add('prodid', '-//DSCPL Spiritual Assistant//EN')
    cal.add('version', '2.0')
    
    for reminder in reminders:
        event = icalendar.Event()
        event.add('summary', reminder['title'])
        event.add('description', reminder['description'])
        
        start_time = datetime.fromisoformat(reminder['date'])
        end_time = start_time + timedelta(hours=1)
        
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pytz.UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=pytz.UTC)
            
        event.add('dtstart', start_time)
        event.add('dtend', end_time)
        event.add('dtstamp', datetime.now(pytz.UTC))
        event['uid'] = f"{reminder['id']}@dscpl.app"
        
        alarm = icalendar.Alarm()
        alarm.add('action', 'DISPLAY')
        alarm.add('trigger', timedelta(minutes=-30))  
        alarm.add('description', f"Reminder: {reminder['title']}")
        event.add_component(alarm)
        
        cal.add_component(event)
    
    with tempfile.NamedTemporaryFile(suffix='.ics', delete=False) as temp_file:
        temp_file.write(cal.to_ical())
        return temp_file.name

def create_apple_calendar_url(reminder):
    """Create a direct URL to add an event to Apple Calendar"""
    title = quote(reminder['title'])
    desc = quote(reminder['description'])
    start_date = datetime.fromisoformat(reminder['date']).strftime('%Y%m%dT%H%M%S')
    end_date = (datetime.fromisoformat(reminder['date']) + timedelta(hours=1)).strftime('%Y%m%dT%H%M%S')
    
    return f"webcal://p44-caldav.icloud.com/published/2/MTAzNDg5MjQ5MTAzNDg5Mjb-Y3xZIrYL-G_BoTrugneSb0HZcbxGRs9-R_3NuVY?title={title}&desc={desc}&dtstart={start_date}&dtend={end_date}"

def mark_reminder_complete(reminder_id):
    """Mark a reminder as complete and update progress"""
    for i, reminder in enumerate(st.session_state.reminders):
        if reminder["id"] == reminder_id:
            reminder_type = reminder.get("category", "").lower()
            
            st.session_state.reminders.pop(i)
            
            with open(REMINDERS_FILE, 'w') as f:
                json.dump(st.session_state.reminders, f)
            
            if reminder_type in ["devotion", "prayer", "meditation", "accountability"]:
                update_progress(reminder_type)
            
            return True
    return False

def get_upcoming_reminders(days=7):
    today = datetime.now().date()
    upcoming = []
    for reminder in st.session_state.reminders:
        reminder_date = datetime.fromisoformat(reminder["date"]).date()
        days_diff = (reminder_date - today).days
        if 0 <= days_diff <= days and not reminder["completed"]:
            upcoming.append(reminder)
    return upcoming

def get_personalized_video_recommendations(user_query=None, user_history=None, available_videos=None, program_topic=None):
    """
    Uses the LLM to determine which videos would be most relevant based on user query,
    history, available videos, and optionally a program topic.
    
    Args:
        user_query (str): User's search query
        user_history (list): User's chat history
        available_videos (list): List of available videos
        program_topic (str): Optional program topic to focus recommendations
        
    Returns:
        list: Ranked list of relevant videos
    """
    if not available_videos:
        available_videos = fetch_videos()
        
    if not available_videos:
        return []
        
    history_context = ""
    if user_history:
        recent_history = user_history[-15:]  
        user_messages = [item["content"] for item in recent_history if item["role"] == "user"]
        if user_messages:
            history_context = "\n".join(user_messages)
    
    video_metadata = []
    for i, video in enumerate(available_videos):
        video_metadata.append({
            "id": i,
            "title": video.get("title", "Untitled"),
            "description": video.get("description", "No description available"),
        })
    
    program_context = f"\nThe user is currently working on a spiritual program about: {program_topic}" if program_topic else ""
    
    prompt = f"""
    You are a personalized content recommendation system. Based on the user's history, current request, and context,
    rank the following videos from most to least relevant for them.
    
    USER HISTORY:
    {history_context}
    
    CURRENT REQUEST:
    {user_query if user_query else "General spiritual growth and encouragement"}
    {program_context}
    
    AVAILABLE VIDEOS:
    {json.dumps(video_metadata, indent=2)}
    
    For each video, analyze:
    1. How closely it matches the search query (if provided)
    2. How it relates to themes in the user's conversation history
    3. How it might help with their spiritual journey
    
    Return a JSON array with the following structure:
    [
      {{
        "id": [video ID number],
        "relevance_score": [score from 0-100],
        "relevance_reason": [1-2 sentences explaining why this video is relevant]
      }},
      ...
    ]
    
    ONLY return this JSON array, nothing else.
    """
    
    try:
        response = llm.invoke(prompt)
        
        try:
            clean_response = response.content.strip()
            if clean_response.startswith("```") and clean_response.endswith("```"):
                clean_response = clean_response[3:-3].strip()
            if clean_response.startswith("json") or clean_response.startswith("JSON"):
                clean_response = clean_response[4:].strip()
                
            recommendations = json.loads(clean_response)
            
            recommendations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            result = []
            for rec in recommendations:
                if rec["id"] < len(available_videos):
                    video = available_videos[rec["id"]]
                    video["relevance_reason"] = rec.get("relevance_reason", "")
                    result.append(video)
            
            return result
            
        except json.JSONDecodeError:
            return available_videos
            
    except Exception as e:
        return available_videos

def fetch_videos():
    url = "https://api.socialverseapp.com/posts/summary/get?page=1&page_size=10"
    headers = {
        "Flic-Token": "flic_b1c6b09d98e2d4884f61b9b3131dbb27a6af84788e4a25db067a22008ea9cce5",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("posts", [])
        else:
            return []
    except Exception:
        return []

def get_personalized_context(user_query):
    # Only perform retrieval if we have a vector store
    if 'vector_store' not in st.session_state:
        return ""
    
    # Create a retriever
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant memories
    )
    
    # Retrieve relevant conversation history
    relevant_docs = retriever.get_relevant_documents(user_query)
    
    if not relevant_docs:
        return ""
    
    # Format the relevant context
    context_parts = ["Based on our previous conversation:"]
    for doc in relevant_docs:
        context_parts.append(f"- {doc.page_content}")
    
    return "\n".join(context_parts)

def send_whatsapp_message_now(phone_number, message):
    """
    Send a WhatsApp message instantly to the specified phone number.
    
    Args:
        phone_number (str): Phone number with country code (e.g., '+1XXXXXXXXXX')
        message (str): Message content to be sent
    """
    try:
        # Get current time
        now = datetime.now()
        
        # Send message with current time (will send almost immediately)
        pywhatkit.sendwhatmsg(phone_number, message, now.hour, now.minute + 1)
        
        # Close the tab after sending
        time.sleep(5)
        
        # Import pyautogui only if needed to avoid installation issues
        import pyautogui
        pyautogui.hotkey('ctrl', 'w')
        
        print(f"Message sent successfully to {phone_number}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def send_email(sender_email, sender_password, recipient_email, subject, message):
    """Send email using SMTP"""
    try:
        # Set up the MIME
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        # Add message body
        msg.attach(MIMEText(message, 'plain'))
        
        # Determine the email server based on sender's email
        if '@gmail.com' in sender_email:
            server = smtplib.SMTP('smtp.gmail.com', 587)
        elif '@outlook.com' in sender_email or '@hotmail.com' in sender_email:
            server = smtplib.SMTP('smtp.office365.com', 587)
        elif '@yahoo.com' in sender_email:
            server = smtplib.SMTP('smtp.mail.yahoo.com', 587)
        else:
            st.error(f"Unknown email provider for {sender_email}. Using Gmail as default.")
            server = smtplib.SMTP('smtp.gmail.com', 587)
        
        # Start TLS for security
        server.starttls()
        
        # Authentication
        server.login(sender_email, sender_password)
        
        # Send the message
        server.send_message(msg)
        
        # Terminate the session
        server.quit()
        
        return True
            
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        if "gmail" in str(e).lower() and "password" in str(e).lower():
            st.warning(
                "For Gmail, you need to use an 'App Password' rather than your regular password.\n"
                "1. Enable 2-Factor Authentication in your Google Account\n"
                "2. Then create an App Password at: https://myaccount.google.com/apppasswords"
            )
        return False

# ---------- UI COMPONENTS ----------
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/ios/250/FFFFFF/church.png", width=100)
        st.title("DSCPL")
        st.subheader("Your Spiritual Assistant")
        
        st.markdown("---")
        
        # Display upcoming reminders
        st.subheader("📅 Upcoming Reminders")
        upcoming = get_upcoming_reminders()
        if upcoming:
            for i, reminder in enumerate(upcoming):
                reminder_date = datetime.fromisoformat(reminder["date"]).date()
                today = datetime.now().date()
                days_left = (reminder_date - today).days
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if days_left == 0:
                            st.markdown(f"**Today:** {reminder['title']}")
                        else:
                            st.markdown(f"**In {days_left} days:** {reminder['title']}")
                    with col2:
                        if st.button("✅", key=f"complete_{reminder['id']}"):
                            mark_reminder_complete(reminder["id"])
                            st.rerun()
        else:
            st.info("No upcoming reminders")
        
        st.markdown("---")
        
        # Progress Dashboard
        st.subheader("📊 Spiritual Progress")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Devotions", st.session_state.progress["completed_devotions"])
            st.metric("Meditations", st.session_state.progress["completed_meditations"])
        with col2:
            st.metric("Prayers", st.session_state.progress["completed_prayers"])
            st.metric("Streak", st.session_state.progress["accountability_streaks"])
        
        st.markdown("---")
        
        # Enhanced Calendar export options
        st.subheader("📅 Export to Calendar")
        calendar_option = st.radio("Choose your calendar:", ["Google Calendar", "Apple Calendar"])
        
        upcoming_reminders = get_upcoming_reminders(30)  # Get next 30 days of reminders
        
        if calendar_option == "Google Calendar":
            if st.button("Export to Google Calendar", use_container_width=True):
                if not upcoming_reminders:
                    st.warning("No upcoming reminders to export.")
                else:
                    with st.spinner("Connecting to Google Calendar..."):
                        if add_to_google_calendar(upcoming_reminders):
                            st.success("✅ Events successfully added to Google Calendar!")
                        else:
                            st.error("Failed to add events. Please try again.")
        else:  # Apple Calendar
            if st.button("Export to Apple Calendar", use_container_width=True):
                if not upcoming_reminders:
                    st.warning("No upcoming reminders to export.")
                else:
                    with st.spinner("Generating calendar file..."):
                        try:
                            ical_path = generate_ical_file(upcoming_reminders)
                            with open(ical_path, 'rb') as file:
                                st.download_button(
                                    label="Download iCal File",
                                    data=file,
                                    file_name="dscpl_spiritual_program.ics",
                                    mime="text/calendar",
                                    use_container_width=True
                                )
                                st.info("📱 After downloading, open the file to import events to your Apple Calendar")
                        except Exception as e:
                            st.error(f"Error generating calendar file: {str(e)}")
        
        # Settings
        st.subheader("⚙️ Settings")
        st.checkbox("Daily Notifications", value=True, key="notifications_enabled")
        st.checkbox("Video Recommendations", value=True, key="video_recommendations")

def recommend_program_type(user_query, user_history=None):
    """
    Uses the LLM to determine which program type would be most beneficial
    based on user query and history.
    """
    # Prepare context from user history
    history_context = ""
    if user_history:
        recent_history = user_history[-15:]  # Use most recent 15 interactions
        user_messages = [item["content"] for item in recent_history if item["role"] == "user"]
        if user_messages:
            history_context = "\n".join(user_messages)
    
    # Create prompt for the LLM
    prompt = f"""
    You are a spiritual guide. Based on the user's history and current request,
    recommend the most appropriate program type from the following options:
    - Daily Devotion
    - Daily Prayer
    - Daily Meditation
    - Daily Accountability
    
    USER HISTORY:
    {history_context}
    
    CURRENT REQUEST:
    {user_query}
    
    Also suggest a specific topic that would be beneficial for them.
    
    Return your response in JSON format:
    {{
        "program_type": "one of the four options above",
        "topic": "specific topic suggestion",
        "reasoning": "brief explanation of why this is recommended"
    }}
    
    ONLY return this JSON object, nothing else.
    """
    
    try:
        # Get recommendation from LLM
        response = llm.invoke(prompt)
        
        # Parse the response as JSON
        try:
            # Clean the response to handle potential formatting issues
            clean_response = response.content.strip()
            if clean_response.startswith("```") and clean_response.endswith("```"):
                clean_response = clean_response[3:-3].strip()
            if clean_response.startswith("json") or clean_response.startswith("JSON"):
                clean_response = clean_response[4:].strip()
                
            recommendation = json.loads(clean_response)
            return recommendation
        except json.JSONDecodeError:
            # If parsing fails, return a default
            return {
                "program_type": "Daily Devotion",
                "topic": "General Spiritual Growth",
                "reasoning": "This is a good starting point for spiritual development."
            }
    except Exception as e:
        # If any error occurs, return a default
        return {
            "program_type": "Daily Devotion",
            "topic": "General Spiritual Growth",
            "reasoning": "This is a good starting point for spiritual development."
        }
    
def render_progress_section():
    st.subheader("Your Spiritual Journey")
    
    # Create a placeholder for the progress chart
    progress_data = {
        "Category": ["Devotions", "Prayers", "Meditations", "Accountability"],
        "Completed": [
            st.session_state.progress["completed_devotions"],
            st.session_state.progress["completed_prayers"],
            st.session_state.progress["completed_meditations"],
            st.session_state.progress["accountability_streaks"]
        ]
    }
    
    df = pd.DataFrame(progress_data)
    st.bar_chart(df.set_index("Category"))
    
    # Show completed programs
    if st.session_state.progress["programs"]:
        st.subheader("Completed Programs")
        for program in st.session_state.progress["programs"]:
            st.markdown(f"✅ {program['title']} - {program['completion_date']}")

def render_videos_section(program_topic=None):
    """
    Renders the videos section with search and recommendations.
    
    Args:
        program_topic (str): Optional program topic to focus recommendations
    """
    st.subheader("Recommended Videos")
    
    # Add search functionality
    search_query = st.text_input("Search for videos:", key="video_search_query")
    search_button = st.button("Search", key="video_search_button")
    
    # Fetch base videos
    all_videos = fetch_videos()
    if not all_videos:
        st.info("No videos available right now. Please check back later.")
        return
    
    # Get personalized recommendations
    if search_query and search_button:
        # If user is searching, use their search query for personalization
        videos = get_personalized_video_recommendations(
            user_query=search_query,
            user_history=st.session_state.chat_history if 'chat_history' in st.session_state else None,
            available_videos=all_videos,
            program_topic=program_topic
        )
        if not videos:
            st.warning(f"No videos found matching '{search_query}'")
            return
    elif program_topic:
        # If we have a program topic, prioritize that
        videos = get_topic_specific_videos(
            topic=program_topic,
            specific_focus=st.session_state.get("specific_focus", ""),
            custom_goal=st.session_state.get("custom_goal", ""),
            available_videos=all_videos
        )
    else:
        # Otherwise use general history-based personalization
        videos = get_personalized_video_recommendations(
            user_history=st.session_state.chat_history if 'chat_history' in st.session_state else None,
            available_videos=all_videos
        )
    
    # Display personalization explanation
    if program_topic:
        st.info(f"These videos are specifically selected for your '{program_topic}' program.")
    elif 'chat_history' in st.session_state and len(st.session_state.chat_history) > 0:
        st.info("These videos are personalized based on your conversations and preferences.")
    
    # Display videos in a grid
    cols = st.columns(3)
    for idx, video in enumerate(videos[:6]):  # Limit to 6 videos
        with cols[idx % 3]:
            title = video.get("title", "Untitled")
            video_url = video.get("video_link", "")
            thumbnail = video.get("thumbnail_url", "")
            
            st.markdown(f"#### {title}")
            
            # Show relevance reason if available
            if "relevance_reason" in video:
                st.caption(f"*{video['relevance_reason']}*")
                
            if video_url:
                st.video(video_url)
            elif thumbnail:
                st.image(thumbnail)
                
            if st.button("Save for Later", key=f"save_video_{idx}"):
                # Save to library
                if "video_library" not in st.session_state:
                    st.session_state.video_library = []
                    
                st.session_state.video_library.append({
                    "id": str(uuid.uuid4()),
                    "title": title,
                    "url": video_url,
                    "thumbnail": thumbnail,
                    "saved_date": datetime.now().isoformat()
                })
                
                st.success("Video saved to your library!")

def get_topic_specific_videos(topic, specific_focus=None, custom_goal=None, available_videos=None):
    """
    Get videos specifically related to a spiritual topic and user's focus areas.
    
    Args:
        topic (str): Main topic of the program (e.g., "Anxiety", "Peace")
        specific_focus (str): User's specific focus areas
        custom_goal (str): User's personal goal
        available_videos (list): List of available videos
        
    Returns:
        list: Top 5 most relevant videos for the topic
    """
    if not available_videos:
        available_videos = fetch_videos()
    
    if not available_videos:
        return []
    
    # Create video metadata for the LLM to analyze
    video_metadata = []
    for i, video in enumerate(available_videos):
        video_metadata.append({
            "id": i,
            "title": video.get("title", "Untitled"),
            "description": video.get("description", "No description available"),
            "url": video.get("video_link", ""),
            "thumbnail": video.get("thumbnail_url", "")
        })
    
    # Create prompt to find topic-specific videos
    prompt = f"""
    You are a spiritual content curator helping users find relevant videos for their spiritual journey.
    
    SPIRITUAL PROGRAM TOPIC: {topic}
    USER'S SPECIFIC FOCUS: {specific_focus if specific_focus else "General spiritual growth"}
    USER'S GOAL: {custom_goal if custom_goal else "Deepening spiritual practice"}
    
    AVAILABLE VIDEOS:
    {json.dumps(video_metadata, indent=2)}
    
    Analyze each video's title and description. Find the 5 MOST RELEVANT videos that would specifically help with:
    1. The main topic of "{topic}"
    2. The user's specific focus areas
    3. The user's personal goals
    
    Return a JSON array with the following structure:
    [
      {{
        "id": [video ID number],
        "relevance_score": [score from 0-100],
        "relevance_reason": [1-2 sentences explaining why this video is relevant],
        "day_recommendation": [which program day (1-7) this video would best fit]
      }},
      ...
    ]
    
    ONLY return this JSON array, nothing else.
    """
    
    try:
        # Get recommendations from LLM
        response = llm.invoke(prompt)
        
        # Parse the response as JSON
        try:
            # Clean the response
            clean_response = response.content.strip()
            if clean_response.startswith("```") and clean_response.endswith("```"):
                clean_response = clean_response[3:-3].strip()
            if clean_response.startswith("json") or clean_response.startswith("JSON"):
                clean_response = clean_response[4:].strip()
                
            recommendations = json.loads(clean_response)
            
            # Sort by relevance score (highest first)
            recommendations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Create result with full video data and enriched metadata
            result = []
            for rec in recommendations:
                if rec["id"] < len(available_videos):
                    video = available_videos[rec["id"]]
                    video["relevance_reason"] = rec.get("relevance_reason", "")
                    video["day_recommendation"] = rec.get("day_recommendation", 1)
                    result.append(video)
            
            return result
            
        except json.JSONDecodeError:
            # If parsing fails, return a subset of available videos
            return available_videos[:5]
            
    except Exception as e:
        # Fallback to original videos
        return available_videos[:5]

def render_sos_support():
    st.subheader("Emergency Spiritual Support")
    
    # Get user's accountability area if available
    accountability_area = None
    if "current_program" in st.session_state:
        program = st.session_state.current_program
        if "accountability" in program.get("category", "").lower():
            accountability_area = program.get("topic")
    
    # Create a more targeted prompt based on the user's specific struggle
    sos_prompt = f"""
    You are a compassionate spiritual support coach. Someone is going through a difficult moment of temptation or crisis
    {f'related to {accountability_area}' if accountability_area else ''}.
    
    Provide:
    1. A short, powerful scripture of strength specifically relevant to their struggle
    2. An encouraging statement that acknowledges the difficulty but emphasizes God's power
    3. Three practical actions they can take right now instead of giving in to temptation
    4. A short prayer for strength that they can say aloud
    5. A reminder of why their freedom from this is important (identity in Christ)
    
    Format as markdown with clear headings.
    """
    
    with st.spinner("Preparing spiritual support..."):
        sos_response = llm.invoke(sos_prompt)
        st.markdown(sos_response.content)
    
    st.markdown("---")
    
    # Enhanced contact support section
    st.subheader("Contact Support")
    
    # Add saved contacts option
    if "accountability_contacts" not in st.session_state:
        st.session_state.accountability_contacts = []
    
    if st.session_state.accountability_contacts:
        support_person = st.selectbox(
            "Choose your accountability partner:",
            [contact["name"] for contact in st.session_state.accountability_contacts] + ["Someone else..."]
        )
        
        if support_person == "Someone else...":
            support_person = st.text_input("Name of trusted friend/mentor:")
            support_contact = st.text_input("Their phone number or email:")
            
            if st.button("Save as accountability partner"):
                st.session_state.accountability_contacts.append({
                    "name": support_person,
                    "contact": support_contact
                })
                st.success(f"Added {support_person} as an accountability partner!")
    else:
        support_person = st.text_input("Name of trusted friend/mentor:")
        support_contact = st.text_input("Their phone number or email:")
        
        if support_person and support_contact and st.button("Save as accountability partner"):
            st.session_state.accountability_contacts.append({
                "name": support_person,
                "contact": support_contact
            })
            st.success(f"Added {support_person} as an accountability partner!")
    
    # Message content
    default_message = f"I'm struggling with temptation right now and need support. Could you please check in with me?"
    support_message = st.text_area("Your message:", value=default_message)
    
    # Add location sharing option for emergency situations
    share_location = st.checkbox("Share my current location")
    # Send button with more visible styling
    if st.button("🚨 SEND SOS MESSAGE NOW", type="primary", use_container_width=True):
        if "@" in support_contact.lower():
            # Send email
            if send_email(
                sender_email="your_email@gmail.com", 
                sender_password="your_password", 
                recipient_email=support_contact, 
                subject="Emergency SOS", 
                message=support_message
            ):
                st.success(f"Email sent to {support_contact}!")
            else:
                st.error("Failed to send email. Trying alternate methods...")
                try:
                    # Send WhatsApp message
                    send_whatsapp_message_now(support_contact, default_message)
                    st.success(f"WhatsApp message sent to {support_contact}!")
                except Exception as e:
                    st.error(f"Failed to send WhatsApp message: {str(e)}")
                    st.warning("All message delivery methods failed. Please try again or use another contact method.")
        else:
            try:
                # Send WhatsApp message
                send_whatsapp_message_now(support_contact, default_message)
                st.success(f"Emergency message sent to {support_contact}!")
            except Exception as e:
                st.error(f"Failed to send message: {str(e)}")
            
        # Log this in accountability tracking
        if "accountability_events" not in st.session_state:
            st.session_state.accountability_events = []
            
        st.session_state.accountability_events.append({
            "type": "sos",
            "timestamp": datetime.now().isoformat(),
            "contacted": support_person,
            "outcome": "message_sent"
        })
        
        # Offer follow-up
        st.info("Remember to log the outcome of this encounter for your accountability tracking.")
        
        # Offer call option
        st.markdown(f"### Would you like to call {support_person} directly?")
        if st.button("📞 Call Now"):
            # This would use a telephony API in a real application
            st.info(f"Initiating call to {support_person}...")

def render_sos_button():
    # Make button more visible with styling
    st.markdown("""
    <style>
    .sos-button {
        background-color: #ff3b30;
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        cursor: pointer;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # SOS Button with enhanced visibility
    if st.button("🚨 SOS - I Need Help Now!", key="sos_button", type="primary", use_container_width=True):
        st.session_state.sos_clicked = True

def render_chat_interface():
    # Add custom CSS for fixed input at bottom
    st.markdown("""
    <style>
    /* Main container styles */
    .main {
        padding-bottom: 80px; /* Space for input box */
    }
    
    /* Fixed footer for input */
    .fixed-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 10px 60px;
        border-top: 1px solid #e6e6e6;
        z-index: 1000;
    }
    
    /* Adjust for Streamlit sidebar */
    @media (min-width: 768px) {
        .fixed-input {
            left: 21rem; /* Adjust based on sidebar width */
        }
    }
    
    /* For when sidebar is collapsed */
    [data-testid="stSidebar"][aria-expanded="false"] ~ .fixed-input {
        left: 3.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create main chat display area
    st.subheader("Chat with DSCPL")
    
    # Initialize memory if needed
    if 'vector_store' not in st.session_state:
        initialize_vector_memory()
    
    # Initialize messages list if not exists
    if "displayed_messages" not in st.session_state:
        st.session_state.displayed_messages = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.displayed_messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
    
    # Create a placeholder for new messages
    message_placeholder = st.empty()
    
    # Create HTML container for fixed input
    st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
    
    # Input for new message
    prompt = st.chat_input("What's on your heart today?")
    
    # Close the HTML container
    st.markdown('</div>', unsafe_allow_html=True)
    
    if prompt:
        # Display user message
        with message_placeholder.container():
            st.chat_message("user").write(prompt)
            st.session_state.displayed_messages.append({"role": "user", "content": prompt})
            
            # Add to history and update vector memory
            add_to_chat_history("user", prompt)
            
            # Get response with context-aware conversation handling
            with st.spinner("DSCPL is reflecting..."):
                response = handle_conversation(prompt)
            
            # Display assistant response
            st.chat_message("assistant").write(response)
            st.session_state.displayed_messages.append({"role": "assistant", "content": response})
            add_to_chat_history("assistant", response)
        

def get_devotion_prompt(program_length, selected_topic, custom_goal, specific_focus, history_context, topic_videos):
    # Format video recommendations as markdown
    video_recommendations = ""
    for i, video in enumerate(topic_videos[:min(len(topic_videos), 7)]):
        day_rec = video.get("day_recommendation", i+1)
        if day_rec <= 7:  # Only include for first 7 days
            video_recommendations += f"""
Day {day_rec} Video Recommendation:
- Title: {video.get('title', 'Untitled')}
- Why it's relevant: {video.get('relevance_reason', 'Supports your spiritual journey')}
- URL: {video.get('video_link', '[Video URL will be displayed in app]')}
"""

    return f"""
    You are a spiritual mentor designing a {program_length} Devotion program on {selected_topic}.
    
    The user's goal is: "{custom_goal}"
    They want to focus on: "{specific_focus}"
    
    USER HISTORY CONTEXT:
    {history_context}
    
    AVAILABLE VIDEO RECOMMENDATIONS:
    {video_recommendations}
    
    Create a personalized Devotion program that follows this EXACT format for each day:
    
    # [INSPIRING PROGRAM TITLE]
    
    ## Program Overview
    [Brief overview connecting to their specific situation and goals - 2-3 sentences]
    
    ## Program Schedule
    
    ### Day 1: [Day Title]
    
    #### 5-minute Bible Reading
    **Scripture:** [Specific Bible reference chosen for their needs]
    [2-3 verses quoted directly]
    
    #### Short Prayer
    "[Short prayer related to the day's theme and the user's specific needs]"
    
    #### Faith Declaration
    "[A declaration statement to speak aloud that affirms God's truth related to their situation]"
    
    #### Recommended Video
    *[Use the Day 1 video recommendation from the list above. If none exists for Day 1, suggest an appropriate video title that would complement the day's theme]*
    
    ### Day 2: [Day Title]
    [Same format as Day 1, using Day 2 video if available]
    
    ### Day 3: [Day Title]
    [Same format as Day 1, using Day 3 video if available]
    
    For the remaining days, provide a brief outline of themes.
    
    Format your response in markdown with clear headings as shown above.
    """

def get_prayer_prompt(program_length, selected_topic, custom_goal, specific_focus, history_context, topic_videos):
    # Format video recommendations as markdown
    video_recommendations = ""
    for i, video in enumerate(topic_videos[:min(len(topic_videos), 7)]):
        day_rec = video.get("day_recommendation", i+1)
        if day_rec <= 7:  # Only include for first 7 days
            video_recommendations += f"""
Day {day_rec} Video Recommendation:
- Title: {video.get('title', 'Untitled')}
- Why it's relevant: {video.get('relevance_reason', 'Supports your prayer journey')}
- URL: {video.get('video_link', '[Video URL will be displayed in app]')}
"""

    return f"""
    You are a spiritual mentor designing a {program_length} Prayer program on {selected_topic}.
    
    The user's goal is: "{custom_goal}"
    They want to focus on: "{specific_focus}"
    
    USER HISTORY CONTEXT:
    {history_context}
    
    AVAILABLE VIDEO RECOMMENDATIONS:
    {video_recommendations}
    
    Create a personalized Prayer program following the ACTS model that includes the following EXACT format for each day:
    
    # [INSPIRING PROGRAM TITLE]
    
    ## Program Overview
    [Brief overview connecting to their specific situation and goals - 2-3 sentences]
    
    ## Program Schedule
    
    ### Day 1: [Day Title]
    
    #### Scripture Foundation
    **Scripture:** [Specific Bible reference chosen for their needs]
    [2-3 verses quoted directly]
    
    #### ACTS Prayer Guide
    
    **Adoration:** [Prompt for praising God, related to the day's theme]
    
    **Confession:** [Gentle prompt for reflective repentance related to their situation]
    
    **Thanksgiving:** [Specific gratitude prompt connected to their life circumstances]
    
    **Supplication:** [Guidance for requests, aligned with their stated goals]
    
    #### Daily Prayer Focus
    [Specific prompt for who or what to pray for today]
    
    #### Recommended Video
    *[Use the Day 1 video recommendation from the list above. If none exists for Day 1, suggest an appropriate video title that would complement the day's theme]*
    
    ### Day 2: [Day Title]
    [Same format as Day 1, using Day 2 video if available]
    
    ### Day 3: [Day Title]
    [Same format as Day 1, using Day 3 video if available]
    
    For the remaining days, provide a brief outline of themes.
    
    Format your response in markdown with clear headings as shown above.
    """

def get_meditation_prompt(program_length, selected_topic, custom_goal, specific_focus, history_context, topic_videos):
    # Format video recommendations as markdown
    video_recommendations = ""
    for i, video in enumerate(topic_videos[:min(len(topic_videos), 7)]):
        day_rec = video.get("day_recommendation", i+1)
        if day_rec <= 7:  # Only include for first 7 days
            video_recommendations += f"""
Day {day_rec} Video Recommendation:
- Title: {video.get('title', 'Untitled')}
- Why it's relevant: {video.get('relevance_reason', 'Supports your meditation practice')}
- URL: {video.get('video_link', '[Video URL will be displayed in app]')}
"""

    return f"""
    You are a spiritual mentor designing a {program_length} Meditation program on {selected_topic}.
    
    The user's goal is: "{custom_goal}"
    They want to focus on: "{specific_focus}"
    
    USER HISTORY CONTEXT:
    {history_context}
    
    AVAILABLE VIDEO RECOMMENDATIONS:
    {video_recommendations}
    
    Create a personalized Christian Meditation program that follows this EXACT format for each day:
    
    # [INSPIRING PROGRAM TITLE]
    
    ## Program Overview
    [Brief overview connecting to their specific situation and goals - 2-3 sentences]
    
    ## Program Schedule
    
    ### Day 1: [Day Title]
    
    #### Scripture Focus
    **Scripture:** [Specific Bible reference chosen for their needs]
    [2-3 verses quoted directly]
    
    #### Meditation Prompts
    - What does this reveal about God?
    - [Specific reflection question related to their situation]
    - How can I live this out today?
    
    #### Breathing Guide
    1. Inhale for 4 seconds, focusing on [specific word/phrase from scripture]
    2. Hold for 4 seconds, reflecting on [brief prompt]
    3. Exhale for 4 seconds, releasing [specific concern/emotion related to their situation]
    4. Repeat for 5 minutes
    
    #### Recommended Video
    *[Use the Day 1 video recommendation from the list above. If none exists for Day 1, suggest an appropriate video title that would complement the day's theme]*
    
    ### Day 2: [Day Title]
    [Same format as Day 1, using Day 2 video if available]
    
    ### Day 3: [Day Title]
    [Same format as Day 1, using Day 3 video if available]
    
    For the remaining days, provide a brief outline of themes.
    
    Format your response in markdown with clear headings as shown above.
    """

def get_accountability_prompt(program_length, selected_topic, custom_goal, specific_focus, history_context, topic_videos):
    # Format video recommendations as markdown
    video_recommendations = ""
    for i, video in enumerate(topic_videos[:min(len(topic_videos), 7)]):
        day_rec = video.get("day_recommendation", i+1)
        if day_rec <= 7:  # Only include for first 7 days
            video_recommendations += f"""
Day {day_rec} Video Recommendation:
- Title: {video.get('title', 'Untitled')}
- Why it's relevant: {video.get('relevance_reason', 'Supports your accountability journey')}
- URL: {video.get('video_link', '[Video URL will be displayed in app]')}
"""

    return f"""
    You are a spiritual mentor designing a {program_length} Accountability program on {selected_topic}.
    
    The user's goal is: "{custom_goal}"
    They want to focus on: "{specific_focus}"
    
    USER HISTORY CONTEXT:
    {history_context}
    
    AVAILABLE VIDEO RECOMMENDATIONS:
    {video_recommendations}
    
    Create a personalized Accountability program that follows this EXACT format for each day:
    
    # [INSPIRING PROGRAM TITLE]
    
    ## Program Overview
    [Brief overview connecting to their specific situation and goals - 2-3 sentences]
    
    ## Program Schedule
    
    ### Day 1: [Day Title]
    
    #### Scripture for Strength
    **Scripture:** [Specific Bible reference chosen for their struggle]
    [2-3 verses quoted directly]
    
    #### Truth Declarations
    1. "[Declaration statement that directly counters their specific temptation/struggle]"
    2. "[Declaration statement affirming their identity in Christ]"
    3. "[Declaration statement about God's power in their situation]"
    
    #### Alternative Actions
    Instead of [specific temptation/vice they mentioned], try:
    - [Healthy action 1]
    - [Healthy action 2]
    - [Healthy action 3]
    
    #### SOS Moment Plan
    If temptation becomes overwhelming:
    1. Pray: "[Short emergency prayer]"
    2. Scripture to meditate on: [Brief, memorizable verse]
    3. Action: [Specific distraction technique]
    4. Reach out: Consider contacting a trusted friend using the in-app DM
    
    #### Recommended Video
    *[Use the Day 1 video recommendation from the list above. If none exists for Day 1, suggest an appropriate video title that would complement the day's theme]*
    
    ### Day 2: [Day Title]
    [Same format as Day 1, using Day 2 video if available]
    
    ### Day 3: [Day Title]
    [Same format as Day 1, using Day 3 video if available]
    
    For the remaining days, provide a brief outline of themes.
    
    Format your response in markdown with clear headings as shown above.
    """

def render_program_creation(category, selected_topic=None):
    st.subheader(f"Create Your {category} Program")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Program customization
        program_length = st.selectbox("Program Length:", ["7 days", "14 days", "30 days"], 
                                    key=f"{category.lower().replace(' ', '_')}_program_length_{selected_topic}")
        custom_goal = st.text_input("Your personal goal for this program:", key=f"personal_goal_{category.replace(' ', '_')}_{selected_topic.replace(' ', '_')}")
        specific_focus = st.text_area("Any specific areas you'd like to focus on:",  key=f"personal_goal_area_{category.replace(' ', '_')}_{selected_topic.replace(' ', '_')}")
        
        # Store these in session state for video recommendations
        st.session_state.custom_goal = custom_goal
        st.session_state.specific_focus = specific_focus
        
        # Add calendar sync option
        calendar_sync = st.checkbox("Add to my calendar automatically", value=True, 
                                  key=f"calendar_sync_{category.replace(' ', '_')}_{selected_topic.replace(' ', '_')}")
        
        # Let user choose which calendar service
        if calendar_sync:
            calendar_service = st.radio("Choose calendar service:", 
                                     ["Google Calendar", "Apple Calendar"],
                                     key=f"calendar_service_{category.replace(' ', '_')}_{selected_topic.replace(' ', '_')}")
        
        if st.button("Generate Program", use_container_width=True, key=f"personal_goal_button_{category.replace(' ', '_')}_{selected_topic.replace(' ', '_')}"):
            with st.spinner("Creating your personalized spiritual program..."):
                # Get user history context
                history_context = ""
                if 'chat_history' in st.session_state:
                    recent_history = st.session_state.chat_history[-15:]  # Use most recent 15 interactions
                    user_messages = [item["content"] for item in recent_history if item["role"] == "user"]
                    if user_messages:
                        history_context = "\n".join(user_messages)
                
                # Get topic-specific videos first
                available_videos = fetch_videos()
                topic_videos = get_topic_specific_videos(
                    topic=selected_topic,
                    specific_focus=specific_focus,
                    custom_goal=custom_goal,
                    available_videos=available_videos
                )
                
                # Select the appropriate prompt based on category
                if "Daily Devotion" in category:
                    prompt_template = get_devotion_prompt(program_length, selected_topic, custom_goal, specific_focus, history_context, topic_videos)
                elif "Daily Prayer" in category:
                    prompt_template = get_prayer_prompt(program_length, selected_topic, custom_goal, specific_focus, history_context, topic_videos)
                elif "Daily Meditation" in category:
                    prompt_template = get_meditation_prompt(program_length, selected_topic, custom_goal, specific_focus, history_context, topic_videos)
                elif "Daily Accountability" in category:
                    prompt_template = get_accountability_prompt(program_length, selected_topic, custom_goal, specific_focus, history_context, topic_videos)
                else:
                    # Fallback to a generic prompt if category doesn't match any specific format
                    prompt_template = f"""
                    You are a spiritual mentor designing a {program_length} {category.lower()} program on {selected_topic}.
                    
                    The user's goal is: "{custom_goal}"
                    They want to focus on: "{specific_focus}"
                    
                    USER HISTORY CONTEXT:
                    {history_context}
                    
                    Based on all the above, create a highly personalized spiritual program that addresses their specific needs and challenges.
                    
                    Include:
                    1. An inspiring title for the program that resonates with their personal journey
                    2. A brief overview that connects to their specific situation and goals
                    3. A breakdown of the first 3 days with specific activities and guidance
                    4. A brief outline of the remaining days
                    
                    Format your response in markdown with clear headings.
                    """
                
                program_response = llm.invoke(prompt_template)
                
                # Store program details
                program_id = str(uuid.uuid4())
                program_start = datetime.now().isoformat()
                program_details = {
                    "id": program_id,
                    "title": f"{selected_topic} {category} Program",
                    "category": category,
                    "topic": selected_topic,
                    "length": program_length,
                    "goal": custom_goal,
                    "start_date": program_start,
                    "content": program_response.content,
                    "recommended_videos": topic_videos[:5]  # Store top 5 recommended videos with the program
                }
                
                # Add to session state
                if "current_program" not in st.session_state:
                    st.session_state.current_program = program_details
                
                # Create reminders for the program
                days = int(program_length.split()[0])
                reminders_created = []
                
                for day in range(1, days + 1):
                    reminder_date = (datetime.now() + timedelta(days=day-1)).isoformat()
                    reminder = create_reminder(
                        f"Day {day}: {selected_topic} {category}",
                        reminder_date,
                        f"Continue your spiritual journey - Day {day}",
                        category.lower().replace("daily ", "")
                    )
                    reminders_created.append(reminder)
                
                # Handle calendar integration if selected
                calendar_added = False
                if calendar_sync:
                    if calendar_service == "Google Calendar":
                        with st.spinner("Adding program to Google Calendar..."):
                            calendar_added = add_to_google_calendar(reminders_created)
                    else:  # Apple Calendar
                        with st.spinner("Generating Apple Calendar file..."):
                            try:
                                ical_path = generate_ical_file(reminders_created)
                                with open(ical_path, 'rb') as file:
                                    st.download_button(
                                        label="Download Calendar File",
                                        data=file,
                                        file_name=f"{selected_topic.replace(' ', '_')}_{category.replace(' ', '_')}.ics",
                                        mime="text/calendar"
                                    )
                                calendar_added = True
                                st.info("📱 Open the downloaded file to add these events to your Apple Calendar")
                            except Exception as e:
                                st.error(f"Could not create calendar file: {str(e)}")
                                calendar_added = False
                
                # Show program
                st.markdown(program_response.content)
                
                # Success message about calendar
                if calendar_sync and calendar_added:
                    st.success(f"✅ Program scheduled in your {calendar_service}! You'll receive reminders for each day.")
                
                # Display recommended videos specifically for this program
                st.subheader("Videos For Your Program")
                st.info("These videos are specifically selected to complement your spiritual program.")
                
                # Display top 3 videos
                video_cols = st.columns(3)
                for idx, video in enumerate(topic_videos[:3]):
                    with video_cols[idx]:
                        title = video.get("title", "Untitled")
                        video_url = video.get("video_link", "")
                        thumbnail = video.get("thumbnail_url", "")
                        
                        st.markdown(f"#### {title}")
                        st.caption(f"*{video.get('relevance_reason', '')}*")
                        
                        if video_url:
                            st.video(video_url)
                        elif thumbnail:
                            st.image(thumbnail)
                        
                        st.write(f"Recommended for: Day {video.get('day_recommendation', idx+1)}")
                        
                        if st.button("Save for Later", key=f"save_program_video_{idx}"):
                            if "video_library" not in st.session_state:
                                st.session_state.video_library = []
                            
                            st.session_state.video_library.append({
                                "id": str(uuid.uuid4()),
                                "title": title,
                                "url": video_url,
                                "thumbnail": thumbnail,
                                "saved_date": datetime.now().isoformat(),
                                "program": selected_topic
                            })
                            
                            st.success("Video saved to your library!")
                
                # Add start button
                if st.button("Begin Program Now", use_container_width=True):
                    st.session_state.program_started = True
                    st.session_state.active_tab = "program"
                    st.experimental_rerun()
    
    with col2:
        st.markdown("### Program Benefits")
        st.markdown("""
        ✅ Daily spiritual guidance
        
        ✅ Scripture-based content
        
        ✅ Personalized to your needs
        
        ✅ Calendar integration
        
        ✅ Progress tracking
        
        ✅ Community support
        """)

def render_active_program():
    if "current_program" not in st.session_state:
        st.warning("No active program found. Create one first!")
        return
        
    program = st.session_state.current_program
    st.title(f"Your {program['topic']} {program['category']} Program")
    
    # Calculate current day
    try:
        start_date = datetime.fromisoformat(program["start_date"])
        today = datetime.now()
        days_elapsed = (today - start_date).days + 1
        total_days = int(program["length"].split()[0])
        current_day = min(days_elapsed, total_days)
    except Exception as e:
        st.error(f"Error calculating program days: {str(e)}")
        current_day = 1
        total_days = int(program["length"].split()[0])
    
    # Progress bar
    progress = current_day / total_days
    st.progress(progress)
    st.write(f"Day {current_day} of {total_days}")
    
    # Display program content
    st.markdown(program["content"])
    
    # Daily reflection
    st.subheader(f"Day {current_day} Reflection")
    reflection = st.text_area("What did you learn today? How did God speak to you?")
    
    # Debug: Show current category
    st.write(f"Debug - Program Category: '{program['category']}'")
    
    if st.button("Save Reflection & Complete Day"):
        # Initialize reflection storage
        if "reflections" not in st.session_state:
            st.session_state.reflections = []
            
        # Save reflection
        st.session_state.reflections.append({
            "day": current_day,
            "program_id": program["id"],
            "reflection": reflection,
            "date": datetime.now().isoformat()
        })
        
        # Initialize progress if it doesn't exist
        if "progress" not in st.session_state:
            st.session_state.progress = {
                "completed_devotions": 0,
                "completed_prayers": 0,
                "completed_meditations": 0,
                "accountability_streaks": 0,
                "programs": []
            }
        
        # Update progress based on category - use simpler checking
        category = program["category"].lower()
        st.write(f"Updating progress for category: {category}")
        
        if "devotion" in category:
            update_progress("devotion")
            st.write("Devotion progress updated!")
        elif "prayer" in category:
            update_progress("prayer")
            st.write("Prayer progress updated!")
        elif "meditation" in category:
            update_progress("meditation")
            st.write("Meditation progress updated!")
        elif "accountability" in category:
            update_progress("accountability")
            st.write("Accountability progress updated!")
        else:
            st.write(f"No matching category found for: {category}")
        
        st.success(f"Day {current_day} completed! Keep going!")
        
        # If program completed
        if current_day >= total_days:
            program["completion_date"] = datetime.now().isoformat()
            
            # Make sure the programs list exists
            if "programs" not in st.session_state.progress:
                st.session_state.progress["programs"] = []
                
            st.session_state.progress["programs"].append({
                "title": program["title"],
                "completion_date": datetime.now().strftime("%Y-%m-%d")
            })
            
            save_progress()
            st.balloons()
            st.success("Congratulations! You've completed the program! 🎉")

def update_progress(activity_type):
    """Update user progress and save it to file"""
    # Initialize progress if it doesn't exist
    if "progress" not in st.session_state:
        st.session_state.progress = {
            "completed_devotions": 0,
            "completed_prayers": 0,
            "completed_meditations": 0,
            "accountability_streaks": 0,
            "programs": []
        }
    
    # Update the appropriate counter
    if activity_type == "devotion":
        st.session_state.progress["completed_devotions"] += 1
    elif activity_type == "prayer":
        st.session_state.progress["completed_prayers"] += 1
    elif activity_type == "meditation":
        st.session_state.progress["completed_meditations"] += 1
    elif activity_type == "accountability":
        st.session_state.progress["accountability_streaks"] += 1
    
    # Save progress immediately
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(st.session_state.progress, f)
        return True
    except Exception as e:
        st.error(f"Failed to save progress: {str(e)}")
        return False

def initialize_vector_memory():
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Initialize vector store
    if 'vector_store' not in st.session_state:
        # Load past conversations if available
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
                
            # Convert history to documents for vector store
            documents = []
            for entry in history:
                if entry.get("role") and entry.get("content"):
                    doc_content = f"{entry['role']}: {entry['content']}"
                    if entry.get("timestamp"):
                        doc_content += f" (at {entry['timestamp']})"
                    documents.append(doc_content)
            
            # Create text chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            texts = text_splitter.create_documents(documents)
            
            # Create vector store from texts
            if texts:
                st.session_state.vector_store = FAISS.from_documents(texts, embeddings)
            else:
                st.session_state.vector_store = FAISS.from_texts(["Initial memory"], embeddings)
        else:
            # Initialize with empty memory
            st.session_state.vector_store = FAISS.from_texts(["Initial memory"], embeddings)
    
    # Create retriever
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant memories
    )
    
    # Create memory object
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    
    return memory


# ---------- MAIN APPLICATION ----------
def main():
    # Apply custom CSS
    st.markdown("""
        <style>
        .main .block-container {padding-top: 2rem;}
        h1, h2, h3 {color: #3a86ff;}
        .stButton button {background-color: #3a86ff; color: white;}
        .stProgress .st-bo {background-color: #3a86ff;}
        </style>
    """, unsafe_allow_html=True)
    
    # Set up sidebar
    render_sidebar()
    
    # Main content
    st.title("DSCPL - Your Spiritual Assistant")

    # Personalized greeting if we have user history
    if 'chat_history' in st.session_state and len(st.session_state.chat_history) > 0:
        # Use LLM to generate personalized greeting based on history
        recent_history = st.session_state.chat_history[-15:]
        
        greeting_prompt = f"""
        Based on these recent interactions with the user, generate a personalized greeting.
        If you know their name, use it. Keep it warm but concise (1-2 sentences).
        
        Recent interactions:
        {json.dumps([item for item in recent_history if item["role"] == "user"], indent=2)}
        
        Example: "Welcome back, [Name]! Ready to continue your journey on [topic they mentioned]?"
        Or: "Great to see you again! How's your progress with [something they mentioned]?"
        
        ONLY return the greeting, nothing else.
        """
        
        try:
            greeting_response = llm.invoke(greeting_prompt)
            st.write(greeting_response.content.strip('"'))
        except:
            st.write("What do you need today?")
    else:
        st.write("What do you need today?")
    
    # Show SOS support if activated
    if st.session_state.get("sos_clicked", False):
        render_sos_support()
        if st.button("Return to Main Menu"):
            st.session_state.sos_clicked = False
            st.rerun()
        return
    
    # Main navigation tabs
    tab_labels = ["🏠 Home", "📖 Daily Devotion", "🙏 Daily Prayer", 
                "🧘 Daily Meditation", "🛡️ Daily Accountability", 
                "💬 Just Chat", "📊 Progress", "📹 Videos", "📚 My Library"]

    tabs = st.tabs(tab_labels)
    
    
    # Home tab
    with tabs[0]:
        st.header("Welcome to DSCPL")
        st.write("What do you need today?")
        
        # SOS Button
        render_sos_button()
        
        # Quick access cards - 2x2 grid
         # Use LLM to suggest the most relevant program based on history
        if 'chat_history' in st.session_state and len(st.session_state.chat_history) > 0:
            try:
                recommendation = recommend_program_type(
                    "What would benefit me most right now?",
                    st.session_state.chat_history
                )
                
                # Show personalized recommendation
                st.info(f"**Recommendation:** Try a {recommendation['program_type']} on '{recommendation['topic']}' - {recommendation['reasoning']}")
                
                if st.button("Start Recommended Program"):
                    # Set session state to navigate to the appropriate tab with the recommendation
                    st.session_state.recommended_program = recommendation
                    st.rerun()
            except:
                pass
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📖 Start Devotion", use_container_width=True):
                st.session_state.active_tab = "devotion"
                st.rerun()
            if st.button("🧘 Begin Meditation", use_container_width=True):
                st.session_state.active_tab = "meditation"
                st.rerun()
        with col2:
            if st.button("🙏 Open Prayer", use_container_width=True):
                st.session_state.active_tab = "prayer"
                st.rerun()
            if st.button("🛡️ Seek Accountability", use_container_width=True):
                st.session_state.active_tab = "accountability"
                st.rerun()
        st.subheader("Today's Spiritual Focus")
        today_reminders = [r for r in get_upcoming_reminders(1) 
                        if datetime.fromisoformat(r["date"]).date() == datetime.now().date()]
        if today_reminders:
            for reminder in today_reminders:
                st.info(f"**{reminder['title']}**: {reminder['description']}")
                if st.button("Complete", key=f"today_{reminder['id']}"):
                    mark_reminder_complete(reminder["id"])
                    st.success("Great job! Keep growing spiritually!")
                    st.rerun()
        else:
            st.write("No activities scheduled for today. Why not start a new program?")
        if "current_program" in st.session_state and st.session_state.get("program_started", False):
            st.subheader("Continue Your Active Program")
            program = st.session_state.current_program
            st.write(f"**{program['title']}** - Day {calculate_program_day(program['start_date'])}")
            if st.button("Open Today's Content", key="open_program"):
                st.session_state.active_tab = "program"
                st.rerun()
    with tabs[1]:
        st.header("Daily Devotion")
        devotion_options = [
            "Dealing with Stress", "Overcoming Fear", "Conquering Depression",
            "Relationships", "Healing", "Purpose & Calling", "Anxiety", "Something else..."
        ]
        devotion_type = st.radio("Choose your devotion experience:", [
            "Devotional Reading",
            "Watch Video Verses",
            "Recreate the Bible (Verse-by-Verse)"
        ])
        if devotion_type == "Devotional Reading":
            selected_topic = st.selectbox("Select a topic:", devotion_options, key="selected_topic")
            if selected_topic == "Something else...":
                selected_topic = st.text_input("Enter your topic:")
            if selected_topic:
                render_program_creation("Daily Devotion", selected_topic)
        elif devotion_type == "Watch Video Verses":
            st.subheader("Video Devotionals")
            selected_topic = st.selectbox("Select a topic:", devotion_options, key="video_devotion_topic")
            if selected_topic == "Something else...":
                selected_topic = st.text_input("Enter your topic:", key="custom_video_topic")
            custom_goal = st.text_input("Your personal goal:", key="video_devotion_goal")
            if st.button("Find Videos", key="find_devotion_videos"):
                st.write(f"Showing videos related to: {selected_topic}")
                videos = fetch_videos()
                if videos:
                    st.success(f"Found videos for your {selected_topic} devotional journey!")
                    cols = st.columns(2)
                    for i, video in enumerate(videos[:4]):
                        with cols[i % 2]:
                            title = video.get("title", f"{selected_topic} Devotional")
                            video_url = video.get("video_link", "")
                            st.markdown(f"### {title}")
                            st.write(f"Supports your goal: {custom_goal}")
                            if video_url:
                                st.video(video_url)
                            if st.button("Add to My Library", key=f"lib_devotion_{i}"):
                                if "video_library" not in st.session_state:
                                    st.session_state.video_library = []
                                st.session_state.video_library.append({
                                    "id": str(uuid.uuid4()),
                                    "title": title,
                                    "url": video_url,
                                    "category": "devotion",
                                    "topic": selected_topic,
                                    "saved_date": datetime.now().isoformat()
                                })
                                st.success("Added to your library!")
                else:
                    st.warning("Unable to load videos. Please try again later.")
        elif devotion_type == "Recreate the Bible (Verse-by-Verse)":
            bible_passage = st.text_input("Enter a Bible passage (e.g., John 3:16):")
            if bible_passage and st.button("Create Modern Retelling"):
                with st.spinner("Creating your modern Bible story..."):
                    prompt = f"""
                    You are a spiritual storyteller. Retell {bible_passage} in a modern, engaging way.
                    
                    Include:
                    1. The original verse quoted at the beginning
                    2. A modern retelling that captures the essence and message
                    3. A brief application for today's life
                    
                    Use vivid language, relatable situations, and keep it engaging.
                    """
                    response = llm.invoke(prompt)
                    st.markdown(response.content)
                    if st.button("Save to My Library"):
                        st.success("Story saved to your favorites!")
    with tabs[2]:
        st.header("Daily Prayer")
        prayer_options = [
            "Personal Growth", "Healing", "Family/Friends",
            "Forgiveness", "Finances", "Work/Career", "Something else..."
        ]
        selected_topic = st.selectbox("What would you like to pray about today?", prayer_options, key="prayer_topic")
        if selected_topic == "Something else...":
            selected_topic = st.text_input("Enter your prayer topic:")
        if selected_topic:
            render_program_creation("Daily Prayer", selected_topic)
    with tabs[3]:
        st.header("Daily Meditation")
        meditation_options = [
            "Peace", "God's Presence", "Strength", "Wisdom", "Faith", "Something else..."
        ]
        selected_topic = st.selectbox("Select a meditation focus:", meditation_options, key="meditation_topic")
        if selected_topic == "Something else...":
            selected_topic = st.text_input("Enter your meditation topic:")
        if selected_topic:
            render_program_creation("Daily Meditation", selected_topic)
    with tabs[4]:
        st.header("Daily Accountability")
        
        accountability_options = [
            "Pornography", "Alcohol", "Drugs", "Sex", "Addiction", "Laziness", "Something else..."
        ]
        selected_topic = st.selectbox("What area would you like accountability in?", accountability_options, key="accountability_topic")
        if selected_topic == "Something else...":
            selected_topic = st.text_input("Enter your accountability area:")
        if selected_topic:
            render_program_creation("Daily Accountability", selected_topic)
            st.markdown("---")
            st.subheader("Need immediate help?")
            if st.button("🚨 SOS - I Need Help Now!", key="sos_accountability", use_container_width=True):
                st.session_state.sos_clicked = True
                st.rerun()
    with tabs[5]:
        render_chat_interface()
    with tabs[6]:
        render_progress_section()
    with tabs[7]:
        render_videos_section()
    with tabs[8]:
        st.header("My Library")
        library_tabs = st.tabs(["Saved Videos", "Saved Devotions", "Saved Programs"])
        with library_tabs[0]:
            if "video_library" in st.session_state and st.session_state.video_library:
                st.subheader("Your Saved Videos")
                search = st.text_input("Search your video library:", key="lib_search")
                
                videos = st.session_state.video_library
                if search:
                    videos = [v for v in videos if search.lower() in v.get("title", "").lower()]
                
                if not videos:
                    st.info("No saved videos match your search.")
                else:
                    for i in range(0, len(videos), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i+j < len(videos):
                                video = videos[i+j]
                                with cols[j]:
                                    st.markdown(f"### {video.get('title')}")
                                    if video.get('url'):
                                        st.video(video.get('url'))
                                    elif video.get('thumbnail'):
                                        st.image(video.get('thumbnail'))
                                    
                                    if st.button("Remove", key=f"remove_vid_{video.get('id')}"):
                                        st.session_state.video_library.remove(video)
                                        st.success("Video removed from library")
                                        st.rerun()
            else:
                st.info("You haven't saved any videos yet. Find videos in the Videos tab!")
        with library_tabs[1]:
            st.info("Your saved devotions will appear here")
        with library_tabs[2]:
            if "progress" in st.session_state and "programs" in st.session_state.progress:
                programs = st.session_state.progress["programs"]
                if programs:
                    for program in programs:
                        with st.expander(f"{program.get('title')} - {program.get('completion_date')}"):
                            st.write(f"Completed on: {program.get('completion_date')}")
                            st.button("View Details", key=f"view_prog_{program.get('title')}")
                else:
                    st.info("You haven't completed any programs yet.")
            else:
                st.info("Start and complete programs to see them here!")
    if st.session_state.get("active_tab") == "program" and "current_program" in st.session_state:
        st.markdown("---")
        render_active_program()

def handle_conversation(user_input):
    context = get_personalized_context(user_input)
    prompt = f"""
    You are DSCPL, a compassionate spiritual assistant.
    
    {context}
    
    Respond to the user's current request: "{user_input}"
    
    Keep your response encouraging, scripture-based when appropriate, and focused on their spiritual growth.
    If you previously learned information about the user (like their name, interests, or concerns),
    be sure to reference that information when relevant.
    """
    response = llm.invoke(prompt)
    return response.content
def calculate_program_day(start_date):
    start = datetime.fromisoformat(start_date)
    today = datetime.now()
    return (today - start).days + 1

def render_videos_section():
    st.subheader("Recommended Videos")
    search_query = st.text_input("Search for videos:", key="video_search_query")
    search_button = st.button("Search", key="video_search_button")
    videos = fetch_videos()
    if not videos:
        st.info("No videos available right now. Please check back later.")
        return
    if search_query and search_button:
        filtered_videos = [video for video in videos if search_query.lower() in video.get("title", "").lower()]
        if filtered_videos:
            videos = filtered_videos
        else:
            st.warning(f"No videos found matching '{search_query}'")
    cols = st.columns(3)
    for idx, video in enumerate(videos[:6]):  
        with cols[idx % 3]:
            title = video.get("title", "Untitled")
            video_url = video.get("video_link", "")
            thumbnail = video.get("thumbnail_url", "")          
            st.markdown(f"#### {title}")
            if video_url:
                st.video(video_url)
            elif thumbnail:
                st.image(thumbnail)       
            if st.button("Save for Later", key=f"save_video_{idx}"):
                # Save to library
                if "video_library" not in st.session_state:
                    st.session_state.video_library = []
                
                st.session_state.video_library.append({
                    "id": str(uuid.uuid4()),
                    "title": title,
                    "url": video_url,
                    "thumbnail": thumbnail,
                    "saved_date": datetime.now().isoformat()
                })
                st.success("Video saved to your library!")
if __name__ == "__main__":
    main()