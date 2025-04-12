import streamlit as st
import os
import random
import requests
import re
import google.generativeai as genai


genai.configure(api_key="AIzaSyBJ65jV-r2Sn4sinZ99LW1E05n9vNJcRzY")
llm = genai.GenerativeModel("models/gemini-1.5-flash")

st.set_page_config(page_title="DSCPL - Chat", layout="centered")


st.markdown("""
    <style>
    .main {background-color: #0e0e0e;}
    .stButton>button {
        background-color: #2c2c2c;
        color: #fff;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .chat-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #e0dede;
    }
    .chat-bubble-user {
        background-color: #2a2a2a;
        padding: 0.75rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        text-align: right;
        color: white;
    }
    .chat-bubble-bot {
        background-color: #333333;  /* Darker background for LLM response */
        padding: 0.75rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        text-align: left;
        color: #e0dede;
        white-space: pre-line;
    }
    .video-container {
        margin-top: 1rem;
        border-radius: 12px;
        background-color: #2a2a2a;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def generate_response(prompt):
    try:
        with st.spinner("Thinking..."):
            response = llm.generate_content(prompt)
            if not response.parts:
                raise ValueError("No content")
            text = response.text.strip()
            text = re.sub(r'\n{3,}', '\n\n', text)  
            text = re.sub(r' +', ' ', text)         
            text = text.replace('\n ', '\n')       
            return text
    except Exception:
        return "Please try rephrasing."


def postprocess_devotion(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    text = re.sub(r'(?<=\n)(\d\.\s)(.*?)(?=\n|$)', r'**\1\2**', text) 
    return text


def get_recommended_video(topic=None):
    url = "https://api.socialverseapp.com/posts/summary/get?page=1&page_size=1000"
    headers = {
        "Flic-Token": "flic_b1c6b09d98e2d4884f61b9b3131dbb27a6af84788e4a25db067a22008ea9cce5"
    }
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            posts = res.json().get("posts", [])
            if topic:
                topic = topic.lower()
                filtered = [p for p in posts if topic in p.get("title", "").lower() or topic in p.get("description", "").lower()]
                post = random.choice(filtered) if filtered else random.choice(posts)
            else:
                post = random.choice(posts)

            title = post.get("title")
            thumbnail = post.get("video_thumbnail") or post.get("image_url")
            video_url = post.get("video_url") or post.get("video_link")
            return title, thumbnail, video_url
    except Exception:
        st.warning("Could not fetch video.")
    return None, None, None


if 'latest_response' not in st.session_state:
    st.session_state.latest_response = None


st.markdown("<h1 style='text-align: center; color: #D4AF37;'>DSCPL</h1>", unsafe_allow_html=True)
st.markdown('<div class="chat-box">🤖 What do you need?</div>', unsafe_allow_html=True)


categories = ["Daily Devotion", "Daily Prayer", "Daily Meditation", "Daily Accountability", "Just Chat"]
category = st.radio("Select a category", categories, horizontal=True)


topics_dict = {
    "Daily Devotion": ["Dealing with Stress", "Overcoming Fear", "Conquering Depression", "Relationships", "Healing", "Purpose & Calling", "Anxiety", "Something else..."],
    "Daily Prayer": ["Personal Growth", "Healing", "Forgiveness", "Finances", "Work/Career", "Something else..."],
    "Daily Meditation": ["Peace", "God's Presence", "Strength", "Wisdom", "Faith", "Something else..."],
    "Daily Accountability": ["Pornography", "Alcohol", "Drugs", "Sex", "Addiction", "Laziness", "Something else..."]
}


days_options = ["Today Only", "3 Days", "Custom"]
multi_day_categories = ["Daily Devotion", "Daily Prayer", "Daily Meditation", "Daily Accountability"]

duration, topic = None, None

if category in multi_day_categories:
    st.markdown(f"<div class='chat-box'>🙏 How many days would you like to commit to {category.lower()}?</div>", unsafe_allow_html=True)
    duration = st.radio("Choose Duration", days_options, horizontal=True, label_visibility="collapsed")
    if duration == "Custom":
        custom_days = st.number_input("Enter number of days", min_value=1, step=1)
        duration = f"{custom_days} Days"

    st.markdown(f"<div class='chat-box'>🧠 What topic would you like for your {category.lower()}?</div>", unsafe_allow_html=True)
    topic = st.selectbox("Choose Topic", topics_dict[category])
    if topic == "Something else...":
        topic = st.text_input("What's on your heart today?")

    if topic and st.button("Generate ✨"):
        
        if category == "Daily Devotion":
            prompt = f"""You are a spiritual content writer.

Generate a devotional for {duration.lower()} on the topic: "{topic}".

The format **must** strictly be:

**5-Minute Bible Reading:**
<include 2 Bible verses with references, each on a new line>

**Short Prayer:**
<write a heartfelt prayer in 3-4 lines>

**Declaration:**
1. <short bold declaration>
2. <short bold declaration>
3. <short bold declaration>
4. <short bold declaration>
5. <short bold declaration>

- DO NOT add any title, heading, or intro before these sections.
- Use double newlines between sections.
- Keep declarations **numbered and bolded** like shown."""
        elif category == "Daily Prayer":
            prompt = f"""You are a spiritual guide.

Create a {duration.lower()} daily prayer using the ACTS format on the topic: "{topic}".

Format:
**Adoration:**
<2-3 sentences>

**Confession:**
<2-3 sentences>

**Thanksgiving:**
<2-3 sentences>

**Supplication:**
<2-3 sentences>

**Daily Prayer Focus:**
<one short, actionable focus>

- DO NOT add headings, intros, or commentary.
- Maintain clean formatting and newline spacing."""
        elif category == "Daily Meditation":
            prompt = f"""You are a meditation instructor.

Write a {duration.lower()} Christian meditation plan on "{topic}".

Format:
**Scripture Focus:**
<one scripture with reference>

**Meditation Prompts:**
<2-3 bullet point prompts for reflection>

**Breathing Guide:**
<simple 3-4 step guide for breathing during meditation>

- Do not include intros or extra titles.
- Keep whitespace between sections."""
        elif category == "Daily Accountability":
            prompt = f"""You are a Christian accountability partner.

Provide a {duration.lower()} accountability guide on the issue of: "{topic}".

Format:
**Scripture of Strength:**
<one empowering verse>

**Truth Declarations:**
1. <bold, empowering statement>
2. <bold, empowering statement>
3. <bold, empowering statement>

**Alternative Actions:**
<list 2-3 better actions to take when tempted>

**SOS Feature:**
<quick 1-2 line guidance for reaching out or emergency prayer>

- No extra fluff or titles. Keep clean layout with line breaks."""

        output = postprocess_devotion(generate_response(prompt))
        title, thumb, link = get_recommended_video(topic)

        
        video_html = ""
        if title and link:
            video_html += f"🎬 <b>Recommended Video:</b> <i>{title}</i><br>"
            if thumb:
                video_html += f"<img src='{thumb}' style='width:100%; border-radius: 12px;'><br>"
            video_html += f"""
            <iframe width="100%" height="320" src="{link}" 
            frameborder="0" allowfullscreen style="border-radius:12px;"></iframe>
            """

        st.session_state.latest_response = f"<div class='chat-bubble-user'>{category} → {duration} → {topic}</div>" + \
                                           f"<div class='chat-bubble-bot'>{output}</div>" + \
                                           f"<div class='video-container'>{video_html}</div>"

elif category == "Just Chat":
    user_query = st.text_input("Ask your question:")
    if user_query and st.button("Send"):
        response = generate_response(user_query)
        title, thumb, link = get_recommended_video(user_query)

        
        video_html = ""
        if title and link:
            video_html += f"🎬 <b>Recommended Video:</b> <i>{title}</i><br>"
            if thumb:
                video_html += f"<img src='{thumb}' style='width:100%; border-radius: 12px;'><br>"
            video_html += f"""
            <iframe width="100%" height="320" src="{link}" 
            frameborder="0" allowfullscreen style="border-radius:12px;"></iframe>
            """

        st.session_state.latest_response = f"<div class='chat-bubble-user'>{user_query}</div>" + \
                                           f"<div class='chat-bubble-bot'>{response}</div>" + \
                                           f"<div class='video-container'>{video_html}</div>"


if st.session_state.latest_response:
    st.markdown(st.session_state.latest_response, unsafe_allow_html=True)
