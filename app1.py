import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import requests

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

st.title("DSCPL-AI AGENT")
st.subheader("What do you need today?")

main_choice = st.radio("Choose a category:", [
    "Daily Devotion",
    "Daily Prayer",
    "Daily Meditation",
    "Daily Accountability",
    "Just Chat"
], index=None)

if main_choice == "Daily Devotion":
    user_choice = st.radio("Pick your devotion experience:", [
        "Devotional Reading",
        "    → Watch Video Verses",
        "    → Recreate the Bible (Verse-by-Verse)",
        "    → Inspiration Video Example"
    ])
else:
    user_choice = main_choice

choice_map = {
    "    → Watch Video Verses": "Watch Video Verses",
    "    → Recreate the Bible (Verse-by-Verse)": "Recreate the Bible (Verse-by-Verse)",
    "    → Inspiration Video Example": "Inspiration Video Example"
}
user_choice = choice_map.get(user_choice, user_choice)

if user_choice:
    st.write(f"You selected: {user_choice}")

    with st.expander("Customize Your Experience"):
        custom_goal = st.text_input("Set your own prayer or meditation goal:")
        program_length = st.selectbox("Choose your program length:", ["7 days", "14 days", "30 days"])
        ask_question = st.text_input("Do you have any specific questions for this program?")
        dashboard_enabled = st.checkbox("Enable Progress Dashboard")

    topics = {
        "Devotional Reading": [
            "Dealing with Stress", "Overcoming Fear", "Conquering Depression",
            "Relationships", "Healing", "Purpose & Calling", "Anxiety", "Something else..."
        ],
        "Daily Prayer": [
            "Personal Growth", "Healing", "Family/Friends",
            "Forgiveness", "Finances", "Work/Career", "Something else..."
        ],
        "Daily Meditation": [
            "Peace", "God's Presence", "Strength", "Wisdom", "Faith", "Something else..."
        ],
        "Daily Accountability": [
            "Pornography", "Alcohol", "Drugs", "Sex", "Addiction", "Laziness", "Something else..."
        ]
    }

    selected_topic = None
    if user_choice in topics:
        selected_topic = st.selectbox(f"Choose a topic for {user_choice}:", topics[user_choice])

    bible_passage = None
    if user_choice == "Recreate the Bible (Verse-by-Verse)":
        bible_passage = st.text_input("Enter a Bible passage (e.g., John 1:1):")

    if st.button("Continue to Weekly Plan"):
        st.success(f"{program_length} Spiritual Plan")

        weekly_prompt = f"""
You are a spiritual mentor designing a {program_length} devotional program for someone focused on '{selected_topic or user_choice}'.

Here are their customization inputs:
- Personal Goal: {custom_goal}
- Question: {ask_question}
- Progress Dashboard Enabled: {dashboard_enabled}

Generate a concise daily focus with:
- Title
- Short Scripture
- Daily Action Challenge
- Reflection Thought

Format:
Day 1: Title
Scripture: ...
Challenge: ...
Reflection: ...

Repeat for each day of the program.
End with a weekly encouragement statement that aligns with their goal and question.
"""

        response = llm.invoke(weekly_prompt)
        st.markdown(" Weekly Outline")
        st.markdown(response.content)

    start_program = st.radio("Would you like to begin?", ["Yes", "No"], index=1)

    prompt_map = {
        "Devotional Reading": """
You are a spiritual mentor. Generate today's devotion for the topic '{topic}'.

Include:
- A Bible verse (5-minute reading)
- A short prayer
- A faith declaration
- A suggested video title

Format:
Scripture: ...
Prayer: ...
Declaration: ...
Video: ...
""",
        "Daily Prayer": """
You are a Christian prayer coach. Create a prayer using the ACTS model for the topic '{topic}'.

Include:
- Adoration
- Confession
- Thanksgiving
- Supplication
- A prayer prompt (e.g., Pray for someone who hurt you)

Format:
Adoration: ...
Confession: ...
Thanksgiving: ...
Supplication: ...
Prompt: ...
""",
        "Daily Meditation": """
You are a Christian meditation guide. Create a meditation session on '{topic}'.

Include:
- Scripture Focus
- Meditation Prompts
- Breathing Guide

Format:
Scripture: ...
Prompts: ...
Breathing: ...
""",
        "Daily Accountability": """
You are a Christian accountability partner. Create a devotional entry for someone struggling with '{topic}'.

Include:
- Scripture for strength
- Truth declaration
- Alternative action
- SOS encouragement

Format:
Scripture: ...
Declaration: ...
Action: ...
SOS: ...
"""
    }

    if start_program == "Yes":
        st.success("Your program has started! Daily notifications and reminders will be scheduled.")

        if user_choice in prompt_map and selected_topic:
            prompt = prompt_map[user_choice].format(topic=selected_topic)
            llm_response = llm.invoke(prompt)

            st.markdown("Today's Spiritual Guidance")
            st.text(llm_response.content)

        elif user_choice == "Watch Video Verses":
            video_prompt = """
Generate a short summary of an inspiring Bible passage and suggest a creative title and video concept that can be shared as a 1-minute devotional video. 
Include:
- Bible Verse
- Title
- Video Concept
- Takeaway Message
"""
            llm_response = llm.invoke(video_prompt)
            st.markdown(" Inspiring Video Verses")
            st.markdown("LLM Generated Idea")
            st.text(llm_response.content)

        elif user_choice == "Recreate the Bible (Verse-by-Verse)" and bible_passage:
            story_prompt = f"""
Retell the Bible story of {bible_passage} in a modern, creative way.
Keep the original message intact but make it more engaging for a young audience.
Use storytelling style, modern language, and emotional appeal.
"""
            llm_response = llm.invoke(story_prompt)
            st.markdown(" Recreated Bible Passage")
            st.text(llm_response.content)

        elif user_choice == "Inspiration Video Example":
            idea_prompt = """
Suggest an idea for a Christian inspirational video. 
Include:
- Title
- Main Message
- Bible Reference
- Emotional Theme
- Call to Action
"""
            llm_response = llm.invoke(idea_prompt)
            st.markdown(" Inspirational Video Idea")
            st.text(llm_response.content)

        elif user_choice == "Just Chat":
            chat_input = st.text_input("What’s on your heart today?")
            if chat_input:
                llm_response = llm.invoke(chat_input)
                st.markdown(" Chat Response")
                st.write(llm_response.content)

        if user_choice in ["Watch Video Verses", "Inspiration Video Example"]:
            st.markdown(" Top Community Video Posts (via SocialVerse)")

            url = "https://api.socialverseapp.com/posts/summary/get?page=1&page_size=1000"
            headers = {
                "Flic-Token": "flic_b1c6b09d98e2d4884f61b9b3131dbb27a6af84788e4a25db067a22008ea9cce5",
                "Content-Type": "application/json"
            }

            try:
                api_response = requests.get(url, headers=headers, timeout=10)
                if api_response.status_code == 200:
                    data = api_response.json()
                    posts = data.get("posts", [])

                    if not posts:
                        st.info("No video posts available right now. Please check back later.")
                    else:
                        for post in posts:
                            title = post.get("title", "Untitled")
                            post_summary = post.get("post_summary", {})
                            if isinstance(post_summary, dict):
                                description = post_summary.get("description", "")
                            else:
                                description = ""

                            video_url = post.get("video_link", "")
                            thumbnail = post.get("thumbnail_url", "")

                            with st.container():
                                st.markdown(f"#### {title}")
                                if description:
                                    st.markdown(description[:500] + "...")
                                if video_url:
                                    st.video(video_url)
                                elif thumbnail:
                                    st.image(thumbnail, width=300)
                                else:
                                    st.markdown("No video or image available.")
                                st.markdown("---")
                else:
                    st.error("Failed to fetch posts from SocialVerse.")
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
