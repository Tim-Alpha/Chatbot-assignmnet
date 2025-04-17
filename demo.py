from typing import List, Dict, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from datetime import datetime  # Add this line
import random   

# Configuration
API_KEY = "AIzaSyDL6QfkZ6aTiiuC270rkTs-Tb8nt0dPOfk"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=API_KEY,
    temperature=0.7,
    max_output_tokens=567
)

llm_program = ChatGoogleGenerativeAI(  # For program generation
    model="gemini-2.0-flash",
    google_api_key="AIzaSyDL6QfkZ6aTiiuC270rkTs-Tb8nt0dPOfk",
    temperature=0.7
)


class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_step: str
    current_topic: str 

state = {
    "messages": [],
    "current_step": "initial_selection",
    "current_topic": "",
    "active_program": None ,
    "last_motivation_date": ""
}



# # Define the options and responses
# OPTIONS = {
#     "1": "You selected Daily Devotion. Here's your devotion content.",
#     "2": "You selected Daily Prayer. Here's your prayer guide.",
#     "3" : "You selected Daily Meditation. Here's your meditation exercise.",
#     "4": "You selected Daily Accountability. Let's work on accountability together.",
#     "5": "You selected Just Chat. How can I help you today?"
# }

def devotion_topic_selection(state: AgentState):
    state["messages"].append(AIMessage(
        content="Choose a devotion topic:\n"
                "1. Dealing with Stress\n"
                "2. Overcoming Fear\n"
                "3. Conquering Depression\n"
                "4. Relationships\n"
                "5. Healing\n"
                "6. Purpose & Calling\n"
                "7. Anxiety\n"
                "8. Something else...\n"
                "Select a number between (1-8)"
    ))
    return {**state, "current_step": "await_devotion_topic_selection"}

def prayer_topic_selection(state: AgentState):
    state["messages"].append(AIMessage(
        content="Choose a prayer topic:\n"
                "1. Personal Growth\n"
                "2. Healing\n"
                "3. Family/Friends\n"
                "4. Forgiveness\n"
                "5. Finances\n"
                "6. Work/Career\n"
                "7.Something else...\n"
                "Select a number between (1-7)"
    ))
    return {**state, "current_step": "await_prayer_topic_selection"}

def meditation_topic_selection(state: AgentState):
    state["messages"].append(AIMessage(
        content="Choose a Meditation topic:\n"
                "1. Peace\n"
                "2. God's Presence\n"
                "3. Strength\n"
                "4. Wisdom\n"
                "5. Faith\n"
                "6.Something else...\n"
                "Select a number between (1-6)"
    ))
    return {**state, "current_step": "await_meditation_topic_selection"}

def accountability_topic_selection(state: AgentState):      
    state["messages"].append(AIMessage(
        content="Choose a Accountability topic:\n"
                "1. Pornography\n"
                "2. Alcohol\n"
                "3. Drugs\n"
                "4. Sex\n"
                "5. Addiction\n"
                "6. Laziness\n"
                "7.Something else...\n"
                "Select a number between (1-7)"
    ))
    return {**state, "current_step": "await_accountability_topic_selection"}



# def fetch_recommended_video() -> str:
#     url = "https://api.socialverseapp.com/posts/summary/get?page=1&page_size=1000"
#     headers = {"Flic-Token": "flic_b1c6b09d98e2d4884f61b9b3131dbb27a6af84788e4a25db067a22008ea9cce5"}
    
#     # try:
#     #     response = requests.get(url, headers=headers, timeout=10)
#     #     response.raise_for_status()  # Check HTTP errors
#     #     data = response.json()
        
#     #     # Handle list structure correctly
#     #     if data.get('data') and isinstance(data['data'], list):
#     #         first_post = data['data'][0]  # Access first item in list
#     #         return first_post.get('title', '')
#     #     return ''
    
#     try:
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()  # Triggers HTTPError for 4xx/5xx
#     except requests.exceptions.HTTPError as e:
#         if e.response.status_code == 401:
#             print("Token expired/invalid. Regenerate credentials.")
#         elif e.response.status_code == 403:
#             print("Missing permissions. Update token scope.")


import json
import random
from difflib import get_close_matches

def fetch_recommended_video(topic: str) -> str:
    """Fetch video from output.json using fuzzy category matching"""
    try:
        with open('output.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        posts = data.get('posts', [])
        matches = []
        
        # Topic to keyword mapping
        keyword_map = {
            "Dealing with Stress": ["stress", "anxiety", "peace"],
            "Overcoming Fear": ["fear", "courage", "bravery"],
            "Conquering Depression": ["depression", "hope", "joy"],
            "Relationships": ["relationships", "love", "marriage"],
            "Healing": ["healing", "recovery", "health"],
            "Purpose & Calling": ["purpose", "vocation", "mission"],
            "Anxiety": ["anxiety", "worry", "peace"]
        }
        
        for post in posts:
            
            categories = []
            category_data = post.get('category', {})
            
            if isinstance(category_data, list):
                categories = [str(cat.get('name', '')).lower() for cat in category_data if isinstance(cat, dict)]
            elif isinstance(category_data, dict):
                categories = [str(category_data.get('name', '')).lower()]
                
            
            for keyword in keyword_map.get(topic, []):
                
                direct_matches = [cat for cat in categories if keyword in cat]
                if direct_matches:
                    matches.append(post)
                    break
                    
    
                close_matches = get_close_matches(keyword, categories, n=1, cutoff=0.6)
                if close_matches:
                    matches.append(post)
                    break
                    
        if matches:

            chosen = random.choice(matches)
            return f"{chosen.get('title', 'Untitled')} - {chosen.get('video_link', 'No link')}"
            
        return "No matching videos found"
        
    except FileNotFoundError:
        return "Data file not available"
    except json.JSONDecodeError:
        return "Invalid data format"




def generate_devotion_content(topic: str) -> str:
    """
    Generate daily devotion content based on the selected topic.
    """
    video_info = fetch_recommended_video(topic)
    
    prompt = (
        f"Write a short daily devotion about {topic}. Include:\n"
        "- **Bible Verse**: A relevant verse (include book and chapter)\n"
        "- **Reflection**: A thoughtful paragraph connecting the verse to daily life\n"
        "- **Application**: One practical action to apply today\n"
        f"- **Recommended Resource**: {video_info}\n"
        "Use only bold (**) for section headers. Keep it under 200 words."
    )
    
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Error generating devotion content: {str(e)}"



# def generate_devotion_content(topic: str) -> str:
#     """
#     Generate daily devotion content based on the selected topic.
#     """
#     video_title = fetch_recommended_video()

#     prompt = (
#         f"Write a short daily devotion about {topic}. Include:\n"
#         "- A Bible verse (book and chapter)\n"
#         "- A reflection paragraph\n"
#         "- A practical application for the day\n"
#         f"- A recommended Video: {video_title}\n"
#         "Use bold (**text**) for section headers and avoid using * or other markdown symbols.\n"
        
#     )

#     try:
#         return llm.invoke(prompt).content
#     except Exception as e:
#         return f"Error generating devotion content: {str(e)}"





def generate_prayer_content(topic: str) -> str:
    """
    Generate daily prayer content using the ACTS model for the selected topic.
    """

    prompt = (
        f"Create a prayer guide using the ACTS model (Adoration, Confession, Thanksgiving, Supplication) for {topic}. Include:\n"
        "- Daily prayer focus prompt (e.g., Pray for someone who hurt you or wisdom in a difficult situation)"
        "Use bold (**text**) for section headers and avoid using * or other markdown symbols."
        "write under 567 words"
    )
    return llm_program.invoke(prompt).content


def generate_meditation_content(topic: str) -> str:
    prompt = (
        f"Design a 10-minute meditation exercise for {topic}. Include:\n"
        "- Focus scripture (book and verse)\n"
        "- Breathing guidance\n"
        "- Reflection prompts\n"
        "- Closing prayer\n"
        "Use bold (**text**) for section headers and avoid using * or other markdown symbols."
        "write under 567 words"
    )
    return llm_program.invoke(prompt).content

def generate_accountability_content(topic: str) -> str:
    """
    Generate daily prayer content based on the selected topic.
    """

    prompt = (
        f"Create an accountability plan with:\n- Relevant scripture\n- Truth declaration\n- Action steps\n related to {topic}\n"
        "Use bold (**text**) for section headers and avoid using * or other markdown symbols."
        "write under 567 words"

    )
    return llm_program.invoke(prompt).content


def generate_content(category: str) -> str:
    prompts = {
        "5": "Start a spiritual guidance conversation"
        "Use bold (**text**) for section headers and avoid using * or other markdown symbols."
    }
    return llm.invoke(prompts[category]).content


def generate_program(topic: str, days: int) -> str:
    prompt =  (f"""Create a short {days}-day program for {topic} with:
        - Short but clear daily structure
        - Few actionable steps
        - Short final encouragement
        Use bold (**text**) for section headers and avoid using * or other markdown symbols.
        """
        )
    try:
        return llm_program.invoke(prompt).content
    except Exception as e:
        return f"Error generating program: {str(e)}"

def generate_daily_motivation() -> str:
    """
    Generate a daily motivational message.
    """
    prompt = """write a shtort daily motivation message. Write greetings rather than "good morning" or "good evening".
    """

    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Error generating motivation: {str(e)}"


def initial_selection(state: AgentState):
    # Check if motivation was already shown today
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("last_motivation_date") != today:
        daily_motivation = generate_daily_motivation()
        state["messages"].append(AIMessage(content=f"✨ Daily Motivation ✨\n{daily_motivation}"))
        state["last_motivation_date"] = today

    # Show main menu after motivation
    state["messages"].append(AIMessage(
        content="Welcome to DSCPL, your Daily Devotional Chatbot!\n\n"
                "What do you need today?\n"
                "1. Daily Devotion\n"
                "2. Daily Prayer\n"
                "3. Daily Meditation\n"
                "4. Daily Accountability\n"
                "5. Just Chat\n"
                "Select a number between (1-5)"
    ))
    return {**state, "current_step": "await_selection"}


# Function to handle user selection and provide a response
def handle_selection(state: AgentState):
    last_input = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    ).strip().lower()

    # Map natural language inputs to numeric options
    option_map = {
        "devotion": "1", "daily devotion": "1",
        "prayer": "2", "daily prayer": "2",
        "meditation": "3", "daily meditation": "3",
        "accountability": "4", "daily accountability": "4",
        "chat": "5", "just chat": "5"
    }

    if "resume program" in last_input or "daily check-in" in last_input:
        if state.get("active_program"):
            state["current_step"] = "program_active"
            state["messages"].append(AIMessage(content="""Resuming your program...\n
            Daily Program Check-In:\n
            1. Mark today as complete\n
            2. View progress\n
            3. Exit program\n"""))
            return state
        else:
            state["messages"].append(AIMessage(content="No active program found. Returning to the main menu."))
            state["current_step"] = "initial_selection"
            return state
    # Map input to selection
    selected_option = option_map.get(last_input, last_input)

    if selected_option == "1":
        return devotion_topic_selection(state)
    elif selected_option == "2":
        return prayer_topic_selection(state)
    elif selected_option == "3":
        return meditation_topic_selection(state)
    elif selected_option == "4":
        return accountability_topic_selection(state)
    elif selected_option == "5":
        response = generate_content("5")
        state["messages"].append(AIMessage(content=response))
        return {**state, "current_step": "await_just_chat"}

   # Default to Just Chat for unrecognized inputs
    state["messages"].append(AIMessage(
        content="Let's chat! How can I help you today?\n"
                "(You can always type 'menu' to see options)"
    ))
    return {**state, "current_step": "await_just_chat"}




def handle_devotion_topic_selection(state: AgentState):
    last_input = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    ).strip()

    # Map user input to devotion topics
    topic_map = {
        "1": "Dealing with Stress",
        "2": "Overcoming Fear",
        "3": "Conquering Depression",
        "4": "Relationships",
        "5": "Healing",
        "6": "Purpose & Calling",
        "7": "Anxiety",
        "8": "Something else..."
    }

    if last_input in topic_map:
        if last_input == "8":
            state["messages"].append(AIMessage(content="What would you like help with?"))
            return {**state, "current_step": "await_custom_help"}

        state["current_topic"] = topic_map[last_input]  # Store topic
        response = generate_devotion_content(state["current_topic"])
        state["messages"].append(AIMessage(content=response))
        state["messages"].append(AIMessage(content=f"Would you like to start a structured program for {topic_map[last_input]}? (yes/no)"))
        return {**state, "current_step": "await_program_interest"}


    state["messages"].append(AIMessage(content="Please select a valid topic (1-8)"))
    return {**state, "current_step": "await_devotion_topic_selection"}



def handle_prayer_topic_selection(state: AgentState):
    last_input = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    ).strip()

    # Map user input to devotion topics
    topic_map = {
        "1": "Personal Growth",
        "2": "Healing",
        "3": "Family/Friends",
        "4": "Forgiveness",
        "5": "Finances",
        "6": "Work/Career",
        "7": "Something else..."
    }

    if last_input in topic_map:
        if last_input == "7":  # User selected "Something else"
            state["messages"].append(AIMessage(content="What do you need help with?"))
            return {**state, "current_step": "await_custom_help"}  # Transition to custom help mode

    if last_input in topic_map:
        # Generate prayer content for the selected topic
        topic = topic_map[last_input]
        response = generate_prayer_content(topic)
        state["messages"].append(AIMessage(content=response))
        state["messages"].append(AIMessage(content=f"Would you like to start a structured program for {topic_map[last_input]}? (yes/no)"))
        return {**state, "current_step": "await_program_interest"}

    state["messages"].append(AIMessage(content="Please select a valid topic (1-7)."))
    return {**state, "current_step": "await_prayer_topic_selection"}

def handle_meditation_topic_selection(state: AgentState):
    last_input = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    ).strip()

    # Map user input to devotion topics
    topic_map = {
        "1" : "Peace",
        "2" : "God's Presence",
        "3" : "Strength",
        "4" : "Wisdom",
        "5" : "Faith",
        "6" : "Something else..."
    }
    if last_input in topic_map:
        if last_input == "6":  # User selected "Something else"
            state["messages"].append(AIMessage(content="What do you need help with?"))
            return {**state, "current_step": "await_custom_help"}  # Transition to custom help mode

    if last_input in topic_map:
        # Generate meditation content for the selected topic
        topic = topic_map[last_input]
        response = generate_meditation_content(topic)
        state["messages"].append(AIMessage(content=response))
        state["messages"].append(AIMessage(content=f"Would you like to start a structured program for {topic_map[last_input]}? (yes/no)"))
        return {**state, "current_step": "await_program_interest"}


    state["messages"].append(AIMessage(content="Please select a valid topic (1-6)."))
    return {**state, "current_step": "await_meditation_topic_selection"}

def handle_accountability_topic_selection(state: AgentState):
    last_input = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    ).strip()

    # Map user input to devotion topics
    topic_map = {
        "1" : "Pornography",
        "2" : "Alcohol",
        "3" : "Drugs",
        "4" : "Sex",
        "5" : "Addiction",
        "6" : "Laziness",
        "7" : "Something else..."
    }

    if last_input in topic_map:
        if last_input == "7":  # User selected "Something else"
            state["messages"].append(AIMessage(content="What do you need help with?"))
            return {**state, "current_step": "await_custom_help"}  # Transition to custom help mode

    if last_input in topic_map:
        # Generate meditation content for the selected topic
        topic = topic_map[last_input]
        response = generate_accountability_content(topic)
        state["messages"].append(AIMessage(content=response))
        state["messages"].append(AIMessage(content=f"Would you like to start a structured program for {topic_map[last_input]}? (yes/no)"))
        return {**state, "current_step": "await_program_interest"}


    state["messages"].append(AIMessage(content="Please select a valid topic (1-7)."))
    return {**state, "current_step": "await_accountability_topic_selection"}

def handle_just_chat(state: AgentState):
    """
    Handle ongoing conversation in 'Just Chat' mode.
    """
    # Get the last 5 exchanges for context
    history = "\n".join(
        [f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}" 
         for m in state["messages"][-5:]]
    )

    last_input = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    ).strip()

    # Check if the user wants to resume a program
    if "resume program" in last_input or "daily check-in" in last_input:
        state["messages"].append(AIMessage(content="""Resuming your program... \n Daily Program Check-In:\n
            1. Mark today as complete   \n
            2. View progress \n
            3. Exit program\n1"""))
        return {**state, "current_step": "program_active"}

    # Check if the user wants to start a new program
    if "start program" in last_input:                                           
        state["messages"].append(AIMessage(content="What topic would you like to explore?"))
        return {**state, "current_step": "await_program_interest"}

    # Check if the user wants to end the conversation
    if "end conversation" in last_input or "exit" in last_input:
        state["messages"].append(AIMessage(content="God bless you! Type anything to restart."))
        return {**state, "current_step": "end"}

    # Check if the user wants to see options again  
    if "options" in last_input or "what are the five options" in last_input:
        state["messages"].append(AIMessage(content="Here are the five options:\n"
                                                   "1. Daily Devotion\n"
                                                   "2. Daily Prayer\n"
                                                   "3. Daily Meditation\n"
                                                   "4. Daily Accountability\n"
                                                   "5. Just Chat"))
        return {**state, "current_step": "await_selection"}

    # Check if the user wants to see the main menu
    if "main menu" in last_input or "return to main menu" in last_input:
        state["messages"].append(AIMessage(content="Here are the five options:\n"
                                                   "1. Daily Devotion\n"
                                                   "2. Daily Prayer\n"
                                                   "3. Daily Meditation\n"
                                                   "4. Daily Accountability\n"
                                                   "5. Just Chat"))
        return {**state, "current_step": "await_selection"}

    # Generate a response based on user input and conversation history
    prompt = f"""Conversation history: {history}
        Current user message: {last_input}
        Provide a compassionate, faith-based response that:
        - Acknowledges previous discussions
        - Offers spiritual guidance
        - Asks follow-up questions
        - Uses scripture where appropriate"""

    response = llm.invoke(prompt).content
    state["messages"].append(AIMessage(content=response))
    state["messages"].append(AIMessage(content="What else can I help you with?"))
    return {**state, "current_step": "await_just_chat"}


def handle_custom_help(state: AgentState):
    """
    Handle user input when they select 'Something else' and specify their needs.
    """
    # Get the last 5 messages for context
    history = "\n".join(
        [f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}" 
         for m in state["messages"][-5:]]
    )

    last_input = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    ).strip()

    # Generate a conversational response based on user input and history
    prompt = f"""Conversation Context:{history}
    Current User Message: {last_input}
    Provide spiritual guidance that:
    - References previous discussions if relevant
    - Addresses the current concern
    - Offers practical steps and scripture
    - Asks follow-up questions"""

    response = llm.invoke(prompt).content
    state["messages"].append(AIMessage(content=response))
    state["messages"].append(AIMessage(content="Would you like to explore this further?"))
    return {**state, "current_step": "await_custom_help_response"}


def handle_program_interest(state: AgentState):
    last_input = next(
        (msg.content.lower() for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    ).strip()

    if last_input == "yes":
        state["messages"].append(AIMessage(content="Choose program length (7/14/30 days):"))
        return {**state, "current_step": "await_program_length"}

    elif last_input == "no":
        state["messages"].append(AIMessage(content="What else can I help you with?"))
        return {**state, "current_step": "await_just_chat"}

    # Check if the user wants to see options again  
    if "options" in last_input or "what are the five options" in last_input:
        state["messages"].append(AIMessage(content="Here are the five options:\n"
                                                   "1. Daily Devotion\n"
                                                   "2. Daily Prayer\n"
                                                   "3. Daily Meditation\n"
                                                   "4. Daily Accountability\n"
                                                   "5. Just Chat"))
        return {**state, "current_step": "await_selection"}

    # Check if the user wants to see the main menu
    if "main menu" in last_input or "return to main menu" in last_input:
        state["messages"].append(AIMessage(content="Here are the five options:\n"
                                                   "1. Daily Devotion\n"
                                                   "2. Daily Prayer\n"
                                                   "3. Daily Meditation\n"
                                                   "4. Daily Accountability\n"
                                                   "5. Just Chat"))
        return {**state, "current_step": "await_selection"}


    state["messages"].append(AIMessage(content="Returning to main menu..."))
    return {**state, "current_step": "initial_selection"}


def handle_program_length(state: AgentState):
    last_input = next(
        (msg.content for msg in reversed(state["messages"]) 
         if isinstance(msg, HumanMessage)),
        ""
    ).strip()

    if last_input in ["7", "14", "30"]:
        program = {
            "topic": state["current_topic"],
            "length": int(last_input),
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "progress": 0
        }

        state["active_program"] = program
        program_content = generate_program(program['topic'], int(last_input))

        # Add program content and check-in prompt
        state["messages"].append(AIMessage(
            content=f"""🎉 Program Started!
Topic: {program['topic']}
Duration: {last_input} days
{program_content}"""
        ))

        # Add Daily Check-In message
        state["messages"].append(AIMessage(
            content="""Daily Program Check-In:
1. Mark today as complete   
2. View progress
3. Exit program"""
        ))

        return {**state, "current_step": "program_active"}

    # Handle invalid input for program length
    state["messages"].append(AIMessage(content="Invalid input. Please choose 7/14/30"))
    return {**state, "current_step": "await_program_length"}



def final_response(state: AgentState):
    last_input = next(
        (msg.content.lower() for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    )

    if last_input == "yes":
        # Keep conversation history for continuity
        state["messages"].append(AIMessage(
            content="What else can I help you with today?"
        ))
        return {**state, "current_step": "initial_selection"}

    state["messages"].append(AIMessage(
        content="God bless you! Our conversation history will be saved until you restart."
    ))
    return {**state, "current_step": "end"}


def classify_user_intent(user_input: str) -> str:
    """
    Classify the user's intent based on their input and return the corresponding action.
    """
    prompt = f"""
    Interpret the following user input and classify it into one of these intents:
    - "daily_check_in" for checking into an ongoing program (e.g., "daily check-in", "program progress", "update progress").
    - "view_progress" for viewing program progress (e.g., "view progress", "show my progress").
    - "exit_program" for exiting an ongoing program (e.g., "exit program", "stop the program").
    - "devotion" for choosing Daily Devotion (e.g., "devotion", "I want devotion").
    - "prayer" for choosing Daily Prayer (e.g., "prayer", "I want to pray").
    - "meditation" for choosing Daily Meditation (e.g., "meditation", "I want meditation").
    - "accountability" for choosing Daily Accountability (e.g., "accountability", "I need accountability").
    - "just_chat" for Just Chat mode (e.g., "chat", "I just want to talk").
    - "main_menu" for returning to the main menu (e.g., "main menu", "back to main menu").
    - "invalid" if the input does not match any intent.
    User Input: {user_input}
    Return the intent as a single word from the above options.
    """
    try:
        intent = llm.invoke(prompt).content.strip().lower()
        return intent
    except Exception as e:
        print(f"Error classifying intent: {e}")
        return "invalid"


# def handle_progress(state: AgentState, user_input: str) -> AgentState:
#     """
#     Handle progress updates and daily check-in logic for active programs.
#     """
#     check_in_message = "Daily Program Check-In:\n1. Mark today as complete\n2. View progress\n3. Exit program"
#     if not any(msg.content == check_in_message for msg in state["messages"]):
#         state["messages"].append(AIMessage(content=check_in_message))


#     if user_input == "1":  # Mark today as complete
#         state["active_program"]["progress"] += 1
#         progress = state["active_program"]["progress"]
#         length = state["active_program"]["length"]
#         if progress >= length:
#             state["messages"].append(AIMessage(content="🎉 Program completed! Returning to main menu."))
#             state["current_step"] = "initial_selection"
#             state["active_program"] = None
#         else:
#             state["messages"].append(AIMessage(content=f"Progress updated! Day {progress}/{length}."))
#             state["current_step"] = "program_active"

#     elif user_input == "2":  # View progress
#         topic = state["active_program"]["topic"]
#         progress = state["active_program"]["progress"]
#         length = state["active_program"]["length"]
#         state["messages"].append(AIMessage(content=f"Program: {topic}\nProgress: Day {progress}/{length}."))
#         state["current_step"] = "program_active"

#     elif user_input == "3":  # Exit program
#         state["messages"].append(AIMessage(content="Program paused. You can resume later using 'resume program'."))
#         state["current_step"] = "program_active"

#     else:  # Invalid input
#         state["messages"].append(AIMessage(content="Invalid choice. Please select 1, 2, or 3."))
#         state["current_step"] = "program_active"

#     return state


def handle_progress(state: AgentState, user_input: str) -> AgentState:
    """
    Handle progress updates and daily check-in logic for active programs.
    """
    check_in_message = """Daily Program Check-In:
1. Mark today as complete   
2. View progress
3. Exit program"""

    # Add check-in message if missing
    if not any(msg.content == check_in_message for msg in state["messages"]):
        state["messages"].append(AIMessage(content=check_in_message))

    if not state.get("active_program"):
        state["messages"].append(AIMessage(content="No active program found. Returning to the main menu."))
        state["current_step"] = "initial_selection"
        return state

    if user_input == "1":  # Mark today as complete
        state["active_program"]["progress"] += 1
        progress = state["active_program"]["progress"]
        length = state["active_program"]["length"]
        if progress >= length:
            state["messages"].append(AIMessage(content="🎉 Program completed! Returning to main menu."))
            state["current_step"] = "initial_selection"
            state["active_program"] = None
        else:
            state["messages"].append(AIMessage(content=f"Progress updated! Day {progress}/{length}."))
            state["current_step"] = "program_active"

    elif user_input == "2":  # View progress
        topic = state["active_program"]["topic"]
        progress = state["active_program"]["progress"]
        length = state["active_program"]["length"]
        state["messages"].append(AIMessage(content=f"Program: {topic}\nProgress: Day {progress}/{length}."))
        state["current_step"] = "program_active"

    elif user_input == "3":  # Exit program
        state["messages"].append(AIMessage(content="Program paused. You can resume later using 'resume program'."))
        state["current_step"] = "initial_selection"

    else:  # Invalid input
        state["messages"].append(AIMessage(content="Invalid choice. Please select 1, 2, or 3."))

    return state



if __name__ == "__main__":

    state = {
        "messages": [],
        "current_step": "initial_selection",
        "current_topic": "",  
        "prev_messages": [],
        "active_program": None,
        "last_motivation_date": "" 
    }


    displayed_messages = set()

    while True:
        print(f"DEBUG: Current step is {state['current_step']}")  # Debugging the flow

        if state["current_step"] == "initial_selection":
            state = initial_selection(state)

        # Print AI messages that haven't been displayed yet
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.content not in displayed_messages:
                print("\n" + "=" * 50)
                print(msg.content)
                displayed_messages.add(msg.content)  # Mark as displayed

        # Handle user input based on current step
        if state["current_step"] in ["await_selection", "final_response"]:
            user_input = input("\nYour response: ").strip()
            if not user_input:
                state["messages"].append(AIMessage(content="Input cannot be empty. Please try again."))
                continue
            state["messages"].append(HumanMessage(content=user_input))
            state = handle_selection(state) if state["current_step"] == "await_selection" else final_response(state)


        elif state["current_step"] == "await_devotion_topic_selection":
            user_input = input("\nYour response: ")
            state["messages"].append(HumanMessage(content=user_input))
            state = handle_devotion_topic_selection(state)

        elif state["current_step"] == "await_prayer_topic_selection":
            user_input = input("\nYour response: ")
            state["messages"].append(HumanMessage(content=user_input))
            state = handle_prayer_topic_selection(state)

        elif state["current_step"] == "await_meditation_topic_selection":
            user_input = input("\nYour response: ")
            state["messages"].append(HumanMessage(content=user_input))
            state = handle_meditation_topic_selection(state)

        elif state["current_step"] == "await_accountability_topic_selection":
            user_input = input("\nYour response: ")
            state["messages"].append(HumanMessage(content=user_input))
            state = handle_accountability_topic_selection(state)

        elif state["current_step"] == "await_just_chat":
            user_input = input("\nYour response: ")
            state["messages"].append(HumanMessage(content=user_input))
            state = handle_just_chat(state)

        elif state["current_step"] == "await_custom_help":
            user_input = input("\nYour response: ")
            state["messages"].append(HumanMessage(content=user_input))
            state = handle_custom_help(state)

        elif state["current_step"] == "await_custom_help_response":
            user_input = input("\nYour response: ")
            if user_input.lower() == "no":
                state["messages"].append(AIMessage(content="God bless you! Type anything to restart."))
                state = {"messages": [], "current_step": "end"}
            else:
                state["messages"].append(AIMessage(content="What else can I help you with?"))
                state["current_step"] = "await_custom_help"

        elif state["current_step"] == "await_program_interest": 
            user_input = input("\nYour response: ")
            state["messages"].append(HumanMessage(content=user_input))
            state = handle_program_interest(state)

        elif state["current_step"] == "await_program_length":
            user_input = input("\nYour response: ")
            state["messages"].append(HumanMessage(content=user_input))
            state = handle_program_length(state)

        elif state["current_step"] == "program_active":
            user_input = input("\nYour response: ")
            state["messages"].append(HumanMessage(content=user_input))
            state = handle_progress(state, user_input)

        intent = classify_user_intent(user_input)

        if intent == "devotion":
            state = devotion_topic_selection(state)
        elif intent == "prayer":
            state = prayer_topic_selection(state)
        elif intent == "meditation":
            state = meditation_topic_selection(state)
        elif intent == "accountability":
            state = accountability_topic_selection(state)
        elif intent == "just_chat":
            state["messages"].append(AIMessage(content="You selected Just Chat. How can I help you today?"))
            state["current_step"] = "await_just_chat"
        elif intent == "daily_check_in":
            state["messages"].append(AIMessage(content=f"""Daily Program Check-In:
1. Mark today as complete   
2. View progress
3. Exit program"""))
            state["current_step"] = "program_active"
        elif intent == "view_progress":
            if state.get("active_program"):
                progress = state["active_program"]["progress"]
                length = state["active_program"]["length"]
                state["messages"].append(AIMessage(
                    content=f"Program: {state['active_program']['topic']}\nProgress: Day {progress}/{length}"
                ))
            else:
                state["messages"].append(AIMessage(content="No active program found."))
        elif intent == "exit_program":
            state["messages"].append(AIMessage(content="Exiting the program. Returning to the main menu."))
            state["current_step"] = "initial_selection"
        elif intent == "main_menu":
            state = initial_selection(state)
