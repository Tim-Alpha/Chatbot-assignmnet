# DSCPL - Your Spiritual Assistant

Welcome to **DSCPL**, a user-friendly web application designed to support your spiritual journey through daily devotions, prayers, meditations, accountability tools, and personalized video recommendations. Whether you're seeking guidance, inspiration, or a way to stay on track with your spiritual goals, DSCPL is here to help!

This README provides a clear and simple guide to understanding, setting up, and running the DSCPL application. It’s written for everyone—whether you’re a beginner or an experienced developer.

---

## 📖 Table of Contents

- What is DSCPL?
- Features
- Prerequisites
- Installation
- Running the Application
- How to Use DSCPL
- File Structure
- Technologies Used

---

## 🌟 What is DSCPL?

DSCPL (pronounced "Disciple") is a web-based spiritual assistant built to help users grow in their faith and stay connected to their spiritual practices. It offers a variety of tools to make your spiritual journey engaging and personalized:

- **Daily Devotions, Prayers, and Meditations**: Create customized programs tailored to your needs.
- **Accountability Support**: Stay on track with reminders and emergency support for challenging moments.
- **Video Recommendations**: Discover inspiring videos that align with your spiritual goals.
- **Calendar Integration**: Sync your spiritual activities with Google Calendar or Apple Calendar.
- **Progress Tracking**: Monitor your growth with metrics and visualizations.
- **Chat Interface**: Have meaningful conversations with DSCPL for guidance and encouragement.

The app is built using **Streamlit**, a simple Python framework for creating web apps, and integrates with AI tools to provide personalized responses and content.

---

## ✨ Features

Here’s what makes DSCPL special:

1. **Personalized Spiritual Programs**:

   - Choose from Devotion, Prayer, Meditation, or Accountability programs.
   - Customize programs with your goals, focus areas, and preferred duration (7, 14, or 30 days).
   - Get daily guidance with scriptures, prayers, and actionable steps.

2. **Calendar Sync**:

   - Add reminders to Google Calendar or download iCal files for Apple Calendar.
   - Never miss a devotion, prayer, or meditation session.

3. **Video Library**:

   - Watch recommended videos based on your interests and program topics.
   - Save videos to your personal library for later viewing.

4. **Progress Tracking**:

   - See your completed devotions, prayers, meditations, and accountability streaks.
   - View a bar chart of your progress and celebrate completed programs.

5. **Emergency Support (SOS)**:

   - Access immediate spiritual guidance during tough moments.
   - Send messages to accountability partners with optional location sharing.

6. **Conversational AI**:

   - Chat with DSCPL for encouragement, advice, or answers to spiritual questions.
   - The app remembers your past conversations to provide context-aware responses.

7. **User-Friendly Interface**:

   - Navigate easily through tabs for Home, Programs, Chat, Videos, and more.
   - A sidebar shows upcoming reminders, progress, and settings.

---

## 🛠️ Prerequisites

Before you can run DSCPL, make sure you have the following installed on your computer:

- **Python 3.8 or higher**: The programming language used to build the app. Download Python.
- **pip**: Python’s package manager (usually comes with Python).
- **Git**: To clone the repository (optional). Download Git.
- A modern web browser (e.g., Chrome, Firefox, Edge) to view the app.
- A **Google account** for Google Calendar integration (optional).
- An internet connection for fetching videos and AI responses.

---

## 📦 Installation

Follow these steps to set up DSCPL on your computer:

1. **Clone the Repository** (or download the code):

   ```bash
   git clone https://github.com/your-username/dscpl.git
   cd dscpl
   ```

   *Note: Replace* `your-username` *with the actual repository owner, or download the ZIP file and extract it.*

2. **Create a Virtual Environment** (recommended): This keeps your project’s dependencies separate from other Python projects.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies**: The app requires several Python libraries. Install them using the provided `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

   If you don’t have a `requirements.txt`, install these packages manually:

   ```bash
   pip install streamlit langchain langchain_community langchain_huggingface faiss-cpu google-auth-oauthlib google-api-python-client icalendar pytz requests pandas
   ```

4. **Set Up Google Calendar API** (optional, for calendar integration):

   - Go to the Google Cloud Console.
   - Create a new project and enable the **Google Calendar API**.
   - Download the `credentials.json` file and place it in the project’s root folder.
   - Follow the prompts during the app’s first run to authenticate.

5. **Prepare Configuration**:

   - Ensure you have write permissions in the project folder for saving chat history, reminders, and progress.
   - No additional API keys are required for basic functionality, but you’ll need a valid `Flic-Token` for video fetching (included in the code).

---

## 🚀 Running the Application

Once installed, starting DSCPL is easy:

1. **Activate the Virtual Environment** (if not already active):

   ```bash
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. **Run the Streamlit App**:

   ```bash
   streamlit run app.py
   ```

   Replace `app.py` with the name of your main Python file (e.g., `dscpl.py` if renamed).

3. **Open the App**:

   - Streamlit will automatically open a browser window at `http://localhost:8501`.
   - If it doesn’t, open your browser and go to that URL.

4. **Explore DSCPL**:

   - Use the sidebar to view reminders and progress.
   - Navigate through tabs to create programs, watch videos, or chat with the assistant.

---

## 📘 How to Use DSCPL

Here’s a quick guide to get started with DSCPL:

1. **Home Tab**:

   - See today’s reminders and recommended programs.
   - Click the **SOS button** if you need urgent help.
   - Start a devotion, prayer, meditation, or accountability session.

2. **Program Tabs** (Devotion, Prayer, Meditation, Accountability):

   - Choose a topic (e.g., "Dealing with Stress" for Devotion).
   - Customize your program by selecting a duration, goal, and focus area.
   - Generate a program and sync it to your calendar.
   - Follow daily guidance and mark tasks as complete.

3. **Chat Tab**:

   - Type a question or share what’s on your heart.
   - DSCPL responds with encouragement and scripture-based advice.

4. **Videos Tab**:

   - Search for videos or browse recommendations.
   - Save videos to your library for later.

5. **Progress Tab**:

   - View your completed activities in a bar chart.
   - Celebrate finished programs.

6. **Library Tab**:

   - Access saved videos, devotions, and completed programs.
   - Remove items you no longer need.

7. **Sidebar**:

   - Check upcoming reminders and complete them.
   - Export reminders to Google or Apple Calendar.
   - Adjust settings like notifications.

8. **SOS Support**:

   - Click the SOS button for immediate spiritual guidance.
   - Contact an accountability partner if needed.

---

## 📁 File Structure

Here’s how the project is organized:

```
dscpl/
├── app.py                # Main application file (or your chosen name)
├── credentials.json      # Google Calendar API credentials (optional)
├── token.json           # Stores Google Calendar authentication tokens
├── users         #stores user directory
       ├── history.json         # Stores user chat history
       ├── reminders.json       # Stores user reminders
       ├── progress.json        # Stores user progress
├── requirements.txt      # List of Python dependencies
├── README.md            # This file
└── venv/                # Virtual environment folder (created after setup)
```

- **app.py**: Contains all the code for the DSCPL app.
- **JSON files**: Store user data persistently.
- **credentials.json** and **token.json**: Used for Google Calendar integration.

---

## 🧑‍💻 Technologies Used

DSCPL is built with the following tools and libraries:

- **Python 3.8+**: The core programming language.
- **Streamlit**: For the web interface.
- **LangChain**: For AI-driven personalization and memory.
- **HuggingFace Embeddings**: To process and store conversation history (`sentence-transformers/all-MiniLM-L6-v2`).
- **FAISS**: For vector-based memory retrieval.
- **Google Calendar API**: For scheduling reminders.
- **iCalendar**: For Apple Calendar compatibility.
- **Requests**: For fetching videos from an external API.
- **Pandas**: For progress visualization.
- **UUID, Datetime, JSON**: For data management.

---

## 🤝 Contributing

We welcome contributions to make DSCPL even better! Here’s how you can help:

1. **Fork the Repository**:

   - Click the "Fork" button on GitHub to create your own copy.

2. **Make Changes**:

   - Clone your fork: `git clone https://github.com/your-username/dscpl.git`
   - Create a new branch: `git checkout -b my-feature`
   - Add your improvements (e.g., new features, bug fixes).

3. **Submit a Pull Request**:

   - Push your changes: `git push origin my-feature`
   - Open a pull request on the original repository.
   - Describe your changes clearly.

4. **Ideas and Feedback**:

   - Open an issue to suggest features or report bugs.

Please follow good coding practices, like writing clear comments and testing your changes.

---

## 🛠️ Troubleshooting

Here are solutions to common issues:

- **Streamlit doesn’t start**:

  - Ensure you’re in the correct directory and the virtual environment is active.
  - Check that all dependencies are installed: `pip install -r requirements.txt`.

- **Google Calendar integration fails**:

  - Verify that `credentials.json` is in the project folder.
  - Delete `token.json` and re-authenticate.
  - Ensure you’ve enabled the Google Calendar API in Google Cloud Console.

- **Videos don’t load**:

  - Check your internet connection.
  - The API token may be invalid. Contact the repository owner for an updated `Flic-Token`.

- **Chat responses are slow**:

  - The AI model may be processing large histories. Try clearing `history.json` (back it up first).

- **Error messages**:

  - Read the error in the terminal or browser for clues.
  - Search the error online or open an issue for help.

For other issues, check the Issues page or ask for help.
