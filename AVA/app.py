import time
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv
from bson import ObjectId
import random
import os
from flask_socketio import SocketIO, emit
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from flask import Response
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import REGISTRY
from functools import wraps

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ── Flask + extensions ────────────────────────────────────────────────────────
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# ── Prometheus counters (safe re-registration) ────────────────────────────────
def _get_or_create_counter(name, description, labels=None):
    if name not in REGISTRY._names_to_collectors:
        return Counter(name, description, labels or [])
    return REGISTRY._names_to_collectors[name]

def _get_or_create_histogram(name, description, labels=None):
    if name not in REGISTRY._names_to_collectors:
        return Histogram(name, description, labels or [])
    return REGISTRY._names_to_collectors[name]

REQUEST_COUNT   = _get_or_create_counter('ava_request_count',           'Total HTTP requests',                    ['method', 'endpoint'])
REQUEST_LATENCY = _get_or_create_histogram('ava_request_latency_seconds','Latency of HTTP requests in seconds',   ['endpoint'])
REMINDER_SAVED  = _get_or_create_counter('ava_reminder_saved_total',    'Total reminders saved')

def track_metrics(endpoint_name):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            REQUEST_COUNT.labels(method=request.method, endpoint=endpoint_name).inc()
            response = f(*args, **kwargs)
            REQUEST_LATENCY.labels(endpoint=endpoint_name).observe(time.time() - start)
            return response
        return wrapper
    return decorator

socketio  = SocketIO(app)
dt        = datetime.today()
TODAY     = dt.strftime('%A').lower()
scheduler = BackgroundScheduler()
scheduler.start()

app.secret_key = os.getenv("SECRET_KEY", "test-secret-key")

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    mongo_client     = MongoClient(MONGO_URI)
    db               = mongo_client["test"]
    users_collection = db["users"]
else:
    mongo_client     = None
    db               = None
    users_collection = None

if users_collection is None:
    class DummyInsertResult:
        def __init__(self, inserted_id): self.inserted_id = inserted_id

    class DummyUpdateResult:
        def __init__(self, matched_count): self.matched_count = matched_count

    class DummyCollection:
        def __init__(self): self.storage = []

        def find_one(self, query):
            for doc in self.storage:
                if all(doc.get(k) == v for k, v in query.items()):
                    return doc
            return None

        def insert_one(self, doc):
            doc["_id"] = ObjectId()
            self.storage.append(doc)
            return DummyInsertResult(doc["_id"])

        def update_one(self, query, update):
            doc = self.find_one(query)
            if not doc:
                return DummyUpdateResult(0)
            if "$push" in update:
                for k, v in update["$push"].items():
                    doc.setdefault(k, [])
                    if isinstance(v, dict) and "$each" in v:
                        doc[k].extend(v["$each"])
                    else:
                        doc[k].append(v)
            return DummyUpdateResult(1)

        def delete_one(self, query):
            self.storage = [d for d in self.storage
                            if not all(d.get(k) == v for k, v in query.items())]
            return None

    users_collection = DummyCollection()


# ── LangChain agent factory ───────────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template(
    """You are AVA, a compassionate and knowledgeable mental health and health assistant.
Your role is to listen to the user's emotional concerns and provide supportive advice.
Act as a friend or caretaker. Use simple, everyday language and keep a conversational tone.
Avoid technical terms. At the end of each response, ask a specific follow-up question
related to the user's health or well-being. Use strictly 1-2 short sentences (max 50 words).
You are NOT a substitute for a real doctor – always remind users to seek professional
care for serious concerns.

You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat up to 3 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""
)

# One agent executor per logged-in user, keyed by user_id string.
# This keeps each user's LangChain memory completely separate.
_user_agents: dict[str, AgentExecutor] = {}


def _build_agent_executor(chat_history: list) -> AgentExecutor:
    """
    Build a fresh AgentExecutor pre-seeded with the user's MongoDB chat history
    so memory survives server restarts as long as history is in the DB.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise EnvironmentError("API_KEY is not set in .env")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.4,
        convert_system_message_to_human=True,
    )

    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="web_search",
            func=search.run,
            description=(
                "Search the internet for current health information such as symptoms, "
                "conditions, treatments, medications, or mental health resources. "
                "Use this when the user asks something that needs up-to-date factual "
                "medical or health knowledge. Input must be a plain-English search query."
            ),
        )
    ]

    # Seed LangChain memory from MongoDB so context carries across restarts
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=10,
        return_messages=False,
    )
    for msg in chat_history[-20:]:   # seed with last 20 stored messages
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            memory.chat_memory.add_user_message(content)
        elif role == "bot":
            memory.chat_memory.add_ai_message(content)

    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4,
    )


def get_agent_executor() -> AgentExecutor:
    """
    Return (or lazily create) the AgentExecutor for the current user.
    Raises ValueError if no user is logged in.
    """
    user_id = session.get("user_id")
    if not user_id:
        raise ValueError("User not logged in.")

    if user_id not in _user_agents:
        # Load existing chat history from DB to seed memory
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        history = user.get("chat_history", []) if user else []
        _user_agents[user_id] = _build_agent_executor(history)

    return _user_agents[user_id]


def _evict_agent(user_id: str):
    """Remove a user's agent from the cache (call on logout)."""
    _user_agents.pop(user_id, None)


# ── Helper functions ──────────────────────────────────────────────────────────

def process_messages(messages):
    return [
        {
            'role': msg.get('role'),
            'content': msg.get('content'),
            'timestamp': msg.get('timestamp') or datetime.now().isoformat()
        }
        for msg in messages
    ]

def get_user_id():
    user_id = session.get('user_id')
    if not user_id:
        raise ValueError("User ID not found in session.")
    return ObjectId(user_id)

def get_chat_history():
    user_id = get_user_id()
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if user and "chat_history" in user:
        return user["chat_history"]
    return []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/reminders')
def reminders():
    return render_template('reminders.html')


@app.route('/api/initChat', methods=['POST'])
def init_chat():
    """
    Called when the chatbot page loads.
    Returns a greeting – either a random opener for new users or a
    context-aware follow-up for returning users.
    """
    try:
        chat_history = get_chat_history()

        if not chat_history:
            initial_prompts = [
                "Hey there! How are you feeling today?",
                "Hi! I'm here for you. What's on your mind?",
                "Hello! How's everything going for you today?",
                "Hey! I'm ready to listen. How are you feeling?",
                "Hi there! How are you doing, both physically and mentally?",
                "Hello! What's something you'd like to talk about today?",
                "Hi! How's your day been so far?",
                "Hey! It's good to see you. How are you holding up?",
                "Hello! How can I support you today?",
                "Hi! What's been on your mind lately?",
            ]
            bot_response = random.choice(initial_prompts)
        else:
            # Build a short history string and ask the agent for a warm follow-up
            recent = chat_history[-6:]
            history_str = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in recent
            )
            prompt = (
                "As AVA, a mental health assistant, review the recent chat history "
                "and craft a brief, empathetic follow-up greeting for the user's return. "
                "Use exactly 1-2 short sentences.\n\n"
                f"{history_str}"
            )
            agent = get_agent_executor()
            result = agent.invoke({"input": prompt})
            bot_response = result["output"]
            bot_response += " Would you like to continue talking about this, or is there something new on your mind?"

        return jsonify({"reply": bot_response})

    except Exception as error:
        print("Error generating initial response:", error)
        return jsonify({"error": "Error generating initial response"}), 500


@app.route('/api/chat', methods=['POST'])
@track_metrics("api_chat")
def chat():
    data       = request.json
    user_input = data.get('input', '')
    user_id    = data.get('userId')

    try:
        agent      = get_agent_executor()
        result     = agent.invoke({"input": user_input})
        bot_response = result["output"]

        # Persist both sides of the exchange to MongoDB
        if user_id:
            try:
                users_collection.update_one(
                    {'_id': ObjectId(user_id)},
                    {'$push': {
                        'chat_history': {'$each': [
                            {'role': 'user', 'content': user_input,    'timestamp': datetime.now()},
                            {'role': 'bot',  'content': bot_response,  'timestamp': datetime.now()},
                        ]}
                    }}
                )
                print("Chat history saved successfully.")
            except Exception as db_error:
                print("Database error:", db_error)

        return jsonify({"reply": bot_response})

    except Exception as error:
        print("Error generating response:", error)
        return jsonify({"error": "Error generating response"}), 500


@app.route('/save_history', methods=['POST'])
def save_history():
    if request.method == 'POST' and request.is_json:
        data = request.get_json()
        if 'messages' not in data:
            return jsonify({"error": "'messages' field is required."}), 400
        try:
            formatted_messages = process_messages(data['messages'])
            user_id = get_user_id()
            result  = users_collection.update_one(
                {'_id': user_id},
                {'$push': {'chat_history': {'$each': formatted_messages}}}
            )
            if result.matched_count == 0:
                return jsonify({"error": "User not found."}), 404
            return jsonify({"message": "Chat history updated successfully"}), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 400


def send_reminder(reminder):
    socketio.emit('reminder', reminder)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user     = users_collection.find_one({"username": username})

        if user and check_password_hash(user["password"], password):
            session['user_id'] = str(user["_id"])
            flash("Login successful!")

            if user.get('reminders'):
                for reminder in user['reminders']:
                    day = reminder['day'].lower()
                    if day == TODAY:
                        now = datetime.now()
                        t   = reminder['time']
                        reminder_datetime = datetime.combine(
                            now.date(),
                            datetime.strptime(t, "%H:%M").time()
                        )
                        scheduler.add_job(
                            send_reminder,
                            trigger=DateTrigger(run_date=reminder_datetime),
                            args=[reminder],
                        )
            return redirect(url_for('chatbot'))
        else:
            flash("Invalid username or password")
            return render_template('login.html')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username  = request.form['username']
        password1 = request.form['password1']
        password2 = request.form['password2']
        user      = users_collection.find_one({"username": username})

        if user is None:
            if password1 != password2:
                flash("Passwords do not match")
                return redirect(url_for('signup'))
            new_user = {
                "username":     username,
                "password":     generate_password_hash(password1),
                "chat_history": [],
                "reminders":    [],
            }
            users_collection.insert_one(new_user)
            session['user_id'] = str(new_user["_id"])
            return redirect(url_for('chatbot'))
        else:
            flash("Username is taken.")
            return redirect(url_for('signup'))
    return render_template('signup.html')


@app.route('/logout')
def logout():
    user_id = session.pop('user_id', None)
    if user_id:
        _evict_agent(user_id)   # free the agent's memory on logout
    scheduler.remove_all_jobs()
    flash("Logged out successfully")
    return render_template('login.html')


@app.route('/send_messages', methods=['POST'])
def send_messages():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    message = request.form.get('message')
    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    save_to_chat_history(user_id, message)

    try:
        agent        = get_agent_executor()
        result       = agent.invoke({"input": message})
        bot_response = result["output"]
    except Exception as e:
        print("Agent error:", e)
        bot_response = "I'm sorry, I ran into a problem. Please try again."

    save_to_chat_history(user_id, bot_response)
    return jsonify({"reply": bot_response})


def save_to_chat_history(user_id, message):
    timestamp = datetime.now()
    users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$push": {"chat_history": {"message": message, "timestamp": timestamp}}}
    )


@app.route('/save_reminder', methods=['POST'])
@track_metrics("save_reminder")
def save_reminder():
    user_id = session['user_id']
    data    = request.get_json()
    day     = data['day'].lower()
    t       = data['time']

    users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$push": {"reminders": {
            "text":      data['text'],
            "time":      t,
            "day":       day,
            "frequency": data['frequency'],
        }}}
    )

    if day == TODAY:
        now = datetime.now()
        reminder_datetime = datetime.combine(
            now.date(),
            datetime.strptime(t, "%H:%M").time()
        )
        scheduler.add_job(
            send_reminder,
            trigger=DateTrigger(run_date=reminder_datetime),
            args=[data],
        )

    return jsonify({"reply": 'success'})


@app.route('/speed_bar')
def speed_bar():
    return render_template('speed_bar.html')


@app.route("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == '__main__':
    app.run(debug=True, port=3000, use_reloader=False)