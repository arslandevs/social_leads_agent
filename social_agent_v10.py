from langchain_groq import ChatGroq
# from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage
import uuid
import os
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_core.messages.utils import convert_to_messages

import re
# ============================================
# Calender BLOCK
# ============================================
# ─── Calendar tools ────────────────────────────────────────────────────────────
from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.utils import (
    build_resource_service,
    get_google_credentials,
)
from langchain.tools import tool
from datetime import datetime, timedelta

# Google creds
credentials = get_google_credentials(
    token_file="token.json",
    scopes=["https://www.googleapis.com/auth/calendar"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
calendar_toolkit = CalendarToolkit(api_resource=api_resource)
# includes create/list/update, etc.
calendar_tools = calendar_toolkit.get_tools()

# Optional: wrap create tool to make it foolproof
DEFAULT_TZ = "Asia/Karachi"
# "green" in Google Calendar is usually 2 or 10 (depends on API/Calendar)
DEFAULT_COLOR = "2"

CALENDAR_REGEX = re.compile(
    r"\b(calendar|schedule|event|reminder|meeting|walk|run|gym)\b", re.I
)


def is_calendar_intent(text: str) -> bool:
    return bool(CALENDAR_REGEX.search(text)) and any(
        kw in text.lower() for kw in ["schedule", "create", "event", "run", "walk", "meeting", "today", "tomorrow", "at ", ":"]
    )


def normalize_query(text: str) -> str:
    # pad single-digit day in YYYY-MM-D
    text = re.sub(r"(\d{4}-\d{2}-)(\d)(\b)", r"\g<1>0\2", text)
    # inject timezone if missing
    if "timezone" not in text.lower():
        text += " timezone Asia/Karachi"
    # nudge color mapping
    if "green" in text.lower() and "color_id" not in text.lower():
        text += " use color green"
    return text
# ============================================
# AGENT BLOCK
# ============================================


# ─── LLM Setup ─────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0,
)
calendar_prompt = f"""You are a persuasive sales agent for a market-leading software consultancy.
         Be concise, professional, and friendly. Reply shortly, ideally in one line. Reply user queries.

TASKS:
- ALWAYS call a tool to answer (do not invent results).

CURRENT TIME(For Asia/Karachi timezone):
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


IMPORTANT!!!:
- Meetings can only be scheduled after current time.
- Always book meeetings according to Asia/Karachi not PST. But as a reference for the client give them the time in PST.

CONSIDERATIONS:
- All meetings are fixed at 30 minutes.
- If the user says “schedule a 30-minute call now” or similar, offer a slot starting 30 minutes from the current time, confirm if it’s feasible, and mention it’s the only slot available.
- If they say “schedule a meeting” without a time, ask for the start time and state it's the only available slot.
- Earliest available slot is 30 minutes from now, today.
- If they request a different duration, politely clarify that meetings are fixed at 30 minutes.
- Always confirm before booking.
- If timezone is missing, use Asia/Karachi.
- Use AM/PM format for times.
- Be friendly and professional.

Return concise confirmations.
"""

# Tools handed to agent: guarded create + native toolkit
# tools_for_agent = [create_calendar_event_guarded] + calendar_tools

calendar_agent = create_react_agent(
    llm,
    calendar_tools,
    prompt=calendar_prompt,
)
# ─── In-Memory Chat Histories ──────────────────────────────────────────────────
chats_by_session_id: dict[str, InMemoryChatMessageHistory] = {}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chats_by_session_id:
        chats_by_session_id[session_id] = InMemoryChatMessageHistory()
    return chats_by_session_id[session_id]

def call_model(state: MessagesState, config) -> dict:
    session_id = config["configurable"].get("session_id")
    if not session_id:
        raise ValueError("Missing session_id in config.configurable")

    history = get_chat_history(session_id)

    # Normalize to BaseMessages
    new_msgs = convert_to_messages(state["messages"])

    # Let the unified agent decide: chat answer vs. tool use
    # Feed full context (history + current turn)
    context = list(history.messages) + new_msgs
    result = calendar_agent.invoke({"messages": context})
    ai_msg = result["messages"][-1]  # get the AIMessage, not the whole list
    history.add_messages(new_msgs + [ai_msg])
    return {"messages": ai_msg}

# ─── Build & Compile Graph ──────────────────────────────────────────────────────
builder = StateGraph(state_schema=MessagesState)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
graph = builder.compile()
