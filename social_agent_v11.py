# pip install -U langchain langchain-community langchain-core langchain-groq \
#   langchain-google-genai chromadb langgraph pypdf python-dotenv \
#   langchain-google-community

import os
import re
import uuid
from datetime import datetime, timedelta
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
import json
import re
# Add the new tool
from .airtable_tool import customer_context, upsert_customer_and_get_context
# ──────────────────────────────────────────────────────────────────────────────
# LangChain / LangGraph core
# ──────────────────────────────────────────────────────────────────────────────
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_core.messages.utils import convert_to_messages
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import create_react_agent

# ──────────────────────────────────────────────────────────────────────────────
# Vectorstore / RAG bits
# ──────────────────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ──────────────────────────────────────────────────────────────────────────────
# Tools (Google Calendar)
# ──────────────────────────────────────────────────────────────────────────────
from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.utils import (
    build_resource_service,
    get_google_credentials,
)
from langchain.tools import tool


# =============================================================================
# 0) ENV + CONSTANTS
# =============================================================================
load_dotenv()
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "app308OkLVQkVLecK")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE", "customers")

os.environ["AIRTABLE_TOKEN"] = AIRTABLE_TOKEN
os.environ["AIRTABLE_BASE_ID"] = AIRTABLE_BASE_ID
os.environ["AIRTABLE_TABLE"] = AIRTABLE_TABLE

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    raise ValueError(
        "Missing API keys. Please set GOOGLE_API_KEY and GROQ_API_KEY in your .env file."
    )

# Also export for libs that read from env at import time
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

DEFAULT_TZ = "Asia/Karachi"
PERSIST_DIR = "./chroma_db"              # persisted RAG index
PDF_PATH = "./portfolio_projects.pdf"    # change if needed


# =============================================================================
# Invite extract
# =============================================================================

# Add in airtable_tool.py (or a new file and import it)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _name_guess(text: str) -> str | None:
    # very light heuristic: grab 2 consecutive capitalized words
    m = re.search(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", text)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    # fallback: single capitalized token (e.g., "Ada")
    m = re.search(r"\b([A-Z][a-z]{2,})\b", text)
    return m.group(1) if m else None


@tool("customer_context_nl", return_direct=False)
def customer_context_nl(utterance: str) -> str:
    """
    Natural-language customer lookup.
    - Extracts email and/or name from free text.
    - Looks up (email -> name -> partial name). Creates only if both name+email present.
    Returns: human-readable context + STATUS.
    """
    email = None
    m = EMAIL_RE.search(utterance)
    if m:
        email = m.group(0)

    name = _name_guess(utterance)

    # Try to use the stricter existing path
    ctx, created, rec = upsert_customer_and_get_context(
        name=name,
        email=email,
        description=None,
        create_if_missing=True,  # will only create if both present
    )
    return f"{ctx}\nSTATUS: created={'true' if created else 'false'}; found={'true' if rec else 'false'}"


# =============================================================================
# Invite extract
# =============================================================================
INVITE_URL_RE = re.compile(
    r'https?://(?:www\.)?(?:calendar\.google\.com|google\.com)/calendar/event\?[^\s>]+',
    re.I,
)


def _extract_invite_link_from_tool_message(msg: ToolMessage) -> str | None:
    # 1) Try JSON payloads (some toolkits return dicts)
    txt = (msg.content or "").strip()
    if txt.startswith("{") and txt.endswith("}"):
        try:
            data = json.loads(txt)
            # Common fields
            for key in ("htmlLink", "link", "url"):
                if isinstance(data.get(key), str):
                    return data[key]
        except Exception:
            pass
    # 2) Fallback: regex from plain text ("Event created: <url>")
    m = INVITE_URL_RE.search(txt)
    return m.group(0) if m else None


def _find_last_invite_link(messages) -> str | None:
    # Walk backwards to get the most recent calendar result
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name in {"create_calendar_event", "update_calendar_event"}:
            link = _extract_invite_link_from_tool_message(m)
            if link:
                return link
    return None


# =============================================================================
# 1) LLM (shared by agent + RAG)
# =============================================================================
llm = ChatGroq(
    # model="openai/gpt-oss-20b",
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0,
)


# =============================================================================
# 2) RAG: Build / Load Vector Store
# =============================================================================
def build_or_load_vectorstore(pdf_path: str, persist_dir: str) -> Chroma:
    """
    Creates (or loads) a Chroma vectorstore from the given PDF.
    Persists to `persist_dir` so subsequent runs are instant.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # If we already have a persisted DB, load directly
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Otherwise, build from scratch
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. Update PDF_PATH or place your file there."
        )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vs = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vs.persist()
    return vs


vector_store = build_or_load_vectorstore(PDF_PATH, PERSIST_DIR)


# A small, focused prompt for doc-grounded answers
rag_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use ONLY the provided context to answer.\n"
    "If the answer is not contained in the context, say you don't know.\n\n"
    
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def _answer_with_rag(question: str, extra_context: str = "") -> str:
    # Retrieve top-k chunks
    docs = vector_store.similarity_search(question, k=4)
    kb_context = "\n\n".join(d.page_content for d in docs)

    # Prepend customer context so it’s highest priority
    # context = (extra_context.strip() + "\n\n" + kb_context).strip() if extra_context else kb_context

    # Generate

    messages = rag_prompt.format_messages(
        question=question,
        context=kb_context,
        # customer_context=context or "(none)"
    )
    resp = llm.invoke(messages)

    # Sources list
    sources = []
    for d in docs:
        meta = d.metadata or {}
        page = meta.get("page")
        if page is not None:
            sources.append(f"p.{page+1}")
        else:
            sources.append(meta.get("source", "document"))

    unique_sources = ", ".join(sorted(set(sources)))
    answer = resp.content.strip()

    return f"{answer}\n\n[SOURCES: {unique_sources}]"

@tool("rag_search", return_direct=False)
def rag_search(payload: str) -> str:
    """
    Accepts EITHER:
      - JSON string: {"question":"...", "customer_context":"..."}  (preferred)
      - OR a plain question string: "who is Ada Lovelace ..."

    Returns: grounded answer + brief sources.
    """
    import json

    question = ""
    extra = ""

    try:
        text = (payload or "").strip()
        if text.startswith("{"):
            data = json.loads(text)
            question = (data.get("question") or "").strip()
            extra = (data.get("customer_context") or "").strip()
        else:
            # treat whole payload as the question
            question = text
    except Exception as e:
        return f"Invalid input for rag_search: {e}"

    if not question:
        return "rag_search needs a 'question'."

    return _answer_with_rag(question, extra_context=extra)



# =============================================================================
# 3) Google Calendar Tools
# =============================================================================
# Requires: credentials.json (OAuth client) and token.json (user token)
credentials = get_google_credentials(
    token_file="token.json",
    scopes=["https://www.googleapis.com/auth/calendar"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
calendar_toolkit = CalendarToolkit(api_resource=api_resource)
calendar_tools = calendar_toolkit.get_tools()  # create/list/update/etc.


# =============================================================================
# 5) Session Memory + Graph
# =============================================================================
chats_by_session_id: dict[str, InMemoryChatMessageHistory] = {}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chats_by_session_id:
        chats_by_session_id[session_id] = InMemoryChatMessageHistory()
    return chats_by_session_id[session_id]


def call_model(state: MessagesState, config) -> dict:
    """
    Single node that hands control to the ReAct agent with tools.
    It maintains per-session chat history.
    """
    print("===============call_model invoked with state:", state)
    session_id = config["configurable"].get("session_id")

    username = config["configurable"].get("username")
    payload = {
        # "email": "ada@example.com",
        "username": username,
        "create_if_missing": False
    }
    print(f"===============call_model: looking up customer context for {payload}")
    user_details = customer_context(json.dumps(payload))
    print(f"===============call_model: user_details={user_details}")


    # =============================================================================
    # 4) Agent Prompt (with explicit-confirmation guardrail)
    # =============================================================================
    prompt = f"""You are a persuasive sales agent for a market-leading software consultancy.
    Be concise, professional, and friendly. Reply shortly (ideally one line).

    USER DETAILS:
    {user_details}

    CRITICAL RULES:
    - Prefer tools over free-form answers, BUT only call Calendar CREATE/UPDATE tools after explicit user confirmation.
    - If the user's message is about scheduling/meetings, prefer Google Calendar tools.
    - If the user's message is about the portfolio/projects/skills/company info, call the RAG tool (rag_search) with the full question.
    - Do NOT call rag_search for scheduling questions.

    CONSENT & CONFIRMATION (VERY IMPORTANT):
    - Never book a meeting immediately. First propose a concrete time and ask for confirmation.
    - Before creating/updating any calendar event, look at the most recent 5 user messages in the conversation.
    - Proceed to book ONLY if you find an explicit confirmation phrase such as:
    "yes, please book it", "yes book it", "confirm", "that works for me", "works for me", 
    "sounds good", "let's do it", "go ahead", "please schedule it", "book it", "lock it in".
    - Implicit or vague replies like "ok", "sure", "maybe", "I'll check", or silence are NOT confirmation.
    - If you don't find explicit confirmation, ASK: "Can I book it?" and wait.

    TIME & SCHEDULING:
    - Current time (Asia/Karachi): {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    - Meetings can only be scheduled AFTER the current time.
    - All meetings are fixed at 30 minutes.
    - Earliest available slot is 30 minutes from now, today.
    - If they say “schedule a 30-minute call now”, propose a slot starting 30 minutes from now; ask for confirmation; say it's the only slot available.
    - If they say “schedule a meeting” without a time, ask for the start time and state it's the only available slot.
    - If a different duration is requested, clarify meetings are fixed at 30 minutes.
    - Use Asia/Karachi for booking. Also include the equivalent in US/Pacific (PST/PDT) for convenience.

    OUTPUT STYLE:
    - Keep confirmations short and clear.
    - After confirmed booking, reply with the event summary and a link to the invite only nothing else.

    EXAMPLES:
    User: "Schedule a 30-minute call today at 4:30 PM."
    Assistant: "I can do 4:30–5:00 PM Asia/Karachi (which is 4:30 AM–5:00 AM US/Pacific). Can I book it?"

    User: "Yes, please book it."
    Assistant: [NOW call the Calendar CREATE tool to book, then reply with the event summary.]

    User: "Ok"
    Assistant: "Just to confirm—should I go ahead and book 4:30–5:00 PM Asia/Karachi (4:30 AM–5:00 AM US/Pacific)?"
    """

    print(f"===============call_model: addign tools")
    # Combine all tools: Calendar + RAG
    # tools_for_agent = [rag_search] + calendar_tools
    tools_for_agent = [rag_search] + calendar_tools

    print(f"===============call_model: creating agent")
    agent = create_react_agent(
        model=llm,
        tools=tools_for_agent,
        prompt=prompt,
    )

    if not session_id:
        raise ValueError("Missing session_id in config.configurable")

    history = get_chat_history(session_id)

    # Normalize to BaseMessages, merge with history
    new_msgs = convert_to_messages(state["messages"])
    context_msgs = list(history.messages) + new_msgs

    print(f"===============call_model: invokeing agent with context:")
    # Invoke agent; it will choose tools (Calendar vs RAG) per the prompt
    result = agent.invoke({"messages": context_msgs})
    ai_msg: AIMessage = result["messages"][-1]
    print("===============agent output:", result)

    # ---- NEW: ensure the final message includes the invite link if a meeting was created/updated
    invite_link = _find_last_invite_link(result["messages"])
    if invite_link and not INVITE_URL_RE.search(ai_msg.content or ""):
        # Append a short acknowledgment with the link
        updated = (ai_msg.content or "").rstrip() + \
            f"\n\n📩 Invite: {invite_link}"
        ai_msg = AIMessage(content=updated)

    # Persist turn
    history.add_messages(new_msgs + [ai_msg])

    # Return the last AI message as the graph node output
    return {"messages": ai_msg}


# Build graph
builder = StateGraph(state_schema=MessagesState)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
graph = builder.compile()
