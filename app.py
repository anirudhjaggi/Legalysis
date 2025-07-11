import streamlit as st
from Utils.Retriever import Retriever
from Utils.LLM import RAGChatbot
from dotenv import load_dotenv
import os
import json
import uuid
from datetime import datetime

# === Load environment variables ===
load_dotenv()

# === Config ===
st.set_page_config(page_title="Legalysis: Legal Chatbot", page_icon="⚖️", layout="wide")

# === Constants ===
CHAT_HISTORY_FILE = "chat_history.json"

# === Utility functions ===
def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return {}
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        history_data = json.load(f)
        return {k: v for k, v in history_data.items() if v}

def save_chat_history(chat_id, messages):
    data = load_chat_history()
    data[chat_id] = messages
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def delete_chat(chat_id):
    data = load_chat_history()
    if chat_id in data:
        del data[chat_id]
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def download_chat(chat_id):
    data = load_chat_history()
    if chat_id in data:
        return json.dumps(data[chat_id], indent=2, ensure_ascii=False)
    return ""

# === Sidebar: Model Customization ===
# st.sidebar.title("⚖️ Legalysis")

if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
    st.session_state.chat_history = []
    save_chat_history(st.session_state.chat_id, [])

with st.sidebar.expander("⚙️ Model Customizations", expanded=False):
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.5, 0.1)
    max_tokens = st.number_input("Max Tokens", min_value=100, max_value=10000, value=1000, step=100)
    top_k = st.slider("Top-k Documents", 1, 10, 4)

show_sources = st.sidebar.checkbox("Show Source Documents", value=True)



# === Load chatbot ===
@st.cache_resource(show_spinner="Loading retriever and LLM...")
def load_chatbot():
    retriever = Retriever(
        persist_directory="./chroma_db",
        collection_name="legal_documents",
        model_name="law-ai/InLegalBERT"
    )
    chatbot = RAGChatbot(
        retriever=retriever,
        model_name="gemini-1.5-flash",
        temperature=temperature,
        max_tokens=max_tokens
    )
    return chatbot

chatbot = load_chatbot()

# === Chat Session State ===
history_data = load_chat_history()

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.chat_history = []

# === Sidebar Chat History ===

st.sidebar.subheader("\n\nChat History")

search_query = st.sidebar.text_input("🔍 Search Chats")

for cid, messages in sorted(history_data.items(), reverse=True):
    if not messages:
        continue
    title = messages[0]['question'][:40]
    if search_query.lower() not in title.lower():
        continue

    with st.sidebar.container():
        label = f"{title}"
        col1, col2 = st.sidebar.columns([8, 2])
        if col1.button(label, key=f"load_{cid}"):
            st.session_state.chat_id = cid
            st.session_state.chat_history = messages
        
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                delete_chat(cid)
                st.rerun()

    # with st.sidebar.container():
    #     label = f"{title}"
    #     col1, col2, col3 = st.sidebar.columns([8, 2, 2])
    #     if col1.button(label, key=f"load_{cid}"):
    #         st.session_state.chat_id = cid
    #         st.session_state.chat_history = messages
    #     with col2:
    #         st.download_button("⬇️", data=download_chat(cid),
    #                            file_name=f"chat_{cid}.json",
    #                            mime="application/json", key=f"download_{cid}")
    #     with col3:
    #         if st.button("🗑️", key=f"del_{cid}"):
    #             delete_chat(cid)
    #             st.rerun()



# === Main UI ===
st.title("⚖️ Legalysis - Indian Law Assistant")
st.markdown("Ask your legal questions based on the Indian Penal Code (IPC) and get structured AI-backed answers using Gemini.")

user_input = st.chat_input("Ask a legal question about IPC...")

if user_input:
    with st.spinner("Thinking..."):
        response = chatbot.query(user_input, k=top_k)
        message = {
            "question": user_input,
            "answer": response["answer"],
            "sources": response["sources"],
            "num_sources": response["num_sources"]
        }
        st.session_state.chat_history.append(message)
        save_chat_history(st.session_state.chat_id, st.session_state.chat_history)

# === Display Chat Messages ===
for chat in st.session_state.chat_history:
    st.chat_message("user", avatar="🧑").write(chat["question"])
    st.chat_message("assistant", avatar="🤖").write(chat["answer"])

    if show_sources and chat["num_sources"] > 0:
        with st.expander("📚 Sources"):
            for idx, source in enumerate(chat["sources"]):
                st.markdown(f"**Source {idx+1}**\n\n{source['content'][:500]}...")
