import streamlit as st
from Retriever import Retriever
from LLM import RAGChatbot
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














# import streamlit as st
# from Retriever import Retriever
# from LLM import RAGChatbot
# from dotenv import load_dotenv
# import os
# import json
# import uuid
# from datetime import datetime

# # === Load environment variables ===
# load_dotenv()

# # === Config ===
# st.set_page_config(page_title="Legalysis: Legal Chatbot", page_icon="⚖️", layout="wide")

# # === Constants ===
# CHAT_HISTORY_FILE = "chat_history.json"

# # === Sidebar Config ===
# st.sidebar.title("⚙️ Configuration")

# # Customization Controls
# st.sidebar.subheader("🧠 Model Behavior")
# temperature = st.sidebar.slider("Creativity (Temperature)", 0.0, 1.0, 0.5, 0.1)
# max_tokens = st.sidebar.number_input("Max Tokens", min_value=100, max_value=10000, value=1000, step=100)
# top_k = st.sidebar.slider("Top-k Documents", 1, 10, 4)
# strict_mode = st.sidebar.checkbox("🔒 Context-Only Mode (No outside knowledge)", value=False)
# show_sources = st.sidebar.checkbox("📚 Show Source Documents", value=True)

# # === Utility functions ===
# def load_chat_history():
#     if not os.path.exists(CHAT_HISTORY_FILE):
#         return {}
#     with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
#         history_data = json.load(f)
#         history_data = {k: v for k, v in history_data.items() if v}  # filter out empty chats
#         return history_data

# def save_chat_history(chat_id, messages):
#     data = load_chat_history()
#     data[chat_id] = messages
#     with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)

# def delete_chat(chat_id):
#     data = load_chat_history()
#     if chat_id in data:
#         del data[chat_id]
#         with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)

# def download_chat(chat_id):
#     data = load_chat_history()
#     if chat_id in data:
#         return json.dumps(data[chat_id], indent=2, ensure_ascii=False)
#     return ""

# # === Load chatbot ===
# @st.cache_resource(show_spinner="Loading retriever and LLM...")
# def load_chatbot():
#     retriever = Retriever(
#         persist_directory="./chroma_db",
#         collection_name="legal_documents",
#         model_name="law-ai/InLegalBERT"
#     )
#     chatbot = RAGChatbot(
#         retriever=retriever,
#         model_name="gemini-1.5-flash",
#         temperature=temperature,
#         max_tokens=max_tokens,
#         #strict_mode=strict_mode
#     )
#     return chatbot

# chatbot = load_chatbot()

# # === Load chat history ===
# history_data = load_chat_history()

# # === Manage Chat Sessions ===
# if "chat_id" not in st.session_state:
#     st.session_state.chat_id = str(uuid.uuid4())
#     st.session_state.chat_history = []

# # === Sidebar Chat History ===
# st.sidebar.subheader("📜 Chat History")
# if st.sidebar.button("➕ New Chat"):
#     st.session_state.chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
#     st.session_state.chat_history = []
#     save_chat_history(st.session_state.chat_id, [])

# search_query = st.sidebar.text_input("🔍 Search Chats")

# for cid, messages in sorted(history_data.items(), reverse=True):
#     if not messages:
#         continue
#     title = messages[0]['question'][:40]
#     if search_query.lower() not in title.lower():
#         continue

#     with st.sidebar.container():
#         col1, col2, col3 = st.columns([6, 1, 1])
#         if col1.button(title, key=f"load_{cid}"):
#             st.session_state.chat_id = cid
#             st.session_state.chat_history = messages
#         if col2.button("⬇️", key=f"dl_{cid}"):
#             st.download_button(
#                 "Download",
#                 data=download_chat(cid),
#                 file_name=f"chat_{cid}.json",
#                 mime="application/json",
#                 key=f"download_{cid}"
#             )
#         if col3.button("🗑️", key=f"del_{cid}"):
#             delete_chat(cid)
#             st.rerun()

# # === Main UI ===
# st.title("⚖️ Legalysis - Indian Law Assistant")
# st.markdown("Ask your legal questions based on the Indian Penal Code (IPC) and get structured AI-backed answers using Gemini.")

# user_input = st.chat_input("Ask a legal question about IPC...")

# if user_input:
#     with st.spinner("Thinking..."):
#         response = chatbot.query(user_input, k=top_k)
#         message = {
#             "question": user_input,
#             "answer": response["answer"],
#             "sources": response["sources"],
#             "num_sources": response["num_sources"]
#         }
#         st.session_state.chat_history.append(message)
#         save_chat_history(st.session_state.chat_id, st.session_state.chat_history)

# # === Display chat ===
# for chat in st.session_state.chat_history:
#     st.chat_message("user", avatar="🧑").write(chat["question"])
#     st.chat_message("assistant", avatar="🤖").write(chat["answer"])

#     if show_sources and chat["num_sources"] > 0:
#         with st.expander("📚 Sources"):
#             for idx, source in enumerate(chat["sources"]):
#                 st.markdown(f"**Source {idx+1}**\n\n{source['content'][:500]}...")














# import streamlit as st
# from Retriever import Retriever
# from LLM import RAGChatbot
# from dotenv import load_dotenv
# import os
# import json
# import uuid
# from datetime import datetime

# # === Load environment variables ===
# load_dotenv()

# # === Config ===
# st.set_page_config(page_title="Legalysis: Legal Chatbot", page_icon="⚖️", layout="wide")

# # === Constants ===
# CHAT_HISTORY_FILE = "chat_history.json"

# # === Sidebar ===
# st.sidebar.title("⚙️ Configuration")
# show_sources = st.sidebar.checkbox("Show source documents", value=True)
# top_k = st.sidebar.slider("Number of documents to retrieve (k)", 1, 10, 4)

# # === Utility functions ===
# def load_chat_history():
#     if not os.path.exists(CHAT_HISTORY_FILE):
#         return {}
#     with open(CHAT_HISTORY_FILE, "r") as f:
#         history_data = json.load(f)
#         # Clean up empty chats
#         history_data = {k: v for k, v in history_data.items() if v}
#         return history_data

# def save_chat_history(chat_id, messages):
#     data = load_chat_history()
#     data[chat_id] = messages
#     with open(CHAT_HISTORY_FILE, "w") as f:
#         json.dump(data, f, indent=2)

# def delete_chat(chat_id):
#     data = load_chat_history()
#     if chat_id in data:
#         del data[chat_id]
#         with open(CHAT_HISTORY_FILE, "w") as f:
#             json.dump(data, f, indent=2)

# def download_chat(chat_id):
#     data = load_chat_history()
#     if chat_id in data:
#         return json.dumps(data[chat_id], indent=2)
#     return ""

# # === Load chatbot ===
# @st.cache_resource(show_spinner="Loading retriever and LLM...")
# def load_chatbot():
#     retriever = Retriever(
#         persist_directory="./chroma_db",
#         collection_name="legal_documents",
#         model_name="law-ai/InLegalBERT"
#     )
#     chatbot = RAGChatbot(
#         retriever=retriever,
#         model_name="gemini-1.5-flash",
#         temperature=0.5,
#         max_tokens=10000
#     )
#     return chatbot

# chatbot = load_chatbot()

# # === Load chat history ===
# history_data = load_chat_history()
# chat_ids = []
# for cid in history_data.keys():
#     if history_data[cid]:
#         label = history_data[cid][0]['question'][:40] + f"... ({cid})"
#         chat_ids.append((cid, label))
# chat_ids = chat_ids[::-1]  # most recent first

# # === Chat session ID ===
# if "chat_id" not in st.session_state:
#     st.session_state.chat_id = str(uuid.uuid4())
#     st.session_state.chat_history = []

# # === Sidebar chat history ===
# st.sidebar.markdown("---")
# st.sidebar.subheader("📜 Chat History")
# search_query = st.sidebar.text_input("🔍 Search previous chats")

# for cid, label in chat_ids:
#     if search_query.lower() in label.lower():
#         col1, col2, col3 = st.sidebar.columns([6, 1, 1])
#         if col1.button(label, key=f"load_{cid}"):
#             st.session_state.chat_id = cid
#             st.session_state.chat_history = history_data[cid]
#         if col2.button("⬇️", key=f"dl_{cid}"):
#             st.download_button("Download", download_chat(cid), file_name=f"chat_{cid}.json", mime="application/json", key=f"download_{cid}")
#         if col3.button("🗑️", key=f"del_{cid}"):
#             delete_chat(cid)
#             st.rerun()

# # === Main UI ===
# st.title("⚖️ Legalysis - Indian Law Assistant")
# st.markdown("Ask your legal questions based on the Indian Penal Code (IPC) and get AI-backed answers using Gemini.")

# user_input = st.chat_input("Ask a question about Indian law...")

# if user_input:
#     with st.spinner("Thinking..."):
#         response = chatbot.query(user_input, k=top_k)
#         message = {
#             "question": user_input,
#             "answer": response["answer"],
#             "sources": response["sources"],
#             "num_sources": response["num_sources"]
#         }
#         st.session_state.chat_history.append(message)
#         save_chat_history(st.session_state.chat_id, st.session_state.chat_history)

# # === Display chat ===
# for chat in st.session_state.chat_history:
#     st.chat_message("user", avatar="🧑").write(chat["question"])
#     st.chat_message("assistant", avatar="🤖").write(chat["answer"])

#     if show_sources and chat["num_sources"] > 0:
#         with st.expander("📚 Show Sources"):
#             for idx, source in enumerate(chat["sources"]):
#                 st.markdown(f"**Source {idx+1}**\n\n{source['content'][:500]}...")















# import streamlit as st
# import os
# import json
# from datetime import datetime
# from Retriever import Retriever
# from LLM import RAGChatbot
# from dotenv import load_dotenv

# # === Load environment variables ===
# load_dotenv()

# # === Configurations ===
# st.set_page_config(page_title="Legalysis: Legal Chatbot", page_icon="⚖️", layout="wide")
# CHAT_HISTORY_FILE = "chat_history.json"

# # === Load retriever and chatbot ===
# @st.cache_resource(show_spinner="Loading retriever and LLM...")
# def load_chatbot():
#     retriever = Retriever(
#         persist_directory="./chroma_db",
#         collection_name="legal_documents",
#         model_name="law-ai/InLegalBERT"
#     )
#     chatbot = RAGChatbot(
#         retriever=retriever,
#         model_name="gemini-1.5-flash",
#         temperature=0.5,
#         max_tokens=10000
#     )
#     return chatbot

# chatbot = load_chatbot()

# # === Helper: Load & Save Chat History ===
# def load_history():
#     if os.path.exists(CHAT_HISTORY_FILE):
#         try:
#             with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 if isinstance(data, dict):
#                     return data
#         except Exception:
#             pass
#     return {}

# def save_history(history_data):
#     with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
#         json.dump(history_data, f, indent=2, ensure_ascii=False)

# # === Initialize Chat State ===
# history_data = load_history()
# if "current_chat_id" not in st.session_state:
#     st.session_state.current_chat_id = None

# # === Sidebar: Chat History ===
# st.sidebar.title("🔍 Search & History")

# search_term = st.sidebar.text_input("Search previous questions")

# # Select or start new chat
# chat_ids = sorted(history_data.keys(), reverse=True)
# matching_ids = [cid for cid in chat_ids if any(search_term.lower() in msg['question'].lower() for msg in history_data[cid])] if search_term else chat_ids

# for cid in matching_ids:
#     # label = history_data[cid][0]['question'][:40] + f"... ({cid})"
#     if history_data[cid]:  # Ensure there is at least one message
#         label = history_data[cid][0]['question'][:40] + f"... ({cid})"
#         chat_ids.append((cid, label))
#     if st.sidebar.button(label, key=cid):
#         st.session_state.current_chat_id = cid

# if st.sidebar.button("➕ New Chat"):
#     st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
#     history_data[st.session_state.current_chat_id] = []
#     save_history(history_data)

# if st.session_state.current_chat_id:
#     selected_chat = history_data.get(st.session_state.current_chat_id, [])

#     # Download/Delete options
#     st.sidebar.markdown("---")
#     st.sidebar.markdown(f"**Chat ID:** `{st.session_state.current_chat_id}`")

#     if st.sidebar.download_button(
#         "💾 Download Chat",
#         data=json.dumps(selected_chat, indent=2, ensure_ascii=False),
#         file_name=f"chat_{st.session_state.current_chat_id}.json",
#         mime="application/json"
#     ):
#         pass

#     if st.sidebar.button("🗑️ Delete Chat"):
#         del history_data[st.session_state.current_chat_id]
#         save_history(history_data)
#         st.session_state.current_chat_id = None
#         st.rerun()

# # === Main UI ===
# st.title("⚖️ Legalysis - Indian Law Assistant")
# st.markdown("Ask your legal questions based on the Indian Penal Code (IPC) and get structured AI-backed answers using Gemini.")

# user_input = st.chat_input("Ask a legal question about IPC...")

# if user_input and st.session_state.current_chat_id:
#     with st.spinner("Analyzing with Gemini..."):
#         response = chatbot.query(user_input, k=4)
#         entry = {
#             "question": user_input,
#             "answer": response["answer"],
#             "sources": response["sources"],
#             "num_sources": response["num_sources"]
#         }
#         history_data[st.session_state.current_chat_id].append(entry)
#         save_history(history_data)

# # === Display Chat ===
# if st.session_state.current_chat_id:
#     for msg in history_data[st.session_state.current_chat_id]:
#         st.chat_message("user", avatar="🧍").write(msg["question"])
#         st.chat_message("assistant", avatar="🧑‍🤖").write(msg["answer"])

#         if msg["num_sources"] > 0:
#             with st.expander("📚 Sources"):
#                 for i, source in enumerate(msg["sources"]):
#                     st.markdown(f"**Source {i+1}**\n\n{source['content'][:500]}...")
# else:
#     st.info("Select or start a new chat from the sidebar to begin.")
