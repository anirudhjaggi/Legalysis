# 🏛️ Legalysis — AI-Powered Legal Document Assistant

**Legalysis** is an AI-powered legal assistant that helps users summarize, analyze, and interact with legal documents. Built with Streamlit, Gemini, LangChain, and ChromaDB, it’s designed for law students, legal professionals, and researchers who want fast and intelligent insights from legal texts.

---

## 🔍 Features

### 📄 Document Upload & Summarization
- Supports **PDF, DOCX, and TXT** formats.
- Extracts:
  - Case Summary
  - Parties Involved
  - Key Legal Dates
  - IPC Sections and Statutes
  - Obligations and Rights

### 💬 Chat with Legal Documents (RAG-based)
- Ask natural language questions.
- Responses include:
  - Legal reasoning
  - Relevant IPC sections
  - Citations (optional toggle)
  - Actionable summaries

### 🛠️ Model Customization
- Adjustable parameters:
  - `Temperature`
  - `Max Tokens`
  - `Top-k Chunks` retrieved
- Toggle source document visibility.

### 🗂️ Chat History
- View previous sessions.
- Download chat transcripts.
- Delete or reset with “New Chat.”

---

## 🧠 Tech Stack

| Component     | Technology                    |
|---------------|-------------------------------|
| UI            | Streamlit                     |
| LLM           | Google Gemini (via LangChain) |
| Embeddings    | InLegalBERT                   |
| Vector Store  | ChromaDB                      |
| PDF Parsing   | PyMuPDF, docx2txt             |

---
