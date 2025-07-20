import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable

# --- Config ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_SUMMARY_STYLE = "bullet"
MODEL_NAME = "gemini-1.5-flash"

# --- Document Loader ---
def load_document(uploaded_file) -> list[Document]:
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    if suffix == ".pdf":
        loader = PyMuPDFLoader(file_path=tmp_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path=tmp_path)
    elif suffix == ".txt":
        loader = TextLoader(file_path=tmp_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
    documents = loader.load()
    return documents

def get_summary_chain() -> Runnable:
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template("""
You are an AI legal assistant. Given the following Indian legal document (such as a court order, petition, or judgment), your task is to:

1. **Summarize** the document clearly and concisely, using professional legal language appropriate for lawyers, law students, or legal researchers.
2. **Extract and organize key legal information** into a structured format. Only include sections that are relevant or explicitly stated in the document.

---

### Case Summary

Provide a short, well-written summary of the overall matter: who the parties are, what the issue is, what relief was sought, and the outcome or status (if applicable).

---

### Key Legal Information

#### Parties
- Petitioner(s):
- Respondent(s):

#### Court & Jurisdiction
- Name of Court:
- Jurisdiction Type (Civil, Criminal, etc.):

#### Important Dates
- Incident/Dispute Date:
- Filing/Petition Date:
- Hearing/Order Date(s):

#### Legal Provisions Cited
List any relevant sections from IPC, CrPC, NDPS Act, or other applicable laws.

#### Claims and Arguments
Summarize the primary claims, demands, or legal arguments made by each party.

#### Court’s Order / Directions
Mention any legal orders, reliefs granted, or obligations placed on parties.

#### Penalties / Consequences
Include any punishments, fines, or directives for non-compliance, if mentioned.

---

Keep the tone factual, objective, and readable. Use clean markdown formatting with headings and bullet points. Do **not** invent or assume missing data — simply skip sections if the document does not contain that information.

Document:
{content}
""")



    llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model=MODEL_NAME,
        temperature=0.3
    )

    return prompt | llm

    llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model=MODEL_NAME,
        temperature=0.3
    )
    return prompt | llm

# # --- Streamlit App ---
# st.set_page_config(page_title="DocIntel", layout="centered")
# st.title("📄 DocIntel")
# st.subheader("Unlock insights from your documents — summarize and quiz with AI.")

# uploaded_file = st.file_uploader("Upload your document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

# with st.expander("Customize Options"):
#     summary_style = st.selectbox("Summary Style", ["bullet", "paragraph", "both"], index=0)
#     summary_length = st.slider("Number of summary points/paragraphs", min_value=1, max_value=15, value=7)

# if uploaded_file and st.button("Generate"):
#     with st.spinner("Processing your document..."):
#         try:
#             docs = load_document(uploaded_file)
#             full_text = "\n".join([doc.page_content for doc in docs])
#             input_data = {"content": full_text, "num_points": summary_length}
#             summary_chain = get_summary_chain(style=summary_style)
#             summary_result = summary_chain.invoke(input_data)
#             st.markdown("### Summary")
#             text = summary_result.content if hasattr(summary_result, "content") else str(summary_result)
#             st.markdown(
#                 f"<div style='color: #F0F0F0; font-size: 16px; line-height: 1.6; white-space: pre-wrap;'>{text}</div>",
#                 unsafe_allow_html=True
#             )
#         except Exception as e:
#             st.error(f" An error occurred: {e}")
