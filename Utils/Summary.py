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
MODEL_NAME = "gemini-3.8-flash"

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
You are an AI legal assistant. Given the following Indian court **judgment**, 
your task is to extract and summarize the key components in a structured and legally sound manner,
 focusing only on what is explicitly stated.

Follow this format for your output:

---

### Name of the Court
Mention the full name of the court that delivered the judgment.

---

### Bench
List the judge(s) or bench that heard and decided the matter, if available.

---

### Facts
Provide a concise, to-the-point summary of the relevant facts of the case that led to the dispute or proceedings.

---

### Issues Involved
List the legal issues or questions of law that the court addressed in the judgment.

---

### Legal Provisions
Mention the specific sections of the IPC, CrPC, Constitution, or other statutes cited or applied in the case.

---

### Application of Law / Judgment
Describe how the court interpreted or applied the above legal provisions to the facts and issues of the case. Include any reasoning or legal tests the court used.

---

### Conclusion / Implication
State the final decision, verdict, or orders issued by the court. Also include any broader legal implications if they are mentioned in the judgment.

---

**Instructions:**
- Use clean markdown formatting with proper headings and bullet points.
- Keep the tone formal, objective, and easy to understand.
- **Do not add** information that is not present in the document.
- **Skip any section** that is not mentioned explicitly in the judgment.

Judgment:
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

# prompt = PromptTemplate.from_template("""
# You are an AI legal assistant. Given the following Indian legal document (such as a court order, petition, or judgment), your task is to:

# 1. **Summarize** the document clearly and concisely, using professional legal language appropriate for lawyers, law students, or legal researchers.
# 2. **Extract and organize key legal information** into a structured format. Only include sections that are relevant or explicitly stated in the document.

# ---

# ### Case Summary

# Provide a short, well-written summary of the overall matter: who the parties are, what the issue is, what relief was sought, and the outcome or status (if applicable).

# ---

# ### Key Legal Information

# #### Parties
# - Petitioner(s):
# - Respondent(s):

# #### Court & Jurisdiction
# - Name of Court:
# - Jurisdiction Type (Civil, Criminal, etc.):

# #### Important Dates
# - Incident/Dispute Date:
# - Filing/Petition Date:
# - Hearing/Order Date(s):

# #### Legal Provisions Cited
# List any relevant sections from IPC, CrPC, NDPS Act, or other applicable laws.

# #### Claims and Arguments
# Summarize the primary claims, demands, or legal arguments made by each party.

# #### Court’s Order / Directions
# Mention any legal orders, reliefs granted, or obligations placed on parties.

# #### Penalties / Consequences
# Include any punishments, fines, or directives for non-compliance, if mentioned.

# ---

# Keep the tone factual, objective, and readable. Use clean markdown formatting with headings and bullet points. Do **not** invent or assume missing data — simply skip sections if the document does not contain that information.

# Document:
# {content}
# """)