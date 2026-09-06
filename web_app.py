import streamlit as st
import os
import hashlib
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document
from groq import Groq


# =========================================================
# CONFIGURATION
# =========================================================

DOCUMENTS_DIR = Path("documents")
CHROMA_DIR = "chroma_db"

DOCUMENTS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# USER SESSION ID
# =========================================================

# Each browser session gets a unique ID.
# This ID is used to separate documents between users.

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

USER_ID = st.session_state.user_id

USER_DOCUMENTS_DIR = DOCUMENTS_DIR / USER_ID
USER_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_groq():

    # Streamlit Cloud Secrets
    api_key = os.getenv("GROQ_API_KEY")

    # Local .env fallback
    if not api_key:

        st.error(
            "GROQ_API_KEY is missing. "
            "Please configure it in Streamlit Secrets "
            "or your local .env file."
        )

        st.stop()

    return Groq(
        api_key=api_key
    )


embedding_model = load_embedding_model()
groq_client = load_groq()


# =========================================================
# CHROMA DATABASE
# =========================================================

@st.cache_resource
def get_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name="college_documents"
    )

    return collection


collection = get_collection()


# =========================================================
# FILE HASH
# =========================================================

def get_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as f:

        while True:

            data = f.read(
                1024 * 1024
            )

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf(file_path):

    reader = PdfReader(
        str(file_path)
    )

    text = ""

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if page_text:

            text += (
                f"\nPAGE {page_number}\n"
            )

            text += page_text
            text += "\n"

    return text


# =========================================================
# DOCX EXTRACTION
# =========================================================

def extract_docx(file_path):

    doc = Document(
        str(file_path)
    )

    normal_text = []

    # Paragraphs
    for paragraph in doc.paragraphs:

        text = paragraph.text.strip()

        if text:

            normal_text.append(
                text
            )

    table_text = []

    # Tables
    for table in doc.tables:

        for row in table.rows:

            cells = []

            for cell in row.cells:

                cell_text = cell.text.strip()

                if cell_text:

                    cells.append(
                        cell_text
                    )

            if cells:

                table_text.append(
                    " | ".join(cells)
                )

    text = ""

    if normal_text:

        text += "\n".join(
            normal_text
        )

        text += "\n\n"

    if table_text:

        text += "\n".join(
            table_text
        )

    return text


# =========================================================
# TXT EXTRACTION
# =========================================================

def extract_txt(file_path):

    return file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


# =========================================================
# GENERAL EXTRACTION
# =========================================================

def extract_text(file_path):

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        return extract_pdf(
            file_path
        )

    elif extension == ".docx":

        return extract_docx(
            file_path
        )

    elif extension == ".txt":

        return extract_txt(
            file_path
        )

    return ""


# =========================================================
# TIMETABLE DETECTION
# =========================================================

def is_timetable(text):

    words = [

        "timetable",
        "class:",
        "lecture hall",
        "faculty in charge",
        "teaching hours/week",
        "venue",
        "days"

    ]

    text_lower = text.lower()

    matches = 0

    for word in words:

        if word in text_lower:

            matches += 1

    return matches >= 2


# =========================================================
# CREATE TIMETABLE RECORDS
# =========================================================

def create_timetable_records(text):

    lines = [

        line.strip()

        for line in text.splitlines()

        if line.strip()

    ]

    records = []

    # Find all class headings
    class_sections = []

    for i, line in enumerate(lines):

        if "class:" in line.lower():

            class_sections.append(
                i
            )

    for section_index, start in enumerate(
        class_sections
    ):

        if section_index + 1 < len(
            class_sections
        ):

            end = class_sections[
                section_index + 1
            ]

        else:

            end = len(lines)

        section = lines[
            start:end
        ]

        class_line = section[0]

        # Find lecture hall
        lecture_hall = ""

        for line in section:

            if "lecture hall" in line.lower():

                lecture_hall = line

                break

        # Search subject/faculty information
        for i, line in enumerate(
            section
        ):

            line_lower = line.lower()

            if (
                "natural language processing"
                in line_lower
            ):

                nearby = section[
                    max(0, i - 2):
                    min(len(section), i + 8)
                ]

                record = f"""
COLLEGE TIMETABLE RECORD

Class:
{class_line}

Lecture Hall:
{lecture_hall}

Natural Language Processing:
{chr(10).join(nearby)}
"""

                records.append(
                    record.strip()
                )

    return records


# =========================================================
# PROCESS DOCUMENT
# =========================================================

def process_document(file_path):

    text = extract_text(
        file_path
    )

    if not text.strip():

        return 0, 0

    file_hash = get_file_hash(
        file_path
    )

    # Delete old version for THIS USER only
    try:

        old_docs = collection.get(

            where={
                "$and": [
                    {
                        "source":
                        file_path.name
                    },
                    {
                        "user_id":
                        USER_ID
                    }
                ]
            }

        )

        if old_docs["ids"]:

            collection.delete(
                ids=old_docs["ids"]
            )

    except Exception:

        pass

    chunks = []

    # Timetable-specific records
    if is_timetable(text):

        timetable_records = (
            create_timetable_records(
                text
            )
        )

        chunks.extend(
            timetable_records
        )

    # Normal chunks
    splitter = RecursiveCharacterTextSplitter(

        chunk_size=700,

        chunk_overlap=100

    )

    normal_chunks = splitter.split_text(
        text
    )

    chunks.extend(
        normal_chunks
    )

    # Remove duplicate chunks
    unique_chunks = []

    for chunk in chunks:

        if chunk not in unique_chunks:

            unique_chunks.append(
                chunk
            )

    chunks = unique_chunks

    if not chunks:

        return 0, len(text)

    # Embeddings
    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = []
    metadatas = []

    for i, chunk in enumerate(
        chunks
    ):

        ids.append(
            f"{USER_ID}_{file_hash}_{i}"
        )

        metadatas.append({

            "source":
            file_path.name,

            "file_hash":
            file_hash,

            "user_id":
            USER_ID

        })

    collection.add(

        ids=ids,

        documents=chunks,

        embeddings=embeddings,

        metadatas=metadatas

    )

    return len(chunks), len(text)


# =========================================================
# DELETE DOCUMENT
# =========================================================

def delete_document(filename):

    try:

        results = collection.get(

            where={
                "$and": [
                    {
                        "source":
                        filename
                    },
                    {
                        "user_id":
                        USER_ID
                    }
                ]
            }

        )

        if results["ids"]:

            collection.delete(
                ids=results["ids"]
            )

    except Exception as e:

        st.error(
            f"Database error: {e}"
        )

    file_path = (
        USER_DOCUMENTS_DIR /
        filename
    )

    if file_path.exists():

        file_path.unlink()


# =========================================================
# CLEAR ALL USER DOCUMENTS
# =========================================================

def clear_all_documents():

    try:

        user_docs = collection.get(

            where={
                "user_id":
                USER_ID
            }

        )

        if user_docs["ids"]:

            collection.delete(
                ids=user_docs["ids"]
            )

    except Exception:

        pass

    if USER_DOCUMENTS_DIR.exists():

        for file in USER_DOCUMENTS_DIR.iterdir():

            if file.is_file():

                file.unlink()


# =========================================================
# GET USER DOCUMENTS
# =========================================================

def get_uploaded_documents():

    files = []

    if not USER_DOCUMENTS_DIR.exists():

        return files

    for file in USER_DOCUMENTS_DIR.iterdir():

        if file.is_file():

            if file.suffix.lower() in [

                ".pdf",
                ".docx",
                ".txt"

            ]:

                files.append(
                    file.name
                )

    return sorted(files)


# =========================================================
# DOCUMENT MANAGER
# =========================================================

st.sidebar.title(
    "📚 Document Manager"
)

st.sidebar.write(
    "Upload college documents to "
    "add them to your AI knowledge base."
)

uploaded_files = st.sidebar.file_uploader(

    "Upload documents",

    type=[
        "pdf",
        "docx",
        "txt"
    ],

    accept_multiple_files=True

)

if uploaded_files:

    st.sidebar.write(

        f"📄 {len(uploaded_files)} "
        "file(s) selected."

    )

    for uploaded_file in uploaded_files:

        st.sidebar.success(

            f"✅ {uploaded_file.name}"

        )


# =========================================================
# ADD DOCUMENTS
# =========================================================

if st.sidebar.button(

    "➕ Add Documents",

    use_container_width=True

):

    if not uploaded_files:

        st.sidebar.warning(

            "Please upload at least "
            "one document."

        )

    else:

        progress = st.sidebar.progress(
            0
        )

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            file_path = (

                USER_DOCUMENTS_DIR /
                uploaded_file.name

            )

            with open(
                file_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            chunks, characters = (
                process_document(
                    file_path
                )
            )

            if chunks == 0:

                st.sidebar.error(

                    f"❌ {uploaded_file.name}: "
                    "No readable text found."

                )

            else:

                st.sidebar.success(

                    f"✅ {uploaded_file.name}\n"
                    f"Chunks: {chunks}\n"
                    f"Characters: {characters}"

                )

            progress.progress(

                (index + 1) /
                len(uploaded_files)

            )

        st.sidebar.success(

            "🎉 Documents added successfully!"

        )


# =========================================================
# DOCUMENT LIST
# =========================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📂 Your Uploaded Documents"
)

documents = get_uploaded_documents()

if documents:

    for filename in documents:

        col1, col2 = (
            st.sidebar.columns(
                [4, 1]
            )
        )

        col1.write(
            f"📄 {filename}"
        )

        if col2.button(

            "🗑️",

            key=f"delete_{USER_ID}_{filename}"

        ):

            delete_document(
                filename
            )

            st.rerun()

else:

    st.sidebar.info(
        "No documents uploaded yet."
    )


# =========================================================
# CLEAR ALL
# =========================================================

st.sidebar.divider()

if documents:

    if st.sidebar.button(

        "🧹 Clear My Documents",

        use_container_width=True

    ):

        clear_all_documents()

        st.session_state.messages = []

        st.sidebar.success(
            "Your documents were removed."
        )

        st.rerun()


# =========================================================
# MAIN CHAT
# =========================================================

st.title(
    "🎓 College AI Assistant"
)

st.write(
    "Ask questions about your college documents."
)


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# QUESTION
# =========================================================

question = st.chat_input(
    "Ask a question..."
)

if question:

    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.messages.append({

        "role":
        "user",

        "content":
        question

    })

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # =====================================================
    # BUILD CONVERSATION HISTORY
    # =====================================================

    previous_messages = (
        st.session_state.messages[:-1]
    )

    conversation_text = ""

    # Keep last 6 messages
    for message in previous_messages[-6:]:

        conversation_text += (

            f"{message['role'].upper()}: "
            f"{message['content']}\n"

        )


    # =====================================================
    # REWRITE FOLLOW-UP QUESTION
    # =====================================================

    rewrite_prompt = f"""

You are a question rewriting assistant.

The student is asking questions about
college documents.

Conversation history:

{conversation_text}

Current question:

{question}

If the current question is a follow-up,
rewrite it as a complete standalone question.

If the current question is already complete,
return it unchanged.

Return ONLY the rewritten question.
"""

    try:

        rewrite_response = (
            groq_client
            .chat.completions.create(

                model="openai/gpt-oss-20b",

                messages=[

                    {

                        "role":
                        "user",

                        "content":
                        rewrite_prompt

                    }

                ],

                temperature=0

            )
        )

        search_question = (
            rewrite_response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception:

        search_question = question


    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

    query_embedding = (
        embedding_model.encode(
            [search_question]
        ).tolist()
    )

    try:

        results = collection.query(

            query_embeddings=
            query_embedding,

            n_results=20,

            where={
                "user_id":
                USER_ID
            }

        )

    except Exception:

        results = {

            "documents": [[]],

            "metadatas": [[]]

        }

    semantic_docs = results[
        "documents"
    ][0]

    semantic_metadata = results[
        "metadatas"
    ][0]


    # =====================================================
    # KEYWORD SEARCH
    # =====================================================

    keywords = [

        "nlp",
        "natural language processing",

        "ai&ds",
        "ai & ds",
        "ai and ds",
        "aids",

        "7 semester",
        "7th semester",
        "vii semester",

        "teacher",
        "teaches",
        "faculty",
        "professor",
        "lecturer",

        "venue",
        "lab",

        "class",
        "subject",
        "timetable"

    ]

    search_lower = (
        search_question.lower()
    )

    matched_keywords = []

    for keyword in keywords:

        if keyword in search_lower:

            matched_keywords.append(
                keyword
            )


    # =====================================================
    # GET ONLY CURRENT USER DOCUMENTS
    # =====================================================

    try:

        all_documents = collection.get(

            where={
                "user_id":
                USER_ID
            },

            include=[
                "documents",
                "metadatas"
            ]

        )

        all_docs = (
            all_documents["documents"]
        )

        all_metadata = (
            all_documents["metadatas"]
        )

    except Exception:

        all_docs = []

        all_metadata = []


    # =====================================================
    # KEYWORD MATCHING
    # =====================================================

    keyword_docs = []

    for doc, metadata in zip(

        all_docs,

        all_metadata

    ):

        doc_lower = doc.lower()

        score = 0

        for keyword in matched_keywords:

            if keyword in doc_lower:

                score += 1

        # Strong timetable bonus
        if (
            "college timetable record"
            in doc_lower
        ):

            score += 2

        # Strong NLP bonus
        if (
            "natural language processing"
            in doc_lower
        ):

            score += 3

        if score > 0:

            keyword_docs.append(

                (
                    score,
                    doc,
                    metadata

                )

            )


    keyword_docs.sort(

        key=lambda x: x[0],

        reverse=True

    )

    keyword_docs = keyword_docs[:15]


    # =====================================================
    # COMBINE RESULTS
    # =====================================================

    final_docs = []

    final_metadata = []

    # Keyword results first
    for score, doc, metadata in keyword_docs:

        if doc not in final_docs:

            final_docs.append(
                doc
            )

            final_metadata.append(
                metadata
            )

    # Semantic results
    for doc, metadata in zip(

        semantic_docs,

        semantic_metadata

    ):

        if doc not in final_docs:

            final_docs.append(
                doc
            )

            final_metadata.append(
                metadata
            )

    final_docs = final_docs[:15]

    final_metadata = (
        final_metadata[:15]
    )


    # =====================================================
    # CREATE CONTEXT
    # =====================================================

    context_parts = []

    for i, doc in enumerate(
        final_docs
    ):

        source = (
            final_metadata[i]
            .get(
                "source",
                "Unknown document"
            )
        )

        context_parts.append(

            f"""
SOURCE: {source}

{doc}
"""

        )

    context = "\n\n".join(
        context_parts
    )


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    if not context.strip():

        answer = (

            "I don't know based on the "
            "provided documents."

        )

    else:

        answer_prompt = f"""

You are a college AI assistant.

Answer the student's question using ONLY
the provided college documents.

Do NOT use outside knowledge.

If the answer is not present in the
documents, say:

"I don't know based on the provided documents."

==================================================
CONVERSATION
==================================================

{conversation_text}

==================================================
SEARCH QUESTION
==================================================

{search_question}

==================================================
CURRENT QUESTION
==================================================

{question}

==================================================
DOCUMENT CONTEXT
==================================================

{context}

==================================================
RULES
==================================================

1. Use the conversation to understand
   follow-up questions.

2. For timetable questions, match the
   exact semester/class.

3. "NLP" means Natural Language Processing
   when supported by the document.

4. "AI&DS" and "AIDS" refer to AI&DS when
   supported by the document.

5. "7 semester" means VII Semester.

6. If multiple classes have different
   answers, explain the ambiguity.

7. Never invent faculty, venue, subject,
   semester, or timetable information.

8. For chapter questions, give a clear,
   student-friendly explanation.

9. If the information is unavailable,
   say that you don't know based on the
   provided documents.

"""

        response = (
            groq_client
            .chat.completions.create(

                model="openai/gpt-oss-20b",

                messages=[

                    {

                        "role":
                        "user",

                        "content":
                        answer_prompt

                    }

                ],

                temperature=0.2

            )
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )


    # =====================================================
    # DISPLAY ANSWER
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

        # =================================================
        # SOURCES
        # =================================================

        with st.expander(
            "📚 Sources used"
        ):

            if final_docs:

                st.write(

                    "Information retrieved "
                    "from your uploaded documents."

                )

                for i, doc in enumerate(
                    final_docs
                ):

                    source = (
                        final_metadata[i]
                        .get(
                            "source",
                            "Unknown document"
                        )
                    )

                    st.markdown(

                        f"**Source {i + 1} — "
                        f"{source}**"

                    )

                    st.write(
                        doc
                    )

            else:

                st.write(

                    "No relevant information "
                    "was retrieved."

                )


    # =====================================================
    # SAVE ASSISTANT MESSAGE
    # =====================================================

    st.session_state.messages.append({

        "role":
        "assistant",

        "content":
        answer

    })