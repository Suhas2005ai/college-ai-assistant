import os
import shutil
import chromadb

from docx import Document
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# -----------------------------------
# 1. Documents folder
# -----------------------------------

documents_folder = "documents"


# -----------------------------------
# 2. Read all documents
# -----------------------------------

all_text = []
source_names = []


print("Reading college documents...\n")


for filename in os.listdir(documents_folder):

    filepath = os.path.join(
        documents_folder,
        filename
    )

    # Skip folders
    if not os.path.isfile(filepath):
        continue


    text = ""


    # -----------------------------------
    # PDF
    # -----------------------------------

    if filename.lower().endswith(".pdf"):

        print(f"Reading PDF: {filename}")

        reader = PdfReader(filepath)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"


    # -----------------------------------
    # DOCX
    # -----------------------------------

    elif filename.lower().endswith(".docx"):

        print(f"Reading DOCX: {filename}")

        document = Document(filepath)

        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                text += (
                    paragraph.text.strip()
                    + "\n"
                )


    # -----------------------------------
    # TXT
    # -----------------------------------

    elif filename.lower().endswith(".txt"):

        print(f"Reading TXT: {filename}")

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()


    # -----------------------------------
    # Unsupported file
    # -----------------------------------

    else:

        print(
            f"Skipping unsupported file: {filename}"
        )

        continue


    # -----------------------------------
    # Store document
    # -----------------------------------

    if text.strip():

        all_text.append(text)

        source_names.append(filename)


# -----------------------------------
# 3. Check documents
# -----------------------------------

if not all_text:

    raise ValueError(
        "No supported documents found "
        "inside the documents folder."
    )


print(
    f"\nTotal documents found: {len(all_text)}"
)


# -----------------------------------
# 4. Split into chunks
# -----------------------------------

print("\nCreating chunks...")


splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=50
)


chunks = []

metadata_list = []


for text, source in zip(
    all_text,
    source_names
):

    document_chunks = splitter.split_text(
        text
    )


    for chunk in document_chunks:

        chunks.append(chunk)

        metadata_list.append(
            {
                "source": source
            }
        )


print(
    f"Total chunks: {len(chunks)}"
)


# -----------------------------------
# 5. Load embedding model
# -----------------------------------

print("\nLoading embedding model...")


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------------
# 6. Create embeddings
# -----------------------------------

print("\nCreating embeddings...")


embeddings = embedding_model.encode(
    chunks,
    show_progress_bar=True
)


print("Embeddings created!")


# -----------------------------------
# 7. Delete old database
# -----------------------------------

print("\nRemoving old ChromaDB...")


if os.path.exists("chroma_db"):

    shutil.rmtree("chroma_db")


# -----------------------------------
# 8. Create new ChromaDB
# -----------------------------------

print("Creating new ChromaDB...")


client = chromadb.PersistentClient(
    path="./chroma_db"
)


collection = client.create_collection(
    name="college_documents"
)


# -----------------------------------
# 9. Store documents
# -----------------------------------

collection.add(

    ids=[
        str(i)
        for i in range(len(chunks))
    ],

    documents=chunks,

    embeddings=embeddings.tolist(),

    metadatas=metadata_list
)


# -----------------------------------
# 10. Finished
# -----------------------------------

print("\n===================================")
print("       DATABASE CREATED!")
print("===================================")

print(
    f"Documents: {len(all_text)}"
)

print(
    f"Chunks: {len(chunks)}"
)

print(
    "Sources stored with every chunk."
)

print(
    "ChromaDB: ./chroma_db"
)

print("===================================")