import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")


# -----------------------------------
# 2. Connect to Groq
# -----------------------------------

groq_client = Groq(api_key=groq_api_key)


# -----------------------------------
# 3. Load embedding model
# -----------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------------
# 4. Connect to ChromaDB
# -----------------------------------

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="college_documents"
)


# -----------------------------------
# 5. Start chatbot
# -----------------------------------

print("\n===================================")
print("       COLLEGE AI ASSISTANT")
print("===================================")

print("Ask questions about your college documents.")
print("Type 'exit' to stop.\n")


while True:

    # -----------------------------------
    # 6. Get user question
    # -----------------------------------

    question = input("You: ")


    # -----------------------------------
    # 7. Exit
    # -----------------------------------

    if question.lower() == "exit":

        print("\nAssistant: Goodbye! 👋")
        break


    # -----------------------------------
    # 8. Create question embedding
    # -----------------------------------

    question_embedding = embedding_model.encode(
        question
    ).tolist()


    # -----------------------------------
    # 9. Retrieve relevant documents
    # -----------------------------------

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=5
    )


    retrieved_documents = results["documents"][0]


    # -----------------------------------
    # 10. Combine retrieved context
    # -----------------------------------

    context = "\n\n".join(
        retrieved_documents
    )


    # -----------------------------------
    # 11. Ask Groq
    # -----------------------------------

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[

            {
                "role": "system",

                "content": """
You are a college AI assistant.

Your job is to answer questions using ONLY
the information provided in the context.

IMPORTANT RULES:

1. Never invent information.

2. If the context does not contain the answer,
say:
"I don't know based on the provided documents."

3. If the question is about a subject such as
Natural Language Processing (NLP), check whether
different classes or sections have different
faculty members.

4. If different faculty members are shown for
different classes, DO NOT choose one randomly.

5. If the user did not specify the class or section,
explain that the answer depends on the class and
ask the user to specify the class.

6. If the user specifies a class such as AI&DS,
AI&ML, CSE, etc., give the answer relevant to
that class.

7. Keep answers short and clear.

8. Do not mention information that is unrelated
to the user's question.
"""
            },

            {
                "role": "user",

                "content": f"""
Context:

{context}


Question:

{question}
"""
            }

        ]
    )


    # -----------------------------------
    # 12. Get answer
    # -----------------------------------

    answer = response.choices[0].message.content


    # -----------------------------------
    # 13. Display answer
    # -----------------------------------

    print("\nAssistant:", answer)


    # -----------------------------------
    # 14. Display sources
    # -----------------------------------

    print("\nSources used:")

    for i, source in enumerate(
        retrieved_documents,
        1
    ):

        # Display only first 200 characters
        short_source = source[:200]

        # Replace unnecessary new lines
        short_source = short_source.replace(
            "\n",
            " "
        )

        print(
            f"[{i}] {short_source}..."
        )


    print(
        "\n-----------------------------------\n"
    )