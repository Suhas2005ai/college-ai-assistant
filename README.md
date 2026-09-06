# College AI Assistant

College AI Assistant is an AI-powered document question-answering application built using Retrieval-Augmented Generation (RAG).

The application allows users to upload college-related documents and interact with them through a simple chat interface. It retrieves relevant information from the uploaded documents and uses a Large Language Model (LLM) to generate context-based responses.

## Overview

Finding specific information in large academic documents can be time-consuming. This project provides a simple way to interact with those documents using natural language.

The system processes uploaded documents, converts their content into embeddings, stores them in a vector database, and retrieves the most relevant information when a user asks a question.

## Key Features

- Upload and process PDF, DOCX, and TXT documents
- Question answering based on uploaded documents
- Retrieval-Augmented Generation (RAG)
- Semantic search using text embeddings
- Vector storage and similarity search using ChromaDB
- Support for multiple documents
- Context-aware conversational follow-up questions
- Source information for generated responses
- Specialized handling for college timetable information
- Secure API key management
- Streamlit-based web interface
- Cloud deployment using Streamlit Community Cloud

## System Architecture

Document Upload
        ↓
Text Extraction
        ↓
Text Chunking
        ↓
Embedding Generation
        ↓
ChromaDB
        ↓
User Query
        ↓
Query Embedding
        ↓
Similarity Search
        ↓
Relevant Context
        ↓
Groq LLM
        ↓
Generated Response

## RAG Workflow

The application follows a Retrieval-Augmented Generation approach:

1. Documents are uploaded through the web interface.
2. Text is extracted from the uploaded files.
3. The extracted content is divided into smaller chunks.
4. Each chunk is converted into a vector embedding.
5. The embeddings are stored in ChromaDB.
6. When a user submits a question, the question is converted into an embedding.
7. ChromaDB performs similarity search to identify relevant document content.
8. The retrieved content is provided as context to the LLM.
9. The LLM generates a response based on the retrieved information.

This approach helps the application provide responses that are grounded in the user's documents rather than relying only on the model's general knowledge.

## Technologies Used

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- LangChain Text Splitters
- Groq
- PyPDF
- python-docx

## Project Structure

college-ai-assistant/
│
├── web_app.py
├── app.py
├── build_database.py
├── requirements.txt
├── README.md
└── .gitignore

## Installation

Clone the repository:

git clone https://github.com/Suhas2005ai/college-ai-assistant.git

cd college-ai-assistant

Create and activate a virtual environment:

python3 -m venv venv

source venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

Create a .env file and configure the Groq API key:

GROQ_API_KEY=your_api_key_here

Run the application:

python -m streamlit run web_app.py

## Deployment

The application is deployed using Streamlit Community Cloud and connected to the GitHub repository.

The Groq API key is stored securely using Streamlit Secrets and is not included in the source code.

## Learning Outcomes

This project provided hands-on experience with:

- Retrieval-Augmented Generation
- Large Language Models
- Text embeddings
- Vector databases
- Semantic search
- Document processing
- Prompt engineering
- Conversational AI
- API integration
- Streamlit application development
- Cloud deployment

## Future Improvements

- Add authentication and user accounts
- Improve document retrieval and ranking
- Support additional document formats
- Add persistent conversation history
- Improve response evaluation
- Add advanced document management

## Author

Suhas Gowda G R

Engineering Student | Python | AI/ML | Generative AI