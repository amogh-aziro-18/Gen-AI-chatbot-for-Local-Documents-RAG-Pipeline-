# Gen-AI-chatbot-for-Local-Documents-RAG-Pipeline-

This project is a Generative AI chatbot designed to interact with custom PDF documents using a Retrieval-Augmented Generation (RAG) approach. It is built with Streamlit, LangChain, and Groq’s LLaMA models. The chatbot allows users to upload local PDFs—such as financial reports—and ask natural language questions to extract meaningful insights from the document.

The model used in this project is powered by Groq’s LLaMA-3.1 series, known for its fast and contextually accurate responses. The app was tested using a Financial Analysis PDF as the primary data source.\

## Main Features:

1. Allows users to upload and process PDF documents locally

2. Uses RAG architecture for accurate, context-based responses

3. Integrates Groq’s LLaMA-3 models for inference (70B and 8B variants)

4. Provides fast text extraction and document chunking for efficient retrieval

5. Displays extracted metadata such as document title, chapters, and page numbers

6. Simple and interactive chat interface built using Streamlit

7. Light and dark modes for a modern, user-friendly experience

8. Configurable settings in the sidebar for model selection and PDF processing

## Setup Instructions:

1. Create and activate a Python virtual environment.

2. Install all required dependencies using the command “pip install -r requirements.txt”.

3. Create a .env file in the root directory and add your Groq API key as “GROQ_API_KEY=your_api_key_here”.

4. Run the application using the command “streamlit run app.py”.

5. Open the link displayed in your terminal (usually http://localhost:8501
) to access the chatbot.

## How It Works:
Once a PDF is uploaded, the app extracts its content and divides it into manageable chunks. A custom retriever identifies the most relevant sections based on your query. The retrieved context is then passed to the Groq LLaMA model, which generates precise and factual responses based only on the document data.

## Example Use Case:
This project demonstrates how a financial analysis report can be explored interactively. After uploading a financial PDF, users can ask questions like:

1. “What are the main financial highlights mentioned in the document?”

2. “What are the limitations?”

3. “List the risk factors discussed in the report.”

The chatbot retrieves relevant text from the document and answers with high accuracy and context awareness.

## Requirements:
All necessary dependencies are listed in the requirements.txt file.

## Screenshots:
The project includes screenshots that illustrate the working chatbot interface. These show:

1. The chatbot running in light mode and dark mode

2. A sample conversation with the uploaded financial analysis PDF

3. The sidebar options for model configuration and document upload

Each screenshot captures different stages of interaction, demonstrating the chatbot’s functionality and interface.

## Future Enhancements:

1. Integration with FAISS or Chroma for improved vector-based retrieval

2. Multi-PDF support for simultaneous analysis

3. Persistent chat history and improved session management

4. Deployment on Streamlit Cloud or Hugging Face Spaces

## Author:
Developed as part of a Generative AI RAG pipeline project.
This version includes a requirements.txt file, screenshots of the interface, and a working demonstration using a Financial Analysis PDF.
