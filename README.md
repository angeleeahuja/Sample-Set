# Overview
It is an interactive query bot that allows users to upload documents, ask questions, and receive both generated answers and relevant document segments. Powered by Streamlit, Pinecone, Cohere, and LangChain, this application provides an efficient way to retrieve and analyze information from uploaded files.

# Features
1. File Uploads: Users can upload files in PDF format to extract text for querying.
2. Interactive Querying: Ask questions about the uploaded document and receive AI-generated responses.
3. Document Retrieval: View relevant segments from the document alongside the answer.
4. History Tracking: Tracks conversation history for ongoing interactions.

# Technology Stack
1. Streamlit: Web interface for user interaction.
2. Pinecone: Vector storage for document embedding and retrieval.
3. Cohere: Embeddings and language model for text generation.
4. LangChain: Chains together document processing, retrieval, and generation steps.

# Getting Started
Prerequisites:
Before running the application, ensure you have the following installed on your system:
1. Python 3.8 or higher
2. Pip (Python package manager)

Installation Steps:
1. Clone the Repository:

   git clone https://github.com/angeleeahuja/Sample-Set.git
   
   cd Sample-Set

3. Set Up a Virtual Environment (Optional but recommended):

   python3 -m venv venv
   
   venv\Scripts\activate

5. Install the Dependencies:

   pip install -r requirements.txt

6. Configure Environment Variables: Create a .env file in the project root and add the following:

   COHERE_API_KEY= your-cohere-api-key
   
   PINECONE_API_KEY= your-pinecone-api-key

8. Run the Application:

   streamlit run sample_set.py
