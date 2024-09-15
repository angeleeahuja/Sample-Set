import os
import fitz
import mimetypes
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

# Load API keys
load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')

# Function to split PDF text into chunks
def get_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

# Initialize Pinecone vector store
def get_vector_store(docs):
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=os.environ["COHERE_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index('dataindex')

    # Create index if not exists
    if 'dataindex' not in pc.list_indexes().names():
        pc.create_index(
            name='dataindex',
            dimension=1024,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # Store document embeddings
    vectorstore = PineconeVectorStore.from_documents(docs, embeddings, index_name='dataindex', namespace="default_namespace")
    return vectorstore

# Set up Retrieval QA chain using Pinecone and Cohere
def get_retrieval_qa_chain(vectorstore):
    llm = ChatCohere(
        model="command-xlarge-nightly",
        cohere_api_key=os.environ["COHERE_API_KEY"],
    )
    retriever = vectorstore.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain

# Handle the conversation and process the chat with retrieved documents
def process_chat(qa_chain, user_prompt):
    # Retrieve document chunks and generated result
    retrieved_docs = qa_chain.retriever.invoke(user_prompt)
    result = qa_chain.invoke(user_prompt).get('result')
    
    return retrieved_docs, result

# Streamlit main function
def main():
    st.set_page_config(page_title='DocuQuery', page_icon='🤖', layout='wide')
    st.sidebar.title('DocuQuery')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    user_prompt = st.chat_input("Your message here...")
    uploaded_file = st.sidebar.file_uploader('Upload Files', type=['pdf'])
    if uploaded_file is not None:
        mime_type = mimetypes.guess_type(uploaded_file.name)[0]
        if mime_type == 'application/pdf':
            docs = get_text_from_pdf(uploaded_file)
        else:
            st.sidebar.write("Unsupported file type.")
            return

        vector_store = get_vector_store(docs)
        st.session_state.qa_chain = get_retrieval_qa_chain(vector_store)
        st.sidebar.success("File processed successfully.")
    
    if user_prompt and st.session_state.qa_chain:
        retrieved_docs, response = process_chat(st.session_state.qa_chain, user_prompt)

        # Add both user and assistant messages to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_prompt))
        st.session_state.chat_history.append(AIMessage(content=response))

        # Display the retrieved document segments and the assistant's response
        with st.chat_message("assistant"):
            st.write("**Retrieved Document Chunks:**")
            for doc in retrieved_docs:
                st.write(doc.page_content)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

if __name__ == '__main__':
    main()