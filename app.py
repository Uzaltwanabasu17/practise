import os
import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from new import GOOGLE_API_KEY, initialize_embeddings, initialize_generative_model

# Initialize embeddings and generative model
embeddings = initialize_embeddings()
gen_model = initialize_generative_model()

@st.cache_resource
def load_documents():
    documents = []
    for file in os.listdir("./documents"):
        file_path = os.path.join("./documents", file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {file}")
            continue
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    vector_store = Chroma.from_documents(texts, embeddings)
    return vector_store 

# Load pre-trained model and tokenizer for GPT-2 (if you still need it)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")

def get_response(query, chat_history, vector_store):
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=3)
    
    # Prepare the context
    context = "\n".join([doc.page_content for doc in docs])
    
    # Prepare the prompt
    system_prompt = f"You are a helpful assistant. Use the following context to answer the user's question: {context}"
    
    # Prepare the chat history and current query
    full_prompt = f"{system_prompt}\n\n"
    for user, bot in chat_history:
        full_prompt += f"Human: {user}\nAssistant: {bot}\n"
    full_prompt += f"Human: {query}\nAssistant:"

    # Use the initialized generative model here
    response = gen_model.generate_content(full_prompt)
    
    return response.text

# Load documents and create vector store
vector_store = load_documents()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("Document Q&A Chatbot")

user_query = st.text_input("Ask a question about the documents:")

if user_query:
    response = get_response(user_query, st.session_state.chat_history, vector_store)
    st.session_state.chat_history.append((user_query, response))
    
    # Display the latest response
    st.write("Chatbot:", response)

# Display chat history
st.subheader("Chat History")
for user, bot in st.session_state.chat_history:
    st.write(f"User: {user}")
    st.write(f"Chatbot: {bot}")
    st.write("---")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()