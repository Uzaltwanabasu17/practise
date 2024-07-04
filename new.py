import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

GOOGLE_API_KEY = "AIzaSyBp3vbJjBGZh8YX-TLgqSHpuPx9Kj7X00Y"  # Replace with your actual API key

def initialize_embeddings():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        print("Embeddings initialized successfully")
        return embeddings
    except Exception as e:
        print(f"An error occurred while initializing embeddings: {e}")
        return None

def initialize_generative_model():
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        print("Generative model initialized successfully")
        return model
    except Exception as e:
        print(f"An error occurred while initializing generative model: {e}")
        return None