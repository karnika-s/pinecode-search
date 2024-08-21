import streamlit as st
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import nltk

# Download the necessary nltk data
nltk.download('punkt')

# Load environment variables
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "hybrid-search-langchain-pinecone"

# Initialize Pinecone
pc = Pinecone(api_key=API_KEY)

# Create the index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Dimensionality of dense model
        metric="dotproduct",  # Sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# Set up embeddings and BM25 encoder
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()

# Define sample sentences and key phrases
sentences = [
    "In 2023, I visited Paris",
    "In 2022, I visited New York",
    "In 2021, I visited New Orleans",
    "In 2024, I visited Kashmir",
    "In 2020, I visited New Delhi"
]

# Key phrases or patterns to recognize in queries
key_phrases = [
    "where did i visit in 2023",
    "where did i visit in 2022",
    "where did i visit in 2021",
    "where did i visit in 2024",
    "where did i visit in 2020"
]

# Fit the BM25 encoder
bm25_encoder.fit(sentences)
bm25_encoder.dump("bm25_values.json")

# Load the BM25 encoder from the saved file
bm25_encoder = BM25Encoder().load("bm25_values.json")

# Set up the retriever
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

# Add texts to the index
retriever.add_texts(sentences)

# Function to match the query to predefined key phrases
def match_query(query):
    query = query.lower().strip()  # Normalize the query for comparison
    for phrase in key_phrases:
        if phrase in query:
            return True
    return False

# Streamlit app layout
st.title("😎 My Hybrid Search buddy")
st.write("Query from pre-defined data.")
st.write("----------------------------------------------------------")

# User input
query = st.text_input("Enter your query:", placeholder="", key="query")

# Search and display the results
if st.button("Search"):
    if query.strip():
        if match_query(query):  # Check if the query matches any predefined key phrase
            try:
                result = retriever.invoke(query)
                if result:
                    st.subheader("Search Results:")
                    st.write(result[0].page_content)
                else:
                    st.write("No matching results found.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.write("Not found")
    else:
        st.warning("Please enter a query to search.")

st.write("----------------------------------------------------------")
st.write("Created Using Pinecone & Langchain")
