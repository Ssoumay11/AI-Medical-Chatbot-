import os
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 

# Set page config
st.set_page_config(
    page_title="Document QA System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

HF_TOKEN = os.getenv("HUGGING_FACEHUB_API_TOKEN")
hugging_face_repo_id = "google/flan-t5-xxl"

def query_huggingface_api(prompt):
    """Direct API call to Hugging Face Inference API"""
    API_URL = f"https://api-inference.huggingface.co/models/{hugging_face_repo_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.3,
            "do_sample": True,
            "top_p": 0.95
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    return result[0]['generated_text']
                elif 'summary_text' in result[0]:
                    return result[0]['summary_text']
                elif 'answer' in result[0]:
                    return result[0]['answer']
            return "Received unexpected response format from API"
        else:
            return f"API Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"API Error: {str(e)}"

# Load vector database
@st.cache_resource
def load_vector_db():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(
            "vectorstores/db_faiss",
            embedding_model,
            allow_dangerous_deserialization=True  
        )
        return db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None

# Custom prompt template
custom_prompt_template = """Answer the question based on the context below. If you don't know, say you don't know.

Context: {context}

Question: {question}

Answer:"""

def manual_qa_query(question):
    retriever = load_vector_db()
    if retriever is None:
        return {"result": "Vector database not available", "source_documents": []}
    
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs][:3])
        
        # Format the prompt
        prompt = custom_prompt_template.format(context=context, question=question)
        
        # Get response from API
        response = query_huggingface_api(prompt)
        
        return {
            'result': response,
            'source_documents': docs,
            'context': context
        }
    except Exception as e:
        return {"result": f"Error: {str(e)}", "source_documents": []}

# UI Components
st.title("📚 Document QA System")
st.markdown("Ask questions about your documents using AI-powered search")

# Sidebar for settings and history
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Using Hugging Face Flan-T5 model with FAISS vector search")
    
    st.header("📋 Query History")
    for i, (question, answer) in enumerate(st.session_state.qa_history):
        with st.expander(f"Q: {question[:50]}..." if len(question) > 50 else f"Q: {question}"):
            st.write(f"**A:** {answer[:200]}..." if len(answer) > 200 else f"**A:** {answer}")
    
    if st.button("🧹 Clear History"):
        st.session_state.qa_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "💬 Ask a question:",
        placeholder="Type your question here...",
        height=100
    )
    
    if st.button("🔍 Search", type="primary"):
        if question.strip():
            with st.spinner("Searching documents and generating answer..."):
                result = manual_qa_query(question)
                
                # Store in history
                st.session_state.qa_history.append((question, result['result']))
                
                # Display results
                st.subheader("✅ Answer:")
                st.write(result['result'])
                
                # Show context and sources
                with st.expander("📖 View Context Used"):
                    st.text_area("Context:", result.get('context', 'No context available'), height=200)
                
                st.subheader("📄 Source Documents")
                for i, doc in enumerate(result['source_documents']):
                    with st.expander(f"Document {i+1}"):
                        st.write(f"**Content:** {doc.page_content}")
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
        else:
            st.warning("Please enter a question.")

with col2:
    st.info("""
    **ℹ️ About this system:**
    
    - Uses FAISS vector database for document search
    - Employs Hugging Face's Flan-T5 model for answering
    - Searches through your document collection
    - Provides source documents for verification
    
    
    """)

# Footer
st.markdown("---")
