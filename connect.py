import os
import requests
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 

HF_TOKEN = os.getenv("HUGGING_FACEHUB_API_TOKEN")

hugging_face_repo_id = "google/flan-t5-large"  

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

DB_FAISS_PATH = "vectorstores/db_faiss"


custom_prompt_template = """Answer the question based on the context below. If you don't know, say you don't know.

Context: {context}

Question: {question}

Answer:"""


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True  
)


retriever = db.as_retriever(search_kwargs={"k": 3})

def manual_qa_query(question):
   
    docs = retriever.invoke(question)  
    context = "\n".join([doc.page_content for doc in docs][:3]) 
    
   
    prompt = custom_prompt_template.format(context=context, question=question)
    
    print(f"Generated prompt length: {len(prompt)} characters")
    

    response = query_huggingface_api(prompt)
    
    return {
        'result': response,
        'source_documents': docs
    }


user_query = input("Write query here: ")
response = manual_qa_query(user_query)
print("\n" + "="*60)
print("RESULT:", response['result'])
print("="*60)
print(f"\nSOURCE DOCUMENTS: {len(response['source_documents'])} documents found")

for i, doc in enumerate(response['source_documents']):
    print(f"\nDocument {i+1}:")
    print(f"Content: {doc.page_content[:200]}...")
    if hasattr(doc, 'metadata') and doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        