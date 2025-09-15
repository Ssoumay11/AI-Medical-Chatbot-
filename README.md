# 🩺 AI Medical Chatbot

An **AI-powered medical QA system** that helps users query medical PDFs and get **context-aware answers** with full source transparency.  
Built with **Streamlit**, **Hugging Face Transformers**, **LangChain**, and **FAISS** for semantic search.  

---

## ✨ Features
- 📄 **Document-based QA** – Upload and query medical PDFs  
- 🔍 **Semantic Search** – FAISS vector database with Hugging Face embeddings  
- 💡 **Context-Aware Answers** – Powered by Flan-T5 LLM  
- 📚 **Source Transparency** – Provides reference chunks from documents  
- 💻 **Interactive UI** – Streamlit-based interface with query history tracking  
- ⚡ **Error Handling** – Robust design for smooth interactions  

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit** (frontend UI)
- **LangChain** (document processing)
- **Hugging Face** (Flan-T5, Sentence Transformers)
- **FAISS** (vector search)
- **PyPDFLoader** (PDF parsing)

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ai-medical-chatbot.git
cd ai-medical-chatbot


python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

pip install -r requirements.txt

HUGGING_FACEHUB_API_TOKEN=your_token_here
python create.py

📌 Future Improvements

Support for multiple document uploads

Advanced ranking of search results

Integration with medical knowledge bases

🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


---

👉 Do you want me to also generate a **short project description for GitHub** (the one-liner that appears under the repo name)?
