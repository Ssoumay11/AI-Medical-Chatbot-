from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS   

DATA_PATH = "data/"

# Step 1: Load PDFs
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

documents = load_pdf_file(data=DATA_PATH)
print("Length of the document:", len(documents))

# Step 2: Split into chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts_chunks = text_splitter.split_documents(extracted_data)
    return texts_chunks

text_chunks = create_chunks(extracted_data=documents)
print("Length of the text chunks:", len(text_chunks))

# Step 3: Create embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings_model = create_embeddings()

# Step 4: Store in FAISS
DB_FAISS_PATH = "vectorstores/db_faiss"
db = FAISS.from_documents(text_chunks, embeddings_model)
db.save_local(DB_FAISS_PATH)
