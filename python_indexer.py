from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from dotenv import load_dotenv

load_dotenv()


def load_code_chunks():
    loader = DirectoryLoader("my_codebase", glob="**/*.py", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    texts = [doc.page_content for doc in chunks]
    metadata = [doc.metadata for doc in chunks]
    return texts, metadata

def build_faiss_index(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    if not texts:
        raise ValueError("No text chunks to index. Check if your codebase is loaded properly.")
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings)
    
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embeddings, got shape: {embeddings.shape}")
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index, embeddings


if __name__ == "__main__":
    texts, metadata = load_code_chunks()
    model, index, embeddings = build_faiss_index(texts)

    os.makedirs("faiss_store", exist_ok=True)
    faiss.write_index(index, "faiss_store/index.faiss")
    #dump our readings into a pickle file
    with open("faiss_store/data.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metadata": metadata, "model_name": 'sentence-transformers/all-MiniLM-L6-v2'}, f)

    print("✅ FAISS index built and saved.")