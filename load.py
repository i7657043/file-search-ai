
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


def load_documents(file_paths):
    documents = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue

        docs = loader.load()
        documents.extend(docs)

    return documents


def get_openai_embedding_model():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    return embeddings


def split_text(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(documents)


def store_embeddings(chunks, index_name):
    embeddings = get_openai_embedding_model()

    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        index_name=index_name,
        embedding=embeddings
    )

    return vectorstore


def query_vectorstore(query, vectorStore: PineconeVectorStore):
    res = vectorStore.similarity_search(query, k=3)
    return res


def get_vectorstore(index_name):
    embeddings = get_openai_embedding_model()
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings)
    return vectorstore
