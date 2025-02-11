from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


def filter_metadata_for_customer(docs, customer_id):
    for doc in docs:
        source = doc.metadata["source"]
        doc.metadata.clear()
        doc.metadata["source"] = source
        doc.metadata["customer_id"] = customer_id


def load_docs(file_paths):
    customer_id = 0
    all_docs = []

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        customer_id += 1
        filter_metadata_for_customer(docs, f"00{customer_id}")
        all_docs.extend(docs)
    return all_docs


def split_docs(all_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len
    )
    split_docs = text_splitter.split_documents(all_docs)
    return split_docs


load_dotenv()

print("\nLoading docs...")

file_paths = [
    "C:\\fv\\docs\\Moray Council.pdf",
    "C:\\fv\\docs\\Bio.pdf"
]

# pull documents from blob storage, by page, into memory
all_docs = load_docs(file_paths)

split_docs = split_docs(all_docs)

index_name = "search-index"

print("\nCreating embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

print("\nAdding vectors...")
vector_store = PineconeVectorStore.from_documents(
    split_docs, embeddings, index_name=index_name
)

print("\nDone\n")
