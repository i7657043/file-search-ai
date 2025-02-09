# Import the Pinecone library
import time
from pinecone.grpc import PineconeGRPC as Pinecone
from load import load_documents, split_text, store_embeddings, query_vectorstore, get_vectorstore
import os
from dotenv import load_dotenv

load_dotenv()

file_paths = ["C:\\fv\\docs\\Moray Council.pdf"]

documents = load_documents(file_paths)

# for doc in documents:
#     print(f"Metadata: {doc.metadata}\n\n")
#     print(f"Content: {doc.page_content}\n\n")
#     print("-" * 80)

chunks = split_text(documents)

# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}: {chunk.page_content}\n\n")

INDEX_NAME = "docsearch-index"

# index = pinecone.Index(INDEX_NAME)

# print(index.describe_index_stats())

# print("\n\n\n---Storing vectors...---\n\n\n")

# vectorstore = store_embeddings(chunks, INDEX_NAME)

# time.sleep(15)

# print("\n\n\n---Vectors stored---\n\n\n")

# indexToPrint = vectorstore.get_pinecone_index(INDEX_NAME)

# print(indexToPrint.describe_index_stats())

# print(pinecone_index)

vectorstore = get_vectorstore(INDEX_NAME)

query = "how long does the training take for regulation 10 of the Control of Asbestos Regulations 2012?"
res = query_vectorstore(query, vectorstore)

counter = 1
print("Question: " + query + "\n\n")
for doc in res:
    print(f"Content #{counter}: {doc.page_content}\n")
    counter += 1
