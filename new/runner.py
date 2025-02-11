from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv


def print_basic_answers(similar_docs):
    counter = 0
    print("Basic answers\n\"\"\"")
    for doc in similar_docs:
        counter += 1
        print(f"#{counter}: " + doc.page_content + "\n")
    print("\"\"\"")


load_dotenv()

index_name = "search-index"

print("\nCreating embeddings...\n")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

print("\nGetting vector db...\n")
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings)

# this just comes from the context of whichever customer is logged in.
# Can grab from request arg, cookie, jwt, session, anything
customer_id = "001"

query = "Who is XXX Construction apprentice scheme part funded by?"
# query = "The slowly adapting Merkel’s receptors are responsible for what exactly?"
# query = "Is America a continent?"

# get the semantic search answer
print("\nGetting basic answers...\n")
similar_docs = vector_store.similarity_search(
    query, k=3, filter={"customer_id": {"$eq": customer_id}})
print_basic_answers(similar_docs)

# get the LLLM answer
print("\nGetting LLM answer...\n")
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

retriever = vector_store.as_retriever(
    search_kwargs={"filter": {"customer_id": {"$eq": customer_id}}})
llm = ChatOpenAI(model="gpt-4o", temperature=0)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

response = chain.invoke({"input": query})
print("LLM Answer: " + response["answer"] + "\n")
