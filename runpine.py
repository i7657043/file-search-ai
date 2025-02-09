# Import the Pinecone library
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time

# Initialize a Pinecone client with your API key
pc = Pinecone(
    api_key="xxx")

# Define a sample dataset where each item has a unique ID, text, and category
data = [
    {
        "id": "rec11",
        "text": "Apples are a great source of dietary fiber, which supports digestion and helps maintain a healthy gut.",
        "category": "digestive system"
    },
    {
        "id": "rec21",
        "text": "Apples originated in Central Asia and have been cultivated for thousands of years, with over 7,500 varieties available today.",
        "category": "cultivation"
    },
    {
        "id": "rec31",
        "text": "Rich in vitamin C and other antioxidants, apples contribute to immune health and may reduce the risk of chronic diseases.",
        "category": "immune system"
    },
    {
        "id": "rec41",
        "text": "The high fiber content in apples can also help regulate blood sugar levels, making them a favorable snack for people with diabetes.",
        "category": "endocrine system"
    }
]

# Convert the text into numerical vectors that Pinecone can index
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d["text"] for d in data],
    parameters={
        "input_type": "passage",
        "truncate": "END"
    }
)

print(embeddings)

# Create a serverless index
index_name = "example-index2"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

# Target the index
# In production, target an index by its unique DNS host, not by its name
# See https://docs.pinecone.io/guides/data/target-an-index
index = pc.Index(index_name)

# Prepare the records for upsert
# Each contains an 'id', the vector 'values',
# and the original text and category as 'metadata'
records = []
indexVal = 0
for d, e in zip(data, embeddings):
    indexVal += 1
    print("Appending record: " + str(indexVal))
    records.append({
        "id": d["id"],
        "values": e["values"],
        "metadata": {
            "source_text": d["text"],
            "category": d["category"]
        }
    })

print("\n---Printing records---")
print(records)
print("---end records---\n")

# Upsert the records into the index
index.upsert(
    vectors=records,
    namespace="example-namespace2"
)

time.sleep(5)

print(index.describe_index_stats())
