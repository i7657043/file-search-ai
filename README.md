# file-search-ai

Reads in 2 PDFs and allows you to search them, and filter by cusomter-id

### Dependencies

`pip install langchain-pinecone langchain-openai langchain langchain-core python-dotenv`

- I hope these cover everything

## 1. Get API Keys

- Obtain an **OpenAI API key** ($5)
- Obtain a **PineconeDB API key** (Free)

## 2. Configure Environment Variables

- Create a `.env` file in the root directory.
- Fill in the values according to `.env.example`.

## 3. Create a Serverless Index in PineconeDB

Follow the [Pinecone guide](https://docs.pinecone.io/guides/indexes/create-an-index) to create an index with the following settings:

- **Model:** `text-embedding-ada-002`
- **Capacity Mode:** `Serverless`
- **Cloud Provider:** AWS

## 4. Update Python Scripts as this is just a prototype

- Replace the value of `index_name` in both `loader.py` and `runner.py` with the index name you set in Pinecone.
- Replace the value of `query` var in runner.py with your question or use the default
- Replace the value of `customer_id` var in runner.py with the valid or invalid customer_id (`"001"` or `"002"`) depending on what you want to test

## 5. Load Embeddings

Run the following command to set embeddings in the index:

```sh
python loader.py
```

## 6. Query Embeddings

Run the following command to query your documents:

```sh
python runner.py
```
