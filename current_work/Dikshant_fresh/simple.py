from transformers import AutoTokenizer, AutoModel
import torch
import chromadb


LLM_API_KEY = 'AIzaSyCDfqPNXPPfBsW5fgGbmiMfro5loK15wt0'


client = chromadb.PersistentClient(path="./data.db")

collection = client.get_or_create_collection("my_collection")
# Example function to generate embeddings using a transformer model
def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Load a pre-trained transformer model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Generate embeddings for each document
documents = [
    "This is a document about machine learning",
    "This is another document about data science",
    "A third document about artificial intelligence"
]
embeddings = [get_embedding(doc, model, tokenizer).tolist()[0] for doc in documents]

# Add documents, embeddings, metadata, and IDs to the collection
collection.add(
    documents=documents,
    metadatas=[
        {"source": "test1"},
        {"source": "test2"},
        {"source": "test3"}
    ],
    ids=[
        "id1",
        "id2",
        "id3"
    ],
    embeddings=embeddings  # Pass the generated embeddings here
)

# Now you can query the collection and retrieve results based on embeddings
data = collection.query(query_texts=["machine learning"], n_results=2)
print(data)