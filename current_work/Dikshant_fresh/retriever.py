import chromadb

client = chromadb.PersistentClient(path="./data.db")

collection = client.get_or_create_collection("my_collection")

out = collection.query(query_texts=["machine learning"], n_results=2)

print(out)