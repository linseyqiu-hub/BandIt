import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="data/chroma")

essays_col    = client.get_collection("essays",    embedding_function=None)
questions_col = client.get_collection("questions", embedding_function=None)

print("essays count:",    essays_col.count())
print("questions count:", questions_col.count())

# spot check — query essays collection with a sample essay snippet
model = SentenceTransformer("all-MiniLM-L6-v2")
test_query = "The graph shows the percentage of people who used the internet in three countries."
query_embedding = model.encode([test_query]).tolist()

results = essays_col.query(
    query_embeddings=query_embedding,
    n_results=3,
)

print("\nTop 3 retrieved examiner comments:")
for i, doc in enumerate(results["documents"][0]):
    print(f"\n[{i+1}] band={results['metadatas'][0][i]['band_bin']} overall={results['metadatas'][0][i]['overall']}")
    print(doc[:200])