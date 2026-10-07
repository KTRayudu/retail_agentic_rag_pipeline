import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType, PointStruct, SparseVectorParams, Document, models
import openai

# 1. Connect to local Qdrant
print("Connecting to Qdrant...")
qdrant_client = QdrantClient(url="http://localhost:6333")

# 2. Recreate collection
print("Recreating Qdrant collection (Deleting old OpenAI vectors)...")
qdrant_client.recreate_collection(
    collection_name="Amazon-items-collection-02-ollama",
    vectors_config={
        "nomic-embed-text": VectorParams(size=768, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "bm25": SparseVectorParams(modifier=models.Modifier.IDF)
    }
)

qdrant_client.create_payload_index(
    collection_name="Amazon-items-collection-02-ollama",
    field_name="parent_asin",
    field_schema=PayloadSchemaType.KEYWORD
)

# 3. Load dataset
print("Loading dataset...")
file_path = "/home/rayudu/otherwork_assignment_all_jobs/Todo/retail_agentic_rag_pipeline/data/meta_Electronics_2022_2023_with_category_ratings_100_sample_1000.jsonl"
df_items = pd.read_json(file_path, lines=True)

data_to_embed = df_items[["description", "image", "rating_number", "price", "average_rating", "parent_asin"]].to_dict(orient="records")
text_to_embed = [data["description"] for data in data_to_embed]

# 4. Generate embeddings
print("Generating embeddings using Ollama (nomic-embed-text:latest)...")
# Note: Ensure Ollama is running and accessible
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def get_embeddings_batch(text_list, model="nomic-embed-text:latest", batch_size=50):
    all_embeddings = []
    counter = 1
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([embedding.embedding for embedding in response.data])
        print(f"Processed {min(counter * batch_size, len(text_list))} of {len(text_list)}")
        counter += 1
    return all_embeddings

embeddings = get_embeddings_batch(text_to_embed)

# 5. Push to Qdrant
print("Pushing data to Qdrant...")
pointstructs = []
i = 1
for embedding, data in zip(embeddings, data_to_embed):
    pointstructs.append(
        PointStruct(
            id=i,
            vector={
                "nomic-embed-text": embedding,
                "bm25": Document(
                    text=data["description"],
                    model="qdrant/bm25"
                )
            },
            payload=data
        )
    )
    i += 1

batch_size = 500
for j in range(0, len(pointstructs), batch_size):
    batch = pointstructs[j:j+batch_size]
    qdrant_client.upsert(
        collection_name="Amazon-items-collection-01-hybrid-search",
        points=batch,
        wait=True
    )
    print(f"Upserted {min(j+batch_size, len(pointstructs))} points...")

print("Re-indexing complete! Your database is now ready for local RAG.")
