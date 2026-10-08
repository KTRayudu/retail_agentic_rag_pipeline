import openai
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Document


# @traceable(
#     name="embed_query",
#     run_type="embedding",
#     metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"}
# )
# def get_embedding(text, model="text-embedding-3-small"):
#     response = openai.embeddings.create(
#         input=text,
#         model=model,
#     )
# 
#     current_run = get_current_run_tree()
# 
#     if current_run:
#         current_run.metadata["usage_metadata"] = {
#             "input_tokens": response.usage.prompt_tokens,
#             "total_tokens": response.usage.total_tokens,
#         }
# 
#     return response.data[0].embedding

# @traceable(
#     name="embed_query",
#     run_type="embedding",
#     metadata={"ls_provider": "ollama", "ls_model_name": "nomic-embed-text:latest"}
# )
# def get_embedding(text, model="nomic-embed-text:latest"):
#     client = openai.OpenAI(base_url="http://host.docker.internal:11434/v1", api_key="ollama")
#     response = client.embeddings.create(
#         input=text,
#         model=model,
#     )
# 
#     current_run = get_current_run_tree()
# 
#     if current_run:
#         current_run.metadata["usage_metadata"] = {
#             "input_tokens": response.usage.prompt_tokens,
#             "total_tokens": response.usage.total_tokens,
#         }
# 
#     return response.data[0].embedding

from google import genai

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "google", "ls_model_name": "models/embedding-001"}
)
def get_embedding(text, model="models/embedding-001"):
    client = genai.Client()
    response = client.models.embed_content(
        model=model,
        contents=text,
    )

    return response.embeddings[0].values


@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_data(query, qdrant_client=None, k=5):

    query_embedding = get_embedding(query)

    if not qdrant_client:
        qdrant_client = QdrantClient(url="http://qdrant:6333")

    # results = qdrant_client.query_points(
    #     collection_name="Amazon-items-collection-02-ollama",
    #     prefetch=[
    #         Prefetch(
    #             query=query_embedding,
    #             using="nomic-embed-text",
    #             limit=20
    #         ),
    #         Prefetch(
    #             query=Document(
    #                 text=query,
    #                 model="qdrant/bm25"
    #             ),
    #             using="bm25",
    #             limit=20
    #         )
    #     ],
    #     query=FusionQuery(fusion="rrf"),
    #     limit=k,
    # )

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-03",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="gemini-embedding-001",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25"
                ),
                using="bm25",
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


def get_formatted_context(query: str, top_k: int = 5) -> str:

    """Get the top k context, each representing an inventory item for a given query.
    
    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    
    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_data(query, top_k)
    formatted_context = process_context(context)

    return formatted_context