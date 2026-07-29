import json

notebook_path = "/home/rayudu/otherwork_assignment_all_jobs/Todo/retail_agentic_rag_pipeline/notebooks/week_3/2_shopping_assistant_gemini.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

markdown_source = """### Explanation of the Error & Fix

The `AttributeError: 'QueryExpandResponse' object has no attribute 'statements'` occurs because the LangGraph is still using an older version of the `query_expand_node` function (from earlier in the notebook, which expected a `query` string and accessed `parsed.statements`). Meanwhile, the Pydantic model `QueryExpandResponse` was redefined later to use the attribute `expanded_query` instead of `statements`.

Because the older `query_expand_node` was never overwritten in memory (or the cell defining the new one taking `state: State` wasn't executed before compiling the graph), `graph.invoke` passes the `State` dictionary to the old function. It then tries to access `.statements` on a response object that only has `.expanded_query`.

**To fix this end-to-end**, we need to ensure the correct `QueryExpandResponse` model, the correct `query_expand_node` (which accepts `state: State` and returns `expanded_query`), and the graph definition are all executed in sequence. Here is the complete updated code to execute:
"""

code_source = """# 1. Redefine State and Response Schema
class State(BaseModel):
    expanded_query: List[str] = []
    retrieved_context: Annotated[List[str], add] = []
    initial_query: str = ""
    answer: str = ""
    query: str = ""
    k: int = 10

class QueryExpandResponse(BaseModel):
   expanded_query: List[str]

# 2. Redefine the node to take `state: State` and use `parsed.expanded_query`
@traceable(
    name="query_expand_node",
    run_type="llm",
    metadata={"ls_provider": "google", "ls_model_name": "gemini-3.5-flash"}
)
def query_expand_node(state: State) -> dict:
   prompt_template = \"\"\"You are part of a shopping assistant that can answer questions about products in stock.

Instructions:
- You will be given a question and you need to expand it into a list of statements that can be used in contextual search to retrieve relevant products.
- The statements should not overlap in context.
- The answer to the question should contain detailed information about the product and returned with detailed specification in bullet points.

<Question>
{{ query }}
</Question>
\"\"\"
   template = Template(prompt_template)
   prompt = template.render(query=state.initial_query)

   response = gemini_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.5,
            response_mime_type="application/json",
            response_schema=QueryExpandResponse,
        ),
   )
   parsed: QueryExpandResponse = response.parsed

   return {
      "expanded_query": parsed.expanded_query
   }

# 3. Rebuild and Compile the Graph
workflow = StateGraph(State)

workflow.add_node("query_expand_node", query_expand_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("aggregator_node", aggregator_node)

workflow.add_edge(START, "query_expand_node")
workflow.add_conditional_edges("query_expand_node", query_expand_conditional_edges)
workflow.add_edge("retrieve_node", "aggregator_node")
workflow.add_edge("aggregator_node", END)

graph = workflow.compile()

# 4. Invoke the Graph
query = "Can I get a tablet for my kid, a watch for me and a laptop for my wife?"
initial_state = {
    "initial_query": query,
}

result = graph.invoke(initial_state)
print(result["answer"])
"""

nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [line + "\n" if i < len(markdown_source.split('\n')) - 1 else line for i, line in enumerate(markdown_source.split('\n'))]
})

nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" if i < len(code_source.split('\n')) - 1 else line for i, line in enumerate(code_source.split('\n'))]
})

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
