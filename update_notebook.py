import json

notebook_path = "/home/rayudu/otherwork_assignment_all_jobs/Todo/retail_agentic_rag_pipeline/notebooks/week_3/2_shopping_assistant_gemini.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def query_expand_node(query)" in source:
            print("Found old cell!")
            # Comment out the old cell source
            new_source = ["# THIS OLD CODE HAS BEEN COMMENTED OUT BECAUSE IT CAUSED AN ATTRIBUTE ERROR\n"]
            for line in cell["source"]:
                new_source.append("# " + line)
            cell["source"] = new_source

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Finished updating notebook.")
