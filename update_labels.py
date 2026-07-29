import json

notebook_path = "/home/rayudu/otherwork_assignment_all_jobs/Todo/retail_agentic_rag_pipeline/notebooks/week_3/2_shopping_assistant_gemini.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Check for old code
        if "# THIS OLD CODE HAS BEEN COMMENTED OUT" in source:
            # Change it to mention OLD CODE explicitly
            new_source = ["# --- OLD CODE (COMMENTED OUT) ---\n"]
            for line in cell["source"]:
                if not line.startswith("# THIS OLD CODE"):
                    new_source.append(line)
            cell["source"] = new_source
            print("Updated old code label.")
            
        # Check for new code at the bottom (has class State and QueryExpandResponse)
        if "# 1. Redefine State and Response Schema" in source:
            new_source = ["# --- NEW CODE (UPDATED FIX) ---\n"]
            for line in cell["source"]:
                new_source.append(line)
            cell["source"] = new_source
            print("Updated new code label.")

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Finished updating labels.")
