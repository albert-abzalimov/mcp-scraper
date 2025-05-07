

import json
import os

def batch_mcp_servers(mcp_data, max_tools_per_batch=1000, output_dir="mcp_batches"):
    os.makedirs(output_dir, exist_ok=True)
    
    batches = []
    current_batch = []
    current_tool_count = 0

    for mcp in mcp_data:
        tool_count = len(mcp["tools"])

        # If adding this MCP would exceed the limit, save the current batch and start a new one
        if current_tool_count + tool_count > max_tools_per_batch:
            batches.append(current_batch)
            current_batch = []
            current_tool_count = 0

        current_batch.append(mcp)
        current_tool_count += tool_count

    # Add the last batch
    if current_batch:
        batches.append(current_batch)

    # Save each batch to a file
    for i, batch in enumerate(batches, 1):
        with open(os.path.join(output_dir, f"batch_{i}.json"), "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

    print(f"{len(batches)} batches saved in '{output_dir}/'")


with open("translated_details copy.json", "r", encoding='utf-8') as f:
    data = json.load(f)

batch_mcp_servers(data, max_tools_per_batch=1000, output_dir="mcp_batches")