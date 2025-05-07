from together import Together
import json
import os
import time
import logging
import argparse
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


def setup(batch_number, model_name):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    load_dotenv()
    client = Together()

    file_name = f"mcp_batches/batch_{batch_number}.json"
    with open(file_name, "r", encoding='utf-8') as f:
        data = json.load(f)

    # Predefined categories
    purpose_categories = [
        "Web Browser (search)",
        "Access to Database or Data",
        "Workflow Planning and Productivity",
        "Automation",
        "Analytics",
        "Art/Creativity",
        "Local Work (terminal, files, code)",
        # "Tooling",
        "Other (if unsure)"
    ]

    action_categories = [
        "Search/Query",
        "Get/Retrieve/List",
        "Create/add",
        "Delete/remove",
        "Update",
        "Run Process",
    ]

    return data, client, purpose_categories, action_categories, file_name

def mcp_purpose_prompt(mcp_name, description, purpose_categories):
    return f"""
You are a semantic classifier.
Your job is to classify an MCP server into one of the following Purpose categories:
{chr(10).join(f"- {cat}" for cat in purpose_categories)}

MCP Server Name: {mcp_name}
Description: {description}

Which Purpose category does this server fall into? Just return the category.
"""

def tool_action_prompt(tool_id, description, action_categories):
    return f"""
You are a semantic classifier.
Your job is to classify a tool into one of the following Action categories:
{chr(10).join(f"- {cat}" for cat in action_categories)}

Tool Name: {tool_id}
Description: {description}

Which Action category does this tool fall into? Just return the category.
"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=10))
def generate_response(client, prompt):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Classify MCPs and Tools using Gemini.")
    parser.add_argument("batch_number", type=int, help="Batch number to process (e.g., 1 for batch_1.json)")
    parser.add_argument("model_name", type=str, help="Gemini model to use (e.g., gemini-pro)")
    args = parser.parse_args()

    # Load data and model
    data, client, purpose_categories, action_categories, file_name = setup(args.batch_number, args.model_name)
    
    results = []
    MAX_REQUESTS = 1500
    request_count = 0

    try:
        for mcp in tqdm(data, desc="Classifying MCP Servers"):
            if request_count >= MAX_REQUESTS:
                logging.warning("Reached request limit. Saving partial results...")
                break

            mcp_name = mcp["MCP_name"]
            mcp_desc = mcp["MCP_description"]
            logging.info(f"==Classifying MCP: {mcp_name}")

            try:
                prompt = mcp_purpose_prompt(mcp_name, mcp_desc, purpose_categories)
                mcp_purpose = generate_response(client, prompt)
                request_count += 1
                logging.info(f"→ Purpose: {mcp_purpose}")
            except Exception as e:
                logging.error(f"Error during MCP classification: {e}")
                break

            time.sleep(1)

            tool_results = []
            for tool in mcp["tools"]:
                if request_count >= MAX_REQUESTS:
                    logging.warning("Reached request limit during tool processing. Saving partial results...")
                    break

                tool_id = tool["id"]
                tool_desc = tool["description"]
                logging.info(f"  Classifying Tool: {tool_id}")

                try:
                    tool_prompt = tool_action_prompt(tool_id, tool_desc, action_categories)
                    action_cat = generate_response(client, tool_prompt)
                    request_count += 1
                    logging.info(f"  → Action: {action_cat}")
                except Exception as e:
                    logging.error("Quota or rate limit exceeded. Exiting early.")
                    break

                time.sleep(1)

                tool_results.append({
                    "tool_id": tool_id,
                    "tool_description": tool_desc,
                    "action_category": action_cat
                })

            logging.info("Request count: %d", request_count)

            results.append({
                "mcp_name": mcp_name,
                "mcp_description": mcp_desc,
                "purpose_category": mcp_purpose,
                "tools": tool_results
            })

    except KeyboardInterrupt:
        logging.warning("Process interrupted by user. Saving partial results...")

    finally: 
        output_file = f"classified_batch_{args.batch_number}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Classification complete. {request_count} requests used. Results saved to {output_file}.")

if __name__ == "__main__":
    main()
