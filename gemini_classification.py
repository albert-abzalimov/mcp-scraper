import json
import os
import time
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


def setup():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    load_dotenv()

    # Initialize Gemini
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    with open("translated_details.json", "r", encoding='utf-8') as f:
        data = json.load(f)

    # Load JSON data
    # Predefined categories
    purpose_categories = [
        "Web Browser (search)",
        "Access to Database or Data",
        "Workflow Planning and Productivity",
        "Automation",
        "Analytics",
        "Art/Creativity",
        "Local Work (terminal, files, code)",
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

    return data, model, purpose_categories, action_categories

data, model, purpose_categories, action_categories = setup()

# Prompt templates
def mcp_purpose_prompt(mcp_name, description):
    return f"""
You are a semantic classifier.
Your job is to classify an MCP server into one of the following Purpose categories:
{chr(10).join(f"- {cat}" for cat in purpose_categories)}

MCP Server Name: {mcp_name}
Description: {description}

Which Purpose category does this server fall into? Just return the category.
"""

def tool_action_prompt(tool_id, description):
    return f"""
You are a semantic classifier.
Your job is to classify a tool into one of the following Action categories:
{chr(10).join(f"- {cat}" for cat in action_categories)}

Tool Name: {tool_id}
Description: {description}

Which Action category does this tool fall into? Just return the category.
"""

# Retry with exponential backoff on failures
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=10))
def generate_response(prompt):
    response = model.generate_content(prompt)
    return response.text.strip()

# Main processing loop with delay
results = []

MAX_REQUESTS = 1380
request_count = 0

for mcp in tqdm(data, desc="Classifying MCP Servers"):

    if request_count >= MAX_REQUESTS:  # ✅ stop if limit reached
        logging.warning("Reached request limit. Saving partial results...")
        break

    mcp_name = mcp["MCP_name"]
    mcp_desc = mcp["MCP_description"]

    logging.info(f"Classifying MCP: {mcp_name}")

    # Send out the prompt
    prompt = mcp_purpose_prompt(mcp_name, mcp_desc)
    mcp_purpose = generate_response(prompt)
    request_count += 1  # ✅ increment request count

    logging.info(f"→ Purpose: {mcp_purpose}")
    time.sleep(1)  # Rate limiting

    tool_results = []
    for tool in tqdm(mcp["tools"], desc=f"  Tools in {mcp_name}", leave=False):
        if request_count >= MAX_REQUESTS:  # ✅ stop if limit reached
            logging.warning("Reached request limit during tool processing. Saving partial results...")
            break
        tool_id = tool["id"]
        tool_desc = tool["description"]

        logging.info(f"  Classifying Tool: {tool_id}")

        tool_prompt = tool_action_prompt(tool_id, tool_desc)
        action_cat = generate_response(tool_prompt)
        request_count += 1  # ✅ increment request count

        logging.info(f"  → Action: {action_cat}")

        time.sleep(1)  # Rate limiting

        tool_results.append({
            "tool_id": tool_id,
            "action_category": action_cat
        })

    results.append({
        "mcp_name": mcp_name,
        "purpose_category": mcp_purpose,
        "tools": tool_results
    })

# Output result
with open("classified_mcp_data.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Classification complete. {request_count} requests used. Results saved to classified_mcp_data.json.")