import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

GLAMA_SOURCE = "https://glama.ai/mcp/servers"

def normalize_url(url):
    """Normalize URL by removing trailing backslash if present."""
    if url and url.endswith('\\'):
        return url[:-1]
    return url

def remove_duplicates_by_repo_or_context(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        if "mcp_servers" not in data:
            logging.error("No 'mcp_servers' field found in the input file.")
            return

        seen_keys = set()
        seen_contexts = set()
        seen_glama_keys = set()  # Track GLAMA entries separately
        unique_servers = []

        # First process GLAMA sources to ensure they're preserved
        for server in data["mcp_servers"]:
            source = server.get("source")
            
            if source == GLAMA_SOURCE:
                context = server.get("context", "").strip()
                
                # Create a deduplication key for GLAMA sources
                repo = server.get("source_repo")
                server_url = server.get("url")
                glama_key = normalize_url(server_url) if server_url else None

                print(glama_key)
                
                # Only add if we haven't seen this GLAMA entry before
                if glama_key not in seen_glama_keys:
                    seen_glama_keys.add(glama_key)
                    seen_keys.add(glama_key)  # Also mark as seen in the main set
                    if context:
                        seen_contexts.add(context)
                    unique_servers.append(server)
                    print("Added unique GLAMA source: ", glama_key)
                else:
                    print("Duplicate GLAMA source found, skipping: ", glama_key)

        # Then process non-GLAMA sources
        for server in data["mcp_servers"]:
            source = server.get("source")
            
            # Skip GLAMA sources as we've already processed them
            if source == GLAMA_SOURCE:
                continue
                
            context = server.get("context", "").strip()

            # Deduplication key based on source_repo, or fallback to url
            repo = server.get("source_repo")
            if repo:
                dedup_key = (
                    repo.get("platform"),
                    repo.get("owner"),
                    repo.get("repo"),
                    normalize_url(server.get("url")),
                )
            else:
                dedup_key = (normalize_url(server.get("url")),)

            # Additional deduplication by context
            if dedup_key not in seen_keys and context not in seen_contexts:
                seen_keys.add(dedup_key)
                if context:
                    seen_contexts.add(context)
                unique_servers.append(server)
            else: 
                logging.info(f"Duplicate found, skipping: {dedup_key}")

        logging.info(f"Original server count: {len(data['mcp_servers'])}")
        logging.info(f"Unique server count: {len(unique_servers)}")

        data["mcp_servers"] = unique_servers
        data["total_found"] = len(unique_servers)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"Cleaned data saved to {output_file}")

    except Exception as e:
        logging.error(f"Error while removing duplicates: {e}")

if __name__ == "__main__":
    remove_duplicates_by_repo_or_context("mcp_servers_about.json", "mcp_servers_deduped.json")