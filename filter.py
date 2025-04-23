import json
import re
from urllib.parse import urlparse

def normalize_url(url):
    parsed = urlparse(url)
    base_url = parsed.scheme + "://" + parsed.netloc + parsed.path.rstrip('/\\')
    return base_url

def is_obviously_invalid(url):
    lower = url.lower()
    return (
        "discord" in lower or
        "reddit" in lower or
        "twitter" in lower or
        "github.com/.*/issues" in lower or
        "youtube" in lower or
        "blog" in lower or
        "docs" in lower or
        "license" in lower or
        "image" in lower or 
        "png" in lower or
        "screenshot" in lower
    )

def filter_duplicates(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    seen_urls = set()
    seen_domains = set()
    filtered_servers = []

    for entry in data.get("mcp_servers", []):
        norm_url = normalize_url(entry["url"])
        domain = urlparse(norm_url).netloc.lower()

        if (
            norm_url not in seen_urls and
            not is_obviously_invalid(norm_url) and
            (domain == "github.com" or domain not in seen_domains)
        ):
            seen_urls.add(norm_url)
            if domain != "github.com":
                seen_domains.add(domain)
            entry["url"] = norm_url
            filtered_servers.append(entry)

    output = {
        "mcp_servers": filtered_servers,
        "total_filtered": len(filtered_servers)
    }

    with open(output_json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Filtered from {len(data['mcp_servers'])} to {len(filtered_servers)} entries.")

# Example usage:
filter_duplicates("mcp_servers.json", "filtered_mcp_servers.json")
