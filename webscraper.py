import requests
from bs4 import BeautifulSoup
import json
import re
import time
import random
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_scraper.log"),
        logging.StreamHandler()
    ]
)

class MCPServerScraper:
    def __init__(self, starting_urls=None, output_file="mcp_servers_about.json", user_agent=None):
        self.starting_urls = starting_urls or [
            "https://github.com/modelcontextprotocol/servers",
            "https://github.com/topics/model-context-protocol",
            "https://github.com/punkpeye/awesome-mcp-servers",
            "https://glama.ai/mcp/servers"
        ]
        self.output_file = output_file
        self.visited_urls = set()
        self.mcp_servers = []
        self.mcp_keywords = [
            "mcp server",
            "model context protocol server"
        ]
        self.blacklisted_patterns = [
            r'example\.com',
            r'localhost',
            r'127\.0\.0\.1',
            r'\.test$',
            r'/documentation/',
            r'/blog/',
            r'/examples/',
            r'twitter\.com',
            r'x\.com',
            r'reddit\.com',
            # Git repository specific patterns to avoid
            r'/wiki/',
            r'/pulse/',
            r'/graphs/',
            r'/network/',
            r'/forks/',
            r'/stars/',
            r'/watchers/',
            r'/labels/',
            r'/milestones/',
            r'/compare/',
            r'/actions/',
            r'/projects/',
            r'/settings/',
            r'/insights/',
            r'/security/',
            r'/commit/',  # Individual commits usually aren't relevant
            r'/releases/',
            r'/tags/',
            r'/packages/',
            r'/marketplace/',
            r'/orgs/',
            r'/teams/',
            r'/community/',
            r'/discussions/',
            r'/notifications/',
            r'/about/',
            r'/pricing/',
            r'/organizations/',
            r'/site/',
            r'/search/',
            r'/trending/',
            r'/collections/',
            r'/sponsors/',
            r'/explore/',
            r'/discord',
            r'/slack/',
            r'/telegram/',
            r'/matrix/',
            r'/topics/'
            r'/irc/',
            r'/gitter/',
            r'/chat/',
            r'/forum/',
            r'/community/',
            r'/contact/',
            r'/support/',
            r'/help/',
            r'/events/',
            r'/meetups/',
            r'/conferences/',
            r'/webinars/',
            r'/stream/',
            r'/live/',
            r'/social/',
            r'/contributing/',
            r'/CODE_OF_CONDUCT/',
            r'/CONTRIBUTING/',
            r'/LICENSE/',
            r'/FUNDING/',
            r'/SECURITY/',
            r'/CHANGELOG/',
            r'/authors/',
            r'/contributors/',
            r'/maintainers/',
            r'/downloads/',
            r'/calendar/',
            r'/schedule/',
            r'/donate/',
            r'/license/',
            r'/install',
        ]
        self.user_agent = user_agent or "MCPServerScraper/1.0 (+https://github.com/yourusername/mcp-scraper)"
        # Updated to include common git hosting platforms
        self.git_hosting_domains = {
            "github.com",
            "gitlab.com",
            "bitbucket.org",
            "git.sr.ht",
            "gitea.io",
            "codeberg.org",
            "source.puri.sm",  # Librem's GitLab instance
            "git.savannah.gnu.org",
            "git.kernel.org",
            "sourceforge.net"
        }

    def get_headers(self):
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def make_request(self, url):
        try:
            time.sleep(random.uniform(1.0, 3.0))
            response = requests.get(url, headers=self.get_headers(), timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    def is_valid_url(self, url):
        if not url or url in self.visited_urls:
            return False
        parsed = urlparse(url)
        if not parsed.netloc or not parsed.scheme:
            return False

        # Only proceed with URLs from git hosting domains
        if not any(parsed.netloc.endswith(domain) for domain in self.git_hosting_domains):
            return False

        # Check for repository patterns in URL
        repo_patterns = [
            r'/[^/]+/[^/]+/?$',  # Basic repo pattern: username/repo
            r'/[^/]+/[^/]+/tree/',  # Branch or directory
            r'/[^/]+/[^/]+/blob/',  # File
            r'/[^/]+/[^/]+/issues',  # Issues
            r'/[^/]+/[^/]+/pull'    # PRs
        ]

        path = parsed.path
        is_repo_url = any(re.search(pattern, path) for pattern in repo_patterns)

        if not is_repo_url:
            return False

        avoid_extensions = ['.pdf', '.zip', '.jpg', '.png', '.gif', '.mp4', '.mp3', '.xml']
        if any(url.lower().endswith(ext) for ext in avoid_extensions):
            print("url did not pass extension check", url)
            return False

        return True

    def is_trusted_domain(self, url):
        domain = urlparse(url).netloc.lower()
        return any(domain.endswith(trusted) for trusted in self.git_hosting_domains)

    def is_blacklisted(self, url):
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in self.blacklisted_patterns)

    def verify_mcp_server(self, url):
        try:
            test_url = url + "/mcp/" if not url.endswith("/mcp/") else url
            response = requests.head(test_url, headers=self.get_headers(), timeout=5)
            return response.status_code == 200
        except:
            return False

    def extract_about_section(self, html_content, url):
        """Extract the 'About' section from a GitHub repository page."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            about_text = ""

            print("URL is ", url)

            if "github.com" in url:
                about_div = soup.find('div', class_='BorderGrid-cell')
                if about_div:
                    about_p = about_div.find('p')
                    if about_p:
                        about_text = about_p.get_text(separator=' ', strip=True)
            
            # Glama-specific extraction
            elif "glama.ai" in url:
                print("GLAMMMAAAAAAA LETS FIND THAT P")
                # Always take the first <p> element
                first_p = soup.find('p')
                if first_p:
                    about_text = first_p.get_text(separator=' ', strip=True)
                    print("oh yeah about text is this: ", about_text)


            if about_text:
                return {
                    "about": about_text,
                    "source_url": url
                }
            return None
        except Exception as e:
            logging.error(f"Error extracting 'About' section from {url}: {e}")
            return None


    def extract_mcp_server_info(self, html_content, source_url):
        soup = BeautifulSoup(html_content, 'html.parser')
        found_servers = []
        url_patterns = [
            r'https?://[^\s<>"]+/mcp/server(?:/v\d+)?(?:/|$)[^\s<>"]*',
            r'https?://[^\s<>"]+/mcp(?:/v\d+)?/server(?:/|$)[^\s<>"]*',
            r'https?://mcp-server\.[^\s<>"]+\.\w{2,}(?:/|$)[^\s<>"]*',
            r'https?://mcp\.[^\s<>"]+\.\w{2,}/server(?:/|$)[^\s<>"]*',
            r'wss?://[^\s<>"]+/mcp/server(?:/v\d+)?(?:/|$)[^\s<>"]*',
            r'https?://[^\s<>"]+/model-context-protocol/server(?:/v\d+)?(?:/|$)[^\s<>"]*',
            # Still keep the general patterns but we'll filter out client mentions later
            r'https?://[^\s<>"]+/mcp(?:/v\d+)?(?:/|$)[^\s<>"]*',
            r'https?://mcp\.[^\s<>"]+\.\w{2,}(?:/|$)[^\s<>"]*',
            r'wss?://[^\s<>"]+/mcp(?:/v\d+)?(?:/|$)[^\s<>"]*',
            r'https?://[^\s<>"]+/model-context-protocol(?:/v\d+)?(?:/|$)[^\s<>"]*',
        ]
        
        for pattern in url_patterns:
            for url in re.findall(pattern, html_content):
                clean_url = re.sub(r'[.,;:\"\')}]$', '', url)
                
                # Skip URLs that specifically mention clients
                if re.search(r'client', clean_url, re.IGNORECASE):
                    continue
                    
                if (
                    not self.is_blacklisted(clean_url) and
                    clean_url not in {s["url"] for s in found_servers} and
                    "twitter.com" not in clean_url and
                    "x.com" not in clean_url and
                    "reddit.com" not in clean_url
                ):
                    confidence = "high" if self.is_trusted_domain(clean_url) else "medium"
                    found_servers.append({
                        "url": clean_url,
                        "source": source_url,
                        "confidence": confidence,
                        "needs_review": confidence != "high"
                    })

        # Filter text content more precisely for MCP servers
        for tag in soup.find_all(lambda t: t.name in ["code", "pre", "h2", "h3", "p"]):
            text = tag.get_text()
            
            # Look for server mentions but exclude client mentions
            if (re.search(r"\bmcp\s+(server|implementation)\b", text, re.IGNORECASE) or 
                re.search(r"\bmodel[\s-]context[\s-]protocol\s+(server|implementation)\b", text, re.IGNORECASE)):
                
                # Skip if it mentions client
                if re.search(r"\bclient\b", text, re.IGNORECASE):
                    continue
                    
                for a_tag in tag.find_all_next("a", href=True, limit=2):
                    url = urljoin(source_url, a_tag["href"])
                    
                    # Skip URLs that specifically mention clients
                    if re.search(r'client', url, re.IGNORECASE):
                        continue
                        
                    if (
                        not self.is_blacklisted(url) and
                        url not in {s["url"] for s in found_servers} and
                        "twitter.com" not in url and
                        "x.com" not in url and
                        "reddit.com" not in url
                    ):
                        found_servers.append({
                            "url": url,
                            "source": source_url,
                            "confidence": "medium",
                            "needs_review": True,
                            "context": text[:100]
                        })

        for server in found_servers:
            # Add repo information if available
            repo_info = self.extract_repo_info(source_url)
            if repo_info:
                server["source_repo"] = repo_info

        return found_servers

    def extract_links(self, html_content, base_url):
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        # Keywords focused on servers, not clients
        server_keywords = ["mcp server", "model context protocol server", "mcp implementation", 
                        "model context protocol implementation", "mcp-server", "mcp_server"]
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            
            # Skip client-related URLs
            if re.search(r'client', full_url, re.IGNORECASE):
                continue
                
            if self.is_valid_url(full_url):
                text = a_tag.get_text().lower()
                
                # Check if URL or text contains server-related keywords
                if (any(keyword in full_url.lower() for keyword in server_keywords) or 
                    any(keyword in text for keyword in server_keywords) or
                    # Still follow MCP links but exclude client links
                    (("mcp" in full_url.lower() or "model-context-protocol" in full_url.lower()) and 
                    "client" not in full_url.lower() and "client" not in text)):
                    links.append(full_url)
                    
        return links

    def extract_repo_info(self, url):
        """Extract repository details from a git URL."""
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split('/') if p]

            if len(path_parts) >= 2:
                owner = path_parts[0]
                repo = path_parts[1]

                return {
                    "platform": parsed.netloc,
                    "owner": owner,
                    "repo": repo,
                    "full_url": url
                }
            return None
        except:
            return None

    def crawl(self, url, depth=2):
        if depth <= 0 or url in self.visited_urls:
            return
        self.visited_urls.add(url)
        # logging.info(f"Crawling: {url} (depth={depth})")
        html_content = self.make_request(url)
        if not html_content:
            return
        
        # Extract MCP servers
        servers = self.extract_mcp_server_info(html_content, url)
        for server in servers:
            if server["url"] not in {s["url"] for s in self.mcp_servers}:
                if server["confidence"] == "high" and not self.verify_mcp_server(server["url"]):
                    server["confidence"] = "medium"
                    server["needs_review"] = True
                if (".xml" in server["url"]):
                    print("OH OH LOOK HERE!!!! XML FOUND IN SERVER URL", server["url"])
                if (".xml" not in server["url"]):
                    self.mcp_servers.append(server)
                # logging.info(f"Found MCP server: {server['url']} (confidence: {server['confidence']})")
        
        # Extract About section if this looks like a main repository page
        if re.search(r'github\.com/[^/]+/[^/]+/?$', url):
            about_data = self.extract_about_section(html_content, url)
            if about_data:
                repo_info = self.extract_repo_info(url)
                if repo_info:
                    repo_key = f"{repo_info['platform']}/{repo_info['owner']}/{repo_info['repo']}"
                    if not hasattr(self, 'about_sections'):
                        self.about_sections = {}
                    self.about_sections[repo_key] = about_data
                    logging.info(f"Extracted About section from {repo_key}")

        
        links = self.extract_links(html_content, url)
        for link in links[:10]:
            if link not in self.visited_urls:
                self.crawl(link, depth - 1)

    def save_results(self):
        try:
            result_data = {
                "mcp_servers": self.mcp_servers,
                "total_found": len(self.mcp_servers),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add About sections if available
            if hasattr(self, 'about_sections') and self.about_sections:
                result_data["about_sections"] = self.about_sections
                logging.info(f"Including {len(self.about_sections)} About sections in results")

                
            with open(self.output_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            logging.info(f"Results saved to {self.output_file}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

    def run(self, max_depth=2):
        logging.info("Starting MCP server scraper (with false-positive reduction)")
        for url in self.starting_urls:
            self.crawl(url, max_depth)
        logging.info(f"Scraping complete. Found {len(self.mcp_servers)} potential MCP servers.")
        self.save_results()
        return self.mcp_servers


if __name__ == "__main__":
    scraper = MCPServerScraper()
    results = scraper.run(max_depth=2)
    print(f"\nFound {len(results)} potential MCP servers:")
    for i, server in enumerate(results[:10], 1):
        print(f"{i}. {server['url']} (confidence: {server['confidence']})")
    if len(results) > 10:
        print(f"... and {len(results) - 10} more. See {scraper.output_file} for complete results.")
    
    print("in total we found ", len(results), "potential MCP servers")
