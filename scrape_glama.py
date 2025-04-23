import asyncio
import json
from playwright.async_api import async_playwright

async def scrape_glama_servers():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://glama.ai/mcp/servers", timeout=60000)

        # Click "Load More" until all servers are loaded
        while True:
            try:
                load_more = await page.wait_for_selector("button:has-text('Load More')", timeout=5000)
                await load_more.click()
                await page.wait_for_timeout(2000)  # Wait for new content to load
            except:
                break  # No more "Load More" button

        # Extract server cards
        server_cards = await page.query_selector_all("div[data-testid='mcp-server-card']")
        servers = []

        for card in server_cards:
            try:
                name = await card.query_selector_eval("h2", "el => el.textContent.trim()")
                author = await card.query_selector_eval("a[href*='/users/']", "el => el.textContent.trim()")
                context = await card.query_selector_eval("div:has-text('Context') + div", "el => el.textContent.trim()")
                url = await card.query_selector_eval("a[href*='/mcp/servers/']", "el => el.href")

                # Navigate to server detail page to get 'about' info
                detail_page = await browser.new_page()
                await detail_page.goto(url, timeout=60000)
                try:
                    about = await detail_page.query_selector_eval("p", "el => el.textContent.trim()")
                except:
                    about = "No description available."
                await detail_page.close()

                servers.append({
                    "name": name,
                    "author": author,
                    "about": about,
                    "context": context,
                    "url": url
                })
            except Exception as e:
                print(f"Error processing a server card: {e}")
                continue

        # Save to JSON
        with open("mcp_servers.json", "w", encoding="utf-8") as f:
            json.dump(servers, f, indent=2, ensure_ascii=False)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_glama_servers())
