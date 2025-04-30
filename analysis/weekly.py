import pandas as pd
import json
import matplotlib.pyplot as plt

with open('translated_details.json', encoding='utf-8') as f:
    data = json.load(f)

records = []
index = 0
for mcp in data:
    records.append({
        "MCP_index": index,
        "published_date": mcp.get("published_date")
    })
    index += 1
    
df = pd.DataFrame(records)
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
df['week'] = df['published_date'].dt.to_period('W')
df = df[['MCP_index', 'week']]
df = df.groupby("MCP_index").first()
trend = df['week'].value_counts().sort_index()
plt.figure(figsize=(12, 5))
trend.plot(marker='o')
plt.title("MCP Usage Over Time")
plt.xlabel("Month")
plt.ylabel("Total Usage")
plt.grid(True)
plt.tight_layout()
plt.savefig("usage_trend.png")
plt.show()
print(trend)

