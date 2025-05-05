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
        "published_date": mcp.get("published_date"),
        "publisher": mcp.get("publisher")
    })
    index += 1
    
df = pd.DataFrame(records)
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
df['week'] = df['published_date'].dt.to_period('W')
df = df[['MCP_index', 'week', "publisher"]]
df = df.groupby("publisher").first()
trend = df['week'].value_counts().sort_index()

plt.figure(figsize=(12, 5))
trend.plot(marker='o')
plt.title("New MCP Developers by Week")
plt.xlabel("Week")
plt.ylabel("Number of New Developers")
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis/plots/New_MCP_Developers_by_Week.png")
plt.show()
print(trend)

