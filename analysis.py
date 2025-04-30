import json
import umap.umap_ as umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

umap_model = umap.UMAP(n_components=2, random_state=42)
# 1. Load and flatten JSONtryhjukil;'
 
with open('translated_details.json', encoding='utf-8') as f:
    data = json.load(f)

records = []
for mcp in data:
    for tool in mcp.get("tools", []):
        records.append({
            "MCP_name": mcp.get("MCP_name"),
            "MCP_description": mcp.get("MCP_description"),
            "MCP_usage": mcp.get("MCP_usage"),
            "MCP_deployment_status": mcp.get("MCP_deployment_status"),
            "published_date": mcp.get("published_date"),
            "publisher": mcp.get("publisher"),
            "tool_id": tool.get("id"),
            "tool_description": tool.get("description")
        })
    

df = pd.DataFrame(records)
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
df['year'] = df['published_date'].dt.year
df['week'] = df['published_date'].dt.to_period('W')
df['MCP_usage'] = pd.to_numeric(df['MCP_usage'], errors='coerce').fillna(0)

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = ("Tool name: " + df['tool_id'] + ". Description: " + df['tool_description']).tolist()
embeddings = model.encode(texts, show_progress_bar=True)


num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)

# pca = pca(n_components=2)
# reduced = pca.fit_transform(embeddings)

umap_model = umap.UMAP(n_components=2, random_state=42)
reduced = umap_model.fit_transform(embeddings)

df['x'] = reduced[:, 0]
df['y'] = reduced[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='tab10')
plt.title("Tool Descriptions Clustered by Semantic Meaning")
plt.savefig("cluster_plot.png")
plt.show()

# 5. Analyze usage by cluster
usage_by_cluster = df.groupby("cluster")["MCP_usage"].sum().sort_values(ascending=False)
print("\n🧮 Usage by Cluster:\n", usage_by_cluster)

# 6. Deployment status counts
deployment_counts = df["MCP_deployment_status"].value_counts()
print("\n📡 Deployment Status Distribution:\n", deployment_counts)

# # 7. Trend by published month
# trend = df.groupby('month')["MCP_usage"].sum()

# plt.figure(figsize=(12, 5))
# trend.plot(marker='o')
# plt.title("MCP Usage Over Time")
# plt.xlabel("Month")
# plt.ylabel("Total Usage")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("usage_trend.png")
# plt.show()

# 8. Save categorized CSV
df.to_csv("categorized_mcp_tools.csv", index=False)
print("\n Categorized CSV saved as 'categorized_mcp_tools.csv'")