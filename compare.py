import json
import umap.umap_ as umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# 1. Load and flatten JSON
with open('translated_details.json', encoding='utf-8') as f:
    data = json.load(f)

records = []
for mcp in data:
    for tool in mcp.get("tools", []):
        records.append({
            "MCP_name": mcp.get("MCP_name"),
            "MCP_description": mcp.get("MCP_description", ""),
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

# Baseline text: tool ID + description
baseline_texts = (df['tool_id'] + ": " + df['tool_description']).tolist()
baseline_embeddings = model.encode(baseline_texts, show_progress_bar=True)

# Extended text: add MCP description
extended_texts = (df['tool_id'] + ": " + df['tool_description'] + " | " + df['MCP_description']).tolist()
extended_embeddings = model.encode(extended_texts, show_progress_bar=True)

# Clustering
num_clusters = 10
kmeans_base = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_ext = KMeans(n_clusters=num_clusters, random_state=42)

df['cluster_baseline'] = kmeans_base.fit_predict(baseline_embeddings)
df['cluster_extended'] = kmeans_ext.fit_predict(extended_embeddings)

# Compare clustering results
ari = adjusted_rand_score(df['cluster_baseline'], df['cluster_extended'])
silhouette_baseline = silhouette_score(baseline_embeddings, df['cluster_baseline'])
silhouette_extended = silhouette_score(extended_embeddings, df['cluster_extended'])

print(f"\n📊 Adjusted Rand Index (Baseline vs Extended): {ari:.4f}")
print(f"🧩 Silhouette Score - Baseline: {silhouette_baseline:.4f}")
print(f"🧩 Silhouette Score - Extended: {silhouette_extended:.4f}")

# UMAP visualization for extended embeddings
umap_model = umap.UMAP(n_components=2, random_state=42)
reduced = umap_model.fit_transform(extended_embeddings)
df['x'] = reduced[:, 0]
df['y'] = reduced[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x', y='y', hue='cluster_extended', palette='tab10')
plt.title("Clusters with MCP_description Included")
plt.savefig("extended_cluster_plot.png")
plt.show()

# Usage & Deployment Analysis
usage_by_cluster = df.groupby("cluster_extended")["MCP_usage"].sum().sort_values(ascending=False)
print("\n🧮 Usage by Cluster (Extended):\n", usage_by_cluster)

deployment_counts = df["MCP_deployment_status"].value_counts()
print("\n📡 Deployment Status Distribution:\n", deployment_counts)

df.to_csv("categorized_mcp_tools_extended.csv", index=False)
print("\n✅ CSV saved as 'categorized_mcp_tools_extended.csv'")
