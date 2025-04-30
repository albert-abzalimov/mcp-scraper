import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("categorized_mcp_tools.csv")
df = df[['MCP_name', 'week']]
df = df.groupby("MCP_name").first()
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


