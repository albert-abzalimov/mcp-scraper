import pandas as pd
import matplotlib.pyplot as plt

# Load and filter data
df = pd.read_csv("final_classified_mcp_servers.csv")
df = df[["published_date", "Category"]]
df = df[df["Category"] != "Unclassified"]

# Convert to datetime and extract week
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
df['week'] = df['published_date'].dt.to_period('W')

# Count servers per category per week
df = df.groupby(["Category", "week"]).size().reset_index(name='count')

# Pivot for plotting
pivot_df = df.pivot(index='week', columns='Category', values='count').fillna(0)
pivot_df.index = pivot_df.index.to_timestamp()

# Plot and capture the Axes object
ax = pivot_df.plot(marker='o', figsize=(14, 6))

# Customize plot
ax.set_title("Number of New MCP Servers by Category per Week")
ax.set_xlabel("Week")
ax.set_ylabel("Number of New MCP Servers")
ax.grid(True)

# Add legend using the Axes object
ax.legend(title="Category", loc='upper left')

plt.tight_layout()
plt.savefig("analysis/plots/Categories_by_Week.png", bbox_inches='tight')
plt.show()