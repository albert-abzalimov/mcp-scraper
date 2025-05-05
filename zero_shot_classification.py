import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Load tool descriptions
df = pd.read_csv("categorized_mcp_tools.csv")
df = df.dropna(subset=["tool_description"])

# Define candidate categories (edit these to match your domain)
candidate_labels = [
    "search",
    "file management",
    "automation",
    "data analysis",
    "AI agent",
    "problem solving",
    "communication",
    "security",
    "system monitoring"
]

# Load the zero-shot classification model
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

# Run classification
categories = []
scores = []

print("Classifying tool descriptions...")
for desc in tqdm(df["tool_description"].tolist()):
    result = classifier(desc, candidate_labels)
    categories.append(result["labels"][0])  # Top prediction
    scores.append(result["scores"][0])      # Confidence

# Store results
df["predicted_category"] = categories
df["confidence"] = scores

# Save to CSV
df.to_csv("mcp_tools_with_categories.csv", index=False)
print("✅ Saved classified tools to 'mcp_tools_with_categories.csv'")

# Plot the category distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y="predicted_category", order=df["predicted_category"].value_counts().index)
plt.title("Tool Categories (Zero-Shot Classified)")
plt.tight_layout()
plt.savefig("tool_categories_count.png")
print("📊 Saved category plot to 'tool_categories_count.png'")
