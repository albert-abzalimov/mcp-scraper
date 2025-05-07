from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
import re
import os

stop_words = set(stopwords.words('english'))

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Load and flatten JSON
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

# Preprocess tool descriptions
df["clean_tool_description"] = df["tool_description"].fillna("").apply(preprocess)
df["clean_length"] = df["clean_tool_description"].apply(lambda x: len(x.split()))
df = df[df["clean_length"] >= 5]

# # Combine tool_id and description for clarity
# texts = (df["tool_id"] + ": " + df["clean_tool_description"]).tolist()

# # Vectorize and create topic model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
# topic_model = BERTopic(embedding_model=embedding_model, vectorizer_model=vectorizer_model)
# topics, probs = topic_model.fit_transform(texts)
# topic_model.reduce_topics(texts, nr_topics=15)

# df["topic"] = topic_model.transform(texts)[0]

# # Optionally, map custom categories based on topic keywords
# # You may need to review topic_model.get_topic_info() and assign custom labels manually

# topic_info = topic_model.get_topic_info()
# print(topic_info)

# # Simple keyword-based mapping (can be improved)
# custom_labels = {
#     "get/search/query": ["query", "search", "get", "retrieve", "find"],
#     "list": ["list", "display", "show", "file", "view", "directory", "information", "content", "path"],
#     "create": ["create", "add", "generate", "register"],
#     "run": ["run", "execute", "start", "launch"],
#     "update": ["update", "edit", "modify", "change"],
#     "delete": ["delete", "remove", "discard", "destroy"],
#     "images": ["images", "figma", "image", "element", "page", "browser"],
#     "github": ["github", "repository", "code", "commit", "branch", "pull", "test"],
#     "other": ["translate", "reset", "knowledge", "audio", "intelligence"],
# }

# # Assign custom label to each topic based on representative words
# def map_custom_topic(topic_words):
#     for label, keywords in custom_labels.items():
#         for word in keywords:
#             if word in topic_words:
#                 return label
#     return "other"

# # Map topics to custom labels
# topic_keywords = {topic_num: topic_model.get_topic(topic_num) for topic_num in df["topic"].unique()}
# topic_label_map = {
#     topic_num: map_custom_topic(" ".join([word for word, _ in words]))
#     for topic_num, words in topic_keywords.items()
# }
# df["custom_topic"] = df["topic"].map(topic_label_map)

# # Print counts
# print(df["custom_topic"].value_counts())

# # Visualize topics
# fig = topic_model.visualize_topics()
# fig.write_html("topic_visualization.html")

# print("Visualization saved as 'topic_visualization.html'")



custom_keywords = {
    "get": ["get", "retrieve", "find", "returns"],
    "search": ["query", "search", "view", "show", "display"],
    "list": ["list", "file", "directory", "information", "content", "path", "knowledge"],
    "create": ["create", "add", "generate", "register", "new"],
    "run": ["run", "execute", "start", "launch", "task", "order"],
    "update": ["update", "edit", "modify", "change", "reset"],
    "delete": ["delete", "remove", "discard", "destroy", "cancel", "terminate"],
    "images": ["images", "figma", "image", "element", "page", "browser", "graph"],
    "github": ["github", "repository", "code", "commit", "branch", "pull", "test", "environment"],
    "communication": ["communication", "chat", "message", "messages", "translate", "audio"],
}

# Assign initial category based on keyword presence
def assign_initial_label(text):
    for label, words in custom_keywords.items():
        if any(word in text.split() for word in words):
            return label
    return "other"

df["initial_label"] = df["clean_tool_description"].apply(assign_initial_label)

# Optionally remove "other" if you want to train only on labeled data
# df = df[df["initial_label"] != "other"]

# Get texts and labels
docs = df["clean_tool_description"].tolist()
labels = df["initial_label"].tolist()

# Train BERTopic using guided labels (semi-supervised)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")

topic_model = BERTopic(embedding_model=embedding_model, vectorizer_model=vectorizer_model)
# topic_model.fit(docs, y=labels)

# Convert string labels to numeric
label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
numeric_labels = [label_mapping[label] for label in labels]

# Train BERTopic
topic_model.fit(docs, y=numeric_labels)
topic_model.reduce_topics(docs, nr_topics=15)
topics, probs = topic_model.transform(docs)
df["initial_label"] = [inverse_label_mapping[num] for num in numeric_labels]

# Save to DataFrame
df["topic_id"] = topics
df["refined_topic"] = [
    topic_model.get_topic_info().iloc[t]["Name"] if t != -1 else "Other"
    for t in topics
]

# Print topic distribution
print("Topic Distribution:\n")
print(df["refined_topic"].value_counts())

# Save visualization
fig = topic_model.visualize_topics()
fig.write_html("topic_visualization.html")
print("\nInteractive topic visualization saved as 'topic_visualization.html'.")
