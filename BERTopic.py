from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import json
import nltk
import re

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Define stopwords set
stop_words = set(stopwords.words('english'))

# Define a text preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# 1. Load and flatten your JSON
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

# Remove or flag descriptions that are too short after preprocessing

# 2. Preprocess text fields and combine for modeling
df['clean_tool_description'] = df['tool_description'].fillna("").apply(preprocess)

df["clean_length"] = df["clean_tool_description"].apply(lambda x: len(x.split()))
df = df[df["clean_length"] >= 5]  # Only keep rows with 5+ meaningful words
texts = (df["tool_id"] + ": " + df["clean_tool_description"]).tolist()

# 3. Load sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Define custom vectorizer with stopwords and bigrams
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)

# 5. Create BERTopic model with custom vectorizer
topic_model = BERTopic(embedding_model=embedding_model, vectorizer_model=vectorizer_model, top_n_words=10, min_topic_size = 5)

# 6. Fit model and extract topics
topics, probs = topic_model.fit_transform(texts)

# Optional: Reduce number of topics
reduced_topic_model = topic_model.reduce_topics(texts, nr_topics=25)
reduced_topics, reduced_probs = reduced_topic_model.transform(texts)

# Add topic assignments to DataFrame
df["topic"] = topics

# 7. Show topic info
topic_info = topic_model.get_topic_info()
print(topic_info.head())

# 8. Visualize topics
fig = topic_model.visualize_topics()
fig.show()
fig.write_html("topic_visualization.html")

# 9. Save reduced topic visualization (optional)
reduced_fig = reduced_topic_model.visualize_topics()
reduced_fig.write_html("reduced_topic_visualization.html")

# 10. Print top words for each topic
for topic_num in topic_model.get_topics():
    print(f"\n🔍 Topic {topic_num}: ", topic_model.get_topic(topic_num))
