import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Sample dataset for TV show reviews
data = {
    "TV Show": [
        "Show A", "Show B", "Show C", "Show A", "Show B",
        "Show C", "Show A", "Show B", "Show C", "Show A"
    ],
    "Rating": [8, 7, 9, 6, 8, 7, 7, 6, 9, 8],
    "Review": [
        "Amazing show with a gripping plot!",
        "Good show but the pacing is slow.",
        "Absolutely loved the characters!",
        "The storyline is predictable.",
        "Solid performance by the cast.",
        "Great cinematography and visuals.",
        "Decent show, but lacks originality.",
        "Not worth the hype.",
        "Fantastic series, highly recommend!",
        "Enjoyable but could be better."
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Function to analyze sentiment
def analyze_sentiment(review):
    blob = TextBlob(review)
    return blob.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

# Apply sentiment analysis
df['Sentiment Score'] = df['Review'].apply(analyze_sentiment)

# Categorize sentiment
def categorize_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment Category'] = df['Sentiment Score'].apply(categorize_sentiment)

# Print the DataFrame
print(df)

# Calculate average rating and sentiment per TV show
popularity = df.groupby("TV Show").agg(
    Avg_Rating=('Rating', 'mean'),
    Avg_Sentiment=('Sentiment Score', 'mean'),
    Positive_Reviews=('Sentiment Category', lambda x: (x == 'Positive').sum()),
    Total_Reviews=('Review', 'count')
).reset_index()

# Print popularity analysis
print("\nPopularity Analysis:")
print(popularity)

# Visualization
plt.figure(figsize=(10, 6))

# Average rating barplot
plt.subplot(1, 2, 1)
sns.barplot(x="TV Show", y="Avg_Rating", data=popularity, palette="viridis")
plt.title("Average Ratings by TV Show")
plt.ylabel("Average Rating")
plt.ylim(0, 10)

# Sentiment score barplot
plt.subplot(1, 2, 2)
sns.barplot(x="TV Show", y="Avg_Sentiment", data=popularity, palette="coolwarm")
plt.title("Average Sentiment Score by TV Show")
plt.ylabel("Sentiment Score")
plt.ylim(-1, 1)

plt.tight_layout()
plt.show()
