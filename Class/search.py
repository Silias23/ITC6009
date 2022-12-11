import praw
import nltk
import pandas as pd
import re
from typing import List, Tuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Define a function that takes a string and returns a list of words
def tokenize(text: str) -> List[str]:
    # Use regular expressions to remove punctuation and split the text into a list of words
    words = re.findall(r'\b\w+\b', text)
    return words

# Define a function that takes a list of words and returns a list of tuples containing each word and its sentiment score
def get_sentiments(words: List[str]) -> List[Tuple[str, int]]:
    # Create a SentimentIntensityAnalyzer object
    analyzer = SentimentIntensityAnalyzer()

    # Initialize an empty list of word/sentiment score tuples
    word_sentiments = []

    # For each word in the input list, look up its sentiment score and add it to the list
    for word in words:
        # Use the SentimentIntensityAnalyzer to get the sentiment score for the word
        sentiment = analyzer.polarity_scores(word)['compound']
        word_sentiments.append((word, sentiment))

    return word_sentiments

# Define a function that takes a list of posts and a search query, and returns a list of posts that have a similar sentiment to the query
def get_similar_posts(posts: List[str], query: str) -> List[str]:
    # Tokenize the search query
    query_words = tokenize(query)

    # Get the sentiment scores for the query words
    query_sentiments = get_sentiments(query_words)

    # Compute the average sentiment score of the query
    query_sentiment = sum(score for word, score in query_sentiments) / len(query_sentiments)

    # Initialize an empty list of similar posts
    similar_posts = []

    # For each post, tokenize the post and compute its sentiment score
    for post in posts:
        post_words = tokenize(post)
        post_sentiments = get_sentiments(post_words)

        # Compute the average sentiment score of the post
        post_sentiment = sum(score for word, score in post_sentiments) / len(post_sentiments)

        # If the post has a similar sentiment to the query, add it to the list of similar posts
        if abs(post_sentiment - query_sentiment) < 0.5:
            similar_posts.append(post)

    return similar_posts

def search_engine_tfidf(documents, query):
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the documents
    vectorizer.fit(documents)

    # Transform the documents into tf-idf vectors
    vectors = vectorizer.transform(documents)

    # Transform the query into a tf-idf vector
    query_vector = vectorizer.transform([query])

    # Compute the dot product between the query vector and each document vector
    # This will return a list of scores, one for each document
    scores = vectors.dot(query_vector.transpose()).toarray()

    # Create a Pandas DataFrame with the documents and their corresponding scores
    data = {'Document': documents}
    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame(scores, columns = ['Score'])
    df = pd.concat([df1, df2], axis=1)

    # Add a new column to the DataFrame with the sentiment of each document
    sentiments = [sid.polarity_scores(doc)['compound'] for doc in documents]
    df['Sentiment'] = sentiments
    
    # Filter the DataFrame to only include documents with a similar sentiment to the query
    similar_posts = get_similar_posts(df['Document'], query)
    df = df[df['Document'].isin(similar_posts)]

    # Sort the DataFrame by the score in descending order and return the top documents
    return df.sort_values('Score', ascending=False).head(3)['Document'].to_list()


# Define a function that takes in a list of documents and a query,
# and returns the documents that are most relevant to the query using word2vec
def search_engine_word2vec(documents, query):
    # Create a TfidfVectorizer object with the word2vec algorithm
    vectorizer = TfidfVectorizer(use_idf=False, norm=None, smooth_idf=False,
                                 sublinear_tf=False, analyzer='word',
                                 stop_words=None, ngram_range=(1,1),
                                 max_df=1.0, min_df=1, vocabulary=None)

    # Fit the vectorizer on the documents
    vectorizer.fit(documents)

    # Transform the documents into word2vec vectors
    vectors = vectorizer.transform(documents)

    # Transform the query into a word2vec vector
    query_vector = vectorizer.transform([query])

    # Compute the cosine similarity between the query vector and each document vector
    # This will return a list of scores, one for each document
    scores = cosine_similarity(query_vector, vectors).flatten()

    # Create a Pandas DataFrame with the documents and their corresponding scores
    data = {'Document': documents, 'Score': scores}
    df = pd.DataFrame(data)

    # Sort the DataFrame by the score in descending order and return the top documents
    return df.sort_values('Score', ascending=False).head(3)['Document'].to_list()




# authenticate with the Reddit API
reddit = praw.Reddit(client_id='ZUHaoF2lWuT-XhQJM4PORg',
                     client_secret='rp2lOgUdUWWmzj9XhIqKjncOarh4Zg',
                     user_agent='ITC6010')

# get the F1 subreddit
f1_subreddit = reddit.subreddit('formula1')

# collect the titles and bodies of the latest 1000 posts in the subreddit
posts = [post for post in f1_subreddit.hot(limit=100)]
titles = [post.title for post in posts]
bodies = [post.selftext for post in posts]

Mposts = []

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Iterate over the index of the lists
for i in range(len(titles)):
    # Get the item at the current index in each list
    item1 = titles[i]
    item2 = bodies[i]

    # Merge the items into a tuple and append them to the merged list
    Mposts.append(item1 + item2)



# preprocess the titles and bodies by tokenizing and removing stop words
stop_words = set(stopwords.words('english'))
titles_processed = []
bodies_processed = []
# for title in titles:
#     tokens = word_tokenize(title)
#     tokens_filtered = [token for token in tokens if token not in stop_words]
#     titles_processed.append(' '.join(tokens_filtered))
# for body in bodies:
#     tokens = word_tokenize(body)
#     tokens_filtered = [token for token in tokens if token not in stop_words]
#     bodies_processed.append(' '.join(tokens_filtered))
for post in Mposts:
    tokens = word_tokenize(post)
    tokens_filtered = [token for token in tokens if token not in stop_words]
    titles_processed.append(' '.join(tokens_filtered))

query = "worse race ever"

# Search for the query in the documents
results1 = search_engine_tfidf(titles_processed, query)
results2 = search_engine_word2vec(titles_processed, query)


# Print the results
for result in results1:
    print(result)

for result in results1:
    print(result)


    