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



def get_query():
    done = False
    print('What is your question?')
    while not done:
        try:
            prompt = input()
            if prompt and isinstance(prompt,str) and len(prompt)>0:
                done = True       
            else:
                print('this is not a valid question') 
        except:
            print('question needs to be a string')
    return prompt


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
        if abs(post_sentiment - query_sentiment) < 0.2:
            similar_posts.append(post)

    return similar_posts


def inverse_value(x):
    return 1 - x/2


def search_engine_tfidf(documents, query, c):
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    
    document = documents['tokenized'].tolist()
    # Fit the vectorizer on the documents
    vectorizer.fit(document)

    # Transform the documents into tf-idf vectors
    vectors = vectorizer.transform(document)

    # Transform the query into a tf-idf vector
    query_vector = vectorizer.transform([query])

    # Compute the dot product between the query vector and each document vector
    # This will return a list of scores, one for each document
    scores = vectors.dot(query_vector.transpose()).toarray()

    documents['total score'] = scores
    # Create a Pandas DataFrame with the documents and their corresponding scores
    # data = {'Document': documents, 'Score':scores}
    # df = pd.DataFrame(data)

    # Sort the DataFrame by the score in descending order and return the top documents
    return documents,documents.sort_values('total score', ascending=False).head(c)['id'].to_list()


def search_engine_tfidf_withSentiment(documents, query, c):
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    document = documents['tokenized'].tolist()
    # Fit the vectorizer on the documents
    vectorizer.fit(document)

    # Transform the documents into tf-idf vectors
    vectors = vectorizer.transform(document)

    # Transform the query into a tf-idf vector
    query_vector = vectorizer.transform([query])

    # Compute the dot product between the query vector and each document vector
    # This will return a list of scores, one for each document
    scores = vectors.dot(query_vector.transpose()).toarray()

    documents['Score'] = scores


    # Add a new column to the DataFrame with the sentiment of each document
    sentiments = [sid.polarity_scores(doc)['compound'] for doc in document]
    documents['Sentiment'] = sentiments


    query_sentiments = get_sentiments(query.split())

    # Compute the average sentiment score of the query
    query_sentiment = sum(score for word, score in query_sentiments) / len(query_sentiments)
    if query_sentiment != 0:
        documents['distance'] = abs(documents['Sentiment'] - query_sentiment)

        col_a = documents.loc[:, "distance"]
        # Apply the function to the "A" column using the apply() method
        col_a = col_a.apply(inverse_value)
        # Replace the "A" column with the transformed column
        documents.loc[:, "distance_inv"] = col_a
        documents['total score'] = (documents['Score']*0.8 + documents['distance_inv']*0.2)/2
    else:
        documents['total score'] = documents['Score']
        return documents,documents.sort_values('total score', ascending=False).head(c)['id'].to_list()
    # Filter the DataFrame to only include documents with a similar sentiment to the query
    #similar_posts = get_similar_posts(df['Document'], query)
    #df = df[df['Document'].isin(similar_posts)]

    # Sort the DataFrame by the score in descending order and return the top documents
    return documents,documents.sort_values('total score', ascending=False).head(c)['id'].to_list()


# Define a function that takes in a list of documents and a query,
# and returns the documents that are most relevant to the query using word2vec
def search_engine_word2vec(documents, query, c):
    # Create a TfidfVectorizer object with the word2vec algorithm
    vectorizer = TfidfVectorizer(use_idf=False, norm='l2', smooth_idf=False,
                                 sublinear_tf=False, analyzer='word',
                                 stop_words=None, ngram_range=(1,1),
                                 max_df=1.0, min_df=1, vocabulary=None)
    
    document = documents['tokenized'].tolist()
    # Fit the vectorizer on the documents
    vectorizer.fit(document)

    # Transform the documents into word2vec vectors
    vectors = vectorizer.transform(document)

    # Transform the query into a word2vec vector
    query_vector = vectorizer.transform([query])

    # Compute the cosine similarity between the query vector and each document vector
    # This will return a list of scores, one for each document
    scores = cosine_similarity(query_vector, vectors).flatten()
    documents['total score'] = scores
    # Create a Pandas DataFrame with the documents and their corresponding scores

    # Sort the DataFrame by the score in descending order and return the top documents
    return documents,documents.sort_values('total score', ascending=False).head(c)['id'].to_list()


def preprocess(results):

    stop_words = set(stopwords.words('english'))
    titles = [post.title for post in results]
    bodies = [post.selftext for post in results]
    ids = [post.id for post in results]

    documents = pd.DataFrame(titles, columns=['Title'])
    df2 = pd.DataFrame(bodies, columns=['Body'])
    df3 = pd.DataFrame(ids, columns=['id'])
    documents=pd.concat([df3,documents,df2],axis =1)
    
    documents['concat'] = documents['Title'] +' '+ documents['Body']

    documents['preprocessed'] = documents['concat'].apply(lambda x: word_tokenize(x))
    documents['tokenized'] = documents['preprocessed'].apply(lambda x: ' '.join([word for word in x if word not in stop_words]))
    # Mposts = []

    # # Iterate over the index of the lists
    # for i in range(len(titles)):
    #     # Get the item at the current index in each list
    #     item1 = titles[i]
    #     item2 = bodies[i]
    #     # Merge the items into a tuple and append them to the merged list
    #     Mposts.append(item1 + item2)

    # # preprocess the titles and bodies by tokenizing and removing stop words
    # stop_words = set(stopwords.words('english'))
    # posts_processed = []

    # for post in Mposts:
    #     tokens = word_tokenize(post)
    #     tokens_filtered = [token for token in tokens if token not in stop_words]
    #     posts_processed.append(' '.join(tokens_filtered))

    return documents

def get_posts():
    # authenticate with the Reddit API
    reddit = praw.Reddit(client_id='ZUHaoF2lWuT-XhQJM4PORg',
                        client_secret='rp2lOgUdUWWmzj9XhIqKjncOarh4Zg',
                        user_agent='ITC6010')

    # get the F1 subreddit
    f1_subreddit = reddit.subreddit('formula1')

    # collect the titles and bodies of the latest 1000 posts in the subreddit
    results = [post for post in f1_subreddit.hot(limit=1000)]
    posts =[]
    for result in results:
        if result.title and result.selftext:
            posts.append(result)

    return posts

    


nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
posts_orig = get_posts()
posts = preprocess(posts_orig)
while True:

    query = get_query()
    if query == '0':
        break
    else:
# Search for the query in the documents
        post1 = posts.copy()
        df1,id1 = search_engine_word2vec(post1, query,3)
        post2 = posts.copy()
        df2,id2 = search_engine_tfidf(post2, query,3)
        post3 = posts.copy()
        df3,id3 = search_engine_tfidf_withSentiment(post3, query,3)


    print('Results with Word2Vec:')
    selected_rows1 = df1.loc[df1['id'].isin(id1)]
    print(selected_rows1.sort_values('total score', ascending=False)['concat'])
    print('------------------------')
    print('\n')
    print('Results with TFIDF:')
    selected_rows2 = df2.loc[df2['id'].isin(id2)]
    print(selected_rows2.sort_values('total score', ascending=False)['concat'])
    print('------------------------')
    print('\n')
    print('Results with TFIDF with Sentiment Analysis:')
    selected_rows3 = df3.loc[df3['id'].isin(id3)]
    print(selected_rows3.sort_values('total score', ascending=False)['concat'])

