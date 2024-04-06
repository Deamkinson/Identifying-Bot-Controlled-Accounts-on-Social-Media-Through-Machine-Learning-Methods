from dateutil.parser import parse
import streamlit as st
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd
import joblib

nltk.download("popular")

def preprocess_tweet(tweet):
    # Remove links
    tweet = re.sub(r'http\S+', '', tweet)
    
    # Remove @username
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove "retweet" tag
    tweet = tweet.replace('RT', '')
    
    # Remove non-alphanumeric characters
    tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)
    
    # Tokenization
    tokens = word_tokenize(tweet)
    
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmatized_tokens



def convert_timestamp(timestamp):
    """
    Convert a timestamp to seconds since the Unix epoch.

    Parameters:
    - timestamp (str): A timestamp in either datetime string format or milliseconds.

    Returns:
    - float: The converted timestamp in seconds.

    The function attempts to parse the input timestamp. If successful, it calculates
    the timestamp in seconds since the Unix epoch using the `timestamp()` method.
    If parsing fails, it assumes the timestamp is in milliseconds and converts it to
    seconds by dividing by 1000.
    """
    converted_timestamp = 0  # Initialize the variable to store the converted timestamp

    try:
        # Try to parse the timestamp as a datetime string and get the timestamp in seconds
        converted_timestamp = parse(timestamp).timestamp()
    except:
        # If parsing fails, assume it's in milliseconds and convert to seconds
        converted_timestamp = int(timestamp[:-1]) / 1000

    return converted_timestamp



def get_intertime(df):
    """
    Calculate the average time difference between consecutive tweets.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing tweet information.
    
    Returns:
    - float: The average time difference between consecutive tweets in seconds.

    This function takes a DataFrame with a 'created_at' column representing
    the timestamp of each tweet. It calculates the time difference between
    consecutive tweets, sorts the differences, and returns the average time
    difference. If there are no tweets or only one tweet, it returns 0.

    Note: Make sure the 'created_at' column is in a format that can be
    converted to timestamps using the convert_timestamp function.
    """
    tweet_timestamps = []

    for index, row in df.iterrows():
        tweet_timestamps.append(convert_timestamp(str(row["created_at"])))

    tweet_timestamps.sort()
    tts_diff = np.diff(np.array(tweet_timestamps))
    
    return sum(tts_diff) / (len(tts_diff) if len(tts_diff) != 0 else 1)



def write_project_info():

    st.write("""<h3 align="center">DETECTING BOT-CONTROLLED ACCOUNTS ON SOCIAL MEDIA USING MACHINE LEARNING TECHNIQUES</h3>""", unsafe_allow_html=True)

    st.write("""<h5 align="center">Project Completed BY AKINDELE ABDULAZEEZ OLADIMEJI</h5>""", unsafe_allow_html=True)

    st.write("""<p align="center">SUPERVISED BY DR. A.B SAKPERE</p>""", unsafe_allow_html=True)

    st.image("images/social_media_bot.png")  

    st.write("""
    #### Project Introduction

    <p align="justify">Social media platforms have become a cornucopia of bot-controlled accounts. These bots mimic human behavior, often created to manipulate or influence social media conversations. They can spread misinformation, amplify certain messages, and even engage in activities like fake likes, follows, or comments.</p>

    <p align="justify">The dangers they pose lie in their potential to deceive and manipulate public opinion, create echo chambers, and contribute to the spread of false information. Bot-controlled accounts can also be used maliciously to spam, harass, or target individuals or groups, undermining the credibility and trustworthiness of the platform. In this light, this project is aimed at</p> 
            <ul>
                <li><p align="justify">Automated detection of bot-controlled accounts to mitigate the impact of bot-driven campaigns on social media platforms, protecting users from deceptive content and maintaining the integrity of online discussions.</l1></p>
                <li><p align="justify">Developing an interpretable machine learning model to distinguish between human-operated and bot-controlled social media accounts in contrast to prevalent deep learning approaches in existing literature.</l1></p>
            </ul> </p>
    """, unsafe_allow_html=True)

    st.write("""
    #### What Improvements Did We Make?
    <ul>
        <li><p align="justify">Firstly, Our research capitalizes on the most up-to-date dataset, Cresi17, sourced from the Bot Repository, distinguishing it from prior studies that relied on older datasets like Cresi14, Cresi15, and Cresi16.</p></li>
        <li><p align="justify">Furthermore, we introduce a novel methodology that integrates a comprehensive array of user profile features including follower count, favorites, and following, along with tweet metrics such as retweets, mentions, hashtags, and replies, in addition to the raw tweet text. Additionally, we have engineered additional features such as intertime (the average time between successive tweets), average hashtag usage, follower ratio, and more. Our methodology, leveraging this multitude of features, demonstrates superior performance compared to earlier works such as Chandra et al., 2021, and Hayawi et al., 2023, which solely focused on modeling user tweets. The subsequent comparison illustrates the efficacy of our proposed approach against prior methodologies.</p></li>
    </ul>
    <br>
    <table align="center">
    <tr>
        <th>Paper</th>
        <th>Accuracy</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1 Score</th>
    </tr>
        <tr>
        <td>Our Methodology</td>
        <td>98%</td>
        <td>97%</td>
        <td>98%</td>
        <td>96%</td>
    </tr>
    <tr>
        <tr>
        <td>Aljabri et al., 2022</td>
        <td>95%</td>
        <td>93.27%</td>
        <td>93%</td>
        <td>93.12%</td>
    </tr>
        <td>Hayawi et al., 2023</td>
        <td>87%</td>
        <td>89%</td>
        <td>88%</td>
        <td>84%</td>
    </tr>
    <tr>
        <td>Chandra et al., 2021</td>
        <td>82.3%</td>
        <td>84.2%</td>
        <td>Not Specified</td>
        <td>Not Specified</td>
    </tr>
    </table>

    <br>
    <ul>
        <li><p align="justify">Finally, in contrast to the predominant trend in literature, which often employs black box deep learning models such as LSTM, 1d-CNN, etc., as seen in works by Hayawi et al., 2023, Derhab et al. (2021), Kudugunta et al., 2018, etc., we advocate for the use of a Random Forest model. Our choice of Random Forest not only ensures interpretability but also facilitates comprehension for non-technical users, all while surpassing state-of-the-art performance benchmarks. Presented below are some of the key features identified by our Random Forest model.</p></li>
    </ul>
    """, unsafe_allow_html=True)

    st.write("""#### Feature Importance""", unsafe_allow_html=True)

    st.image("images/feature_importance.png")  

    st.write("<br>", unsafe_allow_html=True)



def feature_engineering(base_df, eng_df, columns):
    tweet_count = eng_df.shape[0]
    retweets = eng_df["retweeted_status_id"].sum() / tweet_count
    replies = eng_df["in_reply_to_status_id"].sum() / tweet_count

    favoriteC = base_df["favourites_count"].values[0] / tweet_count
    hashtag = eng_df["num_hashtags"].sum() / tweet_count
    url = eng_df["num_urls"].sum() / tweet_count
    mentions = eng_df["num_mentions"].sum() / tweet_count

    ffratio = base_df["friends_count"].values[0] / (base_df["followers_count"].values[0] if base_df["followers_count"].values[0] != 0 else 1)

    intertime = get_intertime(eng_df)

    # Extract users tweets
    tweets = eng_df['tweet_text'].values
    tweets = ' '.join([str(tweet) for index, tweet in enumerate(tweets) if index <= 10])

    # Create a list with user features
    usr_features = [retweets, replies, favoriteC, hashtag, url, mentions, intertime, ffratio, tweets]

    # Create a list with user features
    engineered_features = pd.DataFrame([usr_features], columns=columns)

    return engineered_features



def take_user_inputs():
    columns = ["retweets", "replies", "favoriteC", "hashtag", "url",
           "mentions", "intertime", "ffratio", "tweets"]
    
    st.sidebar.header("Profile Information")

    statuses_count = st.sidebar.number_input("The total number of tweets (including retweets) made by the user", min_value=0, max_value=100_000_000)
    
    followers_count = st.sidebar.number_input("The number of followers the user has", min_value=0, max_value=100_000_000)
    
    friends_count = st.sidebar.number_input("The number of other users the user is following", min_value=0, max_value=100_000_000)
    
    favourites_count = st.sidebar.number_input("The number of tweets the user has liked", min_value=0, max_value=100_000_000)
    
    listed_count = st.sidebar.number_input("The number of public lists that the user is a member of", min_value=0, max_value=100_000_000)

    base_input = {
        "statuses_count": statuses_count,
        "followers_count": followers_count,
        "friends_count": friends_count,
        "favourites_count": favourites_count,
        "listed_count": listed_count
    }

    base_df = pd.DataFrame([base_input])

    st.sidebar.header("Tweet Information Description")

    st.sidebar.write("No of URLs: This is the number of URLs included in the tweet")
    st.sidebar.write("No of Mentions: This is the number of user mentions in the tweet")
    st.sidebar.write("No of Hashtags: This represents the number of hashtags used in the tweet")

    st.write("#### Tweet Information")
    st.write("""<p align="justify">Unfortunately, due to the recent changes to Twitter's API payment policy, we're unable to access the information automatically through the API as they did when compiling the dataset (Cresi17). Therefore, we kindly request that you provide the information manually. It's worth noting that individuals with enterprise access to the Twitter API can still retrieve this information effortlessly.</p>""", unsafe_allow_html=True)

    num_tweets = st.number_input("Specify the number of tweets", min_value=0, max_value=100)

    if num_tweets > 0:
        tweet_info = {
            "retweeted_status_id": [],
            "in_reply_to_status_id": [],
            "num_mentions": [],
            "num_hashtags": [],
            "num_urls": [],
            "created_at": [],
            "tweet_text": []
        }

        for i in range(1, num_tweets + 1):
            st.write(f"Information for tweet {i}:")
            # Create columns for tweet inputs
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                retweeted_status_id = st.number_input(f"No of retweets", min_value=0, max_value=10000000000, key=f"retweets_{i}")
                in_reply_to_status_id = st.number_input(f"No of replies", min_value=0, max_value=10000000000, key=f"replies_{i}")
            with col2:
                num_mentions = st.number_input(f"No of mentions", min_value=0, max_value=10000000000, key=f"mentions_{i}")
                num_hashtags = st.number_input(f"No of hashtags", min_value=0, max_value=10000000000, key=f"hashtags_{i}")
            with col3:
                num_urls = st.number_input(f"No of URLs", min_value=0, max_value=10000000000, key=f"urls_{i}")
                created_at_date = st.date_input(f"Date", key=f"date_{i}")
            with col4:
                created_at_time = st.time_input(f"Time", key=f"time_{i}")
                tweet_text = st.text_input(f"Tweet text", key=f"text_{i}")

            created_at = f"{created_at_date} {created_at_time}"

            tweet_info["retweeted_status_id"].append(retweeted_status_id)
            tweet_info["in_reply_to_status_id"].append(in_reply_to_status_id)
            tweet_info["num_mentions"].append(num_mentions)
            tweet_info["num_hashtags"].append(num_hashtags)
            tweet_info["num_urls"].append(num_urls)
            tweet_info["created_at"].append(created_at)
            tweet_info["tweet_text"].append(tweet_text)

            # Add an empty space for separation between tweets
            st.write("")  
            st.write("")  

    
        eng_df = pd.DataFrame(tweet_info)

        engineered_df = feature_engineering(base_df, eng_df, columns)
    
    else:
        usr_features = [10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 16.0, 190805.7372881356, "Nil"]
        columns = [
            "retweets", "replies", "favoriteC", "hashtag", "url", "mentions", "intertime", "ffratio", "tweets"
        ]
        engineered_df = pd.DataFrame([usr_features], columns=columns)

    final_df = pd.concat([engineered_df, base_df], axis=1)

    final_df.head()

    return final_df



def predict_input(user_input):
    model = joblib.load("trained_model.joblib")

    num_cols = user_input.select_dtypes(include=np.number).columns.tolist()

    scaler = model['scaler']
    scaled_df = pd.DataFrame()
    scaled_output = scaler.fit_transform(user_input[num_cols])
    scaled_df[num_cols] = scaled_output

    vectorizer = model['vectorizer']
    classifier = model['classifier']

    tweets_transformed = vectorizer.transform(user_input['tweets'])
    tweets_transformed_df = pd.DataFrame(tweets_transformed.toarray(), columns=vectorizer.get_feature_names_out())

    X = pd.concat([scaled_df, tweets_transformed_df], axis=1)

    prediction = classifier.predict(X)

    return prediction


if __name__ == "__main__":
    write_project_info()

    user_input = take_user_inputs()

    predict_bot = st.button("Detect Account Type")

    if predict_bot:
        prediction = predict_input(user_input)

        if prediction[0] == 1:
            st.write("Detected account type: Bot-controlled account")
        else:
            st.write("Detected account type: Human-operated account")

