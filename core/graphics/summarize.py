# Imports.
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
from core.utils import UtilMethods


class Results(UtilMethods):

    def __init__(self):
        ...

    # Method for plotting the timelines for sentiment and syntactic analysis.
    @staticmethod
    def plot_timelines(mode):
        # First, we load the datasets for news and tweets.
        print("Loading datasets...")
        tweets_df = pd.read_csv("./outputs/twitter_raw/tweets_hydrate_syntactic_sentiment.csv")
        news_df = pd.read_csv("./outputs/news/pretty_news_dataset_syntactic_sentiment.csv")
        print("... done.")
        # Pre-processing the data according to the mode selected.
        if(mode == "sentiment"):
            print("Selected: sentiment")
            # For news.
            news_sent_df = news_df.copy()
            news_sent_df = news_sent_df.drop(['Show','Snippet','NOUN','ADJ','VERB','PROPN','SENT','SIZE','blob score', 'vader score'],1)
            # For tweets.
            tweets_sent_df = tweets_df.copy()
            tweets_sent_df = tweets_sent_df.drop(['i','id','day','tweet','NOUN','ADJ','VERB','PROPN','SENT','SIZE','blob score', 'vader score'],1)
            # Generating the desired dataframe for news.
            query_news = """SELECT    year, month,
                                      SUM(CASE WHEN `vader category`='pos' THEN 1 ELSE 0 END) AS v_pos_count,
                                      SUM(CASE WHEN `vader category`='neg' THEN 1 ELSE 0 END) AS v_neg_count,
                                      SUM(CASE WHEN `vader category`='neu' THEN 1 ELSE 0 END) AS v_neu_count,
                                      SUM(CASE WHEN `blob category`='pos' THEN 1 ELSE 0 END) AS b_pos_count,
                                      SUM(CASE WHEN `blob category`='neg' THEN 1 ELSE 0 END) AS b_neg_count,
                                      SUM(CASE WHEN `blob category`='neu' THEN 1 ELSE 0 END) AS b_neu_count
                           FROM       news_sent_df
                           GROUP BY   year, month;"""
            # Resulting query.
            result_news = ps.sqldf(query_news, locals())
            # Combining month and year columns.
            result_news["date"] = result_news["year"].astype(str) + "_" + result_news["month"].astype(str)
            # Generating the desired dataframe for tweets.
            query_tweets = """SELECT     year, month,
                                  SUM(CASE WHEN `vader category`='pos' THEN 1 ELSE 0 END) AS v_pos_count,
                                  SUM(CASE WHEN `vader category`='neg' THEN 1 ELSE 0 END) AS v_neg_count,
                                  SUM(CASE WHEN `vader category`='neu' THEN 1 ELSE 0 END) AS v_neu_count,
                                  SUM(CASE WHEN `blob category`='pos' THEN 1 ELSE 0 END) AS b_pos_count,
                                  SUM(CASE WHEN `blob category`='neg' THEN 1 ELSE 0 END) AS b_neg_count,
                                  SUM(CASE WHEN `blob category`='neu' THEN 1 ELSE 0 END) AS b_neu_count
                       FROM       tweets_sent_df
                       GROUP BY   year, month;"""
            # Resulting query.
            result_tweets = ps.sqldf(query_tweets, locals())
            # Combining month and year columns.
            result_tweets["date"] = result_tweets["year"].astype(str) + "_" + result_tweets["month"].astype(str)
            # Generating the plots.
            x = result_news["date"].tolist()

            # For news and VADER.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_news["v_pos_count"].tolist(), label='Positive', color='green')
            plt.plot(x, result_news["v_neu_count"].tolist(), label='Neutral', color='blue')
            plt.plot(x, result_news["v_neg_count"].tolist(), label='Negative', color='red')
            plt.xlabel('Year_month', fontsize=20)
            plt.ylabel('Sentiment counts', fontsize=20)
            plt.title('News sentiment counts per year and month for the VADER method', fontsize=20)
            plt.legend()
            plt.xticks(rotation=90)
            plt.tick_params(labelsize=14)
            #plt.yticks(fontize=14)
            plt.savefig("./outputs/images/sent_news_vader.svg", bbox_inches='tight')
            plt.show()

            # For news and TextBlob.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_news["b_pos_count"].tolist(), label='Positive', color='green')
            plt.plot(x, result_news["b_neu_count"].tolist(), label='Neutral', color='blue')
            plt.plot(x, result_news["b_neg_count"].tolist(), label='Negative', color='red')
            plt.xlabel('Year_month', fontsize=20)
            plt.ylabel('Sentiment counts', fontsize=20)
            plt.title('News sentiment counts per year and month for the TextBlob method', fontsize=20)
            plt.legend()
            plt.xticks(rotation=90)
            plt.tick_params(labelsize=14)
            plt.savefig("./outputs/images/sent_news_blob.svg", bbox_inches='tight')
            plt.show()

            # For tweets and VADER.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_tweets["v_pos_count"].tolist(), label='Positive', color='green')
            plt.plot(x, result_tweets["v_neu_count"].tolist(), label='Neutral', color='blue')
            plt.plot(x, result_tweets["v_neg_count"].tolist(), label='Negative', color='red')
            plt.xlabel('Year_month', fontsize=20)
            plt.ylabel('Sentiment counts', fontsize=20)
            plt.title('Tweets sentiment counts per year and month for the VADER method', fontsize=20)
            plt.legend()
            plt.xticks(rotation=90)
            plt.tick_params(labelsize=14)
            plt.savefig("./outputs/images/sent_tweets_vader.svg", bbox_inches='tight')
            plt.show()

            # For tweets and TextBlob.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_tweets["b_pos_count"].tolist(), label='Positive', color='green')
            plt.plot(x, result_tweets["b_neu_count"].tolist(), label='Neutral', color='blue')
            plt.plot(x, result_tweets["b_neg_count"].tolist(), label='Negative', color='red')
            plt.xlabel('Year_month', fontsize=20)
            plt.ylabel('Sentiment counts', fontsize=20)
            plt.title('Tweets sentiment counts per year and month for the TextBlob method', fontsize=20)
            plt.legend()
            plt.xticks(rotation=90)
            plt.tick_params(labelsize=14)
            plt.savefig("./outputs/images/sent_tweets_blob.svg", bbox_inches='tight')
            plt.show()
        elif(mode == "syntactic"):
            print("Selected: syntactic")
            # For news.
            news_synt_df = news_df.copy()
            news_synt_df = news_synt_df.drop(['Station','Show','Snippet','blob category','vader category','blob score', 'vader score'],1)
            # For tweets.
            tweets_synt_df = tweets_df.copy()
            tweets_synt_df = tweets_synt_df.drop(['i','id','tweet','blob category','vader category','blob score', 'vader score'],1)
            # Generating the desired dataframe for news.
            query_news = """SELECT    year, month,
                                      ROUND(AVG(NOUN),2) as avg_noun,
                                      ROUND(AVG(ADJ),2) as avg_adj,
                                      ROUND(AVG(VERB),2) as avg_verb,
                                      ROUND(AVG(SENT),2) as avg_sent,
                                      ROUND(AVG(SIZE),2) as avg_size
                           FROM       news_synt_df
                           GROUP BY   year, month;"""
            # Resulting query.
            result_news = ps.sqldf(query_news, locals())
            # Combining month and year columns.
            result_news["date"] = result_news["year"].astype(str) + "_" + result_news["month"].astype(str)
            # Generating the desired dataframe for tweets.
            query_tweets = """SELECT      year, month,
                                          ROUND(AVG(NOUN),2) as avg_noun,
                                          ROUND(AVG(ADJ),2) as avg_adj,
                                          ROUND(AVG(VERB),2) as avg_verb,
                                          ROUND(AVG(SENT),2) as avg_sent,
                                          ROUND(AVG(SIZE),2) as avg_size
                               FROM       tweets_synt_df
                               GROUP BY   year, month;"""
            # Resulting query.
            result_tweets = ps.sqldf(query_tweets, locals())
            # Combining month and year columns.
            result_tweets["date"] = result_tweets["year"].astype(str) + "_" + result_tweets["month"].astype(str)
            # Generating the plots.
            x = result_news["date"].tolist()

            # For news and main elements.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_news['avg_noun'].tolist(), label = 'Average Nouns')
            plt.plot(x, result_news['avg_adj'].tolist(), label = "Average Adjectives")
            plt.plot(x, result_news['avg_verb'].tolist(), label = "Average Verbs")
            plt.plot(x, result_news['avg_sent'].tolist(), label = "Average Sentences")
            plt.xlabel('Year_month', fontsize=20)
            plt.ylabel('Average count', fontsize=20) # Sentence size es en número de palabras.
            plt.title('Syntactic characteristics of news per year and month', fontsize=20)
            plt.legend()
            plt.xticks(rotation=90)
            plt.tick_params(labelsize=14)
            plt.savefig("./outputs/images/synt_news.svg", bbox_inches='tight')
            plt.show()

            # For news and average size.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_news['avg_size'].tolist(), label='Average Sentence Size')
            plt.xlabel('Year_month', fontsize=20)
            plt.ylabel('Average sentece size', fontsize=20) # Sentence size es en número de palabras.
            plt.title('Average number of words in news per year and month', fontsize=20)
            plt.legend()
            plt.xticks(rotation=90)
            plt.tick_params(labelsize=14)
            plt.savefig("./outputs/images/synt_news_size.svg", bbox_inches='tight')
            plt.show()

            # For tweets and main elements.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_tweets['avg_noun'].tolist(), label = 'Average Nouns')
            plt.plot(x, result_tweets['avg_adj'].tolist(), label = "Average Adjectives")
            plt.plot(x, result_tweets['avg_verb'].tolist(), label = "Average Verbs")
            plt.plot(x, result_tweets['avg_sent'].tolist(), label = "Average Sentences")
            plt.xlabel('Year_month', fontsize=20)
            plt.ylabel('Average count', fontsize=20) # Sentence size es en número de palabras.
            plt.title('Syntactic characteristics of tweets per year and month', fontsize=20)
            plt.legend()
            plt.xticks(rotation=90)
            plt.tick_params(labelsize=14)
            plt.savefig("./outputs/images/synt_tweets.svg", bbox_inches='tight')
            plt.show()

            # For tweets and average size.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_tweets['avg_size'].tolist(), label='Average Sentence Size')
            plt.xlabel('Year_month', fontsize=20)
            plt.ylabel('Average sentece size', fontsize=20) # Sentence size es en número de palabras.
            plt.title('Average number of words in tweets per year and month', fontsize=20)
            plt.legend()
            plt.xticks(rotation=90)
            plt.tick_params(labelsize=14)
            plt.savefig("./outputs/images/synt_tweets_size.svg", bbox_inches='tight')
            plt.show()
        else:
            print("Invalid argument!")
