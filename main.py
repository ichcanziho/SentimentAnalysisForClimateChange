from core import TwitterCleaner
from core import Syntactic
from core import Sentiment
from core import CriticalWords


def twitter_clean():
    twitter = TwitterCleaner(folder="twitter_raw")
    twitter.hydrate_ids()


def extract_syntactic():
    analyzer = Syntactic("outputs/news/pretty_news_dataset.csv", "outputs/news")
    analyzer.extract_syntactic_features(text_col="Snippet")


def extract_sentiments():
    analyzer = Sentiment("outputs/twitter_raw/tweets_hydrate_syntactic.csv", "outputs/twitter_raw")
    analyzer.extract_sentiment_features(text_col="tweet")


def extract_critical_words():
    critical = CriticalWords(input_data="outputs/news/pretty_news_dataset_syntactic_sentiment.csv",
                             output_folder="outputs/analysis",
                             text_column="Snippet",
                             class_column="blob category",
                             method_name="textBlob",
                             n_words=20)
    critical.extract_critical_words()


def main():
    extract_critical_words()


if __name__ == '__main__':
    main()
