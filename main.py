from core import TwitterCleaner
from core import Syntactic
from core import Sentiment


def twitter_clean():
    twitter = TwitterCleaner(folder="twitter_raw")
    twitter.hydrate_ids()


def extract_syntactic():
    analyzer = Syntactic("outputs/news/pretty_news_dataset.csv", "outputs/news")
    analyzer.extract_syntactic_features(text_col="Snippet")


def extract_sentiments():
    analyzer = Sentiment("outputs/twitter_raw/tweets_hydrate_syntactic.csv", "outputs/twitter_raw")
    analyzer.extract_sentiment_features(text_col="tweet")


def main():
    extract_sentiments()


if __name__ == '__main__':
    main()
