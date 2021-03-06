from core import generate_news_csv
from core import TwitterCleaner
from core import Syntactic


def twitter_clean():
    twitter = TwitterCleaner(folder="twitter_raw")
    twitter.hydrate_ids()


def extract_syntactic():
    analyzer = Syntactic("outputs/news/pretty_news_dataset.csv", "outputs/news")
    analyzer.extract_syntactic_features(text_col="Snippet")


def main():
    extract_syntactic()


if __name__ == '__main__':
    main()
