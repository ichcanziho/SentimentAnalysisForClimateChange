from core import generate_news_csv
from core import TwitterCleaner


def twitter_clean():
    twitter = TwitterCleaner(folder="twitter_raw")
    twitter.hydrate_ids()


def main():
    generate_news_csv()


if __name__ == '__main__':
    main()
