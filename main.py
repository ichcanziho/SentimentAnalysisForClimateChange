from core import TwitterCleaner
from core import Syntactic
from core import Sentiment
from core import CriticalWords
from core import Results


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
                             n_words=1000)
    critical.extract_critical_words()
    critical = CriticalWords(input_data="outputs/news/pretty_news_dataset_syntactic_sentiment.csv",
                             output_folder="outputs/analysis",
                             text_column="Snippet",
                             class_column="vader category",
                             method_name="vader",
                             n_words=1000)
    critical.extract_critical_words()
    critical = CriticalWords(input_data="outputs/twitter_raw/tweets_hydrate_syntactic_sentiment.csv",
                             output_folder="outputs/analysis",
                             text_column="tweet",
                             class_column="blob category",
                             method_name="textBlob",
                             n_words=100)
    critical.extract_critical_words()
    critical = CriticalWords(input_data="outputs/twitter_raw/tweets_hydrate_syntactic_sentiment.csv",
                             output_folder="outputs/analysis",
                             text_column="tweet",
                             class_column="vader category",
                             method_name="vader",
                             n_words=100)
    critical.extract_critical_words()


def make_summarize():
    plot = Results()

    news_blob_max_freq = {'neu': 134978, 'pos': 153930, 'neg': 36808}
    news_vader_max_freq = {'neu': 47066, 'pos': 171306, 'neg': 107344}

    tweet_blob_max_freq = {'neu': 171734, 'pos': 78884, 'neg': 44893}
    tweet_vader_max_freq = {'neu': 74748, 'pos': 105733, 'neg': 115030}

    # plot.make_freq(x_axes_file="outputs/analysis/tweets_textBlob_words.csv",
    #                y_axes_file="outputs/analysis/pretty_textBlob_words.csv",
    #                x_max_freq=tweet_blob_max_freq,
    #                y_max_freq=news_blob_max_freq,
    #                sentiments={'pos': 'green', 'neu': 'blue', 'neg': 'red'},
    #                scale=100,
    #                single_axes=True,  # If false, it will not show the words that have only one axe
    #                output_dir="outputs/analysis/freq/blob_results.csv",
    #                interval=(2, 12))  # the number of words to use (-1) = until last
    #
    # # plot.plot_test2(input_file="outputs/analysis/freq/blob_results.csv", save_dir="outputs/images",
    # #                 classifier='Text-Blob', marker='o', size=(10, 10))
    #
    # plot.make_freq(x_axes_file="outputs/analysis/tweets_vader_words.csv",
    #                y_axes_file="outputs/analysis/pretty_vader_words.csv",
    #                x_max_freq=tweet_vader_max_freq,
    #                y_max_freq=news_vader_max_freq,
    #                sentiments={'pos': 'green', 'neu': 'blue', 'neg': 'red'},
    #                scale=100,
    #                single_axes=True,
    #                output_dir="outputs/analysis/freq/vader_results.csv",
    #                interval=(2, 12))
    # plot.plot_test2(input_file="outputs/analysis/freq/vader_results.csv", save_dir="outputs/images",
    #                 classifier='Vader', marker='*', size=(15, 10),
    #                 ys_lims=(-0.05, 0.05, 0.54, 0.76, 0.76, 1.55),
    #                 xs_lims=(-0.05, 0.8, 0.81, 4.0))
    plot.plot_test2(input_file="outputs/analysis/freq/blob_results.csv", save_dir="outputs/images",
                    classifier='TextBlob', marker='*', size=(15, 10))


def main():
    make_summarize()
    # extract_critical_words()
    # Results.plot_timelines("syntactic")


if __name__ == '__main__':
    """
                 sentiment,   words,  used times
    news blob:   neutral,     8585,   134978
                 positive,    8900,   153930
                 negative,    4649,   36808

                 sentiment,   words,  used times
    news vader:  neutral,     5320,   47066
                 positive,    9252,   171306
                 negative,    7651,   107344

                 sentiment,   words,  used times
    tweet blob:  neutral,     19988,   171734
                 positive,    11735,   78884
                 negative,    7847 ,   44893

                 sentiment,   words,  used times
    tweet vader: neutral,     12048,   74748
                 positive,    14834,   105733
                 negative,    13833,   115030
    """
    main()
