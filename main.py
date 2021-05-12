###############################################################################
# Main file of the repository. Reads the config.ini file and performs the     #
# requested action. Available options:                                        #
# CLEAN: Dataset cleaning.                                                    #
# SYNT: Syntactic analysis.                                                   #
# SENT: Sentiment analysis.                                                   #
# PLOTS: Plot the sentiment and syntactic analysis resulting timelines.       #
#                                                                             #
# @author Gabriel Pérez and Jorge Ciprián                                     #
# Last updated: 12-05-2021.                                                   #
###############################################################################

# Imports.
import configparser
from core import TwitterCleaner
from core import NewsCleaner
from core import Syntactic
from core import Sentiment
from core import CriticalWords
from core import SummarySentiments
from core import SummarySyntactic


def twitter_clean():
    twitter = TwitterCleaner(input_path="inputs/raw datasets/Twitter",
                             output_path="outputs/clean datasets/Twitter")
    twitter.clean_tweets()


def news_clean():
    news = NewsCleaner(input_path="inputs/raw datasets/News/",
                       output_path="outputs/clean datasets/News")
    news.clean_news()


def twitter_syntactic_analysis():
    analyzer = Syntactic(input_data="outputs/clean datasets/Twitter/Twitter.csv",
                         output_folder="outputs/clean datasets/Twitter")
    analyzer.extract_syntactic_features(text_col="tweet")


def news_syntactic_analysis():
    analyzer = Syntactic(input_data="outputs/clean datasets/News/News.csv",
                         output_folder="outputs/clean datasets/News")
    analyzer.extract_syntactic_features(text_col="Snippet")


def twitter_sentiment_analysis():
    sentiment_analyzer = Sentiment(input_data="outputs/clean datasets/Twitter/Twitter.csv",
                                   output_folder="outputs/clean datasets/Twitter")
    sentiment_analyzer.extract_sentiment_features(text_col="tweet")

    words_analyzer = CriticalWords(input_data="outputs/clean datasets/Twitter/Twitter_sentiment.csv",
                                   output_folder="outputs/Sentiment Analysis/words/Twitter",
                                   text_column="tweet",
                                   class_column="blob category",
                                   method_name="textBlob",
                                   n_words=1000)
    words_analyzer.extract_critical_words()

    words_analyzer = CriticalWords(input_data="outputs/clean datasets/Twitter/Twitter_sentiment.csv",
                                   output_folder="outputs/Sentiment Analysis/words/Twitter",
                                   text_column="tweet",
                                   class_column="vader category",
                                   method_name="vader",
                                   n_words=1000)
    words_analyzer.extract_critical_words()


def news_sentiment_analysis():
    sentiment_analyzer = Sentiment(input_data="outputs/clean datasets/News/News.csv",
                                   output_folder="outputs/clean datasets/News")
    sentiment_analyzer.extract_sentiment_features(text_col="Snippet")

    words_analyzer = CriticalWords(input_data="outputs/clean datasets/News/News_sentiment.csv",
                                   output_folder="outputs/Sentiment Analysis/words/News",
                                   text_column="Snippet",
                                   class_column="blob category",
                                   method_name="textBlob",
                                   n_words=1000)
    words_analyzer.extract_critical_words()

    words_analyzer = CriticalWords(input_data="outputs/clean datasets/News/News_sentiment.csv",
                                   output_folder="outputs/Sentiment Analysis/words/News",
                                   text_column="Snippet",
                                   class_column="vader category",
                                   method_name="vader",
                                   n_words=1000)
    words_analyzer.extract_critical_words()


def sentiments_graphs():
    ss = SummarySentiments()
    color_map = {'pos': 'green', 'neu': 'blue', 'neg': 'red'}
    ss.make_freq(x_axes_file="outputs/Sentiment Analysis/words/Twitter/critical words/Twitter_vader_words.csv",
                 y_axes_file="outputs/Sentiment Analysis/words/News/critical words/News_vader_words.csv",
                 x_max_freq="outputs/Sentiment Analysis/words/Twitter/counts/Twitter_vader_counts.csv",
                 y_max_freq="outputs/Sentiment Analysis/words/News/counts/News_vader_counts.csv",
                 sentiments=color_map,
                 output_dir="outputs/Sentiment Analysis/words/freq distribution/vader_results_10_words.csv",
                 start=2,
                 window=10)

    ss.make_freq(x_axes_file="outputs/Sentiment Analysis/words/Twitter/critical words/Twitter_textBlob_words.csv",
                 y_axes_file="outputs/Sentiment Analysis/words/News/critical words/News_textBlob_words.csv",
                 x_max_freq="outputs/Sentiment Analysis/words/Twitter/counts/Twitter_textBlob_counts.csv",
                 y_max_freq="outputs/Sentiment Analysis/words/News/counts/News_textBlob_counts.csv",
                 sentiments=color_map,
                 output_dir="outputs/Sentiment Analysis/words/freq distribution/textBlob_results_10_words.csv",
                 start=2,
                 window=10)

    ss.plot_sentiment_summary(input_file="outputs/Sentiment Analysis/words/freq distribution/"
                                         "textBlob_results_10_words.csv",
                              save_dir="outputs/Sentiment Analysis/images",
                              classifier='TextBlob',
                              marker='*',
                              size=(15, 10),
                              font_size=12,
                              ys_lims=(-0.05, 0.2, 0.2, 0.6, 0.6, 1.45),
                              xs_lims=(-0.05, 0.7, 0.7, 3.0))

    ss.plot_sentiment_summary(input_file="outputs/Sentiment Analysis/words/freq distribution/"
                                         "vader_results_10_words.csv",
                              save_dir="outputs/Sentiment Analysis/images",
                              classifier='Vader',
                              marker='*',
                              size=(15, 10),
                              font_size=12,
                              ys_lims=(-0.05, 0.4, 0.4, 0.8, 0.8, 1.6),
                              xs_lims=(-0.05, 1, 1.3, 3.5))

    ss.make_freq(x_axes_file="outputs/Sentiment Analysis/words/Twitter/critical words/Twitter_vader_words.csv",
                 y_axes_file="outputs/Sentiment Analysis/words/News/critical words/News_vader_words.csv",
                 x_max_freq="outputs/Sentiment Analysis/words/Twitter/counts/Twitter_vader_counts.csv",
                 y_max_freq="outputs/Sentiment Analysis/words/News/counts/News_vader_counts.csv",
                 sentiments=color_map,
                 output_dir="outputs/Sentiment Analysis/words/freq distribution/vader_results_2_words.csv",
                 start=0,
                 window=2)

    ss.make_freq(x_axes_file="outputs/Sentiment Analysis/words/Twitter/critical words/Twitter_textBlob_words.csv",
                 y_axes_file="outputs/Sentiment Analysis/words/News/critical words/News_textBlob_words.csv",
                 x_max_freq="outputs/Sentiment Analysis/words/Twitter/counts/Twitter_textBlob_counts.csv",
                 y_max_freq="outputs/Sentiment Analysis/words/News/counts/News_textBlob_counts.csv",
                 sentiments=color_map,
                 output_dir="outputs/Sentiment Analysis/words/freq distribution/textBlob_results_2_words.csv",
                 start=0,
                 window=2)

    ss.plot_mini(textBlob="outputs/Sentiment Analysis/words/freq distribution/textBlob_results_2_words.csv",
                 vader="outputs/Sentiment Analysis/words/freq distribution/vader_results_2_words.csv",
                 save_dir="outputs/Sentiment Analysis/images/summary.svg",
                 size=(10, 5))


def syntactic_graphs():
    SummarySyntactic.plot_timelines("syntactic",
                                    "outputs/clean datasets/Twitter/Twitter_syntactic_sentiment.csv",
                                    "outputs/clean datasets/News/News_syntactic_sentiment.csv",
                                    "outputs/Syntactic Analysis/images/")
    SummarySyntactic.plot_timelines("sentiment",
                                    "outputs/clean datasets/Twitter/Twitter_syntactic_sentiment.csv",
                                    "outputs/clean datasets/News/News_syntactic_sentiment.csv",
                                    "outputs/Syntactic Analysis/images/")


def main():

    # Reading configuration file.
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Reading operation mode.
    mode = config['MODE'].get('mode')
    print("Selected mode: ", mode)

    if(mode == "CLEAN"):
        #print("clean!")
        # Cleaning the datasets
        news_clean()
        twitter_clean()
    elif(mode == "SYNT"):
        #print("synt!")
        # Making syntactic analysis
        news_syntactic_analysis()
        twitter_syntactic_analysis()
    elif(mode == "SENT"):
        #print("sent!")
        # Making sentiment analysis
        news_sentiment_analysis()
        twitter_sentiment_analysis()
    elif(mode == "PLOTS"):
        #print("plots!")
        # Plotting the graphs
        sentiments_graphs()
        syntactic_graphs()
    else:
        print("Invalid mode!")


if __name__ == '__main__':
    main()
