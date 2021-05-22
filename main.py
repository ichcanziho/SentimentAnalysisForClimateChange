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
from core import CorrCausAnalyzer


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

def corr_caus_analysis():
    analyzer = CorrCausAnalyzer(in_path_tweets="outputs/clean datasets/Twitter/Twitter_syntactic_sentiment.csv",
                                in_path_news="outputs/clean datasets/News/News_syntactic_sentiment.csv",
                                out_path="outputs/Correlation/")
    # First, we test for the Granger causality relation.
    # Generating the required time series.
    pos_df, neg_df, neu_df = analyzer.load_datasets()
    # Checking stationarity for all datasets.
    print("Stationarity tests:")
    print("Positive: ")
    pos_flag_news, pos_flag_tweets = analyzer.check_stationarity(pos_df)
    print("Negative: ")
    neg_flag_news, neg_flag_tweets = analyzer.check_stationarity(neg_df)
    print("Neutral: ")
    neu_flag_news, neu_flag_tweets = analyzer.check_stationarity(neu_df)

    # For our case, the datasets are stationary, so we do not need to do any
    # differencing. If you are working with a different dataset, you may
    # need to differentiate one or more columns one or more times as follows:
    # pos_df_trans = analyzer.diff_df(pos_df, col_name="pos_count_tweets")

    # IMPORTANT:  The Granger causality test requires the time series to be
    # stationary. As such, you will not be able to advance further until all
    # the time series are stationary.

    if(not(pos_flag_news & pos_flag_tweets & neg_flag_news & neg_flag_tweets & neu_flag_news & neu_flag_tweets)):
        print("All time series must be stationary to continue!")
    else:
        # Finding best max lag value for each sentiment.
        print("Finding best max lag values:")
        print("Positive:")
        pos_best_aic, pos_best_lag, pos_best_dw = analyzer.best_lag_dw(pos_df)
        print("Negative:")
        neg_best_aic, neg_best_lag, neg_best_dw = analyzer.best_lag_dw(neg_df)
        print("Neutral:")
        neu_best_aic, neu_best_lag, neu_best_dw = analyzer.best_lag_dw(neu_df)

        # Performing the Granger causality tests per sentiment.
        print("Positive:")
        pos_granger = analyzer.grangers_causation_matrix(pos_df, variables = pos_df.columns,
                                                         maxlag=pos_best_lag)
        print("Negative:")
        neg_granger = analyzer.grangers_causation_matrix(neg_df, variables = neg_df.columns,
                                                         maxlag=neg_best_lag)
        print("Neutral:")
        neu_granger = analyzer.grangers_causation_matrix(neu_df, variables = neu_df.columns,
                                                      maxlag=neu_best_lag)
        # Getting the n-lag cross correlation matrices. Default lags = 6.
        # Positive.
        analyzer.n_lag_corr(pos_df, filename="pos_corr.svg")
        # Negative.
        analyzer.n_lag_corr(neg_df, filename="neg_corr.svg")
        # Neutral.
        analyzer.n_lag_corr(neu_df, filename="neu_corr.svg")

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
    elif(mode == "CC"):
        corr_caus_analysis()
    elif(mode == "ALL"):
        # All options in order.
        news_clean()
        twitter_clean()
        news_syntactic_analysis()
        twitter_syntactic_analysis()
        news_sentiment_analysis()
        twitter_sentiment_analysis()
        sentiments_graphs()
        syntactic_graphs()
    else:
        print("Invalid mode!")


if __name__ == '__main__':
    main()
