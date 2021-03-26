# Imports.
import numpy as np
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
from core.utils import UtilMethods
from adjustText import adjust_text
import numpy as np


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
            plt.xlabel('Year_month')
            plt.ylabel('Sentiment counts')
            plt.title('News sentiment counts per year and month for the VADER method')
            plt.legend()
            plt.xticks(rotation=90)
            plt.savefig("./outputs/images/sent_news_vader.svg", bbox_inches='tight')
            plt.show()

            # For news and TextBlob.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_news["b_pos_count"].tolist(), label='Positive', color='green')
            plt.plot(x, result_news["b_neu_count"].tolist(), label='Neutral', color='blue')
            plt.plot(x, result_news["b_neg_count"].tolist(), label='Negative', color='red')
            plt.xlabel('Year_month')
            plt.ylabel('Sentiment counts')
            plt.title('News sentiment counts per year and month for the TextBlob method')
            plt.legend()
            plt.xticks(rotation=90)
            plt.savefig("./outputs/images/sent_news_blob.svg", bbox_inches='tight')
            plt.show()

            # For tweets and VADER.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_tweets["v_pos_count"].tolist(), label='Positive', color='green')
            plt.plot(x, result_tweets["v_neu_count"].tolist(), label='Neutral', color='blue')
            plt.plot(x, result_tweets["v_neg_count"].tolist(), label='Negative', color='red')
            plt.xlabel('Year_month')
            plt.ylabel('Sentiment counts')
            plt.title('Tweets sentiment counts per year and month for the VADER method')
            plt.legend()
            plt.xticks(rotation=90)
            plt.savefig("./outputs/images/sent_tweets_vader.svg", bbox_inches='tight')
            plt.show()

            # For tweets and TextBlob.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_tweets["b_pos_count"].tolist(), label='Positive', color='green')
            plt.plot(x, result_tweets["b_neu_count"].tolist(), label='Neutral', color='blue')
            plt.plot(x, result_tweets["b_neg_count"].tolist(), label='Negative', color='red')
            plt.xlabel('Year_month')
            plt.ylabel('Sentiment counts')
            plt.title('Tweets sentiment counts per year and month for the TextBlob method')
            plt.legend()
            plt.xticks(rotation=90)
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
            plt.xlabel('Year_month')
            plt.ylabel('Average count') # Sentence size es en número de palabras.
            plt.title('Syntactic characteristics of news per year and month')
            plt.legend()
            plt.xticks(rotation=90)
            plt.savefig("./outputs/images/synt_news.svg", bbox_inches='tight')
            plt.show()

            # For news and average size.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_news['avg_size'].tolist(), label='Average Sentence Size')
            plt.xlabel('Year_month')
            plt.ylabel('Average sentece size') # Sentence size es en número de palabras.
            plt.title('Average number of words in news per year and month')
            plt.legend()
            plt.xticks(rotation=90)
            plt.savefig("./outputs/images/synt_news_size.svg", bbox_inches='tight')
            plt.show()

            # For tweets and main elements.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_tweets['avg_noun'].tolist(), label = 'Average Nouns')
            plt.plot(x, result_tweets['avg_adj'].tolist(), label = "Average Adjectives")
            plt.plot(x, result_tweets['avg_verb'].tolist(), label = "Average Verbs")
            plt.plot(x, result_tweets['avg_sent'].tolist(), label = "Average Sentences")
            plt.xlabel('Year_month')
            plt.ylabel('Average count') # Sentence size es en número de palabras.
            plt.title('Syntactic characteristics of tweets per year and month')
            plt.legend()
            plt.xticks(rotation=90)
            plt.savefig("./outputs/images/synt_tweets.svg", bbox_inches='tight')
            plt.show()

            # For tweets and average size.
            plt.figure(figsize=(15, 10))
            plt.plot(x, result_tweets['avg_size'].tolist(), label='Average Sentence Size')
            plt.xlabel('Year_month')
            plt.ylabel('Average sentece size') # Sentence size es en número de palabras.
            plt.title('Average number of words in tweets per year and month')
            plt.legend()
            plt.xticks(rotation=90)
            plt.savefig("./outputs/images/synt_tweets_size.svg", bbox_inches='tight')
            plt.show()
        else:
            print("Invalid argument!")

    @UtilMethods.print_execution_time
    def make_freq(self, x_axes_file, y_axes_file, x_max_freq, y_max_freq,
                  sentiments, output_dir, scale=1, single_axes=True, interval=(0, -1)):

        def get_freq_window(only, word_col, freq_col):
            mask = [value in only for value in word_col]
            word_values = word_col
            word_values = word_values[mask]
            freq_values = freq_col
            freq_values = freq_values[mask]
            return word_values, freq_values

        x_ax = pd.read_csv(x_axes_file)
        y_ax = pd.read_csv(y_axes_file)
        counts = [col for col in x_ax.columns if 'counts' in col]
        words = [word for word in x_ax.columns if 'words' in word]

        if interval[1] == -1:
            x_ax = x_ax[interval[0]:]
            y_ax = y_ax[interval[0]:]
        else:
            x_ax = x_ax[interval[0]: interval[1]]
            y_ax = y_ax[interval[0]: interval[1]]

        x_ax.reset_index(drop=True, inplace=True)
        y_ax.reset_index(drop=True, inplace=True)
        for count in counts:
            sentiment = count.split("_")[0]
            x_freq = x_max_freq[sentiment]
            y_freq = y_max_freq[sentiment]
            x_ax[count] = x_ax[count] / x_freq
            y_ax[count] = y_ax[count] / y_freq

        matrix = []
        for word_col, count in zip(words, counts):
            sentiment = count.split("_")[0]

            if single_axes:
                only_on_x = [word for word in list(x_ax[word_col]) if word not in list(y_ax[word_col])]
                word_values, freq_values = get_freq_window(only_on_x, x_ax[word_col], x_ax[count])
                for w, f in zip(word_values, freq_values):
                    row = [w, f, 0, sentiments[sentiment]]
                    matrix.append(row)

                only_on_y = [word for word in list(y_ax[word_col]) if word not in list(x_ax[word_col])]
                word_values, freq_values = get_freq_window(only_on_y, y_ax[word_col], y_ax[count])
                for w, f in zip(word_values, freq_values):
                    row = [w, 0, f, sentiments[sentiment]]
                    matrix.append(row)

            both_on_x_y = [word for word in list(x_ax[word_col]) if word in list(y_ax[word_col])]
            word_values, freq_values_x = get_freq_window(both_on_x_y, x_ax[word_col], x_ax[count])
            word_values, freq_values_y = get_freq_window(both_on_x_y, y_ax[word_col], y_ax[count])
            for w, fx, fy in zip(word_values, freq_values_x, freq_values_y):
                row = [w, fx, fy, sentiments[sentiment]]
                matrix.append(row)

        frame = pd.DataFrame(data=matrix, columns=["WORDS", "X", "Y", "COLOR"])
        frame.to_csv(f'{output_dir}', index=False)

    @UtilMethods.print_execution_time
    def plot_test2(self, input_file, save_dir, classifier, marker, size=(10, 10),
                   ys_lims=(-0.0005, 0.0005, 0.0050, 0.0068, 0.0068, 0.014),
                   xs_lims=(-0.0005, 0.0065, 0.0066, 0.024), font_size=6):

        def putLegends(axis, classifier, marker):
            feelings = {"positive": 'green', "neutral": "blue", "negative": "red"}
            for name, Color in feelings.items():
                axis.plot(np.NaN, np.NaN, c=Color, label=name, marker='o', linestyle='None')
            axis.legend(loc=1, title="Feelings")

        def plot_limits(frame, x_limit, y_limit, ax, vertical):

            texts = []
            for word, x, y, color in frame.values:
                if y_limit[0] < y <= y_limit[1] and x_limit[0] < x <= x_limit[1]:
                    ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                    chido = ax.text(x, y, word, fontsize=font_size)
                    texts.append(chido)

            # if vertical:
            #     adjust_text(texts, precision=0.000001, ha='center', va='bottom', ax=ax,
            #                 force_points=0.02,
            #                 arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

            ax.set_ylim(y_limit[0], y_limit[1])
            ax.set_xlim(x_limit[0], x_limit[1])

        frame = pd.read_csv(input_file)
        fig, ((ax_00, ax_01),
              (ax_10, ax_11),
              (ax_20, ax_21)) = plt.subplots(3, 2, figsize=size)
        fig.subplots_adjust(hspace=0.05)  # adjust space between axes

        xlim = 0
        ylim = 1
        vertical_text = 2

        yl1, yl2, yl3, yl4, yl5, yl6 = ys_lims
        xl1, xl2, xl3, xl4 = xs_lims

        sq_00 = ((xl1, xl2), (yl5, yl6), False)
        sq_01 = ((xl3, xl4), (yl5, yl6), False)
        sq_10 = ((xl1, xl2), (yl3, yl4), False)
        sq_11 = ((xl3, xl4), (yl3, yl4), False)
        sq_20 = ((xl1, xl2), (yl1, yl2), True)
        sq_21 = ((xl3, xl4), (yl1, yl2), True)

        axes = [ax_00, ax_01, ax_10, ax_11, ax_20, ax_21]
        squares = [sq_00, sq_01, sq_10, sq_11, sq_20, sq_21]

        for axis, sq in zip(axes, squares):
            plot_limits(frame, sq[xlim], sq[ylim], axis, sq[vertical_text])

        ax_20.spines['top'].set_visible(False)
        ax_20.spines['right'].set_visible(False)

        ax_21.spines['top'].set_visible(False)
        ax_21.spines['left'].set_visible(False)
        ax_21.tick_params(left=False)
        ax_21.tick_params(labelleft=False)

        ax_10.spines['top'].set_visible(False)
        ax_10.spines['bottom'].set_visible(False)
        ax_10.spines['right'].set_visible(False)
        ax_10.tick_params(bottom=False)
        ax_10.tick_params(labelbottom=False)

        ax_11.spines['top'].set_visible(False)
        ax_11.spines['bottom'].set_visible(False)
        ax_11.spines['left'].set_visible(False)
        ax_11.tick_params(bottom=False, left=False)
        ax_11.tick_params(labelbottom=False, labelleft=False)

        ax_00.spines['bottom'].set_visible(False)
        ax_00.spines['right'].set_visible(False)
        ax_00.tick_params(bottom=False)
        ax_00.tick_params(labelbottom=False)

        ax_01.spines['bottom'].set_visible(False)
        ax_01.spines['left'].set_visible(False)
        ax_01.tick_params(bottom=False, left=False)
        ax_01.tick_params(labelbottom=False, labelleft=False)
        putLegends(ax_01, classifier, marker)

        d = .8  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)

        ax_20.plot([1, 0], transform=ax_20.transAxes, **kwargs)
        ax_21.plot([0, 1], transform=ax_21.transAxes, **kwargs)
        ax_10.plot([0, 0], [1, 0], transform=ax_10.transAxes, **kwargs)
        ax_11.plot([1, 1], [0, 1], transform=ax_11.transAxes, **kwargs)
        ax_00.plot([0, 1], transform=ax_00.transAxes, **kwargs)
        ax_01.plot([1, 0], transform=ax_01.transAxes, **kwargs)

        plt.suptitle(f'Most used words in tweets and news by sentiment using {classifier}')
        ax_10.set_ylabel("freq on news")

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("freq on tweets")
        plt.show()
        # save_path = f'{save_dir}/{classifier}_summarize.svg'
        # plt.savefig(save_path, bbox_inches='tight')

    @UtilMethods.print_execution_time
    def plot_test_base(self, input_file, save_dir, classifier, marker, size=(10, 10)):

        def putLegends(ax1, classifier, marker):
            feelings = {"positive": 'green', "neutral": "blue", "negative": "red"}
            for name, Color in feelings.items():
                ax1.plot(np.NaN, np.NaN, c=Color, label=name, marker='o', linestyle='None')
            ax1.legend(loc=1, title="Feelings")

        frame = pd.read_csv(input_file)
        fig, ax_1 = plt.subplots(figsize=size)
        font_size = 8
        for word, x, y, color in frame.values:
            ax_1.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
            ax_1.text(x, y, word, fontsize=font_size)
        putLegends(ax_1, classifier, marker)
        plt.suptitle(f'Most used words in tweets and news by sentiment using {classifier}')
        ax_1.set_ylabel("freq on news")
        ax_1.set_xlabel("freq on twitter")
        #save_path = f'{save_dir}/{classifier}_summarize_base.svg'
        #plt.savefig(save_path, bbox_inches='tight')

        plt.show()
