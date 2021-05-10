# Imports.
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
from core.utils import UtilMethods
import numpy as np
from adjustText import adjust_text

###############################################################################
# Class that makes a plot that summarize all the syntactic analysis results   #
#                                                                             #
# @author Jorge Ciprián                                                  #
# Last updated: 05-05-2021.                                                   #
###############################################################################

class SummarySyntactic(UtilMethods):

    def __init__(self):
        ...

    # Method for plotting the timelines for sentiment and syntactic analysis.
    @staticmethod
    def plot_timelines(mode):
        # First, we load the datasets for news and tweets.
        print("Loading datasets...")
        tweets_df = pd.read_csv("outputs/clean datasets/Twitter/Twitter_syntactic_sentiment.csv")
        news_df = pd.read_csv("outputs/clean datasets/News/News_syntactic_sentiment.csv")
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
            plt.savefig("outputs/Syntactic Analysis/images/sent_news_vader.svg", bbox_inches='tight')
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
            plt.savefig("outputs/Syntactic Analysis/images/sent_news_blob.svg", bbox_inches='tight')
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
            plt.savefig("outputs/Syntactic Analysis/images/sent_tweets_vader.svg", bbox_inches='tight')
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
            plt.savefig("outputs/Syntactic Analysis/images/sent_tweets_blob.svg", bbox_inches='tight')
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
            plt.savefig("outputs/Syntactic Analysis/images/synt_news.svg", bbox_inches='tight')
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
            plt.savefig("outputs/Syntactic Analysis/images/synt_news_size.svg", bbox_inches='tight')
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
            plt.savefig("outputs/Syntactic Analysis/images/synt_tweets.svg", bbox_inches='tight')
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
            plt.savefig("outputs/Syntactic Analysis/images/synt_tweets_size.svg", bbox_inches='tight')
            plt.show()
        else:
            print("Invalid argument!")


###############################################################################
# Class that makes a plot that summarize all the sentiments analysis results  #
#                                                                             #
# @author Gabriel Ichcanziho                                                  #
# Last updated: 05-05-2021.                                                   #
###############################################################################

class SummarySentiments(UtilMethods):
    @UtilMethods.print_execution_time
    def make_freq(self, x_axes_file, y_axes_file, x_max_freq, y_max_freq,
                  sentiments, output_dir, scale=100, start=2, window=20):

        def get_freq_window(only, word_col, freq_col):
            mask = [value in only for value in word_col]
            word_values = word_col
            word_values = word_values[mask]
            freq_values = freq_col
            freq_values = freq_values[mask]
            return word_values, freq_values

        x_ax = pd.read_csv(x_axes_file)
        y_ax = pd.read_csv(y_axes_file)
        x_freq_max = pd.read_csv(x_max_freq)
        x_freq_max = list(x_freq_max['freq'])
        y_freq_max = pd.read_csv(y_max_freq)
        y_freq_max = list(y_freq_max['freq'])
        counts_columns = ['pos_counts', 'neu_counts', 'neg_counts']
        words_columns = ['pos_words', 'neu_words', 'neg_words']

        for j, column in enumerate(counts_columns):
            x_ax[column] = (x_ax[column] / x_freq_max[j]) * scale
            y_ax[column] = (y_ax[column] / y_freq_max[j]) * scale


        # splitting only the first n most common words
        window_x = x_ax[start:start + window]
        window_y = y_ax[start:start + window]
        matrix = []
        for word, count in zip(words_columns, counts_columns):
            sentiment = count.split("_")[0]

            current_xs = list(window_x[word])  # first 20 words that belongs to X axis
            current_ys = list(window_y[word])  # first 20 words that belongs to Y axis

            total_ys = list(y_ax[word])  # first 1000 words that belongs to X axis
            total_xs = list(x_ax[word])  # first 1000 words that belongs to Y axis

            # from the first 20 words that belongs to X, look for them in the 1000 Y words
            both_on_x_y = [w for w in current_xs if w in total_ys]

            word_values, freq_values_x = get_freq_window(both_on_x_y, x_ax[word], x_ax[count])
            word_values, freq_values_y = get_freq_window(both_on_x_y, y_ax[word], y_ax[count])
            for w, fx, fy in zip(word_values, freq_values_x, freq_values_y):
                row = [w, fx, fy, sentiments[sentiment]]
                matrix.append(row)

            # from the first 20 words that belongs to X, look for the words not present in the 1000 Y words
            only_on_x = [w for w in current_xs if w not in total_ys]

            word_values_x, freq_values_x = get_freq_window(only_on_x, x_ax[word], x_ax[count])
            for w, fx in zip(word_values_x, freq_values_x):
                row = [w, fx, 0, sentiments[sentiment]]
                matrix.append(row)

            # from the first 20 words that belongs to Y, look for the words not present in the 1000 X words
            only_on_y = [w for w in current_ys if w not in total_xs]

            word_values_y, freq_values_y = get_freq_window(only_on_y, y_ax[word], y_ax[count])
            for w, fy in zip(word_values_y, freq_values_y):
                row = [w, 0, fy, sentiments[sentiment]]
                matrix.append(row)

        frame = pd.DataFrame(data=matrix, columns=["WORDS", "X", "Y", "COLOR"])
        frame.to_csv(f'{output_dir}', index=False)

    @UtilMethods.print_execution_time
    def plot_sentiment_summary(self, input_file, save_dir, classifier, marker, size=(10, 10),
                               ys_lims=(-0.05, 0.05, 0.50, 0.68, 0.68, 1.4),
                               xs_lims=(-0.05, 0.65, 0.66, 2.4), font_size=6):

        def putLegends(axis, classifier, marker):
            feelings = {"positive": 'green', "neutral": "blue", "negative": "red"}
            for name, Color in feelings.items():
                axis.plot(np.NaN, np.NaN, c=Color, label=name, marker='o', linestyle='None')
            axis.legend(loc=1, title="Feelings")

        def plot_limits(frame, x_limit, y_limit, ax, vertical):

            texts = []
            for word, x, y, color in frame.values:
                if y_limit[0] < y <= y_limit[1] and x_limit[0] < x <= x_limit[1]:
                    if word == 'warm' and color == "blue" and classifier == "TextBlob":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x-0.05, y, word, fontsize=font_size)
                        print("entre")
                    elif word == 'world' and color == "blue" and classifier == "TextBlob":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x, y-0.02, word, fontsize=font_size)
                        print("entre")
                    elif word == 'great' and color == "green" and classifier == "TextBlob":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x-0.05, y-0.02, word, fontsize=font_size)
                        print("entre")
                    elif word == 'global' and color == "green" and classifier == "TextBlob":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x, y+0.008, word, fontsize=font_size)
                        print("entre")
                    elif word == 'warm' and color == "red" and classifier == "TextBlob":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x, y-0.02, word, fontsize=font_size)
                        print("entre")
                    elif word == 'fight' and color == "green" and classifier == "TextBlob":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x+0.005, y-0.005, word, fontsize=font_size)
                        print("entre")
                    elif word == 'warm' and color == "green" and classifier == "Vader":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x-0.07, y, word, fontsize=font_size)
                        print("entre")
                    elif word == 'real' and color == "blue" and classifier == "Vader":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x-0.04, y, word, fontsize=font_size)
                        print("entre")
                    elif word == 'real' and color == "green" and classifier == "Vader":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x-0.04, y, word, fontsize=font_size)
                        print("entre")
                    elif word == 'new' and color == "blue" and classifier == "Vader":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x, y-0.02, word, fontsize=font_size)
                        print("entre")
                    elif word == 'world' and color == "blue" and classifier == "Vader":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x-0.07, y+0.008, word, fontsize=font_size)
                        print("entre")
                    elif word == 'global' and color == "green" and classifier == "Vader":
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x, y-0.005, word, fontsize=font_size)
                        print("entre")
                    else:
                        ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
                        chido = ax.text(x, y, word, fontsize=font_size)
                    texts.append(chido)

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

        plt.suptitle(f'Most used words in tweets and news by sentiment using {classifier}', fontsize=20)
        ax_10.set_ylabel("Frequency on news", fontsize=16)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Frequency on tweets", fontsize=16)
        save_path = f'{save_dir}/{classifier}_summarize_pretty2.svg'
        plt.savefig(save_path, bbox_inches='tight', dpi=400)
        plt.show()

    @UtilMethods.print_execution_time
    def plot_mini(self, textBlob, vader, save_dir, size=(10, 10), f_size=12):
        # import seaborn as sns
        # sns.set()

        def putLegends(axis, location=1):
            feelings = {"positive": 'green', "neutral": "blue", "negative": "red"}
            for name, Color in feelings.items():
                axis.plot(np.NaN, np.NaN, c=Color, label=name, marker='o', linestyle='None')
            axis.legend(loc=location, title="Feelings")

        def draw_axes(drawing_ax, input_file, f_size):

            frame = pd.read_csv(input_file)
            texts = []
            for word, x, y, color in frame.values:
                drawing_ax.scatter(x, y, s=50, c=color, alpha=0.5, marker="*")
                t = drawing_ax.text(x, y, word, fontsize=f_size)
                texts.append(t)

            adjust_text(texts, precision=0.0001, ax=drawing_ax, lim=1500, force_points=(5, 5), force_text=(3, 3))

        fig, (ax_00, ax_01) = plt.subplots(1, 2, sharey=True, figsize=size)
        fig.suptitle("Most used words", fontsize=20)

        draw_axes(ax_00, textBlob, f_size)
        draw_axes(ax_01, vader, f_size)
        putLegends(ax_00, location=1)
        putLegends(ax_01, location=2)

        ax_00.set_title("TexBlob Analyzer")
        ax_01.set_title("VADER Analyzer")
        ax_00.set_ylabel("Frequency on news", fontsize=16)
        ax_01.tick_params(left=False)
        ax_00.set_xlabel("Frequency on tweets \n (a) TextBlob", fontsize=16)
        ax_01.set_xlabel("Frequency on tweets \n (b) VADER", fontsize=16)
        plt.savefig(save_dir, bbox_inches='tight', dpi=400)
        plt.show()
