import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unidecode
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from core.utils import UtilMethods
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text



class Sentiment:
    def __init__(self, input_data, output_folder, lower=-0.1, upper=0.1, labels=("neg", "neu", "pos")):
        self.labels = labels
        self.input_path = input_data
        self.save_path = output_folder + "/" + input_data.split("/")[-1].split(".csv")[0] + "_sentiment.csv"
        self.vader_analyser = SentimentIntensityAnalyzer()
        self.lower = lower
        self.upper = upper

    def vader_scores(self, sentence):
        score = self.vader_analyser.polarity_scores(sentence)
        score = score['compound']
        category = self.score_to_categorical(score)
        return score, category

    def blob_score(self, sentence):
        score = TextBlob(sentence)
        score = score.sentiment.polarity
        category = self.score_to_categorical(score)
        return score, category

    def score_to_categorical(self, score):
        if score <= self.lower:
            return self.labels[0]
        elif self.lower < score <= self.upper:
            return self.labels[1]
        elif self.upper < score <= 1:
            return self.labels[2]
        else:
            return "error"

    def get_line(self, sentence):
        blob_score, blob_cat = self.blob_score(sentence)
        vader_score, vader_cat = self.vader_scores(sentence)
        return [blob_score, vader_score, blob_cat, vader_cat]

    def extract_sentiment_features(self, text_col):

        frame = pd.read_csv(self.input_path)
        feature_matrix = []
        for text in tqdm(frame[text_col], desc="extracting sentiments ..."):
            feature_matrix.append(self.get_line(text))

        syntactic_frame = pd.DataFrame(data=feature_matrix, columns=["blob score", "vader score",
                                                                     "blob category", "vader category"])
        frame = pd.concat([frame, syntactic_frame], axis=1)
        frame.to_csv(self.save_path, index=False)


class CriticalWords:

    def __init__(self, input_data, output_folder, text_column, class_column, method_name, n_words=1000):

        self.input_path = input_data
        self.save_words = f'{output_folder}/critical words/{output_folder.split("/")[-1]}_{method_name}_words.csv'
        self.save_counts = f'{output_folder}/counts/{output_folder.split("/")[-1]}_{method_name}_counts.csv'

        self.frame = pd.read_csv(self.input_path)
        self.text_column = text_column
        self.class_column = class_column
        self.method_name = method_name
        self.n_words = n_words

    @staticmethod
    def clean_sentence(sentence):
        sentence = sentence.lower()
        stopwords_english = stopwords.words('english')
        sentence = re.sub(r'\$\w*', '', sentence)
        sentence = re.sub(r'^RT[\s]+', '', sentence)
        sentence = re.sub(r'https?:.*[\r\n]*', '', sentence)
        sentence = re.sub(r'#', '', sentence)
        tokens = word_tokenize(text=sentence)
        sentence_clean = []
        lemming = WordNetLemmatizer()
        for word in tokens:
            if (word not in stopwords_english and  # remove stopwords
                    word not in string.punctuation and word != 'rt'
                    and word != '...' and not word.isdigit()
                    and word.isalpha()):
                stem_word = lemming.lemmatize(word, pos="v")
                sentence_clean.append(unidecode.unidecode(stem_word))

        bad_words = ['amp', 'dont', 'doesnt', 'us', 'shes']
        sentence_clean = [word for word in sentence_clean if word not in bad_words]
        return sentence_clean

    def generate_list_of_words(self, sentences):
        words = []
        for sentence in tqdm(sentences, desc="cleaning sentence"):
            clean = self.clean_sentence(sentence)
            words.append(clean)
        words = sum(words, [])
        return words

    def extract_most_common_words(self, words, sentiment):
        word_freq = FreqDist(words)
        print("for the sentiment", sentiment)
        print("there are", len(word_freq.keys()), "different words")
        print("that were used", sum(word_freq.values()), "times")
        df = pd.DataFrame(
            {f'{sentiment}_words': list(word_freq.keys()), f'{sentiment}_counts': list(word_freq.values())})
        df = df.nlargest(self.n_words, columns=f'{sentiment}_counts')
        df.reset_index(drop=True, inplace=True)
        return df, len(word_freq.keys()), sum(word_freq.values())

    def extract_words_by_sentiment(self, sentiment):
        sentiment_words = self.generate_list_of_words(self.frame[self.text_column]
                                                      [self.frame[self.class_column] == sentiment])
        sentiment_frame, n_words, n_freq = self.extract_most_common_words(sentiment_words, sentiment)
        return sentiment_frame, n_words, n_freq

    def extract_critical_words(self):

        frame_base = pd.DataFrame({})
        n_words, n_freqs, sentiments = [], [], []
        for sentiment in ['pos', 'neu', 'neg']:
            sentiment_frame, n_word, n_freq = self.extract_words_by_sentiment(sentiment)
            n_words.append(n_word)
            n_freqs.append(n_freq)
            sentiments.append(sentiment)
            frame_base = pd.concat([frame_base, sentiment_frame], axis=1)
        print(frame_base)
        counts = pd.DataFrame({"sentiment": sentiments, "words": n_words, "freq": n_freqs})
        s_words = sum(counts['words'])
        counts['% words'] = counts['words'] / s_words * 100
        s_freq = sum(counts['freq'])
        counts['% freq'] = counts['freq'] / s_freq * 100
        counts.to_csv(self.save_counts, index=False)
        frame_base.to_csv(self.save_words, index=False)


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

            # if vertical:
            #     adjust_text(texts, precision=0.0001, ax=ax, lim=1500, force_text=(10, 10),
            #                 force_points=(15, 15), only_move={'points': 'xy', 'text': 'y', 'objects': 'xy'},
            #                 arrowprops=dict(arrowstyle='->', color='red'))
            # else:
            # adjust_text(texts, precision=0.0001, ax=ax, lim=1500, force_points=(5, 5), force_text=(3, 3),
            #             arrowprops=dict(arrowstyle='->', color='red'))

            ax.set_ylim(y_limit[0], y_limit[1])
            ax.set_xlim(x_limit[0], x_limit[1])

        # frame = pd.read_csv(input_file)
        # print(frame)
        # fig, ax = plt.subplots(figsize=size)
        # texts = []
        # for word, x, y, color in frame.values:
        #     ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
        #     chido = ax.text(x, y, word, fontsize=font_size)
        #     texts.append(chido)
        #
        # plt.suptitle(f'Most used words in tweets and news by sentiment using {classifier}', fontsize=20)
        # ax.set_ylabel("Frequency on news", fontsize=16)
        # ax.set_xlabel("Frequency on tweets", fontsize=16)
        # plt.show()

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
            print(frame)
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


def runner():
    # analyzer = Sentiment("../outputs/twitter_raw/tweets_hydrate_syntactic.csv",
    #                      "../outputs/sentiment_analysis/data")
    # analyzer.extract_sentiment_features(text_col="tweet")
    # analyzer = Sentiment("../outputs/news/pretty_news_dataset_syntactic.csv",
    #                      "../outputs/sentiment_analysis/data")
    # analyzer.extract_sentiment_features(text_col="Snippet")

    critical = CriticalWords(input_data=f"../outputs/sentiment_analysis/data/"
                                        f"pretty_news_dataset_syntactic_sentiment.csv",
                             output_folder="../outputs/sentiment_analysis/words/news",
                             text_column="Snippet",
                             class_column="blob category",
                             method_name="textBlob",
                             n_words=1000)
    critical.extract_critical_words()

    critical = CriticalWords(input_data=f"../outputs/sentiment_analysis/data/"
                                        f"pretty_news_dataset_syntactic_sentiment.csv",
                             output_folder="../outputs/sentiment_analysis/words/news",
                             text_column="Snippet",
                             class_column="vader category",
                             method_name="vader",
                             n_words=1000)
    critical.extract_critical_words()

    critical = CriticalWords(input_data=f"../outputs/sentiment_analysis/data/"
                                        f"tweets_hydrate_syntactic_sentiment.csv",
                             output_folder="../outputs/sentiment_analysis/words/tweets",
                             text_column="tweet",
                             class_column="blob category",
                             method_name="textBlob",
                             n_words=1000)
    critical.extract_critical_words()
    critical = CriticalWords(input_data=f"../outputs/sentiment_analysis/data/"
                                        f"tweets_hydrate_syntactic_sentiment.csv",
                             output_folder="../outputs/sentiment_analysis/words/tweets",
                             text_column="tweet",
                             class_column="vader category",
                             method_name="vader",
                             n_words=1000)
    critical.extract_critical_words()
    ss = SummarySentiments()
    color_map = {'pos': 'green', 'neu': 'blue', 'neg': 'red'}
    ss.make_freq(x_axes_file="../outputs/sentiment_analysis/words/tweets/critical words/tweets_vader_words.csv",
                 y_axes_file="../outputs/sentiment_analysis/words/news/critical words/news_vader_words.csv",
                 x_max_freq="../outputs/sentiment_analysis/words/tweets/counts/tweets_vader_counts.csv",
                 y_max_freq="../outputs/sentiment_analysis/words/news/counts/news_vader_counts.csv",
                 sentiments=color_map,
                 output_dir="../outputs/sentiment_analysis/words/freq distribution/vader_results2.csv",
                 start=0,
                 window=2)
    ss.make_freq(x_axes_file="../outputs/sentiment_analysis/words/tweets/critical words/tweets_textBlob_words.csv",
                 y_axes_file="../outputs/sentiment_analysis/words/news/critical words/news_textBlob_words.csv",
                 x_max_freq="../outputs/sentiment_analysis/words/tweets/counts/tweets_textBlob_counts.csv",
                 y_max_freq="../outputs/sentiment_analysis/words/news/counts/news_textBlob_counts.csv",
                 sentiments=color_map,
                 output_dir="../outputs/sentiment_analysis/words/freq distribution/textBlob_results2.csv",
                 start=0,
                 window=2)
    ss.plot_sentiment_summary(input_file="../outputs/sentiment_analysis/words/freq distribution/textBlob_results.csv",
                              save_dir="../outputs/sentiment_analysis/images",
                              classifier='TextBlob',
                              marker='*',
                              size=(15, 10),
                              font_size=12,
                              ys_lims=(-0.05, 0.2, 0.2, 0.6, 0.6, 1.45),
                              xs_lims=(-0.05, 0.7, 0.7, 3.0)
                              )
    ss.plot_sentiment_summary(input_file="../outputs/sentiment_analysis/words/freq distribution/vader_results.csv",
                              save_dir="../outputs/sentiment_analysis/images",
                              classifier='Vader',
                              marker='*',
                              size=(15, 10),
                              font_size=12,
                              ys_lims=(-0.05, 0.4, 0.4, 0.8, 0.8, 1.6),
                              xs_lims=(-0.05, 1, 1.3, 3.5)
                              )
    # ss.plot_mini(textBlob="../outputs/sentiment_analysis/words/freq distribution/textBlob_results2.csv",
    #              vader="../outputs/sentiment_analysis/words/freq distribution/vader_results2.csv",
    #              save_dir="../outputs/sentiment_analysis/images/final.svg",
    #              size=(10, 5))


if __name__ == '__main__':
    runner()
