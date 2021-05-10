###############################################################################
# Class that extract the sentiment of a given text, it uses two sentiment     #
# analyzers                                                                   #
# TextBlob: https://github.com/sloria/TextBlob                                #
# VADER: https://github.com/cjhutto/vaderSentiment                            #
#                                                                             #
# @author Gabriel Ichcanziho                                                  #
# Last updated: 05-05-2021.                                                   #
###############################################################################


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

###############################################################################
# Class that extract the most used words by sentiment in a given dataset      #
# the dataset must be tagged with a sentiment before using this class         #
# The available sentiments are: ["pos", "neg", neu"] it returns a CSV         #
# with the words most used and their frequency by each sentiment              #
#                                                                             #
# @author Gabriel Ichcanziho                                                  #
# Last updated: 05-05-2021.                                                   #
###############################################################################

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
