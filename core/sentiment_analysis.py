import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


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
