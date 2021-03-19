import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unidecode
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import pandas as pd
from tqdm import tqdm


class CriticalWords:

    def __init__(self, input_data, output_folder, text_column, class_column, method_name, n_words=20):

        self.input_path = input_data
        self.save_path = output_folder + "/" + input_data.split("/")[-1].split(".csv")[0].split("_")[0]
        self.save_path += f"_{method_name}_words.csv"
        print(self.save_path)
        self.frame = pd.read_csv(self.input_path)
        self.text_column = text_column
        self.class_column = class_column
        self.n_words = n_words
        self.method_name = method_name

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
        return df

    def extract_words_by_sentiment(self, sentiment):
        sentiment_words = self.generate_list_of_words(self.frame[self.text_column]
                                                      [self.frame[self.class_column] == sentiment])
        sentiment_frame = self.extract_most_common_words(sentiment_words, sentiment)
        return sentiment_frame

    def extract_critical_words(self):

        frame_base = pd.DataFrame({})
        for sentiment in self.frame[self.class_column].unique():
            sentiment_frame = self.extract_words_by_sentiment(sentiment)
            frame_base = pd.concat([frame_base, sentiment_frame], axis=1)
        print(frame_base)
        frame_base.to_csv(self.save_path, index=False)
