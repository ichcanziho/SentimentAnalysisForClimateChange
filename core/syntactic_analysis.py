import spacy
import pandas as pd
from tqdm import tqdm


class Syntactic:
    def __init__(self, input_data, output_folder):
        # python -m spacy download en_core_web_lg
        self.NLP = spacy.load("en_core_web_lg")
        self.input_path = input_data
        self.save_path = output_folder + "/" + input_data.split("/")[-1].split(".csv")[0] + "_syntactic.csv"

    def get_line(self, text, features):
        doc = self.NLP(text)
        features = {key: 0 for key in features}
        for token in doc:
            if token.pos_ in features.keys():
                features[token.pos_] += 1
            if token.is_alpha and 'SIZE' in features.keys():
                features['SIZE'] += 1
        if 'SENT' in features.keys():
            features['SENT'] = len(list(doc.sents))
        output = [int(val) for val in features.values()]
        return output

    def extract_syntactic_features(self, text_col, features=('NOUN', 'ADJ', 'VERB', 'PROPN', 'SENT', 'SIZE')):

        frame = pd.read_csv(self.input_path)
        feature_matrix = []
        for text in tqdm(frame[text_col], desc="exctacting features ..."):
            feature_matrix.append(self.get_line(text, features))

        syntactic_frame = pd.DataFrame(data=feature_matrix, columns=features)
        frame = pd.concat([frame, syntactic_frame], axis=1)
        frame.to_csv(self.save_path, index=False)
