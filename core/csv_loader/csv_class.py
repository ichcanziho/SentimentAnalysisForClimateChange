import pandas as pd


class CsvLoad:
    def __init__(self, input_path):
        self.input_path = input_path

    def load(self):
        return pd.read_csv(self.input_path)

