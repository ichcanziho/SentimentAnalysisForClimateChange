from core import CsvLoad


def run(path):
    loader = CsvLoad(path)
    frame = loader.load()
    print(frame)
