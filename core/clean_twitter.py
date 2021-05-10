###############################################################################
# Class that recover the date of a tweet base on their tweet id, it uses      #
# TWARC library to hydratate the tweets ids.                                  #
# see https://github.com/DocNow/twarc for more details on how to use TWARC    #
#                                                                             #
# @author Gabriel Ichcanziho                                                  #
# Last updated: 05-05-2021.                                                   #
###############################################################################

from core import UtilMethods
import pandas as pd
from twarc import Twarc
from datetime import datetime
from unidecode import unidecode
from tqdm import tqdm
import os


class TwitterCleaner(UtilMethods):

    def __init__(self, input_path="inputs", output_path="outputs"):
        self.id_col = "tweetid"
        self.ids_txt = "ids"
        self.hydrate_name = "Twitter"
        self.input_path = f'{input_path}'
        self.output_path = f'{output_path}'
        self.create_folder(path=self.output_path)
        self.df_name = self.get_files(self.input_path)[0]
        self.data_frame = pd.read_csv(f'{self.input_path}/{self.df_name}')

    @UtilMethods.print_execution_time
    def save_ids(self):
        with open(f'{self.output_path}/{self.ids_txt}.txt', 'a') as f:
            f.write(self.data_frame[[self.id_col]].to_string(header=False, index=False))

    @staticmethod
    def filter_tweet(tweet):

        text = tweet["full_text"]
        id = tweet["id"]
        new_datetime = datetime.strftime(datetime.strptime(tweet["created_at"], '%a %b %d %H:%M:%S +0000 %Y'),
                                         '%Y-%m-%d')
        year = new_datetime.split("-")[0]
        month = new_datetime.split("-")[1]
        day = new_datetime.split("-")[2]
        try:
            text = unidecode(text)
        except UnicodeDecodeError:
            text = text
        text = text.replace('\n', ' ').replace('\r', '').replace("'", "").replace('"', "").replace(",", " ")
        output = f'{id},{year},{month},{day},{text}'
        return output

    @UtilMethods.print_execution_time
    def clean_tweets(self):
        self.save_ids()
        t = Twarc()
        file = open(f'{self.output_path}/{self.hydrate_name}.csv', 'w', encoding="ISO-8859-1")
        file.write("i,id,year,month,day,tweet\n")
        i = 1
        for tweet in tqdm(t.hydrate(open(f'{self.output_path}/{self.ids_txt}.txt')), desc="parsing tweet ..."):
            try:
                output = self.filter_tweet(tweet)
            except UnicodeDecodeError:
                output = "error,error,error,error,error"
            file.write(f'{i},{output}\n')
            i += 1
        file.close()
        os.remove(f'{self.output_path}/{self.ids_txt}.txt')
