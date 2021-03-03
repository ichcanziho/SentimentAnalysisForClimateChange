###############################################################################
# Function that merges all the news CSV, pre-process the merged dataframe and #
# generates the day, month, and year fields. The function also removes        #
# unecessary columns and saves the final dataframe as a CSV file.             #
#                                                                             #
# @author Jorge Ciprián                                                       #
# Last updated: 02-03-2021.                                                   #
###############################################################################

# Imports.
import pandas as pd
from os import listdir
from os.path import isfile, join

def generate_news_csv():
    # Defining source path.
    source_path = "./datasets/TelevisionNews/"
    # Generating file_list.
    file_list = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    # Verifying that only CSV files are saved.
    csv_list = [item for item in file_list if 'csv' in item]
    #print(csv_list)
    dataframe_list = []
    #count = 0
    for file in csv_list:
        full_path = source_path + file
        df = pd.read_csv(full_path)
        dataframe_list.append(df)
        print("Loaded: ", full_path)
    # Merging all read CSVs.
    full_df = pd.concat(dataframe_list, axis=0, ignore_index=True)
    # Displaying snippet of dataframe.
    print(full_df)
    # Getting number of rows and columns.
    rows, cols = full_df.shape
    print("Rows: ", rows)
    print("Columns: ", cols)
    # Removing unnecessary columns.
    full_df = full_df.drop(['URL', 'IAShowID','IAPreviewThumb'], 1)
    # Formatting the day, month, and year columns.
    full_df = pd.concat([full_df.drop('MatchDateTime', axis = 1),
                        (full_df.MatchDateTime.str.split("/| ").str[:3].apply(pd.Series).rename(columns={0:'month', 1:'day', 2:'year'}))],
                        axis = 1)
    print("Final dataframe:")
    #full_df.head()
    print(full_df)
    # Saving final CSV.
    save_path = "./datasets/full_news_dataset.csv"
    full_df.to_csv(save_path, index=False)
    print("Done!")
