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
    source_path = "./inputs/datasets/TelevisionNews/"
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
    # Filtering by the required dates.
    full_df['MatchDateTime'] = pd.to_datetime(full_df['MatchDateTime'])
    start_date = '2015-04-27'
    end_date = '2018-02-21'
    mask = (full_df['MatchDateTime'] >= start_date) & (full_df['MatchDateTime'] <= end_date)
    dates_df = full_df.loc[mask]
    print(dates_df)
    rows, cols = dates_df.shape
    print("Rows: ", rows)
    print("Columns: ", cols)
    filtered_df = dates_df.copy()
    # Removing BBC entries.
    mask_show = (filtered_df['Station'] != "BBCNEWS")
    filtered_df = filtered_df.loc[mask_show]
    # Filtering by keyword "climate change".
    filtered_df['Snippet'] = filtered_df['Snippet'].str.lower()
    filtered_df = filtered_df[filtered_df['Snippet'].str.contains("climate change")]
    print(filtered_df)
    rows, cols = filtered_df.shape
    print("Rows: ", rows)
    print("Columns: ", cols)
    # # Removing unnecessary columns.
    final_df = filtered_df.drop(['URL', 'IAShowID','IAPreviewThumb'], 1)
    # Formatting the day, month, and year columns.
    final_df = pd.concat([final_df.drop('MatchDateTime', axis = 1),
                        (final_df.MatchDateTime.dt.strftime('%Y-%m-%d').str.split("-| ").str[:3].apply(pd.Series).rename(columns={0:'year', 1:'month', 2:'day'}))],
                        axis = 1)
    print("Final dataframe:")
    #full_df.head()
    print(final_df)
    rows, cols = final_df.shape
    print("Rows: ", rows)
    print("Columns: ", cols)
    # # Saving final CSV.
    save_path = "./outputs/news/pretty_news_dataset.csv"
    final_df.to_csv(save_path, index=False)
    print("Done!")

