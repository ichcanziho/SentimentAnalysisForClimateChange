###############################################################################
# Functions to perform Spearman N-lag cross correlation of the sentiment time #
# lines. Saves the obtained correlation matrices as SVG files. Also obtains   #
# the Granger causality between the time series.                              #
#                                                                             #
# @author Jorge Ciprián                                                       #
# Last updated: 21-05-2021.                                                   #
###############################################################################

# Imports.
import numpy as np
import pandas as pd
import seaborn as sns
import pandasql as ps
from pylab import rcParams
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller, zivot_andrews

# Main class. Contains the methods for N-lag Spearman cross-correlation and
# Granger causality.
class CorrCausAnalyzer():
    def __init__(self, in_path_tweets, in_path_news, out_path):
        # Assigning attributes.
        self.in_dir_tweets = in_path_tweets
        self.in_dir_news = in_path_news
        self.out_dir = out_path

    # Method for loading the datasets and doing the required pre-processing
    # for correlation and causality analysis.
    def load_datasets(self):
        print("Loading datasets...")
        # Getting source paths.
        news_source_path = self.in_dir_news
        tweets_source_dir = self.in_dir_tweets
        # Loading dataframes.
        news_df = pd.read_csv(news_source_path)
        tweets_df = pd.read_csv(tweets_source_dir)
        print("... done.")

        # For the news dataset.
        print("Pre-processing news...")
        # Dropping unnecessary columns.
        news_df = news_df.drop(['Show','Snippet','NOUN','ADJ','VERB','PROPN',
                                'SENT','SIZE','blob score', 'vader score'],1)
        # Defining query for generating the time series.
        query_news = """SELECT     year, month,
                      SUM(CASE WHEN `vader category`='pos' THEN 1 ELSE 0 END) AS pos_count,
                      SUM(CASE WHEN `vader category`='neg' THEN 1 ELSE 0 END) AS neg_count,
                      SUM(CASE WHEN `vader category`='neu' THEN 1 ELSE 0 END) AS neu_count
           FROM       news_df
           GROUP BY   year, month;"""
        # Applying query and getting results.
        result_news = ps.sqldf(query_news, locals())
        # Generating date index.
        result_news["date"] = pd.to_datetime(result_news['year'].astype(str) + "-" + result_news['month'].astype(str) + "-1", format='%Y-%m-%d')
        # Dropping month and year.
        result_news = result_news.drop(['year','month'],1)
        # Setting date as index.
        result_news = result_news.set_index('date')
        # We don't have full data for the first and last months: remove them.
        result_news = result_news.loc['2015-05-01':'2018-01-31']
        # Getting the column list.
        news_col_list = result_news.columns.tolist()
        print("... done.")

        # For the Tweets dataset.
        print("Pre-processing tweets...")
        # Dropping unnecessary columns.
        tweets_df = tweets_df.drop(['i','id','day','tweet','NOUN','ADJ','VERB',
                                   'PROPN','SENT','SIZE','blob score', 'vader score'],1)
        # Defining query for generating the time series.
        query_tweets = """SELECT     year, month,
                      SUM(CASE WHEN `vader category`='pos' THEN 1 ELSE 0 END) AS pos_count,
                      SUM(CASE WHEN `vader category`='neg' THEN 1 ELSE 0 END) AS neg_count,
                      SUM(CASE WHEN `vader category`='neu' THEN 1 ELSE 0 END) AS neu_count
           FROM       tweets_df
           GROUP BY   year, month;"""
        # Applying query and getting results.
        result_tweets = ps.sqldf(query_tweets, locals())
        # Generating date index.
        result_tweets["date"] = pd.to_datetime(result_tweets['year'].astype(str) + "-" + result_tweets['month'].astype(str) + "-1", format='%Y-%m-%d')
        # Dropping month and year.
        result_tweets = result_tweets.drop(['year','month'],1)
        # Setting date as index.
        result_tweets = result_tweets.set_index('date')
        # We don't have full data for the first and last months: remove them.
        result_tweets = result_tweets.loc['2015-05-01':'2018-01-31']
        # Getting the column list.
        tweets_col_list = result_tweets.columns.tolist()
        print("... done.")


        # Generating sentiment dataframes with news and tweets.
        pos_df = result_tweets[[tweets_col_list[0]]].join(result_news[[news_col_list[0]]],
                                                          lsuffix='_tweets',
                                                          rsuffix='_news')
        # Negative sentiment.
        neg_df = result_tweets[[tweets_col_list[1]]].join(result_news[[news_col_list[1]]],
                                                          lsuffix='_tweets',
                                                          rsuffix='_news')
        # Neutral sentiment.
        neu_df = result_tweets[[tweets_col_list[2]]].join(result_news[[news_col_list[2]]],
                                                          lsuffix='_tweets',
                                                          rsuffix='_news')
        # Returning readied dataframes.
        return pos_df, neg_df, neu_df

    # Method that checks for stationarity in the time series.For the news
    # time series, it does the ADF test. For the tweets dataset it performs the
    # Zivot-Andrews test to account for the potential structural break in this
    # time series. Assumes it receives the dataframes generated by the
    # load_datasets method; a dataframe with datetime index, a column for tweets
    # and a column of news, in that order.
    def check_stationarity(self, df):
        # Initializing flags.
        flag_news = False
        flag_tweets = False
        # Getting column list.
        col_list = df.columns.tolist()
        # Checking for tweets.
        result_tweets = zivot_andrews(df[col_list[0]])
        # Checking for news.
        result_news = adfuller(df[col_list[1]])

        # Displaying results.
        print("Result for news:")
        print("P-value: ", result_news[1])
        if(result_news[1]<=0.05):
            flag_news = True
            print("Stationary: True")
        else:
            print("Stationary: False")
        print("Result for tweets:")
        print("P-value: ", result_tweets[1])
        if(result_tweets[1]<=0.05):
            flag_tweets = True
            print("Stationary: True")
        else:
            print("Stationary: False")
        print("--------------------------------------")
        return flag_news, flag_tweets

    # Method to differentiate a time series to make it stationary. You may
    # need to call this method more than once until the stationary test
    # checks out. Performs the operation in the indicated column and adjusts
    # all the dataframe to eliminate NA values.
    def diff_df(self, df, col_name):
        # Generating an internal copy of the dataframe.
        df_trans = df.copy()
        # Differentiating.
        df_trans[col_name] = df_trans[col_name].diff()
        # Dropping rows with NA values from the dataframe.
        df_trans = df_trans.dropna()
        return df_trans

    # Method that searches for the best max lag value for the VAR model while
    # ensuring it stays in within a given margin of the Durbin-Watson ideal
    # test scores. Takes into account the AIC as information metric. It assumes
    # stationary data.
    def best_lag_dw(self, df, threshold=0.2):
        model = VAR(df, freq="MS")
        # Assumes stationary data.
        best_aic = 99999
        best_lag = None
        best_dw = None
        # Searching for best lag order.
        for i in range(1,16):
            result = model.fit(i)
            #print("Lag order: ", i, " AIC: ", result.aic)
            # Checking with Durbin-Watson test for autocorrelation as well.
            dw_out = durbin_watson(result.resid)
            #print("DW test: ", dw_out)
            #print(abs(2.0-dw_out[0]))
            if((result.aic < best_aic) and (abs(2.0-round(dw_out[0],2))<=threshold) and (abs(2.0-round(dw_out[1],2))<=threshold)):
                #print("ENTRA")
                best_aic = result.aic
                best_lag = i
                best_dw = dw_out
        print("Best lag order: ", best_lag, " with an AIC score of: ", best_aic)
        print("Durbin-Watson results:")
        for col, val in zip(df.columns, best_dw):
          print(col, ':', round(val, 2))
        print("-------------------------------------------------")
        return best_aic, best_lag, best_dw

    # Method that performs the Granger causality test.
    def grangers_causation_matrix(self, data, variables, maxlag, test='ssr_chi2test', verbose=False):
        # Obtaining the results matrix.
        df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        for c in df.columns:
            for r in df.index:
                test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
        df.columns = [var + '_x' for var in variables]
        df.index = [var + '_y' for var in variables]
        # Providing feedback.
        print("Granger causality test results: ")
        print(df)
        print("-------------------------------------------------")
        return df

    # Method to generate the N-lagged dataframe for cross-correlation.
    def df_derived_by_shift(self, df, lag=0, NON_DER=[]):
        df = df.copy()
        if not lag:
            return df
        cols ={}
        for i in range(1,lag+1):
            for x in list(df.columns):
                if x not in NON_DER:
                    if not x in cols:
                        cols[x] = ['{}_{}'.format(x, i)]
                    else:
                        cols[x].append('{}_{}'.format(x, i))
        for k,v in cols.items():
            columns = v
            dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
            i = 1
            for c in columns:
                dfn[c] = df[k].shift(periods=i)
                i+=1
            df = pd.concat([df, dfn], axis=1)
        return df

    # Method to generate the N-lag Spearman cross-correlation, saving the
    # correlation matrices as SVG files. "n" refers to the number of lags.
    def n_lag_corr(self, df, filename, n=6):
        lag_df = self.df_derived_by_shift(df, n)
        corr_df = lag_df.corr(method='spearman')
        # Plotting and saving SVG file.
        plt.figure(figsize=(25,25))
        title = str(n) + " months"
        plt.title(title, y=1.05, size=16)
        mask = np.zeros_like(corr_df)
        mask[np.triu_indices_from(mask)] = True
        svm = sns.heatmap(corr_df, mask=mask, linewidths=0.1,vmax=1.0, vmin=-1.0,
                          square=True, cmap='coolwarm', linecolor='white', annot=True)
        img_path = self.out_dir + filename
        plt.savefig(img_path, bbox_inches='tight')
