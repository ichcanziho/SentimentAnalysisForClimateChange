# SentimentAnalysisForClimateChange
Syntactic and sentiment analysis on climate change-related news and tweets.

* Code authors: [Gabirel Ichcanziho Pérez Landa](https://github.com/ichcanziho) and [Jorge Francisco Ciprián Sánchez](https://github.com/JorgeFCS).
* Paper authors: [Gabriel Ichcanziho Pérez Landa](https://www.linkedin.com/in/ichcanziho/) and [Jorge Francisco Ciprián Sánchez](https://www.linkedin.com/in/jorge-ciprian-a735b7105/).
* Date of creation: May 10th, 2021.
* Code Author Email: ichcanziho@outlook.com

## Overview

We employ natural language processing techniques such as sentiment and syntactic analysis to analyze selected news and tweets datasets, obtaining the prevailing sentiment and structure of climate change-related news and tweets. We then extract and compare the most relevant words associated with positive, negative, and neutral feelings in both datasets. Finally, we visualize their change over time to extract relevant patterns that can provide insight into the evolution of the climate change-related narrative. For more details on the implementation and results, see the paper [here]().

### System specifications
This work was developed and tested on Ubuntu 16.04.6 LTS with Python 3.6.9, and Windows 10 with Python 3.8.1.

We employ [VADER](https://pypi.org/project/vaderSentiment/) and [TextBlob](https://textblob.readthedocs.io/en/dev/#) for sentiment analysis and [spaCy](https://spacy.io/) for syntactic analysis. We include a *requirements.txt* file for ease of use; to install the required dependencies run the following command:

```sh
$ pip install -r requirements.txt
```

## Data

We analyze the following publicly available datasets:

* **[A New Dataset For Exploring Climate Change Narratives On Television News 2009-2020](https://blog.gdeltproject.org/a-new-dataset-for-exploring-climate-change-narratives-on-television-news-2009-2020/)**: This dataset contains 95,000 entries for news that contain the keywords *climate change* or *global warming* or *climate crisis* or *greenhouse gas* or *greenhouse gases* or *carbon tax* across four stations: BBC News, CNN, MSNBC, and Fox News, from 2009 to 2020. Each entry includes the UTC time, the station, the show, a 15 second clip of the captioning containing the mention, and a URL to the website to view the specific clip containing it.
* **[Twitter Climate Change Sentiment Dataset](https://www.kaggle.com/edqian/twitter-climate-change-sentiment-dataset)**: This dataset, generated by the University of Waterloo, contains tweet IDs on the subject of climate change collected between April 27, 2015 and February 21, 2018. The dataset contains a total of 43,943 tweets. The authors labeled the tweets according to four categories related to climate change belief or denial. For the purposes and scope of the present project, we only consider the tweet text and date, as we perform our own syntactic and sentiment analysis. 

### Data pre-processing
We filter the entries by the subject of *climate change* and adjust the news dataset so that it contains entries for the same period of the tweets one, also removing the entries for the BBC news chain, as these entries start in the year 2017. After performing the said filters, we obtain 16,124 news entries and 28,116 tweet entries.

We provide the raw datasets in the *inputs* directory and the pre-processed datasets in the *outputs* folder. The datasets containing the results for the syntactic and sentiment analysis are the *outputs/clean datasets/News_syntactic_sentiment.csv* and *outputs/clean datasets/Twitter_syntactic_sentiment.csv* files.

## Running the program

We provide 5 options: dataset cleaning, sentiment analysis, syntactic analysis, plotting the results' timelines, and all operations. You can select the operation you with to carry on the *config.ini* file, changing the following line:

```sh
[MODE]
mode = SENT ; change this line.
```
