DATA_DIR = '/Users/Adam/PycharmProjects/Twitter_Sentiment_Analysis/data/GOLD/Subtask_A/'
PROC_DIR = '/Users/Adam/PycharmProjects/Twitter_Sentiment_Analysis/data/processed/'
EVAL1_DIR = '/Users/Adam/PycharmProjects/Twitter_Sentiment_Analysis/data/Dev/'
EVAL2_DIR = '/Users/Adam/PycharmProjects/Twitter_Sentiment_Analysis/data/Final/'

TRAIN = 'twitter-2016train-A.txt'
TEST = 'twitter-2016test-A.txt'
DEV = 'twitter-2016dev-A.txt'
DEVTEST = 'twitter-2016devtest-A.txt'

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]
