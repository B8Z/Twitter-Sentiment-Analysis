import pandas as pd
import re
import string
import nltk
import constants
nltk.download('stopwords')
from nltk.corpus import stopwords

tokens_re = re.compile(r'(' + '|'.join(constants.regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + constants.emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token.lower() if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def main():
    data = {
        'subtask': constants.DATA_DIR + constants.TRAIN,
        'subtasktest': constants.DATA_DIR + constants.TEST,
        'dev': constants.DATA_DIR + constants.DEV,
        'devtest': constants.DATA_DIR + constants.DEVTEST,
        }

    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['rt', 'via']

    for dataset in data:
        #print(dataset)
        with open(data[dataset], 'r') as dataset_f:
            output_data = []
            for line in dataset_f:
                info = line.strip().split('\t')
                id, label, text = info[0], info[1], ' '.join(info[2:])
                if(label == 'positive'):
                    label = 2
                elif(label == 'negative'):
                    label = 0
                else:
                    label = 1
                # process text
                tokens = preprocess(text)
                # remove stopwords and others
                tokens = [term.lower() for term in tokens if term.lower() not in stop]
                # remove hashtags
                tokens = [term for term in tokens if not term.startswith('#')]
                # remove profiles
                tokens = [term for term in tokens if not term.startswith('@')]
                d = {
                    'id': id,
                    'label': label,
                    'text': ' '.join(tokens)
                }
                output_data.append(d)
            df = pd.DataFrame(output_data)
            df.to_csv(constants.PROC_DIR+dataset+'.csv')


if __name__ == '__main__':
    main()
