import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import constants

def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset

def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector


# Load dataset
dataset = load_dataset(constants.PROC_DIR + "train.csv", ['', 'id', 'label', 'text'])
# Remove unwanted columns from dataset
n_dataset = remove_unwanted_cols(dataset, ['', 'id'])
# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
y = np.array(dataset.iloc[:, 0]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)


# Creating a word cloud
words = ' '.join([tweet for tweet in dataset['text']])
wordCloud = WordCloud(width=600, height=400).generate(words)
plt.imshow(wordCloud)
plt.show()

# # create a word frequency dictionary
# wordfreq = Counter(words)
# # draw a Word Cloud with word frequencies
# wordcloud = WordCloud(
#     background_color='white',
#     max_words=2000,
#     stopwords=stopwords
#    ).generate_from_frequencies(wordfreq)
# plt.figure(figsize=(10,9))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# # plt.show()

positive = [r for r in dataset['text'][dataset['label']==2]]
pos = ''.join(positive)
negative = [r for r in dataset['text'][dataset['label']==1]]
neg = ''.join(negative)
neutral = [r for r in dataset['text'][dataset['label']==0]]
neu = ''.join(neutral)


# Training Naive Bayes model
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
y_predict_nb = NB_model.predict(X_test)
print("Naive Bayes Model Accuracy = ", accuracy_score(y_test, y_predict_nb))
print("Naive Bayes Model F1 Score = ", f1_score(y_test, y_predict_nb, average="macro"))

# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
y_predict_lr = LR_model.predict(X_test)
print("Logistic Regression Model Accuracy = ", accuracy_score(y_test, y_predict_lr))
print("Logistic Regression Model F1 Score = ", f1_score(y_test, y_predict_lr, average="macro"))

# Training Logistics Regression model with balanced class weights
LR_model_balanced = LogisticRegression(solver='lbfgs', class_weight='balanced')
LR_model_balanced.fit(X_train, y_train)
y_predict_lr_balanced = LR_model_balanced.predict(X_test)
print("Logistic Regression Model Accuracy Balanced = ", accuracy_score(y_test, y_predict_lr_balanced))
print("Logistic Regression Model F1 Score Balanced = ", f1_score(y_test, y_predict_lr_balanced, average="macro"))

#Support Vector Machine model with imbalanced class weights
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, kernel='linear'))
clf_output = clf.fit(X_train, y_train)
clf_predict = clf_output.predict(X_test)
print("SVM Model Accuracy = ", accuracy_score(y_test, clf_predict))
print("SVM Model F1 Score = ", f1_score(y_test, clf_predict, average="macro"))

#Support Vector Machine model with balanced class weights
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(X_train, y_train)
clf_predict = clf_output.predict(X_test)
print("SVM Model Accuracy Balanced = ", accuracy_score(y_test, clf_predict))
print("SVM Model F1 Score Balanced = ", f1_score(y_test, clf_predict, average="macro"))

#Forest Model with imbalanced class weights
forest_model = RandomForestClassifier(n_estimators=200)
forest_model.fit(X_train,y_train)
forest_model_predict = forest_model.predict(X_test)
print("Forest Model Accuracy = ", accuracy_score(y_test, forest_model_predict))
print("Forest Model F1 Score = ", f1_score(y_test, forest_model_predict, average="macro"))

#Forest Model with balanced class weights
forest_model = RandomForestClassifier(n_estimators=200, class_weight='balanced')
forest_model.fit(X_train,y_train)
forest_model_predict = forest_model.predict(X_test)
print("Forest Model Accuracy Balanced = ", accuracy_score(y_test, forest_model_predict))
print("Forest Model F1 Score Balanced = ", f1_score(y_test, forest_model_predict, average="macro"))

#Class ratio
#Figure out what occurs most, tag everything, and see metrics
#Tag everything as majority class - what performance?
