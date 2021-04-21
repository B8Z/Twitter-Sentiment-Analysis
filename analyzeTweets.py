import pandas as pd
import numpy as np
import scipy
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import constants


###### Functions #######
#Function to load the dataset and save to a pandas dataframe
def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset

#Function to remove unwanted columns from a pandas dataframe
def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset

#Function to get the feature vector from the data
def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

#get p-values to determine statistical significance
def get_p_value(distribution_a, distribution_b):
    #use statistical significance tests to determine which classifier performed the best
    #important -- we are using a paired (dependent) t-test. This is because the samples are
    # the same in both classifiers. Paired t-test is stronger statistically
    # We are using two-sided t-test because we want to know if the systems are different
    # (better or worse)
    t_test_result = scipy.stats.ttest_rel(distribution_a, distribution_b)
    return t_test_result.pvalue

#get predictions for the test set
def get_test_set_predictions(X_train, Y_train, X_test, model):
    #train on full training set and test on test set
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return predictions

###### Process and Split Dataset #######
# Load the dataset
dataset = load_dataset(constants.PROC_DIR + "dev.csv", ['', 'id', 'label', 'text'])
# Remove the unwanted columns from dataset
n_dataset = remove_unwanted_cols(dataset, ['', 'id'])

# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
y = np.array(dataset.iloc[:, 0]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)


###### Process Metrics #######
#Count positive, negative, and neutral tweets
positive = [r for r in dataset['text'][dataset['label']==2]]
pos = ''.join(positive)
negative = [r for r in dataset['text'][dataset['label']==0]]
neg = ''.join(negative)
neutral = [r for r in dataset['text'][dataset['label']==1]]
neu = ''.join(neutral)
#Convert pandas dataframes out of the positive, negative, and neutral tweet lists
positive_df = DataFrame(positive, columns=['text'])
negative_df = DataFrame(negative, columns=['text'])
neutral_df = DataFrame(neutral, columns=['text'])


###### Word Metrics #######
#Figure out what occurs most, tag everything, and see metrics
all_words_frequency = n_dataset.text.str.split(expand=True).stack().value_counts()
positive_words_frequency = positive_df.text.str.split(expand=True).stack().value_counts()
negative_words_frequency = negative_df.text.str.split(expand=True).stack().value_counts()
neutral_words_frequency = neutral_df.text.str.split(expand=True).stack().value_counts()
print(all_words_frequency)
#Generate word clouds
wordCloudpos = WordCloud(width=600, height=400).generate(pos)
plt.imshow(wordCloudpos)
plt.show()


# ###### Class Ratio #######
# #Tag everything as majority class - what performance?
all_classes = len(positive) + len(negative) + len(neutral)
print("Number of positive tweets: ", len(positive))
print("Number of negative tweets: ", len(negative))
print("Number of neutral tweets: ", len(neutral))
print("Number of tweets: ", all_classes)
print("Positive class ratio: ", len(positive) / all_classes)
print("Negative class ratio: ", len(negative) / all_classes)
print("Neutral class ratio: ", len(neutral) / all_classes)


# ###### MODELS #######
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

#Forest Model with imbalanced class weights
forest_model = RandomForestClassifier(n_estimators=200)
forest_model.fit(X_train,y_train)
forest_model_predict = forest_model.predict(X_test)
print("Forest Model Accuracy = ", accuracy_score(y_test, forest_model_predict))
print("Forest Model F1 Score = ", f1_score(y_test, forest_model_predict, average="macro"))

#Support Vector Machine model with imbalanced class weights
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, kernel='linear'))
clf_output = clf.fit(X_train, y_train)
clf_predict = clf_output.predict(X_test)
print("SVM Model Accuracy = ", accuracy_score(y_test, clf_predict))
print("SVM Model F1 Score = ", f1_score(y_test, clf_predict, average="macro"))

# Training Logistics Regression model with balanced class weights
LR_model_balanced = LogisticRegression(solver='lbfgs', class_weight='balanced')
LR_model_balanced.fit(X_train, y_train)
y_predict_lr_balanced = LR_model_balanced.predict(X_test)
print("Logistic Regression Model Accuracy Balanced = ", accuracy_score(y_test, y_predict_lr_balanced))
print("Logistic Regression Model F1 Score Balanced = ", f1_score(y_test, y_predict_lr_balanced, average="macro"))


#Support Vector Machine model with balanced class weights
clf2 = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf2.fit(X_train, y_train)
clf_predict = clf_output.predict(X_test)
print("SVM Model Accuracy Balanced = ", accuracy_score(y_test, clf_predict))
print("SVM Model F1 Score Balanced = ", f1_score(y_test, clf_predict, average="macro"))

#Forest Model with balanced class weights
forest_model2 = RandomForestClassifier(n_estimators=200, class_weight='balanced')
forest_model2.fit(X_train,y_train)
forest_model_predict2 = forest_model2.predict(X_test)
print("Forest Model Accuracy Balanced = ", accuracy_score(y_test, forest_model_predict2))
print("Forest Model F1 Score Balanced = ", f1_score(y_test, forest_model_predict2, average="macro"))

predictions_a = get_test_set_predictions(X_train, y_train, X_test, LR_model)
predictions_b = get_test_set_predictions(X_train, y_train, X_test, LR_model_balanced)
p = get_p_value(predictions_a, predictions_b)
print ("p-value comparing test set performance of linear regression vs. linear regression balanced = ", p)
#compare your p-value to alpha to make a conclusion
predictions_a = get_test_set_predictions(X_train, y_train, X_test, clf)
predictions_b = get_test_set_predictions(X_train, y_train, X_test, clf2)
p = get_p_value(predictions_a, predictions_b)
print ("p-value comparing test set performance of support vector machine vs. support vector machine balanced = ", p)
#compare your p-value to alpha to make a conclusion
predictions_a = get_test_set_predictions(X_train, y_train, X_test, forest_model)
predictions_b = get_test_set_predictions(X_train, y_train, X_test, forest_model2)
p = get_p_value(predictions_a, predictions_b)
print ("p-value comparing test set performance of forest model vs. forest model balanced = ", p)

predictions_a = get_test_set_predictions(X_train, y_train, X_test, forest_model)
predictions_b = get_test_set_predictions(X_train, y_train, X_test, LR_model)
p = get_p_value(predictions_a, predictions_b)
print ("p-value comparing test set performance of forest model vs. linear regression model = ", p)

predictions_a = get_test_set_predictions(X_train, y_train, X_test, forest_model)
predictions_b = get_test_set_predictions(X_train, y_train, X_test, clf)
p = get_p_value(predictions_a, predictions_b)
print ("p-value comparing test set performance of forest model vs. support vector machine = ", p)

predictions_a = get_test_set_predictions(X_train, y_train, X_test, LR_model)
predictions_b = get_test_set_predictions(X_train, y_train, X_test, clf)
p = get_p_value(predictions_a, predictions_b)
print ("p-value comparing test set performance of linear regression vs. support vector machine = ", p)
#compare your p-value to alpha to make a conclusion


# ###### Create Confusion Matrices #######
cm_lr = confusion_matrix(y_test, y_predict_lr, normalize='all')
print("linear regression confusion matrix \n", cm_lr)

cm_svm = confusion_matrix(y_test, clf_predict, normalize='all')
print("support vector machine confusion matrix \n", cm_svm)

cm_forest = confusion_matrix(y_test, forest_model_predict, normalize='all')
print("random forest confusion matrix \n", cm_forest)
