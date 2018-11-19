#importing libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

#reading the data
doc = pd.read_csv('sampled_with_names.csv',header=None)
doc.columns = [['class','text']]
doc = doc.dropna(axis=0, subset=['text'])

doc['class'].value_counts()

#Creating feature vectors
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',ngram_range=(1,2))
features = tfidf.fit_transform(doc['text'].values.astype(str)).toarray()

#saving the TfidfVectorizer as pickle file
joblib.dump(tfidf, 'tfidf.pkl')

labels = doc['class']
random_state = 10

#splitting the data into train and test sets
train_labels,test_labels,train_features,test_features = train_test_split(labels,features,test_size = 0.2,random_state=random_state, stratify = doc['class'])

#defining RandomForestClassifier model and fitting to the training data
clf_rf  = RandomForestClassifier(random_state=random_state, n_estimators = 100,class_weight='balanced')
clf_rf.fit(train_features,train_labels)

#predicting the test data
prediction=clf_rf.predict(test_features)
accuracy_score(prediction,test_labels)

#saving the classifier
joblib.dump(clf_rf, 'rf_clf.pkl')