import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

col_names = ['Outlook','Temperature','Humidity','Wind','Play ball','Play ball bin']
# load dataset
data = pd.read_csv("simple_dt.csv", header=0, names=col_names)
data.columns = col_names
head = data.head()

#split dataset in features and target variable
feature_cols = ['Outlook','Temperature','Humidity','Wind']

#get_dummies builds categorical conditions for each feature (e.g. Temperature -> Hot, Cold, etc.)
X = pd.get_dummies(data[feature_cols]) # Features
y = data['Play ball bin'] # Target variable

features_expanded = list()

#in a situation in which the entire CSV is the training set, no need to split.
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#I had to loop through each item in my exploded categories as it was throwing
#in some random variables if I just used X.  Don't know why.
for item in X:
    features_expanded.append(item)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(X, y)

#Predict the response for test dataset
y_pred = clf.predict(X)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = features_expanded)#,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('dt_simple.png')
Image(graph.create_png())