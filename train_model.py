
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("dataset.csv")

X = data["question"]
y = data["case_type"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = {
"SVM":LinearSVC(),
"NaiveBayes":MultinomialNB(),
"RandomForest":RandomForestClassifier()
}

best_model=None
best_acc=0

for name,clf in models.items():
    pipe = Pipeline([
    ("tfidf",TfidfVectorizer(stop_words="english")),
    ("clf",clf)
    ])
    
    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test)
    acc=accuracy_score(y_test,pred)
    
    print(name,"accuracy:",acc)
    
    if acc>best_acc:
        best_acc=acc
        best_model=pipe

print("Best accuracy:",best_acc)

pickle.dump(best_model,open("model.pkl","wb"))
print("Best model saved")
