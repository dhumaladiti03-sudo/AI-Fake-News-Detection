import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
fake = pd.read_csv("dataset/Fake.csv")

true = pd.read_csv("dataset/True.csv")
fake["label"] = 0
true["label"] = 1
data = pd.concat([fake, true])

print(data.head())
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
tfidf = TfidfVectorizer(

stop_words='english',

max_df=0.7

)

X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)
model = PassiveAggressiveClassifier(

max_iter=50

)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print("Accuracy:", score)
news = input("Enter News : ")

vector = tfidf.transform([news])

prediction = model.predict(vector)

if prediction[0] == 0:

    print("Fake News")

else:

    print("Real News")