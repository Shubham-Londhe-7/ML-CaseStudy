import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def MarvellousLogistics(datasetpath):
    df = pd.read_csv(datasetpath)

    print("Dimension dataframe :",df.shape)
    print("Initial data is :\n",df.head())

    df['Gender'] = df['Gender'].map({'Female' : 0, 'Male' : 1})

    print("Encoded data is :\n",df.head())

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='Height', y='Weight', hue='Gender', palette='Set1')
    plt.title("Marvellous Gender Predictor")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.show()

    X = df[['Height', 'Weight']]
    Y = df['Gender']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    Accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy is :",Accuracy*100)

    Conf_Matrix = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix :\n",Conf_Matrix)

    report = classification_report(Y_test, Y_pred)

    print("Classification report is :\n",report)

def main():
    MarvellousLogistics("weight-height.csv")

if __name__ == "__main__":
    main()