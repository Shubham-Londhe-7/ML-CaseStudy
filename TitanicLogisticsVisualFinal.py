import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import seaborn as sns
from seaborn import countplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def MarvellousTitanicLogistic(Datapath):
    line = '-'*100
    df = pd.read_csv(Datapath)
    print(line)
    print("Dataset loaded successfully :")
    print(df.head())

    print(line)
    print("Dimensions of dataset is : ", df.shape)
    print(line)

    df.drop(columns=['Passengerid', 'zero'], inplace=True)
    print(line)
    print("Dimensions of dataset is : ", df.shape)
    print(line)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    figure()
    target = "Survived"
    countplot(data = df, x = target).set_title('Survived VS Non Survived')
    # show()

    figure()
    target = "Survived"
    countplot(data=df, x=target, hue='Sex').set_title("Based on Gender")
    # show()

    figure()
    target = "Survived"
    countplot(data=df, x=target, hue='Pclass').set_title("Based on Passenger Class")
    # show()

    figure()
    df['Age'].plot.hist().set_title("Age Report")
    # show()

    figure()
    df['Fare'].plot.hist().set_title("Fare Report")
    # show()

    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot= True, cmap='coolwarm')
    plt.title("Feature Correlation HeatMap")
    # plt.show()

    x = df.drop(columns=['Survived'])
    y = df['Survived']

    print(line)
    print("Dimensions of Target : ",x.shape)
    print("Dimensions of Labels : ",y.shape)
    print(line)

    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(line)
    print("Accuracy is : ",accuracy)
    print("Confusion Matrix :")
    print(cm)
    print(line)


def main():
    MarvellousTitanicLogistic('MarvellousTitanicDataset.csv')

if __name__ == '__main__':
    main()