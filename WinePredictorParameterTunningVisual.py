import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def WinePredictor(Datapath):
    line = '-'*100
    df = pd.read_csv(Datapath)

    df.dropna(inplace=True)

    x = df.drop(columns=['Class'])
    y = df['Class']

    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scale, y, train_size=0.2, random_state=42)


    accuracy_scores = []
    K_Range = range(1,21)

    for k in K_Range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

    plt.figure(figsize=(8,5))
    plt.plot(K_Range, accuracy_scores, marker = 'o', linestyle = '--')
    plt.title("Accuracy VS K Value")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(K_Range)
    plt.show()
    
    
    best_K = K_Range[accuracy_scores.index(max(accuracy_scores))]
    print("Best Value of K is : ",best_K)

    model = KNeighborsClassifier(n_neighbors=best_K)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(line)
    print("Final best accuracy is : ",accuracy*100)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print(cm)

    

def main():
    WinePredictor('WinePredictor.csv')

if __name__ == '__main__':
    main()