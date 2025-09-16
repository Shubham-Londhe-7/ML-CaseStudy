import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def MarvellosCancerDetection():
    cancer = load_breast_cancer()

    X = cancer.data
    Y = cancer.target

    data = pd.DataFrame(X, columns=cancer.feature_names)
    data['target'] = Y

    print(data.shape)
    print('Classes : ', dict(zip(cancer.target_names, [0,1])))

    scaler = StandardScaler()
    X_Scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_Scaled, Y, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', C=1)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print('Accuracy : ',accuracy_score(Y_test, Y_pred)*100)
    print('Confusion matrix :\n', confusion_matrix(Y_test, Y_pred))


    
def main():
    MarvellosCancerDetection()

if __name__ == "__main__":
    main()