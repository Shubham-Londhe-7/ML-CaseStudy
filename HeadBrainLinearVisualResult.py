import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def MarvellousHeadBrainLinear(Datapath):
    line = '*'*80
    df = pd.read_csv(Datapath)

    print(line)
    print("First few records of the dataset are :")
    print(line)
    print(df.head())
    print(line)

    print("Statistical information of the dataset :")
    print(line)
    print(df.describe())
    print(line)

    x = df[['Head Size(cm^3)']]
    y = df[['Brain Weight(grams)']]

   
    print("Independent Variable are : Head Size")
    print("Dependent Variable are : Brain Weight")

    print("Total records in dataset : ",x.shape)
    print(line)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print("Dimensions of Training dataset : ",x_train.shape)
    print("Dimensions of Testing dataset : ",x_test.shape)
    print(line)

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Visual Representation :")

    plt.figure(figsize=(8, 5))
    plt.scatter(x_test,y_test, color = 'blue', label = 'Actual')
    plt.plot(x_test.values.flatten(), y_pred, color = 'red', linewidth = 2, label = 'Regression Line')
    plt.xlabel("Head Size(cm^3)")
    plt.ylabel("Brain Weight(grams)")
    plt.title("Marvellous Head Brain Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Result of Case Study :")
    print("Slope of Line (m) : ",model.coef_[0])
    print("Intercept (c) : ",model.intercept_)
    print("Mean Squared Error is : ",mse)
    print("Root Mean Squared Error : ",rmse)
    print("R Square Value : ",r2)
    print(line)

def main():
    MarvellousHeadBrainLinear("MarvellousHeadBrain.csv")

if __name__ == "__main__":
    main()