from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
test = np.array(float(input("Enter the test value: "))).reshape(-1,1)
# print(df.describe())
x = df.iloc[:,0]
y = df.iloc[:,1]

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
obj = LinearRegression()
obj.fit(x,y)
out = obj.predict(test)
print(f'Score of Student studing {float(test)} hrs/day is {str(out).replace("[[","").replace("]]","")}')