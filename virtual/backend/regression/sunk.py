import pandas 
from sklearn import linear_model
import warnings

warnings.simplefilter(action="ignore")
red = pandas.read_csv("regression/titanic.csv", header=0, delimiter=',', encoding="utf-8-sig")
red.to_numpy()

X = red[['pclass', 'sex', 'age']]

for i, x in enumerate(X['sex']):
    if x == "male": 
        X['sex'][i] = 1
    else: 
        X['sex'][i] = 0
    
y = red['survived']

regr = linear_model.LinearRegression()
regr.fit(X, y)

def pred(pclass, s, a): 
    if pclass != '' and s != '' and a != '': 
        predquality = regr.predict([[float(pclass), float(s), float(a)]])
        print(predquality)
        return predquality
    else:  
        print("something is null") 