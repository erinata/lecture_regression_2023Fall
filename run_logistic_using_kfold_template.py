import pandas 
from sklearn import linear_model

import kfold_template

dataset = pandas.read_csv("dataset.csv")

target = dataset.iloc[:,1].values
data = dataset.iloc[:,3:9].values

machine = linear_model.LogisticRegression()

kfold_template.run_kfold(data, target, machine, 4, True, True)