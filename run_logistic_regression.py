import pandas

from sklearn import linear_model

dataset = pandas.read_csv("dataset.csv")

print(dataset)

target = dataset.iloc[:,1].values
data = dataset.iloc[:,3:9].values


machine = linear_model.LogisticRegression()
machine.fit(data,target)


new_dataset = pandas.read_csv("new_dataset.csv")
new_dataset = new_dataset.values

prediction = machine.predict(new_dataset)

print(prediction)



