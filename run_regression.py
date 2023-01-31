import pandas

from sklearn import linear_model


dataset = pandas.read_csv("dataset.csv")
print(dataset)

target = dataset.iloc[:,0].values
print(target)

data = dataset.iloc[:,3:9].values
print(data)

machine = linear_model.LinearRegression()
machine.fit(data, target)

print(machine)

new_dataset = pandas.read_csv("new_dataset.csv")
new_dataset = new_dataset.values

prediction = machine.predict(new_dataset)

print(prediction)


