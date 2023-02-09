import pandas 
from sklearn import linear_model

import kfold_template

dataset = pandas.read_csv("dataset.csv")

target = dataset.iloc[:,2].values
data = dataset.iloc[:,3:9].values

machine = linear_model.LogisticRegression()
# machine = linear_model.PoissonRegressor()
all_return_values = kfold_template.run_kfold(data, target, machine, 4, True, False, False)

all_return_values = [i[0] for i in all_return_values]
print(all_return_values)
print(sum(all_return_values)/len(all_return_values))


machine = linear_model.LogisticRegression()
machine.fit(data, target)

new_dataset = pandas.read_csv("new_dataset.csv")
new_dataset = new_dataset.values
prediction = machine.predict(new_dataset)
prediction_probability = machine.predict_proba(new_dataset)
prediction_probability = prediction_probability.round(3)
print(prediction)
print(prediction_probability)

print("Done")
