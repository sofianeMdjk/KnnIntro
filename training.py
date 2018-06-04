import pandas as pd
import preporcessing as pre
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

#Specifying column names
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

#loading dataset (internet connexion required)
dataset = pd.read_csv(url, header=None, names=col_names)

# mapping classes to integer values

classes = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
dataset['species_num'] = [classes[i] for i in dataset.species]

#getting inputs and target
inputs = dataset.drop(['species', 'species_num'], axis=1)
target = dataset.species_num

#normalizing inputs
input = pre.dataNormalization(inputs)

#splitting data into train and test
trainInput,testInput,trainTarget,testTarget = train_test_split(input,target,test_size=0.25)

#training clissifier
classifier = KNeighborsClassifier(n_neighbors=4)
## Fit the model on the training data.
classifier.fit(trainInput, trainTarget)
## See how the model performs on the test data.
accuracy = classifier.score(testInput, testTarget)
print(accuracy)


