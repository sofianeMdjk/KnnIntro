import pandas as pd

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

## Specifying column names
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_dataset = pd.read_csv(url, header=None, names=col_names)


