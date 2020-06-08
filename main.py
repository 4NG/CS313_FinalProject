from Information import *
from Preprocessing import *
from Classification import *
from sklearn.model_selection import train_test_split
import pandas as pd

"""
Pre-processing data
return: a csv file
"""
input_file = input("Nhap file dau vao:")
output_file = input("Nhap file dau ra:")

data = Preprocessing(output_file)
data.preprocessing(input_file)

"""
Drop label
Separate output file to training set and test set
"""
data = pd.read_csv(output_file)
X = data.drop(['RainTomorrow'], axis=1)
y = data['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


"""
Classification
"""
clf = Classification(X_train, X_test, y_train, y_test)
clf.logistic_regression()
clf.random_forest()
clf.support_vector_machine()
clf.decisiontree()

