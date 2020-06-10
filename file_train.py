"""
Classify a file
"""
from Preprocessing import *
from joblib import load

file = input("Nhap file: ")
output_file = input("Nhap ten file output: ")

# file pre-processing
f = Preprocessing(output_file)
f.preprocessing(file)

# classification
file_data = pd.read_csv(output_file)
file_data.info()
file_data_test = file_data.drop(['RainTomorrow'], axis=1)

log_reg = load('LogisticRegression1.joblib')
print(log_reg.predict(file_data_test))

rf = load('RandomForest1.joblib')
print(rf.predict(file_data_test))

svm = load('SupportVectorMachine1.joblib')
print(svm.predict(file_data_test))

dt = load('DecisionTree1.joblib')
print(dt.predict(file_data_test))