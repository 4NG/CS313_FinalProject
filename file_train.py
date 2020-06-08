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
file_data_test = file_data.drop(['RainTomorrow'])

log_reg = load('LogisticRegression.joblib')
print(log_reg.predict(file_data_test))

rf = load('RandomForest.joblib')
print(rf.predict(file_data_test))

svm = load('SupportVectorMachine.joblib')
print(svm.predict(file_data_test))

dt = load('DecisionTree.joblib')
print(dt.predict(file_data_test))