"""
Classify a sample
"""
from Preprocessing import *
from joblib import load
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# input all features
date = input("YY/MM/DD: ")
location = input("Location: ")
min_temp = float(input("MinTemp: "))
max_temp = float(input("MaxTemp: "))
rainfall = float(input("Rainfall: "))
evaporation = float(input("Evaporation: "))
sunshine = float(input("Sunshine: "))
windgustdir = input("WindGustDir: ")
windgustspeed = float(input("WindGustSpeed: "))
winddir9am = input("WindDir9am: ")
winddir3pm = input("WindDir3am: ")
windspeed9am = float(input("WindSpeed9am: "))
windspeed3pm = float(input("WindSpeed3am: "))
humidity9am = float(input("Humidity9am: "))
humidity3pm = float(input("Humidity3pm: "))
pressure9am = float(input("Pressure9am: "))
pressure3pm = float(input("Pressure3pm: "))
cloud9am = float(input("Cloud9am: "))
cloud3pm = float(input("Cloud3pm: "))
temp9am = float(input("Temp9am: "))
temp3pm = float(input("Temp3pm: "))
raintoday = input("RainToday: ")
risk_mm = input("RISK_MM: ")
raintomorrow = input("RainTomorrow: ")

# create a pandas dataframe
data = {'Date': [date],
        'Location': [location],
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall],
        'Evaporation': [evaporation],
        'Sunshine': [sunshine],
        'WindGustDir': [windgustdir],
        'WindGustSpeed': [windgustspeed],
        'WindDir9am': [winddir9am],
        'WindDir3pm': [winddir3pm],
        'WindSpeed9am': [windspeed9am],
        'WindSpeed3pm': [windspeed3pm],
        'Humidity9am': [humidity9am],
        'Humidity3pm': [humidity3pm],
        'Pressure9am': [pressure9am],
        'Pressure3pm': [pressure3pm],
        'Cloud9am': [cloud9am],
        'Cloud3pm': [cloud3pm],
        'Temp9am': [temp9am],
        'Temp3pm': [temp3pm],
        'RainToday': [raintoday],
        'RISK_MM': [risk_mm],
        'RainTomorrow': [raintomorrow]}

sample = pd.DataFrame(data, columns=['Date', 'Location', 'MinTemp', 'MaxTemp', 'RainFall',
                                     'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed',
                                     'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm',
                                     'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                                     'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow'])
# sample pre-processing
sample.drop(['RISK_MM'], axis=1, inplace=True)

sample['Date'] = pd.to_datetime(sample['Date'])
sample['Year'] = sample['Date'].dt.year
sample['Month'] = sample['Date'].dt.month
sample['Day'] = sample['Date'].dt.day
sample.drop('Date', axis=1, inplace=True)

nominal = []
list_columns = sample.columns.tolist()
list_columns.remove('RainTomorrow')
for i in list_columns:
    if sample[i].dtypes == 'object':
        nominal.append(i)

sample = pd.get_dummies(sample, columns=nominal)

scale = MinMaxScaler()
sample = pd.DataFrame(scale.fit_transform(sample), columns=sample.columns)

sample.drop(['RainTomorrow'], axis=1, inplace=True)

# predict
log_reg = load('LogisticRegression.joblib')
print(log_reg.predict(sample))

rf = load('RandomForest.joblib')
print(rf.predict(sample))

svm = load('SupportVectorMachine.joblib')
print(svm.predict(sample))

dt = load('DecisionTree.joblib')
print(dt.predict(sample))


