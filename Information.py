import matplotlib.pyplot as plt
import seaborn as sns


class Information:

    def __init__(self, input_data):
        self.input_data = input_data

    def information(self):
        # view 5 first rows of data
        print("5 dong dau cua du lieu:")
        print(self.input_data.head())
        print("\n")

        # view 5 last rows of data
        print("5 dong cuoi cua du lieu:")
        print(self.input_data.tail())
        print("\n")

        # view 5 samples
        print("5 mau du lieu ngau nhien:")
        print(self.input_data.sample(5))
        print("\n")

        # describe data
        print("Mo ta du lieu:")
        print(self.input_data.describe())
        print("\n")

        # view all columns
        print("Cac cot cua du lieu:")
        print(self.input_data.columns.tolist())
        print("\n")

        # data types of columns
        numeric = []
        nominal = []
        list_columns = self.input_data.columns.tolist()
        for i in list_columns:
            if self.input_data[i].dtypes == 'float64':
                numeric.append(i)
            else:
                nominal.append(i)
        print("\n")

        # view nominal(categorical) columns
        print("Thuoc tinh nominal:")
        print(self.input_data[nominal].head())
        print("\n")

        # view frequency of nominal(categorical) variables
        for i in nominal:
            print(self.input_data[i].value_counts())
            print("\n")

        # check missing value in categorical variables
        print("So mau bi thieu:")
        print(self.input_data[nominal].isnull().sum())
        print("\n")

        # view numeric columns
        print("Thuoc tinh numeric:")
        print(self.input_data[numeric].head())
        print("\n")

        # view frequency of numeric(categorical) variables
        for i in numeric:
            print(self.input_data[i].value_counts())
            print("\n")

        # check missing value in numeric variables
        print("So mau bi thieu:")
        print(self.input_data[numeric].isnull().sum())
        print("\n")

    def visualization(self):
        # view frequency distribution of RainTomorrow variable
        f, ax = plt.subplots(figsize=(6, 8))
        ax = sns.countplot(x='RainTomorrow', data=self.input_data, palette='Set1')
        plt.show()

        # draw boxplots
        # view outliers
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 2)
        fig = self.input_data.boxplot(column='MinTemp')
        fig.set_title('')
        fig.set_ylabel('MaxTemp')

        plt.subplot(2, 2, 3)
        fig = self.input_data.boxplot(column='MaxTemp')
        fig.set_title('')
        fig.set_ylabel('MaxTemp')

        plt.subplot(2, 2, 4)
        fig = self.input_data.boxplot(column='Rainfall')
        fig.set_title('')
        fig.set_ylabel('Rainfall')

        plt.subplot(2, 2, 5)
        fig = self.input_data.boxplot(column='WindGustSpeed')
        fig.set_title('')
        fig.set_ylabel('WindGustSpeed')

        plt.subplot(2, 2, 6)
        fig = self.input_data.boxplot(column='WindSpeed9am')
        fig.set_title('')
        fig.set_ylabel('WindSpeed9am')

        plt.subplot(2, 2, 7)
        fig = self.input_data.boxplot(column='Humidity9am')
        fig.set_title('')
        fig.set_ylabel('Humidity9am')





