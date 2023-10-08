import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler , LabelEncoder



'''Class Model menggunakan algoritma Decision Tree'''
class Model:
    '''Menerima string berupa lokasi file csv'''
    def __init__(self, csvfile):
        self.csvfile = csvfile
    
    '''Ini adalah process mengolah data menjadi bentuk yang diterima oleh model'''
    def textProcessing(self):
        self.opencsv = pd.read_csv(self.csvfile)
        self.opencsv.drop('User ID', axis=1, inplace=True)
        self.x = self.opencsv[['Gender', 'Age', 'AnnualSalary']]
        self.y = self.opencsv[['Purchased']]
        
        # one hot encoding data Gender menjadi bentuk biner
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.x.loc[:, 'Gender'] = self.encoder.fit_transform(self.x[['Gender']])

        # menggunakan min max scaller untuk mengubah data umur dan gaji jadi skala 0 - 1
        self.scaler = MinMaxScaler(feature_range=(0,1))
        data_x = self.x[['Age', 'AnnualSalary']]
        self.scaler.fit(data_x)
        self.x.loc[:, ['Age', 'AnnualSalary']] = self.scaler.transform(data_x)
        
    '''Menampilkan data'''
    def printData(self):
        print(self.opencsv.info())
        print(self.opencsv.head())
    
    '''Proses pemisahan data menjadi data training dan temporary'''
    def modelTrain(self):
        self.x_train , self.x_temp , self.y_train , self.y_temp = train_test_split(self.x, self.y, test_size=0.1, random_state=123)
        self.tree = DecisionTreeClassifier()
        self.tree = self.tree.fit(self.x_train, self.y_train)
    
    '''Membagi data temporary menjadi data test dan validasi'''
    def testModel(self):
        x_test , self.x_val , y_test , self.y_val = train_test_split(self.x_temp, self.y_temp, test_size=0.5, random_state=123)
        y_predict = self.tree.predict(x_test)
        accuracy = accuracy_score(y_test, y_predict)
        print(f"Accuracy Score : {accuracy:.2f}")
    
    '''Melakukan validasi dengan cross validation'''
    def crossVal(self):
        result = cross_val_score(self.tree, self.x_val, self.y_val, cv=5)
        print(f"Result Cross Validation KFold : {result}")
    
    '''Model bisa menerima input berupa (gender , umur , gaji) note : gaji dalam hitungan dollar'''
    def startPredict(self, gender, age, annual_salary):
       gender_encode = self.encoder.transform([[gender]])
       age_salary = self.scaler.transform([[age, annual_salary]])
    #    print(f"Gender encode : {gender_encode[0][0]}\nAge :  {age_salary[0][0]}\nAnnual Salary : {age_salary[0][1]}\n")
       final = self.tree.predict([[gender_encode[0][0], age_salary[0][0], age_salary[0][1]]])
       info_decision = ''
       if final[0] != 0:
           info_decision = 'Buy'
       else:
           info_decision = "Not Buy"
       print(f'\nGender : {gender}\nAge:{age}\nSalary : {annual_salary}\nDecision : {info_decision}')
    #    print(f"Decision Buy : {final[0]}")
       
    
    
    
        

carsBuy = Model('car_data.csv')
carsBuy.textProcessing()
carsBuy.modelTrain()
carsBuy.testModel()
carsBuy.crossVal()
carsBuy.startPredict('Male', 25, 10000)
# print(carsBuy.startPredict('Male', 25, 100000))
        