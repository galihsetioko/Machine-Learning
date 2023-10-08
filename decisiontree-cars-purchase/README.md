This is a machine learning model using a decision tree algorithm to classify consumer car buying behavior based on age and annual salary.
The data is obtained from the Kaggle website.
Here are some details about the program
- I used one hot encoder to convert 'Female' and 'Male' category data into binary.
- I used minmax scaller for data normalization (converting data into a range between 0 -1 so as to speed up model training).
- The validation process uses K-Fold Cross Validation with 5 folds.
- The score accuracy results are: 0.94

Columns and rows of the dataset:
1.	Gender
2.	Age
3.	Annual Salary

How to use it is by changing the parameters 
carsBuy.startPredict(Gender, Age, Annual Salary)
in the model.py file
