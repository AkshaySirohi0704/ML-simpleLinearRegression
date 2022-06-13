#Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd


# salary.csv contains one feature (Linear)
# Year of expierence and salary 
print('-------------------------------------------------')
print('Seprating features and dependent variable . . . ')
dataset = pd.read_csv('salary.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
# Spliting
from sklearn.model_selection import train_test_split
print('-------------------------------------------------')
print('splitting . . . ')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Displaying
print('-------------------------------------------------')
print('Length of X_train', len(X_train))
print(X_train)
print('-------------------------------------------------')
print('Length of X_test', len(X_test))
print(X_test)
print('-------------------------------------------------')
print('Length of y_train', len(y_train))
print(y_train)
print('-------------------------------------------------')
print('Length of y_test', len(y_test))
print(y_test)
print('-------------------------------------------------')

# Machine learns
from sklearn.linear_model import LinearRegression
print('-------------------------------------------------')
print('Machine is learning . . .')
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting
print('-------------------------------------------------')
print('Predicting . . .')
y_pred = regressor.predict(X_test)


print('-------------------------------------------------')
print('Visualising the training set')
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print('-------------------------------------------------')
print('Visualising the testing set')
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

