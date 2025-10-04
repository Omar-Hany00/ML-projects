import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

students=pd.read_csv(r"C:\Users\Omar\Downloads\archive\StudentPerformanceFactors.csv")
students=students.dropna()

x=students[['Hours_Studied','Attendance','Sleep_Hours']]
y=students['Exam_Score']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=10)

model=LinearRegression()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(students['Exam_Score'])
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

plt.figure(figsize=(8,5),dpi=120)
plt.scatter(y_test, y_pred, color='blue',s=17)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='green',label='Regression line')
plt.xticks(np.arange(60,105,5))
plt.yticks(np.arange(60,105,5))
plt.xlabel('Real scores')
plt.ylabel('Predicted scores')
plt.title('Students Score Sredictions')
plt.legend()
plt.savefig('Students score predictions.jpg',dpi=120)
plt.show()