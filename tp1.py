import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


data = p.read_csv(r"C:\Users\cyber\Downloads\grades.csv")
print("\033[91m dimension:\033[0m\t ",data.shape)
print("\033[91m file infromation:\033[0m \n", data.info())
print("\033[91m data description:\033[0m \n",data.describe())
print("\033[91m show only the top 5 line:\033[0m \n",data.head())
print("\033[91m let's count :\033[0m \n",data.value_counts())

x = data["happiness"]
y = data[["lifeexp","unemployment"]]


x_train,y_train,x_test,y_test = train_test_split(x,y,test_size=0.3)
model = KNeighborsClassifier(n_neighbors = 12)
print("\033[91m x_train values:\033[0m \n",x_train)
print("\033[91m y_train values:\033[0m \n",y_train)
print("\033[91m x_test values:\033[0m \n",x_test)
print("\033[91m y_test values:\033[0m \n",y_test)
model.fit(x_train,y_train)
model.predict(x_test)
