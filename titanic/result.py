import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers

# Read data
train_data = pd.read_csv(r"C:\Users\0720k\myapplications\open_ml_learning_record\titanic\train.csv")
test_data = pd.read_csv(r"C:\Users\0720k\myapplications\open_ml_learning_record\titanic\test.csv")
for_result_test_data = pd.read_csv(r"C:\Users\0720k\myapplications\open_ml_learning_record\titanic\test.csv")

# Remove un-use columns

def drop_columns(df, list_columns=[]):
  return df.drop(list_columns, axis =1 )

list_columns =  ["PassengerId", "Name", "Ticket", "Fare", "Cabin", "Embarked"]
train_data = drop_columns(train_data, list_columns)
list_columns_test =  ["PassengerId", "Name", "Ticket", "Fare", "Cabin", "Embarked"]
test_data = drop_columns(test_data, list_columns_test)

# Remove nan in exist columns

train_data.fillna(train_data.mean(),inplace=True)
test_data.fillna(test_data.mean(),inplace=True)

# Split purpose

train_data_y = train_data.Survived
list_columns_survived =  ["Survived"]
train_data = drop_columns(train_data, list_columns_survived)

# one-hot vector

def one_hot_vector(df, list_OHV_columns):
  for column_name in list_OHV_columns:
    df = pd.get_dummies(df, columns=[column_name], prefix="oh"+column_name, sparse=True)
  return df

one_hot_columns = ["Pclass"]
train_data = one_hot_vector(train_data, one_hot_columns)
test_data = one_hot_vector(test_data, one_hot_columns)


# label encoding
sex_le = LabelEncoder()
train_data.Sex = sex_le.fit_transform(train_data.Sex)
test_data.Sex = sex_le.fit_transform(test_data.Sex)

# Standardization

sc = StandardScaler()
train_data.Age = sc.fit_transform(train_data.Age.values.reshape(-1, 1))
test_data.Age = sc.transform(test_data.Age.values.reshape(-1,1))

# to numpy
X = train_data.values
y = train_data_y.values

# Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


model = Sequential()
model.add(Dense(100, activation = "relu", input_dim = X_train.shape[1], name = "layer_1", kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(50, activation = "sigmoid", name = "layer_2", kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(25, activation = "sigmoid", name = "layer_3" , kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1, activation = "sigmoid", name ="output_layer")) 
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])

model.fit(X_train, y_train, epochs=5000, batch_size=16)


score = model.evaluate(X_train, y_train)
print("")
print("Train loss:{0}".format(score[0]))
print("Train accuracy:{0}".format(score[1]))
t_score = model.evaluate(X_test, y_test)
print("")
print("Test loss:{0}".format(t_score[0]))
print("Test accuracy:{0}".format(t_score[1]))

result = model.predict(test_data.values)

df_suv = pd.DataFrame(result, columns = ["Survived"])
df_suv.Survived = round(df_suv.Survived)
df_suv.Survived = df_suv.Survived.astype('int8')
df_r = pd.concat([for_result_test_data, df_suv], axis=1)
df_r.drop(columns=[ "Name", "Ticket", "Fare", "Cabin", "Embarked",'Sex', 'Age', 'SibSp', 'Parch', 'Pclass'],inplace =True)
df_r.to_csv("test_result.csv", index = False)