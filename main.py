import time
import GUI
from Perceptron import Perceptron
import numpy as np
from Adaline import Adaline
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Loading El Data
dataset = pd.read_excel("C:/Users/Malak/Documents/Semester 7/Neural Networks & Deep Learning/17/Dry_Bean_Dataset.xlsx", engine='openpyxl')

# Fill El Missing Values with El Mean
dataset = dataset.fillna({'Area': dataset['Area'].mean(), 'Perimeter': dataset['Perimeter'].mean(),
                          'MajorAxisLength': dataset['MajorAxisLength'].mean(),
                          'MinorAxisLength': dataset['MinorAxisLength'].mean(),
                          'Class': dataset['Class'].mode()})

# Retrieve Data From GUI
time.sleep(3)
data = GUI.gui_data

# Split Date into variables
feature1 = data[0]
feature2 = data[1]
class1 = data[2]
class2 = data[3]
learning_rate = data[4]
epoch = data[5]
mse = data[6]
basis = data[7]
algorithm = data[8]

print("feature1: " + feature1 + "		" + "class1: " + class1)
print("feature2: " + feature2 + "		" + "class2: " + class2)
print("learning_rate: " + learning_rate)
print("epoch: " + epoch+"		"+"mse: " + mse+"		"+"basis: " + basis)
print("algorithm: " + algorithm)


# Assuming to gui take only two features and only two classes
def selected_choices(dataset, class1, class2, feature1, feature2):
    dataframe = pd.DataFrame(dataset, columns=['Class', feature1, feature2])
    if class1 != "SIRA" and class2 != "SIRA":
        dataframe = dataframe.loc[dataframe['Class'] != "SIRA"]

    elif class1 != "CALI" and class2 != "CALI":
        dataframe = dataframe.loc[dataframe['Class'] != "CALI"]

    elif class1 != "BOMBAY" and class2 != "BOMBAY":
        dataframe = dataframe.loc[dataframe['Class'] != "BOMBAY"]

    dataframe = dataframe.reset_index(drop=True)
    x = dataframe.iloc[:, 1:]
    y = dataframe['Class']

    return x, y


# Encoding function
def Encoding(y, class1, class2):
    # Step 1: Define a mapping dictionary for replacement
    mapping = {class1: -1, class2: 1}

    # Step 2: Use the replace function to replace values
    y = y.replace(mapping)
    y = y.reset_index(drop=True)
    return y.astype(int)


# normalization or standarization
def scaling(x, method):
    if (method == 1):
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
    elif (method == 2):
        scaler = StandardScaler()
        x = scaler.fit_transform(x)  # calc mean and standard division
    return x


x, y = selected_choices(dataset, class1, class2, feature1, feature2)
# shuffle and split data
class_A_mask = (y == class1)
class_B_mask = (y == class2)

# Shuffle the indices of each class
np.random.seed(30)
class_A_indices = np.random.choice(np.where(class_A_mask)[0], size=30, replace=False)
class_B_indices = np.random.choice(np.where(class_B_mask)[0], size=30, replace=False)


# Select 30 random samples from each class for training
train_indices = np.concatenate((class_A_indices, class_B_indices))
X_train, y_train = x.iloc[train_indices], y.iloc[train_indices]
y_train = Encoding(y_train, class1, class2)

# Select the remaining samples for testing
test_indices = np.setdiff1d(range(len(x)), train_indices)
X_test, y_test = x.iloc[test_indices], y.iloc[test_indices]

y_test = Encoding(y_test, class1, class2)
# indices from 0 +++
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

X_train_stand = scaling(X_train, 2)
# print(X_train)
X_test_stand = scaling(X_test, 2)


# back to dataframe format
df_train = pd.DataFrame()
df_test = pd.DataFrame()
df_train[feature1] = X_train_stand[:, 0]
df_train[feature2] = X_train_stand[:, 1]
df_test[feature1] = X_test_stand[:, 0]
df_test[feature2] = X_test_stand[:, 1]
X_test1 = df_test.iloc[39:40, :]
print("x shape: ", df_train.shape)
print("y shape: ", y_train.shape)

if algorithm == "Perceptron":
    # perceptron algorithms
    perceptron = Perceptron(num_of_features=2, learning_rate=float(learning_rate), epochs=int(epoch),
                            mse_threshold=float(mse), add_bias=basis)
    perceptron.train(df_train, y_train)
    print("--------------------------------------- train")
    y_predicted = perceptron.predict(X_test1)
    print("---------------------------------------")
    print(y_predicted)
    print(y_test[39])
    print("---------------------------------------")
    perceptron.draw(df_train, y_train)
    accuracy = perceptron.calc_evaluation(df_test, y_test)
elif algorithm == "Adaline":
    # adaline algorithms
    print("adaline")
    adaline = Adaline(num_of_features=2, learning_rate=float(learning_rate), epochs=int(epoch),
                      mse_threshold=float(mse), add_bias=basis)
    adaline.train(df_train, y_train)
    y_predicted = adaline.predict(X_test1)
    accuracy = adaline.calc_evaluation(df_test, y_test)
    adaline.draw(df_train, y_train)

print(y_predicted)
print(y_test[39])

print(X_test1)
