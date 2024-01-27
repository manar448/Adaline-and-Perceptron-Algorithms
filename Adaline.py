import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def signum(x):
    return 1 if x >= 0 else -1


class Adaline:
    def __init__(self, num_of_features, learning_rate, epochs, mse_threshold, add_bias):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.weights = np.random.randn(num_of_features)
        self.bias = 0
        if add_bias:
            self.bias = np.random.randn()
        self.weights = np.append(self.weights, self.bias)

    def train(self, x, y):
        n = float(len(x))
        names_features = x.columns
        x1 = x[names_features[0]]
        x2 = x[names_features[1]]
        print(self.weights[1])
        print(self.weights[0])
        total_error = 0
        for epoch in range(self.epochs):
            for i in range(len(x1)):
                # net value
                y_i = self.weights[0] * x1[i] + self.weights[1] * x2[i] + self.weights[2]
                # class value
                error = y[i] - y_i
                self.weights[0] += self.learning_rate * error * x1[i]
                self.weights[1] += self.learning_rate * error * x2[i]
                self.weights[2] += self.learning_rate * error
                total_error += 0.5 * (error ** 2)
            MSE = total_error / n
            print(MSE)
            if MSE <= self.mse_threshold:
                break

    def draw(self, x, y):

        # Assuming you have trained the perceptron and obtained the weights and bias
        weights = np.array([self.weights[0], self.weights[1]])  # Replace w1 and w2 with your actual weight values
        bias = self.weights[2]  # Replace b with your actual bias value

        # Generate x values for plotting the line
        x_line = np.linspace(-1, 1, 100)

        # Calculate y values for the decision boundary line
        # y_line = (weights[0] * x_line + bias) / weights[1]
        y_line = -(weights[1] * x_line + bias) / weights[0]
        # Plot the decision boundary line
        plt.plot(x_line, y_line, color='red', label='Decision Boundary ')

        names_features = x.columns
        # Scatter plot for class 0 (negative class)
        plt.scatter(x[y == 1][names_features[0]], x[y == 1][names_features[1]], color='blue', label='Class 0')
        # Scatter plot for class 1 (positive class)
        plt.scatter(x[y == -1][names_features[0]], x[y == -1][names_features[1]], color='green', label='Class 1')
        # Add labels and legend to the plot
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        # Display the plot
        plt.show()

    # done test
    def predict(self, x):
        x = x.reset_index(drop=True)
        names_features = x.columns
        x1 = x[names_features[0]]
        x2 = x[names_features[1]]
        return signum(self.weights[0] * x1[0] + self.weights[1] * x2[0] + self.weights[2])

    def calc_evaluation(self, x_test, y_test):
        # calculate confusion matrix
        # display true positives(TP), true negatives(TN), false positives(FP), and false negatives(FN) produced by the model on the test data.
        # 2x2 matrix for binary classification
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(x_test)):
            x = x_test.iloc[i:i + 1, :]
            y_predicted = self.predict(x)
            # print(y_predicted)
            if y_predicted == 1 and y_test.iloc[i] == 1:
                TP += 1
            elif y_predicted == -1 and y_test.iloc[i] == -1:
                TN += 1
            elif y_predicted == 1 and y_test.iloc[i] == -1:
                FP += 1
            elif y_predicted == -1 and y_test.iloc[i] == 1:
                FN += 1

        print(TP, TN, FP, FN)
        # calculate accuracy
        # calculate accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        print("Confusion Matrix:")
        print("   TP: {}   FP: {}".format(TP, FP))
        print("   FN: {}   TN: {}".format(FN, TN))
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        # create confusion matrix dataframe
        confusion_matrix = pd.DataFrame({'Actual Positive': [TP, FN], 'Actual Negative': [FP, TN]},
                                        index=['Predicted Positive', 'Predicted Negative'])

        # plot heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu')
        plt.show()

        return accuracy


# linear

'''
# Test the classifier with the remaining samples
y_pred = adaline.predict(X_test)

# Calculate the confusion matrix and overall accuracy
confusion_matrix = confusion_matrix(y_test, y_pred)
overall_accuracy = accuracy_score(y_test, y_pred)
'''
