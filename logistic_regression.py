# Librairies
import pandas as pd
import math
import matplotlib.pyplot as plt  # <-- Ajout pour le graphique

# Load the dataset
dataframe = pd.read_csv('data/diabetes.csv')

# Define variables
w0 = 0.0
w1 = 0.0
w2 = 0.0
w3 = 0.0
w4 = 0.0
w5 = 0.0
w6 = 0.0
w7 = 0.0
w8 = 0.0
w9 = 0.0

loss = 0
learning_rate = 0.0001
epochs = 100
delta = 0
losses = []  # <-- Liste pour enregistrer les pertes

# Method linear scoring
def linear_score(x1, x2, x3, x4, x5, x6, x7, x8):
    z = w0 + x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6 + x7*w7 + x8*w8
    return z

#Sigmoid function
def sigmoid_function(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)

# Cross entropy
def cross_entropy(y, p, eps=1e-15):
    p = max(min(p, 1 - eps), eps)
    loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return loss

# Gradient calcul
def gradient_calcul(y, p):
    error = p - y
    return error

# Calcul of gradient by weights
def grad(error, x):
    return error * x

# Weight update
def weight_update(learning_rate, w, grad):
    return w - learning_rate * grad

# Training while
for j in range(epochs):
    epoch_loss = 0
    for i in range(len(dataframe)):
        x1 = dataframe.loc[i, "Pregnancies"]
        x2 = dataframe.loc[i, "Glucose"]
        x3 = dataframe.loc[i, "BloodPressure"]
        x4 = dataframe.loc[i, "SkinThickness"]
        x5 = dataframe.loc[i, "Insulin"]
        x6 = dataframe.loc[i, "BMI"]
        x7 = dataframe.loc[i, "DiabetesPedigreeFunction"]
        x8 = dataframe.loc[i, "Age"]
        true_class = dataframe.loc[i, "Outcome"]

        z = linear_score(x1, x2, x3, x4, x5, x6, x7, x8)
        p = sigmoid_function(z)
        error = gradient_calcul(true_class, p)
        epoch_loss += cross_entropy(true_class, p)

        grad0 = grad(error, 1)
        grad1 = grad(error, x1)
        grad2 = grad(error, x2)
        grad3 = grad(error, x3)
        grad4 = grad(error, x4)
        grad5 = grad(error, x5)
        grad6 = grad(error, x6)
        grad7 = grad(error, x7)
        grad8 = grad(error, x8)

        w0 = weight_update(learning_rate, w0, grad0)
        w1 = weight_update(learning_rate, w1, grad1)
        w2 = weight_update(learning_rate, w2, grad2)
        w3 = weight_update(learning_rate, w3, grad3)
        w4 = weight_update(learning_rate, w4, grad4)
        w5 = weight_update(learning_rate, w5, grad5)
        w6 = weight_update(learning_rate, w6, grad6)
        w7 = weight_update(learning_rate, w7, grad7)
        w8 = weight_update(learning_rate, w8, grad8)

    epoch_loss /= len(dataframe)
    losses.append(epoch_loss)

    if loss != 0:
        delta = loss - epoch_loss
        if delta < 4e-6:
            print("Elbow is done stop")
            break
    loss = epoch_loss
    if delta != 0:
        print("epochs", j, ":", "Loss :", loss, "Delta :", delta)
    else:
        print("epochs", j, ":", "Loss :", loss)

print("Final weights", w0, w1, w2, w3, w4, w5, w6, w7, w8)

# Model testing on a new data
z = linear_score(3, 88, 58, 11, 54, 24.8, 0.267, 22)
diabet_probability = sigmoid_function(z)
print("Diabet probability :", diabet_probability)

# Graphic elbow method
plt.plot(range(len(losses)), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Elbow Method")
plt.grid(True)
plt.show()
