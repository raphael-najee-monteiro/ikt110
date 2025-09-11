import csv
import numpy as np
import random

# open file
file_path = "datasets/train.csv"

xs = []
ys = []
theta = np.zeros(3)

with open(file_path, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        x1 = float(row["x1"])
        x2 = float(row["x2"])
        y = float(row["y"])
        xs.append([x1, x2])
        ys.append(y)


# define function
def func(x, theta):
    return theta[0] + theta[1] * x[0] + theta[2] * x[1]

def loss_L2(y_pred, y):
    return (y - y_pred) ** 2


# training
max_iter = 10000
best_loss = np.inf
best_params = None


for i in range(max_iter):
    # select parameters
    theta_try = [random.uniform(-10, 10) for _ in range(3)]
    loss = 0.0

    for x, y in zip(xs, ys):
        y_pred = func(x, theta_try)
        loss += loss_L2(y, y_pred)



    # evaluate with threshold
    if loss < best_loss:
        best_loss = loss
        best_params = theta_try
        print("new best found: ", best_loss)



print("best loss:", best_loss)
print("best params:", best_params)