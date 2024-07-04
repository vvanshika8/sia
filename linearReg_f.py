import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def generate_data(use_randn=True, size=1000, x_min=0, x_max=100):
    if use_randn:
        x = np.random.randn(size)
    else:
        x = np.random.randint(x_min, x_max, size)
        x = (x - np.mean(x)) / np.std(x) # Normalization for randint
    m_gt = 2
    c_gt = 4
    y = m_gt * x + c_gt
    return x, y, m_gt, c_gt
'''
def normalize_data(x, y):
    scaler_x = StandardScaler().fit(x.reshape(-1, 1))
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    x_norm = scaler_x.transform(x.reshape(-1, 1)).flatten()
    y_norm = scaler_y.transform(y.reshape(-1, 1)).flatten()
    return x_norm, y_norm'''
def plot_data_and_ground_truth(x, y, y_gt):
    plt.scatter(x, y, label='Data')
    plt.plot(x, y_gt, label='Ground Truth (y = mx + c)', color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Generated Data and Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.show()
# Scikit-learn linear regression function
def sk_linear_regression(x, y):
    # Reshape x if it's a 1D array
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    predicted_m = model.coef_[0]
    predicted_c = model.intercept_
    print("\n----- Scikit-learn Linear Regression:")
    print("Predicted Slope (m):", predicted_m)
    print("Predicted y-intercept (c):", predicted_c)
# Closed-form solution function
def closed_form_solution(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator_m = np.sum((x - x_mean) * (y - y_mean))
    denominator_m = np.sum(np.square(x - x_mean))
    m = numerator_m / denominator_m
    c = y_mean - m * x_mean
    return m, c
# Gradient descent function
def gradient_descent(x, y, learning_rate=0.01, n_iterations=1000):
    n = float(len(x))
    m = np.random.rand()
    c = np.random.rand()
    for i in range(n_iterations):
        y_predicted = m * x + c
        dm = (-2.0 / n) * np.sum(x * (y - y_predicted))
        dc = (-2.0 / n) * np.sum(y - y_predicted)
        m -= learning_rate * dm
        c -= learning_rate * dc
    # progress
        if i % 100 == 0:
            print(f"Iteration {i}: m = {m}, c = {c}, Loss = {np.mean(np.square(y - y_predicted))}")
    return m, c
# GeneratiNg data
use_randn = True # Change to False to use randint
x, y, m_gt, c_gt = generate_data(use_randn=use_randn)
# If we're using randint, normalize data
'''
if not use_randn:
    x, y = normalize_data(x, y)'''
# ground truth LINE:
y_gt = m_gt * x + c_gt
# Plot data and ground truth
plot_data_and_ground_truth(x, y, y_gt)
# Scikit-learn linear regression
sk_linear_regression(x, y)
# Closed-form solution
m_closed, c_closed = closed_form_solution(x, y)
print("\n----- Closed-form Solution:")
print("Slope (m):", m_closed)
print("y-intercept (c):", c_closed)
# Gradient descent
learning_rate = 0.01
n_iterations = 1000
m_gd, c_gd = gradient_descent(x, y, learning_rate, n_iterations)
print("\n----- Gradient Descent Result:")
print("Slope (m):", m_gd)
print("y-intercept (c):", c_gd)