import matplotlib.pyplot as plt
import numpy as np
def f(x):
    return x*x + 6*x + 8
def f_prime(x):
    return 2*x + 6
def gradient_descent(starting_point, learning_rate, num_iterations):
    x = starting_point
    for i in range (num_iterations):
        gradient = f_prime(x)
        x = x - learning_rate * gradient
        print (f'Iteration {i+1}: x = {x}, f(x) = {f(x)}')

learning_rates = [0.001, 0.01, 0.1, 1.0]
starting_point = 0
num_iterations = 50

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    f_values = gradient_descent(starting_point, lr, num_iterations)
    plt.plot(f_values, label=f"Learning rate = {lr}")
plt.title("Gradient Descent với các tốc độ học khác nhau")
plt.xlabel("Số bước lặp")
plt.ylabel("Giá trị của f(x)")
plt.legend()
plt.grid(True)
plt.show()
