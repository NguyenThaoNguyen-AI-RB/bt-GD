import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 6*x + 8

def f_derivative(x):
    return 2*x + 6

def step_decay(initial_lr, decay_rate, decay_steps, iteration):
    return initial_lr * (decay_rate ** (iteration // decay_steps))

def exponential_decay(initial_lr, decay_rate, iteration):
    return initial_lr * np.exp(-decay_rate * iteration)

def gradient_descent_lr_scheduler(starting_point, initial_learning_rate, num_iterations, decay_strategy, decay_rate, decay_steps=None):
    x = starting_point
    learning_rate = initial_learning_rate
    x_values = []
    f_values = []
    
    for i in range(num_iterations):
        if decay_strategy == "step":
            learning_rate = step_decay(initial_learning_rate, decay_rate, decay_steps, i)
        elif decay_strategy == "exponential":
            learning_rate = exponential_decay(initial_learning_rate, decay_rate, i)
        
        x = x - learning_rate * f_derivative(x)
        x_values.append(x)
        f_values.append(f(x))
        
        print(f"Iteration {i+1}: x = {x}, f(x) = {f(x)}, learning_rate = {learning_rate}")
    
    return f_values

starting_point = 0
initial_learning_rate = 0.1
num_iterations = 50

step_decay_results = gradient_descent_lr_scheduler(starting_point, initial_learning_rate, num_iterations, decay_strategy="step", decay_rate=0.5, decay_steps=10)
exp_decay_results = gradient_descent_lr_scheduler(starting_point, initial_learning_rate, num_iterations, decay_strategy="exponential", decay_rate=0.1)

plt.figure(figsize=(10, 6))
plt.plot(step_decay_results, label="Step Decay")
plt.plot(exp_decay_results, label="Exponential Decay")
plt.title("Gradient Descent với Step Decay và Exponential Decay")
plt.xlabel("Số bước lặp")
plt.ylabel("Giá trị của f(x)")
plt.legend()
plt.grid(True)
plt.show()