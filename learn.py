import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

# Define a simple function with two learnable parameters
class SimpleFunction(nn.Module):
    def __init__(self):
        super(SimpleFunction, self).__init__()
        # Initialize parameters a and b as learnable variables
        self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float))  # Start with a=1.0
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float))  # Start with b=0.0

    def forward(self, x):
        # Define the function f(x) = a * x + b + y

        return torch.sqrt(self.a * x[0] + self.b + x[1])

# Create data
x = torch.tensor(np.arange(1, 10))  # Input values
y = torch.tensor(np.arange(1, 10))
fx = torch.sqrt(torch.tensor(2 * np.arange(1,10) + 1 + np.arange(1, 10)))  # Target values (e.g., y = 2*x + 1)

# Initialize the model, loss function, and optimizer
model = SimpleFunction()
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Stochastic gradient descent

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    predictions = model((x, y))
    loss = criterion(predictions, fx)

    # Backward pass
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters

    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, a: {model.a.item():.4f}, b: {model.b.item():.4f}")

# Final parameters
print(f"Learned parameters: a = {model.a.item()}, b = {model.b.item()}")
