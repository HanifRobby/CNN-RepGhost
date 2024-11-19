import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from repghost.model.repghost import RepGhostModule

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
trainset = datasets.MNIST(root='./data', train=True,
                          download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False,
                         download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# CNN Model with RepGhostModule
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Replace Conv2d with RepGhostModule from original code
        self.repghost1 = RepGhostModule(1, 32, kernel_size=3, stride=1)
        self.repghost2 = RepGhostModule(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Debugging tensor dimensions
        # print(f"Input shape: {x.shape}")
        x = F.relu(self.repghost1(x))
        # print(f"After repghost1: {x.shape}")
        x = F.max_pool2d(F.relu(self.repghost2(x)), 2)
        # print(f"After repghost2 and pooling: {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"After flatten: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Initialize model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Net().to(device)

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 5
for epoch in range(epochs):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)  # Move data to GPU if available
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(trainloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

# Evaluate model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')
