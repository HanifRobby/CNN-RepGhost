import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Define RepGhostModule


class RepGhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, relu=True):
        super(RepGhostModule, self).__init__()
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(oup, oup, kernel_size, 1, kernel_size //
                      2, groups=oup, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return x1 + x2


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

# Define CNN Model with RepGhostModule
# Define CNN Model with RepGhostModule


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Replace Conv2d with RepGhostModule
        self.repghost1 = RepGhostModule(1, 32, kernel_size=3, stride=1)
        self.repghost2 = RepGhostModule(32, 64, kernel_size=3, stride=1)
        # Update input size to match flattened output
        self.fc1 = nn.Linear(12544, 128)  # Updated input size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.repghost1(x))
        # print("After repghost1:", x.shape)  # Debugging tensor shape
        x = F.max_pool2d(F.relu(self.repghost2(x)), 2)
        # print("After repghost2 and pooling:", x.shape)
        x = torch.flatten(x, 1)
        # print("After flatten:", x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 5
for epoch in range(epochs):
    model.train()
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in testloader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')
