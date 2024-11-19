import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from repghost.model.repghost import RepGhostModule
from sklearn.metrics import precision_recall_fscore_support, classification_report
from fvcore.nn import FlopCountAnalysis

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
        x = F.relu(self.repghost1(x))
        x = F.max_pool2d(F.relu(self.repghost2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Initialize model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Net().to(device)

# Print Total Parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

# Calculate FLOPs
input_tensor = torch.randn(1, 1, 28, 28).to(device)
flops = FlopCountAnalysis(model, input_tensor)
print(f"FLOPs: {flops.total()}")

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
y_true = []
y_pred = []
with torch.no_grad():
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        y_true.extend(target.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Accuracy
accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')

# Precision, Recall, F1-Score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))
