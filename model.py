import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

# ----------------------------
# PATHS
# ----------------------------
train_dir = r"D:\crowd_panic_project\datasets\final_dataset\train"
test_dir = r"D:\crowd_panic_project\datasets\final_dataset\test"

# ----------------------------
# IMAGE TRANSFORMS
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# ----------------------------
# LOAD DATA
# ----------------------------
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

print("Classes:", train_data.classes)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ----------------------------
# LOSS + OPTIMIZER
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ----------------------------
# TRAINING
# ----------------------------
epochs = 8

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss}")

# ----------------------------
# SAVE MODEL
# ----------------------------
torch.save(model.state_dict(), "crowd_behavior_model.pth")

print("Model saved")

# ----------------------------
# TEST ACCURACY
# ----------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("Test Accuracy:", accuracy)

