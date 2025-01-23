import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import decode_image
from torch.utils.data import DataLoader, TensorDataset

root = "/path_to_your_root/"
train_txt = "/rootpath/trainval.txt"
test_txt = "/rootpath/train.txt"
val_txt = "/rootpath/val.txt"

def load_data(text_file, root, transform = None):

    image_list = []
    label_list= []

    with open(text_file, "r") as f:
        for line in f:
            text_contains = line.strip().split()

            image_name, labels = text_contains[0], int(text_contains[1])
            image = decode_image(root + image_name + ".jpg")
            #print(image)

            if transform:
                image = transform(image)

            image_list.append(image)
            label_list.append(labels)

    return torch.stack(image_list), torch.tensor(label_list)

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding  = 1,)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding  = 1,)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding  = 1,)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding  = 1,)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding  = 1,)
        self.conv6 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding  = 1,)
        self.conv7 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding  = 1,)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding  = 1,)
        self.conv9 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding  = 1,)
        self.conv10 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding  = 1,)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding  = 1,)
        self.conv12 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding  = 1,)
        self.conv13 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding  = 1,)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)


        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 20)




    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)

        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        x = self.pool3(x)

        x = torch.relu(self.conv8(x))
        x = torch.relu(self.conv9(x))
        x = torch.relu(self.conv10(x))
        x = self.pool4(x)

        x = torch.relu(self.conv11(x))
        x = torch.relu(self.conv12(x))
        x = torch.relu(self.conv13(x))
        x = self.pool5(x)

        x = x.view(-1, 512 * 7 * 7)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

train_images, train_labels = load_data(train_txt, root, transform)
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)


test_images, test_labels = load_data(test_txt, root, transform)
test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

val_images, val_labels = load_data(val_txt, root, transform)
val_dataset = TensorDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

train_losses = []
val_losses = []
val_accuracies = []


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        #print(model(images))
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss/len(train_loader))

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss +=loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct / total)


    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"Val loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {correct/total:.4f}")

plt.figure(figsize = (12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label = "Train loss", marker = "*")
plt.plot(range(1, num_epochs + 1), val_losses, label = "Validation loss", marker = "*")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label = "Validation Accuracy", marker = "*", color = "green")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

