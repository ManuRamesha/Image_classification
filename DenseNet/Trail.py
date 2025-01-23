import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import decode_image
from torch.utils.data import DataLoader, TensorDataset
import math

import torch.nn.functional as F


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





class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck=True, drop_rate=0.0):
        super(DenseLayer, self).__init__()
        self.bottleneck = bottleneck
        self.drop_rate = drop_rate

        # Bottleneck layer reduces the number of input channels
        inter_channels = 4 * growth_rate if bottleneck else in_channels

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        if self.bottleneck:
            out = self.conv1(out)
            out = self.bn2(out)
            out = F.relu(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck=True, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck, drop_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes=20, growth_rate=12, depth=40, reduction=0.5, bottleneck=True, drop_rate=0.0):
        super(DenseNet, self).__init__()

        # Calculate number of layers in each block
        num_dense_layers = (depth - 4) // 3
        if bottleneck:
            num_dense_layers //= 2

        # Initial convolution layer
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Dense Block 1
        self.block1 = DenseBlock(num_dense_layers, num_channels, growth_rate, bottleneck, drop_rate)
        num_channels += num_dense_layers * growth_rate
        self.trans1 = TransitionLayer(num_channels, int(num_channels * reduction))
        num_channels = int(num_channels * reduction)

        # Dense Block 2
        self.block2 = DenseBlock(num_dense_layers, num_channels, growth_rate, bottleneck, drop_rate)
        num_channels += num_dense_layers * growth_rate
        self.trans2 = TransitionLayer(num_channels, int(num_channels * reduction))
        num_channels = int(num_channels * reduction)

        # Dense Block 3
        self.block3 = DenseBlock(num_dense_layers, num_channels, growth_rate, bottleneck, drop_rate)
        num_channels += num_dense_layers * growth_rate

        # Final batch normalization
        self.bn = nn.BatchNorm2d(num_channels)

        # Classification layer
        self.fc = nn.Linear(num_channels, num_classes)

        # Weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)







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
model = DenseNet(num_classes=20, growth_rate=12, depth=40, reduction=0.5, bottleneck=True, drop_rate=0.0).to(device)
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

