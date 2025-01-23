import torch
from torchvision import transforms
from torchvision.io import decode_image
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os
import matplotlib.pyplot as plt

img = decode_image("rootpath/2007_000027.jpg")

# print(img)

root = "/path_to_your_root/"
train_txt = "/rootpath/trainval.txt"
test_txt = "/rootpath/train.txt"
val_txt = "/rootpath/val.txt"

def load_data(text_file, root, transform = None):

    image_list = []
    label_list = []

    with open(text_file, "r") as f:
        for line in f:
            text_contains = line.strip().split()
            # print(text_contains[0]+ "        " + text_contains[1])

            image_name, labels = text_contains[0], int(text_contains[1])

            # print("/home/manu/AI_ML_DL/IMAGE_CLASSIFICATION/VOCdevkit/VOC2012/JPEGImages/"+image_name+".jpg")
            # print(os.path.join(root, f"{image_name}.jpg"))
            image = decode_image(root + image_name + ".jpg")
            # print(image)

            if transform:
                image = transform(image)

            image_list.append(image)
            label_list.append(labels)

    return torch.stack(image_list), torch.tensor(label_list)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size = 5, stride = 1, padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5, stride = 1)
        # self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # print("After Conv1:", x.shape)
        x = self.pool1(x)
        # print("After Pool1:", x.shape)
        x = torch.relu(self.conv2(x))
        # print("After Conv2:", x.shape)
        x = self.pool1(x)
        # print("After Pool2:", x.shape)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        # print("After Flatten:", x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),  # Resize to 32x32 for LeNet
    transforms.ToTensor(),       # Convert image to tensor
    # transforms.Lambda(lambda x: x.float()),
    transforms.Normalize((0.5,), (0.5,))
])

# print(transform(img))

train_images, train_labels = load_data(train_txt, root, transform)

# print(train_images)

train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_images, test_labels = load_data(test_txt, root, transform)
test_dataset = TensorDataset(test_images, test_labels)
test_Loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

val_images, val_labels = load_data(val_txt, root, transform)
val_dataset = TensorDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# num_epochs = 100
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")



train_losses = []
val_losses = []
val_accuracies = []

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    train_losses.append(running_loss/len(train_loader))

     #Validation after each epoch
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    if epoch == 50:
        learning_rate = 0.001
    if epoch == 80:
        learning_rate = 0.1


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

    print(learning_rate)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"Val loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {correct/total:.4f}")


#Plot

plt.figure(figsize = (12,6))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label = 'Train Loss', marker = "o")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker = "o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label = 'Validation Accuracy', marker = 'o', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
