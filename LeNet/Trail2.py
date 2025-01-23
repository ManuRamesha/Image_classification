import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim


images_folder = "/root_path/JPEGImages"
annotations_folder = "/root_path/Annotations"

classes = ["aeroplane",
"bicycle",
"bird",
"boat",
"bottle",
"bus",
"car",
"cat",
"chair",
"cow",
"diningtable",
"dog",
"horse",
"motorbike",
"person",
"pottedplant",
"sheep",
"sofa",
"train",
"tvmonitor"]


# Label mapping
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
# print(class_to_idx)


# Function to parse XML and extract labels
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    file_name = root.find("filename").text
    class_name = root.find("object").find("name").text
    return file_name, class_to_idx[class_name]

# Prepare a list of file-label pairs
def prepare_dataset(images_folder, annotations_folder):
    data = []
    for annotation_file in os.listdir(annotations_folder):
        annotation_path = os.path.join(annotations_folder, annotation_file)
        file_name, label = parse_annotation(annotation_path)
        image_path = os.path.join(images_folder, file_name)
        data.append((image_path, label))
    return data

dataset = prepare_dataset(images_folder, annotations_folder)

# print(dataset[5])



class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32 for LeNet
    transforms.ToTensor(),       # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Create dataset and dataloader

full_dataset = CustomDataset(dataset, transform=transform)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
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

    #Validation after each epoch
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


    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"Val loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {correct/total:.4f}")

#save the model
torch.save(model.state_dict(),"lenet5_rgb.pth")




