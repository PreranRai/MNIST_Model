import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Step 1: Setup
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------
# Step 2: MNIST Datasets
# -----------------------
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# -----------------------
# Step 3: Directory to Save Models
# -----------------------
os.makedirs("saved_models", exist_ok=True)

# -----------------------
# Step 4: Binary Classifier
# -----------------------
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

# -----------------------
# Step 5: Train + Save 10 Models
# -----------------------
def train_and_save_models(loader):
    classifiers = []
    for digit in range(10):
        print(f"\nTraining classifier {digit} vs non-{digit}")
        model = BinaryClassifier().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(3):  # adjust epochs for better accuracy
            model.train()
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                labels = (target == digit).float().unsqueeze(1)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), f"saved_models/digit_{digit}.pth")
        classifiers.append(model)
    return classifiers

# -----------------------
# Step 6: Load Models
# -----------------------
def load_models():
    models = []
    for digit in range(10):
        model = BinaryClassifier().to(device)
        model.load_state_dict(torch.load(f"saved_models/digit_{digit}.pth", map_location=device))
        model.eval()
        models.append(model)
    return models

# -----------------------
# Step 7: Multiclass Prediction (Argmax)
# -----------------------
def multiclass_predict(models, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            probs = [m(data).cpu().detach().numpy() for m in models]  # <-- detach() added
            probs = np.concatenate(probs, axis=1)
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(target.numpy())
    return np.array(all_preds), np.array(all_labels)

def evaluate(models, loader=test_loader):
    y_pred, y_true = multiclass_predict(models, loader)
    accuracy = (y_pred == y_true).mean()
    print("\nAccuracy:", accuracy)
    return y_pred, y_true

# -----------------------
# Step 8: Show Sample Predictions
# -----------------------
def show_samples(models, loader=test_loader, num_samples=10):
    examples = iter(loader)
    images, labels = next(examples)
    images = images.to(device)
    probs = [m(images).cpu().detach().numpy() for m in models]  # <-- detach() added
    probs = np.concatenate(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    plt.figure(figsize=(12, 4))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap="gray")
        plt.title(f"Pred: {preds[i]}, True: {labels[i].item()}")
        plt.axis("off")
    plt.show()

# -----------------------
# Step 9: Custom Handwritten Images (Folder)
# -----------------------
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))
    img_tensor = transforms.ToTensor()(img)
    img_tensor = transforms.Normalize((0.5,), (0.5,))(img_tensor)
    return img_tensor.unsqueeze(0)

def predict_custom_folder(models, folder="my_digits"):
    for fname in os.listdir(folder):
        if not (fname.endswith(".png") or fname.endswith(".jpg")):
            continue
        fpath = os.path.join(folder, fname)
        img_tensor = load_and_preprocess_image(fpath).to(device)
        probs = [m(img_tensor).cpu().detach().numpy() for m in models]  # <-- detach() added
        probs = np.concatenate(probs, axis=1)
        pred = np.argmax(probs, axis=1)[0]
        print(f"{fname} → Predicted digit: {pred}")

# -----------------------
# Step 10: Custom Large Dataset (CSV)
# -----------------------
class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

def predict_csv_dataset(models, csv_file):
    dataset = CSVDataset(csv_file, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    y_pred, y_true = evaluate(models, loader)
    print("\nPredictions on CSV dataset complete.")
    return y_pred, y_true

# -----------------------
# Step 11: Main Execution
# -----------------------
if __name__ == "__main__":
    # Train models only if not already saved
    if not os.listdir("saved_models"):
        classifiers = train_and_save_models(train_loader)
    else:
        classifiers = load_models()

    print("\nEvaluating on MNIST Test Set:")
    evaluate(classifiers)
    show_samples(classifiers, num_samples=10)

    print("\nPredicting on custom handwritten folder images:")
    predict_custom_folder(classifiers, folder="my_digits")

    # If you have a CSV dataset in future
    if os.path.exists("data_labels.csv"):
        print("\nPredicting on large CSV dataset:")
        predict_csv_dataset(classifiers, "data_labels.csv")
