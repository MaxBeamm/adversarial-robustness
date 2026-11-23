
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# A simple CNN for recognizing MNIST digits
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def main():
    # Use the GPU if available, otherwise use the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Convert MNIST images into PyTorch tensors
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = MNISTModel().to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model for 3 epochs
    for epoch in range(3):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            predictions = model(images)
            loss = loss_function(predictions, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/3 - Loss: {average_loss:.4f}")

    # Check how well the model recognizes unseen digits
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            predicted_labels = predictions.argmax(dim=1)

            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")

    # Save the trained model for our adversarial attacks
    torch.save(model.state_dict(), "models/mnist_model.pth")
    print("Model saved to models/mnist_model.pth")


if __name__ == "__main__":
    main()
