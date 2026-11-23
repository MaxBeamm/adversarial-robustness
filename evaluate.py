
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mnist_cnn import MNISTModel
from attacks.fgsm import fgsm_attack


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the MNIST test dataset
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Load our previously trained neural network
    model = MNISTModel().to(device)
    model.load_state_dict(
        torch.load("models/mnist_model.pth", map_location=device, weights_only=True)
    )
    model.eval()

    epsilon = 0.1

    total_correct_original = 0
    total_successful_attacks = 0
    total_originally_correct = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Check which images the model classifies correctly
        with torch.no_grad():
            original_predictions = model(images).argmax(dim=1)

        originally_correct = original_predictions == labels

        # Generate adversarial images using FGSM
        adversarial_images = fgsm_attack(model, images, labels, epsilon)

        with torch.no_grad():
            adversarial_predictions = model(adversarial_images).argmax(dim=1)

        # An attack succeeds if a previously correct prediction becomes wrong
        successful_attacks = originally_correct & (adversarial_predictions != labels)

        total_correct_original += originally_correct.sum().item()
        total_originally_correct += originally_correct.sum().item()
        total_successful_attacks += successful_attacks.sum().item()

    attack_success_rate = 100 * total_successful_attacks / total_originally_correct
    original_accuracy = 100 * total_correct_original / len(test_data)

    print(f"Original accuracy: {original_accuracy:.2f}%")
    print(f"FGSM epsilon: {epsilon}")
    print(f"Successful attacks: {total_successful_attacks}")
    print(f"Attack success rate: {attack_success_rate:.2f}%")


if __name__ == "__main__":
    main()
