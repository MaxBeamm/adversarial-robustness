
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mnist_cnn import MNISTModel
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack",
        choices=["fgsm", "pgd"],
        default="fgsm"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = MNISTModel().to(device)
    model.load_state_dict(
        torch.load(
            "models/mnist_model.pth",
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    epsilon = 0.1
    total_originally_correct = 0
    total_successful_attacks = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            original_predictions = model(images).argmax(dim=1)

        originally_correct = original_predictions == labels

        if args.attack == "fgsm":
            adversarial_images = fgsm_attack(
                model, images, labels, epsilon
            )
        else:
            adversarial_images = pgd_attack(
                model, images, labels, epsilon,
                step_size=0.01, steps=20
            )

        with torch.no_grad():
            adversarial_predictions = model(adversarial_images).argmax(dim=1)

        successful_attacks = (
            originally_correct & (adversarial_predictions != labels)
        )

        total_originally_correct += originally_correct.sum().item()
        total_successful_attacks += successful_attacks.sum().item()

    original_accuracy = 100 * total_originally_correct / len(test_data)
    attack_success_rate = (
        100 * total_successful_attacks / total_originally_correct
    )

    print(f"Original accuracy: {original_accuracy:.2f}%")
    print(f"Attack: {args.attack.upper()}")
    print(f"Epsilon: {epsilon}")
    print(f"Successful attacks: {total_successful_attacks}")
    print(f"Attack success rate: {attack_success_rate:.2f}%")


if __name__ == "__main__":
    main()
