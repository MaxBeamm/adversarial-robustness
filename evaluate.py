
import argparse
import time
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

    total_adversarial_correct = 0
    total_l2 = 0.0
    total_linf = 0.0

    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

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

        total_adversarial_correct += (adversarial_predictions == labels).sum().item()

        # Measure perturbations for successful attacks only
        differences = (adversarial_images - images).flatten(1)
        l2 = differences.norm(p=2, dim=1)
        linf = differences.abs().max(dim=1).values
        total_l2 += l2[successful_attacks].sum().item()
        total_linf += linf[successful_attacks].sum().item()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    original_accuracy = 100 * total_originally_correct / len(test_data)
    attack_success_rate = (
        100 * total_successful_attacks / total_originally_correct
    )

    print(f"Original accuracy: {original_accuracy:.2f}%")
    print(f"Attack: {args.attack.upper()}")
    print(f"Epsilon: {epsilon}")
    print(f"Successful attacks: {total_successful_attacks}")
    print(f"Attack success rate: {attack_success_rate:.2f}%")
    print(f"Accuracy after attack: {100 * total_adversarial_correct / len(test_data):.2f}%")
    if total_successful_attacks > 0:
        print(f"Average L2 distance: {total_l2 / total_successful_attacks:.4f}")
        print(f"Average Linf distance: {total_linf / total_successful_attacks:.4f}")
    print(f"Execution time: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    main()
