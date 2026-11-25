import time
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.mnist_cnn import MNISTModel
from attacks.cw import cw_attack


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_loader = DataLoader(
        test_data, batch_size=64, shuffle=False
    )

    model = MNISTModel().to(device)

    model.load_state_dict(
        torch.load(
            "models/mnist_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()
    model.requires_grad_(False)

    # Select correctly classified images
    images, labels = next(iter(test_loader))

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        predictions = model(images).argmax(dim=1)

    correct = predictions == labels

    images = images[correct][:10]
    labels = labels[correct][:10]

    # Select the next digit as the target
    target_labels = (labels + 1) % 10

    c = 1.0
    steps = 10000
    learning_rate = 0.01

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    adversarial_images, successful = cw_attack(
        model,
        images,
        target_labels,
        c=c,
        steps=steps,
        learning_rate=learning_rate
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    # Calculate L2 perturbation distances
    distances = torch.sqrt(
        ((adversarial_images - images) ** 2)
        .flatten(1).sum(dim=1)
    )

    successful_count = successful.sum().item()
    success_rate = 100 * successful_count / len(images)

    print(f"Images tested: {len(images)}")
    print(f"C: {c}")
    print(f"Steps: {steps}")
    print(f"Successful attacks: {successful_count}")
    print(f"Attack success rate: {success_rate:.2f}%")

    if successful_count > 0:
        average_l2 = distances[successful].mean().item()
        print(f"Average L2 distance: {average_l2:.4f}")

    print(f"Execution time: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    main()