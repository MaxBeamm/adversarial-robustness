import argparse
import time
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mnist_cnn import MNISTModel
from attacks.cw import cw_attack, cw_binary_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--search-steps", type=int, default=0)
    args = parser.parse_args()
    if args.search_steps < 0:
        parser.error("Search steps cannot be negative.")

    if min(args.samples, args.steps, args.batch_size) <= 0 or args.c <= 0:
        parser.error("Samples, steps, batch size and c must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_data = datasets.MNIST(
        root="data", train=False, download=True,
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(
        "models/mnist_model.pth", map_location=device, weights_only=True
    ))
    model.eval()
    model.requires_grad_(False)

    total_tested = 0
    total_successful = 0
    total_l2 = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Only test correctly classified images
        with torch.no_grad():
            predictions = model(images).argmax(dim=1)

        correct = predictions == labels
        remaining = args.samples - total_tested
        images = images[correct][:remaining]
        labels = labels[correct][:remaining]

        if len(images) == 0:
            continue

        target_labels = (labels + 1) % 10
        if args.search_steps > 0:
            adversarial_images, successful = cw_binary_search(
                model, images, target_labels, c=args.c,
                steps=args.steps, search_steps=args.search_steps
            )
        else:
            adversarial_images, successful = cw_attack(
                model, images, target_labels,
                c=args.c, steps=args.steps, learning_rate=0.01
            )

        distances = torch.sqrt(
            ((adversarial_images - images) ** 2).flatten(1).sum(dim=1)
        )

        total_tested += len(images)
        total_successful += successful.sum().item()
        total_l2 += distances[successful].sum().item()

        print(f"Progress: {total_tested}/{args.samples}", flush=True)
        if total_tested >= args.samples:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    if total_tested == 0:
        print("No correctly classified images found.")
        return

    if total_tested < args.samples:
        print("Reached the end of the dataset.")

    print("\nFinal Results")
    print(f"Images tested: {total_tested}")
    print(f"Initial C: {args.c}")
    print(f"Search rounds: {args.search_steps}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Successful attacks: {total_successful}")
    print(f"Attack success rate: {100 * total_successful / total_tested:.2f}%")

    if total_successful > 0:
        print(f"Average L2 distance: {total_l2 / total_successful:.4f}")

    print(f"Execution time: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    main()
