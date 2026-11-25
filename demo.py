import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.mnist_cnn import MNISTModel
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.cw import cw_binary_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--successful', action='store_true')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(
        "models/mnist_model.pth", map_location=device, weights_only=True
    ))
    model.eval()

    test_data = datasets.MNIST(
        root="data", train=False, download=True,
        transform=transforms.ToTensor()
    )
    loader = DataLoader(test_data, batch_size=100, shuffle=False)

    # Select examples in dataset order
    selected = [[], [], [], []]
    selected_targets = []
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            correct = model(images).argmax(dim=1) == labels
        images, labels = images[correct], labels[correct]
        if len(images) == 0:
            continue
        if not args.successful:
            images, labels = images[:5-count], labels[:5-count]

        model.requires_grad_(True)
        fgsm_images = fgsm_attack(model, images, labels, 0.1)
        pgd_images = pgd_attack(
            model, images, labels, 0.1, step_size=0.01, steps=20
        )
        if args.successful:
            with torch.no_grad():
                both = ((model(fgsm_images).argmax(dim=1) != labels)
                        & (model(pgd_images).argmax(dim=1) != labels))
            images, labels = images[both], labels[both]
            fgsm_images, pgd_images = fgsm_images[both], pgd_images[both]
        if len(images) == 0:
            continue

        targets = (labels + 1) % 10
        model.requires_grad_(False)
        cw_images, successful = cw_binary_search(
            model, images, targets, c=1.0, steps=1000, search_steps=9
        )
        if not args.successful:
            successful = torch.ones_like(successful)
        indices = successful.nonzero(as_tuple=True)[0][:5-count]
        for group, values in zip(selected, [images, fgsm_images, pgd_images, cw_images]):
            group.append(values[indices].detach())
        selected_targets.append(targets[indices])
        count += len(indices)
        if count == 5:
            break

    if count == 0:
        print("No matching examples found.")
        return
    image_groups = [torch.cat(group) for group in selected]
    images = image_groups[0]
    targets = torch.cat(selected_targets)
    names = ["Original", "FGSM", "PGD", "C&W L2"]
    fig, axes = plt.subplots(len(images), 4, figsize=(9, 11), squeeze=False)

    for column, group in enumerate(image_groups):
        with torch.no_grad():
            predictions = model(group).argmax(dim=1)
        for row in range(len(images)):
            ax = axes[row, column]
            ax.imshow(group[row].detach().cpu().squeeze(), cmap="gray",
                      vmin=0, vmax=1)
            title = f"{names[column]}: {predictions[row].item()}"
            if column == 3:
                title += f" (target {targets[row].item()})"
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    title = "Selected successful examples" if args.successful else "MNIST adversarial examples"
    fig.suptitle(title + " (predicted labels)")
    fig.tight_layout()
    Path("results").mkdir(exist_ok=True)
    filename = "results/demo_successful.png" if args.successful else "results/demo.png"
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.show()


if __name__ == "__main__":
    main()
