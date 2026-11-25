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

    # Select the first five correctly classified images
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        correct = model(images).argmax(dim=1) == labels
    images = images[correct][:5]
    labels = labels[correct][:5]

    fgsm_images = fgsm_attack(model, images, labels, 0.1)
    pgd_images = pgd_attack(
        model, images, labels, 0.1, step_size=0.01, steps=20
    )
    targets = (labels + 1) % 10
    model.requires_grad_(False)
    cw_images, successful = cw_binary_search(
        model, images, targets, c=1.0, steps=1000, search_steps=9
    )

    image_groups = [images, fgsm_images, pgd_images, cw_images]
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

    fig.suptitle("MNIST adversarial examples (predicted labels)")
    fig.tight_layout()
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/demo.png", dpi=200, bbox_inches="tight")
    print("Saved results/demo.png")
    plt.show()


if __name__ == "__main__":
    main()
