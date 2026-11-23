
import torch
import torch.nn.functional as F


def pgd_attack(model, images, labels, epsilon, step_size=0.01, steps=20):
    original_images = images.clone().detach()
    adversarial_images = original_images.clone()

    for step in range(steps):
        adversarial_images.requires_grad = True

        predictions = model(adversarial_images)
        loss = F.cross_entropy(predictions, labels)

        gradient = torch.autograd.grad(
            loss, adversarial_images
        )[0]

        # Move pixels in the direction that increases the loss
        adversarial_images = adversarial_images + step_size * gradient.sign()

        # Keep changes within the allowed epsilon range
        perturbation = torch.clamp(
            adversarial_images - original_images,
            min=-epsilon,
            max=epsilon
        )

        adversarial_images = original_images + perturbation

        # Keep pixel values between 0 and 1
        adversarial_images = torch.clamp(adversarial_images, 0, 1)
        adversarial_images = adversarial_images.detach()

    return adversarial_images
