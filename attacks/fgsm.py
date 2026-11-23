
import torch
import torch.nn.functional as F


def fgsm_attack(model, images, labels, epsilon):
    # Make a copy so we can calculate gradients for the images
    images = images.clone().detach()
    images.requires_grad = True

    # Get predictions and calculate the loss
    predictions = model(images)
    loss = F.cross_entropy(predictions, labels)

    # Calculate how each pixel affects the loss
    model.zero_grad()
    loss.backward()

    # Change pixels in the direction that increases the loss
    gradient = images.grad
    adversarial_images = images + epsilon * gradient.sign()

    # Keep pixel values between 0 and 1
    adversarial_images = torch.clamp(adversarial_images, 0, 1)

    return adversarial_images.detach()
