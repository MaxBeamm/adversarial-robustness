import torch


def cw_attack(model, images, target_labels, c=1.0,
              steps=200, learning_rate=0.01):

    original_images = images.clone().detach()

    # Transform images into the optimization space
    scaled_images = original_images * 2 - 1
    scaled_images = torch.clamp(scaled_images, -0.999999, 0.999999)

    w = torch.atanh(scaled_images).detach()
    w.requires_grad = True

    optimizer = torch.optim.Adam([w], lr=learning_rate)

    best_images = original_images.clone()

    best_distances = torch.full(
        (len(images),), float("inf"), device=images.device
    )

    successful = torch.zeros(
        len(images), dtype=torch.bool, device=images.device
    )

    for step in range(steps):

        # Convert back to valid pixel values
        adversarial_images = (torch.tanh(w) + 1) / 2

        predictions = model(adversarial_images)

        # Calculate squared L2 distance
        l2_distance = (
            (adversarial_images - original_images) ** 2
        ).flatten(1).sum(dim=1)

        # Get the score of the target class
        target_scores = predictions.gather(
            1, target_labels.unsqueeze(1)
        ).squeeze(1)

        # Get the highest score among the other classes
        other_scores = predictions.clone()
        other_scores.scatter_(1, target_labels.unsqueeze(1), -1e9)

        highest_other_scores = other_scores.max(dim=1).values

        # Calculate the C&W objective
        attack_loss = torch.clamp(
            highest_other_scores - target_scores, min=0
        )

        loss = (l2_distance + c * attack_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the best successful adversarial image
        with torch.no_grad():

            current_images = (torch.tanh(w) + 1) / 2
            current_predictions = model(current_images).argmax(dim=1)

            distances = (
                (current_images - original_images) ** 2
            ).flatten(1).sum(dim=1)

            better = (
                (current_predictions == target_labels)
                & (distances < best_distances)
            )

            best_distances[better] = distances[better]
            best_images[better] = current_images[better]
            successful[better] = True

    return best_images, successful