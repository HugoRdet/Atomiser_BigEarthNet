import torch
from sklearn.metrics import average_precision_score, f1_score


def accuracy(y: torch.Tensor, y_hat: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute the accuracy for multi-label classification, focusing only on active labels.

    Args:
        y (torch.Tensor): Ground truth labels (batch_size, num_classes), multi-hot encoded.
        y_hat (torch.Tensor): Predicted probabilities (batch_size, num_classes).
        threshold (float): Threshold to decide positive predictions.

    Returns:
        float: Percentage of correctly predicted active labels among total active labels.
    """
    y_pred = (y_hat >= threshold).float()

    # Only consider active labels (y == 1)
    active_labels = y.sum().item()
    true_positives = ((y_pred == 1) & (y == 1)).sum().item()

    # Avoid division by zero
    accuracy = (true_positives / active_labels) * 100 if active_labels > 0 else 0.0
    return accuracy

def accuracy_per_class(y: torch.Tensor, y_hat: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Compute the accuracy for each class in multi-label classification.

    Args:
        y (torch.Tensor): Ground truth labels (batch_size, num_classes), multi-hot encoded.
        y_hat (torch.Tensor): Predicted probabilities (batch_size, num_classes).
        threshold (float): Threshold to decide positive predictions.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing accuracy for each class as percentages.
    """
    y_pred = (y_hat >= threshold).float()

    # True positives per class
    true_positives = ((y_pred == 1) & (y == 1)).sum(dim=0).float()

    # Total positives in ground truth per class
    total_positives = y.sum(dim=0).float()

    # Avoid division by zero
    per_class_accuracy = torch.where(
        total_positives > 0,
        (true_positives / total_positives) * 100,
        torch.zeros_like(total_positives),
    )
    return per_class_accuracy

from sklearn.metrics import average_precision_score

def average_precision(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """
    Compute the Average Precision (AP) for each class in multi-label classification.

    Args:
        y (torch.Tensor): Ground truth labels (batch_size, num_classes), multi-hot encoded.
        y_hat (torch.Tensor): Predicted probabilities (batch_size, num_classes).

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the AP for each class.
    """
    # Convert tensors to numpy arrays for sklearn
    y_np = y.cpu().numpy()
    y_hat_np = y_hat.cpu().numpy()

    # Compute AP for each class
    ap_per_class = [
        average_precision_score(y_np[:, i], y_hat_np[:, i])
        if y_np[:, i].sum() > 0 else 0.0  # Handle cases with no positive samples
        for i in range(y.shape[1])
    ]

    return torch.tensor(ap_per_class, device=y.device)

    def f1_score_per_class(y: torch.Tensor, y_hat: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Compute the F1 score for each class in multi-label classification.

    Args:
        y (torch.Tensor): Ground truth labels (batch_size, num_classes), multi-hot encoded.
        y_hat (torch.Tensor): Predicted probabilities (batch_size, num_classes).
        threshold (float): Threshold to decide positive predictions.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the F1 score for each class.
    """
    # Apply threshold to predictions
    y_pred = (y_hat >= threshold).float()

    # Convert tensors to numpy for sklearn
    y_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Compute F1 score for each class
    f1_per_class = [
        f1_score(y_np[:, i], y_pred_np[:, i])
        if y_np[:, i].sum() > 0 else 0.0  # Handle cases with no positive samples
        for i in range(y.shape[1])
    ]

    return torch.tensor(f1_per_class, device=y.device)

