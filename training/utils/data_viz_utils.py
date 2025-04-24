import torch
from sklearn.metrics import average_precision_score, f1_score


def accuracy(y: torch.Tensor, y_hat: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute the micro-average accuracy for multi-label classification:
    (TP + TN) / (TP + TN + FP + FN)

    Returns:
        float: Percentage of correct predictions across all (batch_size * num_classes).
    """
    y_pred = (y_hat >= threshold).float()

    TP = ((y_pred == 1) & (y == 1)).sum().float()
    TN = ((y_pred == 0) & (y == 0)).sum().float()
    FP = ((y_pred == 1) & (y == 0)).sum().float()
    FN = ((y_pred == 0) & (y == 1)).sum().float()

    total = TP + TN + FP + FN
    if total == 0:
        return 0.0
    
    acc = (TP + TN) / total
    return acc.item() * 100


def multi_label_accuracy(
    y: torch.Tensor, 
    y_hat: torch.Tensor, 
    threshold: float = 0.5
) -> float:
    """
    Computes the micro-level multi-label accuracy across a batch:
        (TP + TN) / (TP + TN + FP + FN)

    Args:
        y (torch.Tensor): Ground truth labels (batch_size, num_classes), multi-hot (0 or 1).
        y_hat (torch.Tensor): Model outputs (batch_size, num_classes) of raw scores 
                              or probabilities, to be thresholded.
        threshold (float): Value to binarize y_hat (>= threshold => 1).

    Returns:
        float: The overall accuracy across all (batch * num_classes) label decisions.
    """
    # Binarize predictions
    y_pred = (y_hat >= threshold).float()
    
    # Compute true/false positives/negatives
    TP = ((y_pred == 1) & (y == 1)).sum()
    TN = ((y_pred == 0) & (y == 0)).sum()
    FP = ((y_pred == 1) & (y == 0)).sum()
    FN = ((y_pred == 0) & (y == 1)).sum()
    
    # Compute accuracy
    total = TP + TN + FP + FN
    if total == 0:
        return 0.0  # Edge case: no labels at all
    
    accuracy = (TP + TN).float() / total.float()
    
    # Return a Python float (move to CPU if needed)
    return accuracy.item()*100




def accuracy_per_class(y: torch.Tensor, y_hat: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Returns a tensor of shape [num_classes] with the accuracy for each class.
    """
    num_classes = y.shape[1]
    y_pred = (y_hat >= threshold).float()

    accuracies = []
    for c in range(num_classes):
        # True labels and predictions for class c
        y_c = y[:, c]       # shape [batch_size]
        y_pred_c = y_pred[:, c]
        
        TP = ((y_pred_c == 1) & (y_c == 1)).sum()
        TN = ((y_pred_c == 0) & (y_c == 0)).sum()
        FP = ((y_pred_c == 1) & (y_c == 0)).sum()
        FN = ((y_pred_c == 0) & (y_c == 1)).sum()

        total = TP + TN + FP + FN
        if total == 0:
            acc_c = 0.0
        else:
            acc_c = (TP + TN).float() / total.float()

        accuracies.append(acc_c)

    # Stack into a tensor: shape [num_classes]
    return torch.stack(accuracies, dim=0)*100

def precision_per_class(y: torch.Tensor, y_hat: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Returns a tensor of shape [num_classes] with the precision (%) for each class.
    """
    num_classes = y.shape[1]
    y_pred = (y_hat >= threshold).float()

    precisions = []
    for c in range(num_classes):
        # True labels and predictions for class c
        y_c = y[:, c]       # shape [batch_size]
        y_pred_c = y_pred[:, c]
        
        TP = ((y_pred_c == 1) & (y_c == 1)).sum()
        FP = ((y_pred_c == 1) & (y_c == 0)).sum()

        total = TP + FP
        if total == 0:
            # Use a torch scalar tensor
            prec_c = torch.tensor(0.0, device=y.device)
        else:
            prec_c = TP.float() / total.float()

        precisions.append(prec_c)

    # Stack into a tensor: shape [num_classes]
    return torch.stack(precisions, dim=0) * 100




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

    return torch.tensor(ap_per_class, device=y.device)*100

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

