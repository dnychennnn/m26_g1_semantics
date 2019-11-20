import torch


def make_classification_map(semantic_output_batch):
    """Select the class with maximum confidence.

    Args:
        semantic_output_batch (torch.Tensor): Class confidences of shape (batch_size, num_classes, height, width,).

    Returns:
        A torch.tensor of shape (batch_size, height, width,) with integer class labels.
    """
    return torch.argmax(semantic_output_batch, dim=1)
