from torch import Tensor
from torchmetrics import JaccardIndex

def get_iou(pred: Tensor, target: Tensor, num_classes: int, device: str) -> Tensor:
    """
    Calculates the mean Intersection over Union of the model's prediction across all classes.

    Parameters
    ----------
    pred: torch.Tensor
        The predictions of the model.
        The expected shape is (batch size, height, width) if pred is an int tensor.
        If pred is a float tensor, the expected shape is (batch size, num classes, height, width).

    target: torch.Tensor
        The ground truth tensor of shape (batch size, height, width).

    num_classes: int
        The number of target classes.

    device: str
        The device by which the prediction and target tensors are located.
        One of ['cpu', 'cuda'].

    Returns
    -------
    miou: torch.Tensor
        The mean Intersection over Union (mIOU) of the prediction across all classes.
    """

    jaccard = JaccardIndex(task="multiclass", num_classes=num_classes, average="macro").to(device)
    miou = jaccard(pred, target)

    return miou