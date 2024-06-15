import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

def get_network(backbone: str, num_classes: int) -> DeepLabV3:
    """
    Creates a DeepLab3 model with the specified backbone and number of classes.

    Parameters
    ----------
    backbone: str
        The backbone to be used. One of ['resnet50', 'resnet101'].

    num_classes: int
        The number of classes to be segmented.

    Returns
    -------
    model: torchvision.models.segmentation.deeplabv3.DeepLabV3
        The DeepLabV3 model to be trained.

    Raises
    ------
    ValueError
        If the backbone is not one of ['resnet50', 'resnet101'].
    """

    if backbone == "resnet50":
        model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
        model.backbone.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), 
                padding=(3, 3), bias=False
            )

    elif backbone == "resnet101":
        model = deeplabv3_resnet101(weights=None, num_classes=num_classes)
        model.backbone.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), 
                padding=(3, 3), bias=False
            )
        
    else:
        raise ValueError("Backbone must be one of ['resnet50', 'resnet101']")

    return model