import torch

def map_to_class(mask: torch.Tensor) -> torch.Tensor:
    """
    Maps each function to its respective class number based on its closest 
    color in the class map.

    Parameters
    ----------
    mask: torch.Tensor
        A mask of shape (batch size, 3, height, width)

    Returns
    -------
    mapped_mask: torch.Tensor
        a mask where each pixel is appropriately mapped to its class number 
        of shape (batch size, height, width).
    """
    
    class_map = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # prepare mask for broadcasting
    mask = mask.permute(0, 2, 3, 1).unsqueeze(3) # (batch size, height, width, 1, channels)
    sqrd_dist = torch.sum((class_map - mask) ** 2, dim=-1) # (batch size, height, width, 4)
    mapped_mask = torch.argmin(sqrd_dist, dim=-1) # (batch size, height, width)

    return mapped_mask
