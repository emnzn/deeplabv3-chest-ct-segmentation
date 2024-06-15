import os
import torch
from tqdm import tqdm
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from utils import map_to_class, get_iou, compute_stats, CustomDataset, \
    get_network, get_args, save_inference

def run_inference(
        model: DeepLabV3, 
        dataloader: DataLoader, 
        criterion: torch.nn.modules.loss, 
        device: str, num_classes: int
        ) -> Tuple[float, float]:
    
    """
    Performs model inference for a given epoch.

    Parameters
    ----------
    model: torchvision.models.segmentation.deeplabv3.DeepLabV3
        The DeepLabV3 model to be trained.

    dataloader: torch.utils.data.dataloader.DataLoader
        The dataloader to extract minibatches of images and masks.

    criterion: torch.nn.modules.loss
        The loss function.
        CrossEntropyLoss was used in this project.

    device: str
        One of ['cpu', 'cuda'].

    num_classes: int
        The number of target classes.

    Returns
    -------
    epoch_loss: float
        The loss for the epoch.

    epoch_miou: float
        The mean Intersection over Union (mIOU) for the epoch.
    """

    results_table = {
        "img_name": [],
        "loss": [],
        "miou": [],
        "prediction": []
    }

    running_loss = 0
    running_miou = 0

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for (img, mask, img_name) in tqdm(dataloader, desc="Running inference"):
            img = img.to(device)
            mapped_mask = map_to_class(mask).to(device)

            logits = model(img)["out"]
            probabilities = F.softmax(logits, dim=1)

            pred = torch.argmax(probabilities, dim=1)

            results_table["img_name"].extend(img_name)
            results_table["prediction"].extend(pred.cpu().numpy())

            for i in range(len(img_name)):
                loss = criterion(logits[i].unsqueeze(0), mapped_mask[i].unsqueeze(0))
                miou = get_iou(pred[i], mapped_mask[i], num_classes, device)

                results_table["loss"].append(loss.cpu().item())
                results_table["miou"].append(miou.cpu().item())

            running_loss += loss.detach().cpu().item()
            running_miou += miou.detach().cpu().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_miou = running_miou / len(dataloader)

    return epoch_loss, epoch_miou, results_table

def main():
    data_dir = os.path.join("..", "data")
    arg_dir = os.path.join("config", "inference_config.yaml")
    args = get_args(arg_dir)

    num_classes = args["num_classes"]
    results_dir = os.path.join("..", "assets", "inference", args["id"])
    model_dir = os.path.join("..", "assets", "models", args["id"])

    train_img_dir = os.path.join(data_dir, "train", "images")
    test_img_dir = os.path.join(data_dir, "test", "images")
    test_mask_dir = os.path.join(data_dir, "test", "masks")

    mean, std = compute_stats(train_img_dir)

    test_dataset = CustomDataset(
        img_dir=test_img_dir, mask_dir=test_mask_dir,
        mean=mean, std=std, inference=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torch.load(os.path.join(model_dir, args["weights"]), map_location=torch.device(device))

    test_loader = DataLoader(dataset=test_dataset, batch_size=args["batch_size"], shuffle=False, drop_last=False)

    model = get_network(args["backbone"], num_classes).to(device)
    model.load_state_dict(weights)
    
    criterion = torch.nn.CrossEntropyLoss()

    loss, miou, results_table = run_inference(model, test_loader, criterion, device, num_classes)

    save_inference(results_table, results_dir)

    print(f"\nInference Results:")
    print(f"Loss: {loss} | mIOU: {miou:.4f}")

if __name__ == "__main__":
    main()
