import os
import torch
from tqdm import tqdm
from typing import Tuple
from datetime import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from utils import map_to_class, get_iou, compute_stats, CustomDataset, \
    get_network, get_args, save_args, set_seed

def train(
        model: DeepLabV3,
        dataloader: DataLoader,
        optimizer: torch.optim, 
        criterion: torch.nn.modules.loss,
        device: str, num_classes: int
        ) -> Tuple[float, float]:
    
    """
    Trains the model for one epoch.

    Parameters
    ----------
    model: torchvision.models.segmentation.deeplabv3.DeepLabV3
        The DeepLabV3 model to be trained.

    dataloader: torch.utils.data.dataloader.DataLoader
        The dataloader to extract minibatches of images and masks.

    optimizer: torch.optim
        The gradient descent optimizer.
        Adam was used in this project.

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

    model.train()
    running_loss = 0
    running_miou = 0

    for (img, mask) in tqdm(dataloader, desc="Training Model"):
        optimizer.zero_grad()
        img = img.to(device)
        mapped_mask = map_to_class(mask).to(device)

        logits = model(img)["out"]
        probabilities = F.softmax(logits, dim=1)

        loss = criterion(logits, mapped_mask)
        pred = torch.argmax(probabilities, dim=1)

        miou = get_iou(pred, mapped_mask, num_classes, device)
        running_loss += loss.detach().cpu().item()
        running_miou += miou.detach().cpu().item()

        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_miou = running_miou / len(dataloader)

    return epoch_loss, epoch_miou

def validate(
        model: DeepLabV3, 
        dataloader: DataLoader, 
        criterion: torch.nn.modules.loss, 
        device: str, num_classes: int
        ) -> Tuple[float, float]:
    
    """
    Performs model validation for a given epoch.

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

    model = model.to(device)
    model.eval()
    running_loss = 0
    running_miou = 0

    with torch.no_grad():
        for (img, mask) in tqdm(dataloader, desc="Validating Model"):
            img = img.to(device)
            mapped_mask = map_to_class(mask).to(device)

            logits = model(img)["out"]
            probabilities = F.softmax(logits, dim=1)

            loss = criterion(logits, mapped_mask)
            pred = torch.argmax(probabilities, dim=1)

            miou = get_iou(pred, mapped_mask, num_classes, device)
            running_loss += loss.detach().cpu().item()
            running_miou += miou.detach().cpu().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_miou = running_miou / len(dataloader)

    return epoch_loss, epoch_miou

def main():
    data_dir = os.path.join("..", "data")
    arg_dir = os.path.join("config", "train_config.yaml")
    id = datetime.now().strftime("%m-%d-%Y-%H:00")
    results_dir = os.path.join("runs", id)
    writer = SummaryWriter(results_dir)

    args = get_args(arg_dir)
    save_args(args, results_dir)
    set_seed(args["seed"])
    
    model_dir = os.path.join("..", "assets", "models", id)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    train_img_dir = os.path.join(data_dir, "train", "images")
    train_mask_dir = os.path.join(data_dir, "train", "masks")

    val_img_dir = os.path.join(data_dir, "val", "images")
    val_mask_dir = os.path.join(data_dir, "val", "masks")
    
    mean, std = compute_stats(train_img_dir)

    train_dataset = CustomDataset(
        img_dir=train_img_dir, mask_dir=train_mask_dir,
        mean=mean, std=std
    )

    val_dataset = CustomDataset(
        img_dir=val_img_dir, mask_dir=val_mask_dir,
        mean=mean, std=std
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(dataset=train_dataset, batch_size=args["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args["batch_size"], shuffle=False, drop_last=False)

    model = get_network(args["backbone"], args["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)

    losses_val, mious_val = [], []

    for epoch in range(1, args["epochs"] + 1):
        print(f"Epoch [{epoch}/{args['epochs']}]")

        train_loss, train_miou = train(model, train_loader, optimizer, criterion, device, args["num_classes"])

        print("\nTrain Statistics:")
        print(f"Loss: {train_loss} | mIOU: {train_miou:.4f}")

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/mIOU", train_miou, epoch)

        val_loss, val_miou = validate(model, val_loader, criterion, device, args["num_classes"])

        print("\nValidation Statistics:")
        print(f"Loss: {val_loss} | mIOU: {val_miou:.4f}")

        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/mIOU", val_miou, epoch)

        if len(losses_val) > 0 and val_loss < min(losses_val):
            print("New minimum loss — model saved")
            torch.save(model.state_dict(), os.path.join(model_dir, f"{args['backbone']}_backbone_lowest_loss.pth"))

        if len(mious_val) > 0 and val_miou > max(mious_val):
            print("New maximum mIOU - model saved")
            torch.save(model.state_dict(), os.path.join(model_dir, f"{args['backbone']}_backbone_highest_miou.pth"))

        if epoch % 10 == 0:
            print("Latest model updated")
            torch.save(model.state_dict, os.path.join(model_dir, f"{args['backbone']}_backbone_latest_model.pth"))

        losses_val.append(val_loss)
        mious_val.append(val_miou)
        
        scheduler.step(val_loss)

        print("___________________________________________________________________\n")

    torch.save(model.state_dict, os.path.join(model_dir, f"{args['backbone']}_backbone_latest_model.pth"))

if __name__ == "__main__":
    main()