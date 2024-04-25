from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from dataloader import SHHA_loader
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def data_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data, 0)
    return [data, target]



def train(args):
    # Define dataloader with the transform
    train_dataset = SHHA_loader(args.data_path, "train", args.output_size)
    test_dataset = SHHA_loader(args.data_path, "test", args.output_size)

    train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        args.batch_size,
        False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=data_collate,
    )

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    # criterion = nn.SmoothL1Loss()

    criterion = nn.MSELoss()
    training_losses = []
    for epoch in range(args.num_epoch):
        model.train()
        whole_loss = 0
        for batch_idx, inputs in enumerate(train_loader):
            images, gt = inputs
            if torch.cuda.is_available():
                images = images.cuda()
            # Standardizing the crowd count
            crowd_count = (
                (torch.tensor([len(g) for g in gt], dtype=torch.float))
            ).unsqueeze(1)
            if torch.cuda.is_available():
                crowd_count = crowd_count.cuda()

            # TODO Forward
            outputs = model(images)
            # Loss
            loss = criterion(outputs, crowd_count)
            whole_loss += loss.item()
            # TODO Backward
            optimizer.zero_grad()
            loss.backward()
            # TODO Update parameters
            optimizer.step()
            # TODO Print log info
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}/{args.num_epoch} Batch {batch_idx}/{len(train_loader)} Loss:{loss.item()}"
                )
        scheduler.step()
        training_losses.append(whole_loss)
        print("whole_loss", whole_loss)
    plt.plot(training_losses, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()
    # Save model checkpoints
    model.eval()
    total_mae, total_mse, count = 0, 0, 0

    for batch_idx, inputs in enumerate(test_loader):
        images, gt = inputs

        if torch.cuda.is_available():
            images = images.cuda()
        crowd_count = (
            (torch.tensor([len(g) for g in gt], dtype=torch.float))
        ).unsqueeze(1)
        if torch.cuda.is_available():
            crowd_count = crowd_count.cuda()
        outputs = model(images)
        # TODO Test model performance
        print(
            "output:",
            outputs,
            "crowd_count:",
            crowd_count,
            "loss:",
            criterion(outputs, crowd_count),
        )
        outputs_denormalized = outputs
        crowd_count_denormalized = crowd_count
        # print("crowd_count:", crowd_count_denormalized)
        # print("outputs:", outputs_denormalized)
        total_mae += (
            torch.abs(outputs_denormalized - crowd_count_denormalized).sum().item()
        )
        total_mse += (
            ((outputs_denormalized - crowd_count_denormalized) ** 2).sum().item()
        )
        count += crowd_count_denormalized.size(0)
    print(f"MAE:{total_mae/count},MSE:{total_mse/count}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./ShanghaiTech_Crowd_Counting_Dataset/part_A_final",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--output_size", type=int, default=512)
    args = parser.parse_args()
    train(args)

