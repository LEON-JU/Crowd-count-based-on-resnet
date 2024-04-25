from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import SHHA_loader
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def data_collate(batch):
		data = [item[0] for item in batch]
		target = [item[1] for item in batch]
		data = torch.stack(data, 0)
		return [data, target]

def eval(args):
		train_dataset = SHHA_loader(args.data_path, "train", args.output_size)
		test_dataset = SHHA_loader(args.data_path, "test", args.output_size)
		train_loader = DataLoader(
				train_dataset, args.batch_size, True,
				num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=data_collate)
		test_loader = DataLoader(
				test_dataset, args.batch_size, False,
				num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=data_collate)


		# 设置超参数，导入模型
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		learning_rate = 0.001
		model = models.resnet18(pretrained=False)

		# 修改模型的输出层最后一层，使其输出一个值
		input_size = model.fc.in_features
		model.fc = nn.Linear(input_size, 1)

		model.load_state_dict(torch.load('./Model/crowd50.pth'))
		model = model.to(device)
		model.eval()
		sum_mae = 0
		sum_mse = 0
		n = 0
		for batch_idx, inputs in enumerate(test_loader):
			images, gt = inputs
			gt_lengths = [len(component) for component in gt]
			number = torch.tensor(gt_lengths, dtype=torch.float).unsqueeze(1)

			images = images.to(device)
			count = number.to(device)
			outputs = model(images)

		
			sum_mae += F.l1_loss(outputs, count, reduction='sum').item()
			sum_mse += F.mse_loss(outputs, count, reduction='sum').item()
			n += outputs.shape[0]
		print("MAE: ", sum_mae / n)
		print("MSE: ", sum_mse / n)


if __name__ == "__main__":
		parser = ArgumentParser()
		parser.add_argument('--data_path', type=str, default="./ShanghaiTech_Crowd_Counting_Dataset/part_A_final")
		parser.add_argument('--batch_size', type=int, default=4)
		parser.add_argument('--num_workers', type=int, default=4)
		parser.add_argument('--num_epoch', type=int, default=500)
		parser.add_argument('--output_size', type=int, default=512)
		args = parser.parse_args()
		
		eval(args)