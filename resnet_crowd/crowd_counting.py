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

'''
def draw_and_save(images, coords, save_path, batch_idx):
		std = torch.tensor([0.229, 0.224, 0.225])
		mean = torch.tensor([0.485, 0.456, 0.406])
		for i in range(images.shape[0]):
				image = images[i].permute((1, 2, 0))
				image = image * std + mean
				image = image.numpy()
				coord = coords[i]
				fig, ax = plt.subplots(1)
				ax.imshow(image)
				ax.plot(coord[:, 0], coord[:, 1], 'ro')
				plt.savefig(f"{save_path}/image_{batch_idx+i}.png")
				plt.close()
'''

'''
能把训练结果跑出来是下降的就可以了，不需要多准确
现在跑的结果是震荡的，不知道为什么
'''

def train(args):
		# loader，这部分不需要修改
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
		model = model.to(device)
		
		# 设置损失函数和优化器
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		schedulor = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

		# 作业要求统计的内容

		losses = []
		# 开始训练
		print("training on" + str(device))
		model.train()
		for epoch in range(args.num_epoch):
			epoch_loss = 0
			for batch_idx, inputs in enumerate(train_loader):
				images, gt = inputs

				# 获取ground truth中的人头数，gt是一个list，其中每个元素是一个tensor，表示一张图片中的人头坐标，这里只需要知道有多少个人头就可以了
				gt_lengths = [len(component) for component in gt]
				number = torch.tensor(gt_lengths, dtype=torch.float).unsqueeze(1)

				images = images.to(device)
				count = number.to(device)


				# TODO Forward
				outputs = model(images)

				# TODO Backward
				loss = criterion(outputs, count)
				epoch_loss += loss.item()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# TODO Print log info 
				if batch_idx % 10 == 0:
					print(f"Epoch: {epoch} / {args.num_epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
			losses.append(epoch_loss)
			schedulor.step()


		plt.plot(losses, label="Training loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.title("Loss curve")
		plt.legend()
		plt.savefig("loss_curve.png")

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
		with open('./TrainData/data.txt', 'w') as f:
			f.write('MAE:' + str(sum_mae / n) + '\n')
			f.write('MSE:' + str(sum_mse / n) + '\n')

		# 训练完毕
		# 保存模型
		print('Saving Model...')
		Path = './Model/crowd50.pth'
		torch.save(model.state_dict(), Path)



# 这部分是提供的代码，仅default部分需要修改。
if __name__ == "__main__":
		parser = ArgumentParser()
		parser.add_argument('--data_path', type=str, default="./ShanghaiTech_Crowd_Counting_Dataset/part_A_final")
		parser.add_argument('--batch_size', type=int, default=4)
		parser.add_argument('--num_workers', type=int, default=4)
		parser.add_argument('--num_epoch', type=int, default=500)
		parser.add_argument('--output_size', type=int, default=512)
		args = parser.parse_args()
		
		train(args)
		


