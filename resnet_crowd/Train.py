import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import resnet50


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 50 #50轮
batch_size = 64 #50步长
learning_rate = 0.001 #学习率0.001

# 图像预处理
transform = transforms.Compose([
		transforms.Pad(4),
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(32),
		transforms.ToTensor()])

# 导入数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
																						 train=True, 
																						 transform=transform,
																						 download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data/',
																						train=False, 
																						transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
																					 batch_size=batch_size,
																					 shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
																					batch_size=batch_size,
																					shuffle=False)



model = resnet50().to(device)


# 损失函数和优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

def test(model, test_loader):
		correct = 0
		total = 0
		with torch.no_grad():
				for data in test_loader:
						images, labels = data
						images = images.to('cuda')
						labels = labels.to('cuda')
						outputs = model(images)
						_, predicted = torch.max(outputs.data, 1)
						total += labels.size(0)
						correct += (predicted == labels).sum().item()

		accuracy = 100 * correct / total
		return accuracy

# 训练网络模型
print('Start Training...')
print(device)
total_step = len(train_loader)

loss_data = []
accuracy_data = []
for epoch in range(num_epochs):
		
		for i, data in enumerate(train_loader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data
				inputs = inputs.to(device)
				labels = labels.to(device)
				
				# zero the parameter gradients
				optimizer.zero_grad()
				
				# forward + backward + optimize
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				
				# print statistics
				if (i+1) % 100 == 0:
						accuracy = test(model, test_loader)
						loss_data.append(loss.item())
						accuracy_data.append(accuracy)
						print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}, Accuracy: {:.2f}%"
									 .format(epoch+1, num_epochs, i+1, total_step, loss.item(), accuracy))

print('Finished Training')

# 保存训练数据
with open('./TrainData/50data.txt', 'w') as f:
	for i in range(len(loss_data)):
		f.write(str(loss_data[i]) + ',' + str(accuracy_data[i]) + '\n')
print('Training data saved to 50data.txt')

# 保存模型
print('Saving Model...')
Path = './Model/ResNet50.pth'
torch.save(model.state_dict(), Path)

# 分别画两张图并保存到data文件夹
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.plot(loss_data)
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig('./TrainData/50loss.png')
plt.figure()
plt.plot(accuracy_data)
plt.xlabel('step')
plt.ylabel('accuracy')
plt.savefig('./TrainData/50accuracy.png')


