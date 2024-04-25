import matplotlib.pyplot as plt

# Load the data from the file
with open('./TrainData/50data.txt', 'r') as f:
	data = f.readlines()

# Load the data from the file
with open('./TrainData/50pdata.txt', 'r') as f:
	datap = f.readlines()

# Extract the loss and accuracy values from the data
loss_data = []
accuracy_data = []
for line in data:
	loss, accuracy = line.strip().split(',')
	loss_data.append(float(loss))
	accuracy_data.append(float(accuracy))

loss_datap = []
accuracy_datap = []
for line in datap:
	loss, accuracy = line.strip().split(',')
	loss_datap.append(float(loss))
	accuracy_datap.append(float(accuracy))


# Plot loss curves
plt.figure()
plt.plot(loss_data, color='royalblue', label='scratch')
plt.plot(loss_datap, color='darkorange', label='pretrained')
plt.title('Loss Curves')
plt.xlabel('step')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./TrainData/losscompare.png')

plt.figure()
plt.plot(accuracy_data, color='royalblue', label='scratch')
plt.plot(accuracy_datap, color='darkorange', label='pretrained')
plt.title('Accuracy Curves')
plt.xlabel('step')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./TrainData/accuracycompare.png')
