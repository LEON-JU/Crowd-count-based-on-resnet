# draw the loss and accuracy curve using ./TrainData/50pdata.txt
import matplotlib.pyplot as plt

# Load the data from the file
with open('./TrainData/50pdata.txt', 'r') as f:
	data = f.readlines()

# Extract the loss and accuracy values from the data
loss_data = []
accuracy_data = []
for line in data:
	loss, accuracy = line.strip().split(',')
	loss_data.append(float(loss))
	accuracy_data.append(float(accuracy))

# Plot the loss and accuracy curves
plt.figure()
plt.plot(loss_data)
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig('./TrainData/50ploss.png')
plt.figure()
plt.plot(accuracy_data)
plt.xlabel('step')
plt.ylabel('accuracy')
plt.savefig('./TrainData/50paccuracy.png')
