import torch
import torchvision.transforms as transforms
from CNNClassifier import CNN
from CIFAR10DataLoader import CIFAR10DataLoader
import argparse

# arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--useCuda', default='Y', type=str, help='use cuda or not')
parser.add_argument('--dataPath', default='./data', type=str, help='data path where data to be stored')
parser.add_argument('--workers', default=2, type=int, help='number of workers')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer name')

args = parser.parse_args()
print("useCuda: ", args.useCuda)
print("dataPath: ", args.dataPath)
print("workers: ", args.workers)
print("optimizer: ", args.optimizer)

# Data
print('==> Preparing data..')
transform_config = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_batch = 128
test_batch = 100
epochs = 5

print("init data loader class")
data_loader = CIFAR10DataLoader()
print("load data")
train, train_loader, test, test_loader = data_loader.load(root=args.dataPath, transform=transform_config,
                                                          train_batch_size=train_batch,
                                                          test_batch_size=test_batch, num_workers=args.workers)
print("data loaded successfully")

# Model
print('==> Building model..')
net = CNN()

# if is_gpu:
#     net = net.cuda()
#     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#     cudnn.benchmark = True

running_loss = 0.0
correct = 0
total = 0
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train_losses = []
train_accuracies = []
best_training_accuracy = 0

# train model for epochs times, store training losses, training accuracies
for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data[0], data[1]
        # print(inputs.shape)
        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        # inputs, labels = data[0].to(device), data[1].to(device)
        if args.useCuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            net.to(device)

        #     wrap them in Variable
        #     inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.forward(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        print("total: ", total)
        correct += predicted.eq(labels).sum().item()
        print("correct: ", correct)
        #
        # Normalizing the loss by the total number of train batches
        running_loss /= len(train_loader)
        train_losses.append(running_loss)
        # accuracy
        train_accuracy = 100 * correct / total
        print("train_accuracy: ", train_accuracy)
        train_accuracies.append(train_accuracy)

        if train_accuracy > best_training_accuracy:
            best_training_accuracy = train_accuracy
            PATH = './cifar_net.pth'
            torch.save({
                'model': net.state_dict()
            }, PATH)
print('==> Finished Training ...')
print("losses")
print(train_losses)
print("train_accuracies")
print(train_accuracies)
print("best_training_accuracy: ", best_training_accuracy)

net = CNN()
stored_result = torch.load(PATH)
# load model
net.load_state_dict(stored_result['model'])

# prediction on training dataset
correct = 0
total = 0
best_training_accuracy = 0
with torch.no_grad():
    for i, data in enumerate(train_loader):
        images, labels = data[0], data[1]
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        print("train.total: ", total)
        correct = (predicted == labels).sum().item()
        print("train.correct: ", correct)

        train_accuracy = 100 * correct / total
        if train_accuracy > best_training_accuracy:
            best_training_accuracy = train_accuracy
print("best_training_accuracy: ", best_training_accuracy)

# Test whole dataset
print("############Test whole dataset##########")
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test accuracy(10000 images): %d %%' % (
        100 * correct / total))
print('==> Finished Testing ...')
