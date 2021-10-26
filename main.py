import torch
import torchvision.transforms as transforms
from CNNClassifier import CNN
from cifar10classification.cifar10classification.CIFAR10DataLoader import CIFAR10DataLoader

useCuda = 'Y'
data_path = './data'
workers = 2
optimizer = 'sgd'
print("Define transformer")
transform_config = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
batch = 128
epochs = 5

print("init data loader class")
data_loader = CIFAR10DataLoader()
print("load data")
train, train_loader, test, test_loader = data_loader.load(root=data_path, transform=transform_config, batch_size=batch,
                                                          num_workers=workers)
print("data loaded successfully")

is_gpu = False
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

for i, data in enumerate(train_loader):
    print("i: ", i)
    # get the inputs
    inputs, labels = data[0], data[1]
    # print(inputs.shape)
    # device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    # inputs, labels = data[0].to(device), data[1].to(device)
    # if is_gpu:
    #     inputs = inputs.cuda()
    #     labels = labels.cuda()
    # net.to(device)

#     # wrap them in Variable
#     inputs, labels = Variable(inputs), Variable(labels)

    # zero the parameter gradients
    optimizer.zero_grad()
#
#     # forward + backward + optimize
    outputs = net.forward(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
#
    # print statistics
    running_loss += loss.item()
    print("running_loss: ", running_loss)

    _, predicted = outputs.max(1)
    total += labels.size(0)
    print("total: ", total)
    correct += predicted.eq(labels).sum().item()
    print("correct: ", correct)
#
    # Normalizing the loss by the total number of train batches
    running_loss /= len(train_loader)
    # accuracy
    train_accuracy = 100 * correct / total
#
#     print('Epoch {}, train Loss: {:.3f}'.format(epochs, loss.item()), "Training Accuracy: %d %%" % train_accuracy)

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
print('==> Finished Training ...')


dataiter = iter(test_loader)
images, labels = dataiter.next()

net = CNN()
net.load_state_dict(torch.load(PATH))

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
        print("test.total: ", total)
        correct += (predicted == labels).sum().item()
        print("test.correct: ", correct)

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

print('==> Finished Testing ...')