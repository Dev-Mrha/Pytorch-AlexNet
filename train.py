from torchvision import transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import torch
from AlexNet import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

input_size = 224 * 224
num_classes = 101
num_epoch = 150
batch_size = 32

writer_train = SummaryWriter('log_SEQL/train')
writer_valid = SummaryWriter('log_SEQL/valid')
writer_test = SummaryWriter('log_SEQL/test')

img_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)

train_data = datasets.ImageFolder(root='data/Caltech101/train/', transform=img_transform)
valid_data = datasets.ImageFolder(root='data/Caltech101/valid/', transform=img_transform)
test_data = datasets.ImageFolder(root='data/Caltech101/test/', transform=img_transform)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

x_list = []
y_list = []
z_list = []


def train(epoch, model, criterion, optimizer):
    for batch_idx, (x, y) in enumerate(train_loader):
        # x = x.reshape(-1, 224 * 224).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        x = x.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        y = y.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if batch_idx % 50 == 0:
            print("Epoch [%d/%d], Iter [%d] Loss: %.4f" % (epoch + 1, num_epoch, batch_idx + 1, loss.data.item()))
        if batch_idx == 0:
            y_list.append(loss.data.item())
            writer_train.add_scalar('loss', loss.data.item(), epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def valid(epoch, model, optimizer, best_acc):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            image, labels = data
            # image = image.reshape(-1, 224 * 224).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            image = image.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            labels = labels.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(image)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    z_list.append(100.0 * correct / total)
    writer_valid.add_scalar('accuracy', (100.0 * correct / total), epoch)
    print('Accuracy on valid set: %.6lf %%' % (100.0 * correct / total))
    if 100.0 * correct / total > best_acc:
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': num_epoch}
        torch.save(state, 'AlexNet-best.pth')
    return 100.0 * correct / total


cnt = 0


def trainn(model, lr, name, pth):
    best_acc = 0
    txtName = name + ".txt"
    f = open(txtName, "w")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global x_list
    global y_list
    global z_list
    x_list = []
    y_list = []
    z_list = []
    for i in range(num_epoch):
        x_list.append(i)
        train(i, model, criterion, optimizer)
        best_acc = max(best_acc, valid(i, model, optimizer, best_acc))
    f.write(str(y_list))
    f.write("\n")
    f.write(str(z_list))
    # cnt = cnt + 1

    # state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': num_epoch}
    # torch.save(state, pth)


model = AlexNet()
model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
trainn(model, 0.001, "Adam0.001", "AlexNet-Adam-0.001.pth")
pic = plt.figure(cnt)
plt, ax = plt.subplots()
ax.plot(x_list, y_list, "r")
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title(name)
ax2 = ax.twinx()
ax2.plot(x_list, z_list, "g")
ax2.set_ylabel('accuracy')

plt.savefig(name + '.jpg')
plt.show()

writer_train.close()
writer_valid.close()
writer_test.close()
