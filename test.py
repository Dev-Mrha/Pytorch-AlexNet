from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from AlexNet import *
import torch
import numpy as np


input_size = 224 * 224
num_classes = 101

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

test_data = datasets.ImageFolder(root='data/Caltech101/test/', transform=img_transform)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

model = AlexNet().to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
checkpoint = torch.load('AlexNet-best.pth')
# checkpoint = torch.load('best_model_mlp7_Adam_0.001.pth',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            image, labels = data
            image = image.reshape(-1, 28 * 28).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            labels = labels.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(image)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print('Accuracy on test set: %.6lf %%' % (100.0*correct/total))

test()