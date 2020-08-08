import torch
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from SKModule import SKNet
from visdom import Visdom
import numpy as np
'''模型训练'''

batch_size = 10
epochs = 50
learning_rate = 9e-6
seed = 123456
torch.manual_seed(seed)
device = torch.device("cuda:0")

data_path = "../data/cat_vs_dog"
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.CenterCrop((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = MyDataset(data_path, transform)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

''' two classification'''
net = SKNet(2)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
net.to(device)
criterion.to(device)
total_loss = []
viz = Visdom()
viz.line([[0., 0.]], [0.], win="train", opts=dict(title="train&&val loss",
                                                  legend=['train', 'val']))
viz.line([0.], [0.], win="acc", opts=dict(title="accuracy",
                                          legend=['acc']))
for epoch in range(epochs):
    net.train()
    total_loss.clear()
    for batch, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        logits = net(input)

        loss = criterion(logits, label)
        total_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%50==0:
            print("epoch:{} batch:{} loss:{}".format(epoch, batch, loss.item()))

    net.eval()
    correct = 0
    test_loss = 0
    for input, label in test_loader:
        input, label = input.to(device), label.to(device)
        logits = net(input)

        '''crossentropy'''
        test_loss += criterion(logits, label).item() * input.shape[0]
        pred = logits.argmax(dim=1)

        correct += pred.eq(label).float().sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    viz.line([[float(np.mean(total_loss)), test_loss]], [epoch], win="train", update="append")
    viz.line([acc], [epoch], win='acc', update='append')
    torch.save(net.state_dict(), "model/resnet18_{}.pkl".format(epoch))

