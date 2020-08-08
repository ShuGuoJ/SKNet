import torch
from visdom import Visdom
from PIL import Image
from SKModule import SKNet
from torchvision import transforms
from matplotlib import pyplot as plt


class SaveOutput(object):
    def __init__(self):
        super(SaveOutput, self).__init__()
        self.output = []

    def __call__(self, module, input, out):
        self.output.append(out)

    def clear(self):
        self.output.clear()

def denormalize(input, mean, std):
    mean = torch.tensor(mean, device=device).view(1,3,1,1)
    std = torch.tensor(std, device=device).view(1,3,1,1)
    out = input * std + mean
    return out

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
viz = Visdom()
model_path = 'model/resnet18_43.pkl'
net = SKNet(2)

net.load_state_dict(torch.load(model_path))
net.to(device)

img = Image.open('test.jpg')
transform_1 = transforms.Compose([transforms.Resize((256,256)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                                  ])
transform_2 = transforms.Compose([transforms.CenterCrop((256,256)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                                  ])
input_1 = transform_1(img)
input_2 = transform_2(img)
attention_1_hook = SaveOutput()
attention_2_hook = SaveOutput()
net.layer1[0].conv2.attention_1.register_forward_hook(attention_1_hook)
net.layer1[0].conv2.attention_2.register_forward_hook(attention_2_hook)
# input:[2,3,256,256]
input = torch.stack([input_1, input_2], dim=0)
input = input.to(device)
net.eval()
net(input)
# attention_1:[2,channel]
attention_1 = attention_1_hook.output[0]
attention_2 = attention_2_hook.output[0]
attention = torch.stack([attention_1, attention_2], dim=-1)
# attention_softmax:[b,channel, 2]
attention_softmax = torch.softmax(attention, dim=-1)
# [b,channel]
channel_attention = attention_softmax[...,1]
# [channel, b]
channel_attention = channel_attention.transpose(0,1).contiguous()
viz.images(denormalize(input, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), win='image')
viz.line(channel_attention, list(range(channel_attention.shape[0])), win='channel_attention', opts=dict(title='channel_attention',
                                                                                                        legend=['resize', 'centercrop']))
