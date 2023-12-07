# modified from: https://github.com/LJY-HY/cifar_pytorch-lightning/blob/master/models/classifiers.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as F

import torchvision.transforms as transforms
import torch.utils.data
from torchvsion.models import resnet18, resnet34, resnet50
import torchvision.models.resnet as Resnet

import pytorch_lightning as pl


class CIFAR100_LIGHTNING(pl.LightningModule):
    def __init__(self, norm):
        super(CIFAR100_LIGHTNING, self).__init__()
        self.norm_type = norm
        if self.norm_type == 'frob':
            self.target_norms = self.get_targert_frob_norms()
            self.automatic_optimization = False
        elif self.norm_type == 'spec':
            self.target_norms = self.get_target_spec_norms()
            self.automatic_optimization = False
        else:
            self.target_norms = None

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        if self.automatic_optimization == False:
            opt = self.optimizers()
            loss = F.cross_entropy(output,target)
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            if self.norm == 'frob':
                for i, p in enumerate(self.parameters()):
                    p.data = p.data / torch.norm(p.data, p='fro') * self.target_norms[i]
            else:
                for i, p in enumerate(self.parameters()):
                    p.data = p.data / self.spectral_norm(p) * self.target_norms[i]
        else:
            loss = F.cross_entropy(output,target)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'val_loss':loss,'correct':correct}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'val_loss':avg_loss}
        print('Validation accuracy : ',sum_correct/10000,'\n\n')
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}    

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss':F.cross_entropy(output,target), 'correct':correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'test_loss': avg_loss}
        print('Test accuracy :',sum_correct/10000,'\n')
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def get_target_frob_norms(self):
        return [torch.norm(w, p='fro') for w in self.parameters()]

    def spectral_norm(self, w, n_steps=10):
        v = torch.randn(w.shape[1], device=w.device)
        for _ in range(n_steps):
            v /= v.norm()
            v = w @ v @ w
        return v.norm().sqrt()

    def get_target_spec_norms(self):
        return [self.spectral_norm(w) for w in self.parameters()]


class CIFAR100_Resnet(CIFAR100_LIGHTNING):
    # This Module is based on Resnet for dataset CIFAR100
    def __init__(self, model_size, norm):
        super(CIFAR100_Resnet, self).__init__(norm)
        if model_size == 18:
            self.model = resnet18(Resnet.Bottleneck, [3,4,6,3], num_classes=100, pretrained=False)
        elif model_size == 34:
            self.model = resnet34(Resnet.Bottleneck, [3,4,6,3], num_classes=100, pretrained=False)
        else:
            self.model = resnet50(Resnet.Bottleneck, [3,4,6,3], num_classes=100, pretrained=False)
        self.model.inplanes=64
        self.model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.linear = nn.Linear(512*Resnet.Bottleneck.expansion, 100)
        del self.model.maxpool
        self.model.maxpool = lambda x : x


class CIFAR100_MLP(CIFAR100_LIGHTNING):
    # This Module is based on MLP for dataset CIFAR100
    def __init__(self, num_layers, width, norm):
        super(CIFAR100_MLP, self).__init__(norm)
        self.num_layers = num_layers
        self.width = width
        self.layer_sizes = [3 * 32 * 32, width]

        net = [nn.flatten(), nn.Linear(self.layer_sizes[0], self.width), nn.ReLU()]

        for i in range(num_layers -2):
            self.layer_sizes.append(width)
            net.append([nn.Linear(width, width), nn.ReLU()])

        self.layer_sizes.append(100)
        net.append(nn.Linear(width, 100))
        self.net = nn.Sequential(*net)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = self.net(x)
        return self.softmax(output)
