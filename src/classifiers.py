# modified from: https://github.com/LJY-HY/cifar_pytorch-lightning/blob/master/models/classifiers.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as F

import torchvision.transforms as transforms
import torch.utils.data
from torchvision.models import resnet18, resnet34, resnet50
import torchvision.models.resnet as Resnet

import pytorch_lightning as pl


class CIFAR100_LIGHTNING(pl.LightningModule):
    def __init__(self, norm):
        super(CIFAR100_LIGHTNING, self).__init__()
        self.norm_type = norm
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        output = self.model(x)
        return output

    def power_iteration(self, A, n_steps=100):
        v = torch.randn(A.shape[1], device=A.device)
        for _ in range(n_steps):
            v /= v.norm()
            v = A@v
        return v.norm().sqrt(), v / v.norm()
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        if self.automatic_optimization == False:
            opt = self.optimizers()
            loss = F.cross_entropy(output,target)
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            if self.norm_type == 'frob':
                for i, named_data in enumerate(self.named_parameters()):
                    name, param = named_data
                    if 'bn' in name or 'bias' in name:
                        continue
                    else:
                        param.data /= (torch.norm(param.data, p='fro') * self.target_norms[i])
            else:
                for i, named_data in enumerate(self.named_parameters()):
                    name, param = named_data
                    if 'bn' in name or 'bias' in name:
                        continue
                    else:
                        first_val, right_vec = self.power_iteration(param.data.T@param.data)
                        left_vec = param.data@right_vec/first_val
                        change = self.target_norms[i] - first_val
                        outer = torch.outer(left_vec, right_vec)
                        param.data += (change * outer)
        else:
            loss = F.cross_entropy(output,target)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return [optimizer]

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        self.validation_step_outputs.append({'val_loss':loss,'correct':correct})
        return {'val_loss':loss,'correct':correct}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        sum_correct = sum([x['correct'] for x in self.validation_step_outputs])
        tensorboard_logs = {'val_loss':avg_loss}
        print('Validation accuracy : ',sum_correct/10000,'\n\n')
        self.validation_step_outputs.clear()
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}    

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        output = {'test_loss':F.cross_entropy(output,target), 'correct':correct}
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        sum_correct = sum([x['correct'] for x in self.test_step_outputs])
        tensorboard_logs = {'test_loss': avg_loss}
        print('Test accuracy :',sum_correct/10000,'\n')
        self.test_step_outputs.clear()
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def init_orthonormal(self):
        layer_idx = 0
        for idx, data in enumerate(self.named_parameters()):
            name, param = data
            if param.requires_grad:
                if 'bn' in name or 'bias' in name:
                    continue
                else:
                    nn.init.orthogonal_(param)
                    in_size = self.layer_sizes[layer_idx]
                    out_size = self.layer_sizes[layer_idx+1]
                    param.data *= (out_size / in_size)**0.5
                    layer_idx += 1

    def get_target_frob_norms(self):
        target_norms = dict()
        for idx, data in enumerate(self.named_parameters()):
            name, param = data
            if param.requires_grad:
                if 'bn' in name or 'bias' in name:
                    continue
                else:
                    frob_norm = torch.norm(param, p='fro')
                    target_norms[idx] = frob_norm.item()
        return target_norms

    def get_target_spec_norms(self):
        target_norms = dict()
        layer_idx = 0
        for idx, data in enumerate(self.named_parameters()):
            name, param = data
            if param.requires_grad:
                if 'bn' in name or 'bias' in name:
                    continue
                else:
                    spec_norm = (self.layer_sizes[layer_idx+1] / self.layer_sizes[layer_idx])**0.5
                    target_norms[idx] = spec_norm
        return target_norms


class CIFAR100_Resnet(CIFAR100_LIGHTNING):
    # This Module is based on Resnet for dataset CIFAR100
    def __init__(self, model_size, norm):
        super(CIFAR100_Resnet, self).__init__(norm)
        if model_size == 'ResNet18':
            self.model = resnet18(Resnet.Bottleneck, [2,2,2,2], pretrained=False)
        elif model_size == 'ResNet34':
            self.model = resnet34(Resnet.Bottleneck, [3,4,6,3], pretrained=False)
        else:
            self.model = resnet50(Resnet.Bottleneck, [3,4,6,3], pretrained=False)
        self.model.inplanes=64
        self.model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.fc = nn.Linear(512, 100)
        del self.model.maxpool
        self.model.maxpool = lambda x : x
        #TODO: get layer sizes
        self.init_orthonormal()

        if self.norm_type == 'frob':
            self.target_norms = self.get_target_frob_norms()
            self.automatic_optimization = False
        elif self.norm_type == 'spec':
            self.target_norms = self.get_target_spec_norms()
            self.automatic_optimization = False
        else:
            self.target_norms = None


class CIFAR100_MLP(CIFAR100_LIGHTNING):
    # This Module is based on MLP for dataset CIFAR100
    def __init__(self, num_layers, width, norm):
        super(CIFAR100_MLP, self).__init__(norm)
        self.num_layers = num_layers
        self.width = width

        self.layer_sizes = [3 * 32 * 32, width]
        net = [nn.Flatten(), nn.Linear(self.layer_sizes[0], self.width), nn.ReLU()]
        for i in range(num_layers -2):
            self.layer_sizes.append(width)
            net.extend([nn.Linear(width, width), nn.ReLU()])
        self.layer_sizes.append(100)
        net.append(nn.Linear(width, 100))
        self.model = nn.Sequential(*net)
        self.softmax = nn.Softmax(dim=1)
        self.init_orthonormal()

        if self.norm_type == 'frob':
            self.target_norms = self.get_target_frob_norms()
            self.automatic_optimization = False
        elif self.norm_type == 'spec':
            self.target_norms = self.get_target_spec_norms()
            self.automatic_optimization = False
        else:
            self.target_norms = None

    def forward(self, x):
        output = self.model(x)
        return self.softmax(output)
