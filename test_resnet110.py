import os
import shutil
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import setup_resnet
import numpy as np

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def pgd(model, x, y,
        eps=4/255, 
        a=8/255, 
        steps=40,
        random_start=False):
    adv_images = x.clone().detach().cuda()
    loss = nn.CrossEntropyLoss()
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1)
    for i in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images).cuda()
        cost = loss(outputs, y.cuda()).cuda()
        grad = torch.autograd.grad(cost, adv_images,
                retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + a * grad.sign()
        delta = torch.clamp(adv_images - x.cuda(), min=-eps, max=eps)
        adv_images = torch.clamp(x.cuda() + delta, min=-1, max=1).detach()

    return adv_images


def accuracy(test_loader, model):
    """
    Evaluate on test set
    """
    model.eval()
    correct = 0.
    total = 0.
    with torch.no_grad():
        for i, (x, target) in enumerate(test_loader):
            target = target.cuda()
            input_var = x.cuda()
            target_var = target.cuda()

            output = model(input_var)
            output = output.float()

            for j, pred in enumerate(output):
                l = pred.argmax()
                total += 1
                if l == target[j]:
                    correct += 1
    return correct/total

def adv_accuracy(model, loader):
    """
    Diff function than accuracy because
    need grad to compute pgd >:(
    """
    model.eval()
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.cuda()
        batchsize = x.shape[0]
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).detach().cpu().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
        total += batchsize
    return total_correct / total

# working with resnet20
load_dir = "save_resnet110/checkpoint.th"

model = torch.nn.DataParallel(setup_resnet.__dict__['resnet110']())
model.cuda()
model.eval()
checkpoint = torch.load(load_dir)
model.load_state_dict(checkpoint['state_dict'])

cudnn.benchmark = True

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

# Don't need train loader
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False,         
                         transform=transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
                            ])),
                         batch_size=128,
                         shuffle=False,
                         num_workers=2)

acc = accuracy(test_loader, model)
print("clean:", acc)


# adversarial image generation
class AdversarialLoader():
    def __init__(self, model, dataloader, attack=pgd):
        self.model = model
        self.dataloader = dataloader
        self.attack = attack
    def __iter__(self):
        for x, y in self.dataloader:
            yield self.attack(self.model, x, y), y
    def __len__(self):
        return len(self.dataloader)

class TruncIterator():
    def __init__(self, iterator, stop):
        self.iterator = iterator
        self.stop = stop
    def __iter__(self):
        end = self.stop if self.stop != -1 else len(self.iterator)
        it = iter(self.iterator)
        for i in range(end):
            yield next(it)
    def __len__(self):
        return self.stop if self.stop != 1 else len(self.iterator)

adv_loader = AdversarialLoader(model, test_loader, pgd)
adv_loader = TruncIterator(adv_loader, -1)
adv_acc = adv_accuracy(model, adv_loader)
print("Adv:", adv_acc)
