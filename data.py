
import torchvision
from torch.utils.data import DataLoader
class myDataLoader():
    def __init__(self,transform):
        self.train_dataset = torchvision.datasets.CIFAR10(root='CIFAR_10', train=True, transform=transform, download=True)
        self.test_dataset = torchvision.datasets.CIFAR10(root='CIFAR_10', train=False, transform=transform, download=True)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=128)


    def get(self):
        return self.train_loader , self.test_loader