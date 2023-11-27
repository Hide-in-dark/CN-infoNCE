import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn  as nn
import time
from submission.CN_InfoNCE import CN_InfoNCE
from  submission.data import myDataLoader
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
#define the transform actions, also as same as SimCLR
transform_aug = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96),
        transforms.Resize((224, 224)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)
transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]
)

if __name__ == '__main__':

    train_loader,test_loader = myDataLoader(transform).get()
    #use ResNet18 as the backbone
    Resnet18 = torchvision.models.resnet18(weights=None)
    #change the latent space dimension
    Resnet18.fc = nn.Linear(Resnet18.fc.in_features,128)
    #add the projection MLP
    model = nn.Sequential(
    Resnet18,
    nn.Linear(Resnet18.fc.out_features,2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(),
    nn.Linear(2048,128),
    ).cuda()
    # training settings
    epochs = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    temperature = 0.07
    #train
    for epoch in range(epochs):
        batch_number = 0
        model.train()
        for data in train_loader:
            batch_number += 1
            img, label_raw = data
            img_aug_1 = torch.zeros((128, 3, 224, 224))
            img_aug_2 = torch.zeros((128, 3, 224, 224))
            for i in range(img.shape[0]):
                img_aug_1[i] = transform_aug(transforms.ToPILImage()(img[i]))
                img_aug_2[i] = transform_aug(transforms.ToPILImage()(img[i]))
            if torch.cuda.is_available():
                img = img.cuda()
                img_aug_1 = img_aug_1.cuda()
                img_aug_2 = img_aug_2.cuda()
            output_aug_1 = torch.nn.functional.normalize(model(img_aug_1)).cuda()
            output_aug_2 = torch.nn.functional.normalize(model(img_aug_2)).cuda()
            logits, label = CN_InfoNCE(output_aug_1, output_aug_2,temperature).loss()
            logits = logits.cuda()
            label = label.cuda()
            loss = torch.nn.CrossEntropyLoss()(logits, label).cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print('time:{} epoch:{} batch:{}/{} loss:{}'.format(
                time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())), epoch, batch_number, len(train_loader),
                loss))

    torch.save(model.state_dict(), 'CN_InfoNCE.pth')
    print("Model save")




