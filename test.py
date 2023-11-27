import torch
import torchvision
from torchvision import transforms
from  submission.data import myDataLoader
from torch import nn
import time
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def judge_predict(tensor):
    predit = torch.zeros(tensor.shape[0]).cuda()
    for i in range(tensor.shape[0]):
        if tensor[i][1] == tensor[i][2]:
            predit[i]=tensor[i][1]
        else:
            predit[i] = tensor[i][0]

    return predit
def test(model,memory_loader):
        with torch.no_grad():
            correct = 0
            total_number = 0
            memory_bank = torch.zeros((len(memory_loader),128)).to(device)
            memory_label = torch.zeros(len(memory_loader)).to(device)
            memory_batch_number=0
            for data,label in memory_loader:
                memory_batch_number +=1
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                output = nn.functional.normalize(output)
                memory_bank = torch.cat((memory_bank,output),dim=0).to(device)
                memory_label = torch.cat((memory_label,label),dim=0).to(device)
                print('time:{}  memory_batch:{}/{} '.format(
                    time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())), memory_batch_number,
                    len(memory_loader)))
            batch_number=0
            for data,label in test_loader:
                batch_number +=1
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                output = nn.functional.normalize(output)
                index = torch.argsort(torch.matmul(output,torch.transpose(memory_bank,0,1)),dim=1,descending=True)[:,:3].to(device)
                candidate=torch.zeros((index.shape[0],index.shape[1])).to(device)
                for i in range(index.shape[0]):
                    for j in range(index.shape[1]):
                         candidate[i][j] =memory_label[index[i][j]]
                predit = judge_predict(candidate).to(device)
                total_number += predit.shape[0] 
                correct = correct+(predit==label).sum().item()
                print('time:{}  test_batch:{}/{} '.format(
                    time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())), batch_number,
                    len(test_loader)))
        print('time:{}  acc:{},correct:{} total_number:{}'.format(time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time())),correct/total_number,correct,total_number))




if __name__ == '__main__':
    Resnet18=torchvision.models.resnet18(weights=None).to(device)
    Resnet18.fc = nn.Linear(Resnet18.fc.in_features, 128)
    model = nn.Sequential(
        Resnet18,
        nn.Linear(Resnet18.fc.out_features, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(2048, 128),
    )
    model.load_state_dict(torch.load(r''))
    model=model.to(device)

    train_loader,test_loader = myDataLoader(transform).get()
    test(model,train_loader)


