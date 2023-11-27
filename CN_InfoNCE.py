import torch

import torch.nn  as nn

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
#define CN_InfoNCE
class CN_InfoNCE():
    def __init__(self,output_aug_1,output_aug_2,temperature):
        self.output_aug_1 = output_aug_1
        self.output_aug_2 = output_aug_2
        self.tempearture = temperature


    def loss(self):
        output = self.output_aug_1
        output_aug = self.output_aug_2
        temperature = self.tempearture
        #concat the two matrix
        output_all = torch.cat((output, output_aug),dim=0).cuda()
        #get the label by itself
        positive = torch.cat(((torch.arange(output.shape[0],output_all.shape[0])),(torch.arange(0,output.shape[0]))),dim=0).cuda()
        labels = torch.zeros((output_all.shape[0],output_all.shape[0])).cuda()
        for i in range(labels.shape[0]):
            labels[i][positive[i]]=1
        #get the center of negative samples
        cluster_neg = torch.zeros((output.shape[0],output.shape[1])).cuda()
        for i in range(output.shape[0]):
            #mask the positive sample
            Mask = torch.zeros((output_all.shape[0],output_all.shape[1])).cuda()
            Mask[i,:]=1
            Mask[i+output.shape[0],:]=1
            cluster_neg[i] = torch.mean(output_all[~Mask.bool()].view(-1,output.shape[1]),dim=0)
        #calculate the similarity between the cluster_negs and positive samples
        cluster_neg = torch.cat((cluster_neg,cluster_neg),dim=0).cuda()
        cluster_sim = torch.sum(torch.mul(output_all,cluster_neg),dim=1).view(-1,1).cuda()
        cluster_label = torch.zeros((256,1)).cuda()
        #add the parameter w
        Ela_co = nn.Parameter(torch.ones((cluster_sim.shape[0],1))).cuda()
        Ela_co.data.clamp_(min=1, max=2)
        cluster_sim = torch.mul(cluster_sim,Ela_co)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        labels = torch.cat((labels,cluster_label),dim=1).cuda()
        cos_similarity_matrix = torch.matmul(output_all,torch.transpose(output_all,0,1))
        cos_similarity_matrix = cos_similarity_matrix[~mask].view(cos_similarity_matrix.shape[0], -1)
        cos_similarity_matrix = torch.cat((cos_similarity_matrix,cluster_sim),dim=1)
        cos_similarity_matrix = cos_similarity_matrix / temperature
        #put the positive on the first position
        positive_sim = cos_similarity_matrix[labels.bool()].view(cos_similarity_matrix.shape[0],-1).cuda()
        neg_sim = cos_similarity_matrix[~labels.bool()].view(cos_similarity_matrix.shape[0],-1).cuda()
        logits = torch.cat([positive_sim,neg_sim],dim=1).cuda()
        #Cross Entropy function needs to know the right label,and in contrastive learning, the label of positive samples is the right label
        # The positive samples is on the first position ,so the label is 0
        labels_logits = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
        return logits, labels_logits












