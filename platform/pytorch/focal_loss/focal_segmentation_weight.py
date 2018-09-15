import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
          
#        print("type", type(input*target.float()))
        logpt = F.log_softmax(input,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())


        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class FocalLossOneChannel(nn.Module):

    """
    The input is predicted masks the size of which is  n,1,h,w.

    target is of size: n,h,w.

    The input is probability to be forground; 

    """

    def __init__(self, gamma=0,alpha=0.5, size_average=True):
        super(FocalLossOneChannel, self).__init__()
        self.gamma = gamma
        self.alphas = torch.Tensor([1-alpha,alpha]) # .cuda() if run on GPU
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(-1)  # N,C,H,W => N,C,H*W
        # when the target used ad index, it is in long int form
        # when in exression with p, it is float
        target = target.view(-1).long()
    
        p = input
        # pt is probability to be forground or backgroud for each pixel
        weights = self.alphas.gather(0,target.data)
        
        pt = 1-target.float() + (2*target.float() - 1)*p 
        print("weights", weights)
        logpt = torch.log(pt) * Variable(weights)
        loss = -1 * (1-pt)**self.gamma * logpt

  
        if self.size_average: return loss.mean()
        else: return loss.sum()
