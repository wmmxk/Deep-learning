import torch
from torch.autograd import Variable 
import torch.nn.functional as F
from focal_segmentation import FocalLoss, FocalLossOneChannel 

h,w = 3,3
y = torch.rand(2,2,h,w)
y = Variable(y)

t = torch.rand(2,h,w)*1
t = t.long()
t = Variable(t)

pt2 = F.softmax(y,dim=1)
pt1Chanel = pt2[:,1:,:,:]
print(t.view(-1,1).size())

error = FocalLoss()(y,t)
print("error", error)


error = FocalLossOneChannel()(pt1Chanel,t)
print("error", error)


alpha=0.2
alphas = torch.Tensor([alpha,1-alpha])
target = torch.LongTensor([[1,0],[0,1]])
weights = alphas.gather(0, target.view(-1))

print(weights)
