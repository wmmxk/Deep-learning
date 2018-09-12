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
print(pt1Chanel.size())

error = FocalLoss()(y,t)
print("error", error)


error = FocalLossOneChannel()(pt1Chanel,t)
print("error", error)


