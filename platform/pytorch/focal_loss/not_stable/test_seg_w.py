import torch
from torch.autograd import Variable 
import torch.nn.functional as F
from focal_segmentation_weight import FocalLoss, FocalLossOneChannel 

torch.manual_seed(1)
h,w = 3,3
n=1
y = torch.rand(n,2,h,w)
y = Variable(y)
t = torch.rand(n,h,w)*1 > 0.5
t = t.long()
t = Variable(t)
print("target", t)

pt2 = F.softmax(y,dim=1)
pt1Chanel = pt2[:,1:,:,:]
#print("prediction 2:", pt2)
print("pred 1 channel", pt1Chanel)


error = FocalLossOneChannel(alpha=0.2)(pt1Chanel,t)
print("error", error)


