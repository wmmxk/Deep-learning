import torch
from torch.autograd import Variable 
import torch.nn.functional as F
from focal_segmentation import FocalLoss, FocalLossOneChannel, FocalLossTrick, FocalLossTorch 

h,w = 3,3
y = torch.rand(2,2,h,w)
y = Variable(y)

t = torch.rand(2,1,h,w)>0.5
t = t.long()
t = Variable(t)

pt2 = F.softmax(y,dim=1)
pt1Chanel = pt2[:,1:,:,:]
print(t.view(-1,1).size())

error = FocalLossTrick()(y[:,1:,:,:],t.float())
print("Focal stable error", error)


error = FocalLossTorch()(y[:,1:,:,:],t.float())
print("Focal loss error by pytorch", error)


# you implemetation assume t is missing the 2nd dimension
#error = FocalLoss()(y,t)
#print("error", error)

#error = FocalLossOneChannel()(pt1Chanel,t)
#print("Focal not stable error", error)




alpha=0.2
alphas = torch.Tensor([alpha,1-alpha])
target = torch.LongTensor([[1,0],[0,1]])
weights = alphas.gather(0, target.view(-1))

