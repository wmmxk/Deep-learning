import numpy as np
def get_accuracy(cls_prob,labels):
    pred = np.argmax(cls_prob,axis =1)
    num_correct = np.sum(np.equal(pred,labels))
    return(100.*num_correct/pred.shape[0])
