from yacs.config import CfgNode as CN


C = CN()
C.MODEL = CN()
C.SAVE = CN()
C.TRAIN = CN()

C.SAVE.MODEL_DIR = "./"


C.TRAIN.BATCH_SIZE = 8
C.TRAIN.LR = 0.001