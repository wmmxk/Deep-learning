from easydict immport Easydict as edict

# the first layer
cfg = edict()

# the second layer
cfg.DATA = edict()
cfg.TRAIN = edict()


# the third layer
cfg.DATA.NUM_CLASS = 10
cfg.TRAIN.WEIGHT_DECAY = 0.1
