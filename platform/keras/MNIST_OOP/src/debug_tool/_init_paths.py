import os.path as osp
import sys
src_dir = osp.dirname(osp.dirname(__file__))

project_dir = osp.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.insert(0,src_dir)
