import os.path as osp
import sys
project_dir = osp.dirname(osp.dirname(__file__))

if project_dir not in sys.path:
    sys.path.insert(0,project_dir)
