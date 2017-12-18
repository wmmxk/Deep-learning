import os.path as osp
import sys
code_dir = osp.dirname(osp.dirname(__file__))
project_dir = osp.dirname(code_dir)
if code_dir not in sys.path:
    sys.path.insert(0,code_dir)

