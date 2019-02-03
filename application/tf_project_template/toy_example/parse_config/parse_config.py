from .config import C as cfg
import os


def test_parse_config():
    print(os.getcwd())
    cfg.merge_from_file('./parse_config/config.yaml')
    print(cfg)

