import yaml
from collections import namedtuple


def convert_to_namedtuple(d):
    """Convert a dict into a namedtuple"""
    if not isinstance(d, dict):
        raise ValueError("Can only convert dicts into namedtuple")
    for k,v in d.items():
        if isinstance(v, dict):
            d[k] = convert_to_namedtuple(v)
    return namedtuple('ConfigDict', d.keys())(**d)

class Config:
    @classmethod
    def load(self, file_name='backup.cfg'):

        with open(file_name, 'r') as f:
            yamlcfg = yaml.load(f)
        return convert_to_namedtuple(yamlcfg)


