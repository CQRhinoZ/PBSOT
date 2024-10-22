# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ..utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""


from .roadgroup import roadgroup
from .CSG import CSG
from .SYSU import SYSUGroup


__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
