import os

import yaml


class Config:
    """Thin wrapper over a YAML file, with dotted lookup and defaults.

    ``yaml.load`` without a ``Loader`` argument is a hard TypeError since PyYAML 5.1,
    which stopped every entry point in this repo from starting; ``safe_load`` is what
    this config (plain scalars, mappings and lists) needs.
    """

    def __init__(self, config_file, defaults=None):
        self.config_file = config_file

        if not os.path.exists(self.config_file):
            raise FileNotFoundError("No such file or directory: {}".format(self.config_file))

        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f) or {}

        if defaults:
            self.config = self._merge(dict(defaults), self.config)

    @staticmethod
    def _merge(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                Config._merge(base[key], value)
            else:
                base[key] = value
        return base

    def get(self, path, default=None):
        """``config.get('TRAIN.CURRICULUM.MAX_DISTORTION', 0.9)``"""
        node = self.config
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def getDict(self):
        return self.config

    def __getitem__(self, key):
        return self.config[key]
