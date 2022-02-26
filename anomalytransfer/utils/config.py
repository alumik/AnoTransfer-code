import os
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configparser

DEFAULT_CONFIGS = [
    os.path.join(PROJECT_PATH, "sample", "configs", "default.conf"),
]

LOCAL_CONFIGS = [
    os.path.join(PROJECT_PATH, "sample", "configs", "local.conf"),
]


def config() -> configparser.ConfigParser:
    config_parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config_list = []
    config_list.extend(DEFAULT_CONFIGS)
    for local_config in LOCAL_CONFIGS:
        if os.path.exists(local_config):
            config_list.append(local_config)
    config_parser.read(config_list)
    return config_parser
