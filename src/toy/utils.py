import logging
from pathlib import Path


def filename_relative_to_dir_root(filename: str) -> Path:
    return Path(filename).resolve().relative_to(DIR_ROOT)


def get_logger_by_filename(filename: str) -> logging.Logger:
    return logging.getLogger(str(filename_relative_to_dir_root(filename)))


DIR_ROOT = (Path(__file__).parent / '..' / '..').resolve()
HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='conf')
