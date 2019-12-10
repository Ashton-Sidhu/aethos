from aethos.config import cfg, DEFAULT_IMAGE_DIR
from aethos.util import _make_dir


def _make_image_dir():  # pragma: no cover

    if not cfg["images"]["dir"]:
        image_dir = DEFAULT_IMAGE_DIR
    else:
        image_dir = cfg["images"]["dir"]

    _make_dir(image_dir)

    return image_dir
