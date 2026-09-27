"""Checkpoint downloads owned by the MatAnyone engine."""

from sammie.model_downloader import DownloadSpec


DOWNLOADS = {
    "matanyone": DownloadSpec(
        url="https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth",
        md5="a50eeaa149a37509feb45e3d6b06f41d",
        dest_dir="checkpoints",
    ),
    "matanyone2": DownloadSpec(
        url="https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth",
        md5="b1d3cfbb7596ecf3b88391198427ca95",
        dest_dir="checkpoints",
    ),
}
