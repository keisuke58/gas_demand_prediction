"""
ヘルパー関数
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def ensure_dir(dir_path: str):
    """
    ディレクトリが存在しない場合は作成
    
    Parameters
    ----------
    dir_path : str
        ディレクトリパス
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    設定ファイルを読み込む
    
    Parameters
    ----------
    config_path : str
        設定ファイルのパス
    
    Returns
    -------
    dict
        設定の辞書
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
