"""
Streamlit Cloud用のエントリーポイント
dashboard/app_deploy.pyをインポート
"""

# Streamlit Cloudでは、このファイルがエントリーポイントになります
# dashboard/app_deploy.pyの内容を実行

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# デプロイ用アプリを実行
exec(open('dashboard/app_deploy.py').read())
