"""
Streamlit Cloud用のエントリーポイント
dashboard/app_deploy.pyをインポート
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# デプロイ用アプリを実行
try:
    # app_deploy.pyの内容を直接インポート
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "app_deploy", 
        project_root / "dashboard" / "app_deploy.py"
    )
    app_deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_deploy)
except Exception as e:
    # フォールバック: execを使用
    with open(project_root / 'dashboard' / 'app_deploy.py', 'r', encoding='utf-8') as f:
        exec(f.read())
