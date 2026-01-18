"""
Streamlit Cloud用のエントリーポイント
dashboard/app_deploy.pyをインポート
エラーハンドリングと診断機能を追加
"""

import sys
import traceback
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 必要なパッケージのインポートチェック
def check_imports():
    """必要なパッケージがインストールされているか確認"""
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'xgboost': 'xgboost',
        'sklearn': 'scikit-learn',
        'shap': 'shap',
        'yaml': 'pyyaml',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    return missing

# パッケージチェック
missing_packages = check_imports()
if missing_packages:
    import streamlit as st
    st.error(f"❌ 以下のパッケージがインストールされていません: {', '.join(missing_packages)}")
    st.info("Streamlit Cloudのログを確認し、requirements.txtを更新してください。")
    st.stop()

# デプロイ用アプリを実行
try:
    # app_deploy.pyの内容を直接インポート
    import importlib.util
    app_deploy_path = project_root / "dashboard" / "app_deploy.py"
    
    if not app_deploy_path.exists():
        import streamlit as st
        st.error(f"❌ app_deploy.pyが見つかりません: {app_deploy_path}")
        st.info("プロジェクト構造を確認してください。")
        st.stop()
    
    spec = importlib.util.spec_from_file_location(
        "app_deploy", 
        app_deploy_path
    )
    
    if spec is None or spec.loader is None:
        raise ImportError("app_deploy.pyの読み込みに失敗しました")
    
    app_deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_deploy)
    
except ImportError as e:
    # インポートエラーの場合
    import streamlit as st
    st.error(f"❌ インポートエラー: {str(e)}")
    st.code(traceback.format_exc())
    st.info("requirements.txtを確認し、すべての依存パッケージがインストールされているか確認してください。")
    st.stop()
    
except FileNotFoundError as e:
    # ファイルが見つからない場合
    import streamlit as st
    st.error(f"❌ ファイルが見つかりません: {str(e)}")
    st.info("プロジェクト構造を確認してください。")
    st.stop()
    
except Exception as e:
    # その他のエラー
    import streamlit as st
    st.error(f"❌ 予期しないエラーが発生しました: {str(e)}")
    st.code(traceback.format_exc())
    
    # フォールバック: execを使用
    try:
        app_deploy_path = project_root / 'dashboard' / 'app_deploy.py'
        if app_deploy_path.exists():
            st.info("フォールバックモードで実行を試みます...")
            with open(app_deploy_path, 'r', encoding='utf-8') as f:
                exec(f.read())
        else:
            st.error(f"app_deploy.pyが見つかりません: {app_deploy_path}")
            st.stop()
    except Exception as e2:
        st.error(f"フォールバックモードでもエラーが発生しました: {str(e2)}")
        st.code(traceback.format_exc())
        st.stop()
