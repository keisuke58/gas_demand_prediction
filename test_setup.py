"""
セットアップと動作確認スクリプト
"""

import sys
from pathlib import Path

print("=" * 60)
print("セットアップと動作確認")
print("=" * 60)

# 1. Pythonバージョンチェック
print("\n[1/6] Pythonバージョンチェック...")
python_version = sys.version_info
print(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
    print("⚠️ 警告: Python 3.9以上を推奨します")
else:
    print("✅ Pythonバージョンは問題ありません")

# 2. 必要なパッケージのチェック
print("\n[2/6] 必要なパッケージのチェック...")
required_packages = [
    'pandas', 'numpy', 'sklearn', 'xgboost', 
    'shap', 'streamlit', 'plotly', 'yaml'
]

missing_packages = []
for package in required_packages:
    try:
        if package == 'sklearn':
            __import__('sklearn')
        elif package == 'yaml':
            __import__('yaml')
        else:
            __import__(package)
        print(f"  ✅ {package}")
    except ImportError:
        print(f"  ❌ {package} (未インストール)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠️ 以下のパッケージがインストールされていません: {', '.join(missing_packages)}")
    print("  実行: pip install -r requirements.txt")
else:
    print("\n✅ すべての必須パッケージがインストールされています")

# 3. ディレクトリ構造のチェック
print("\n[3/6] ディレクトリ構造のチェック...")
required_dirs = [
    'src', 'src/data', 'src/models', 'src/interpretability',
    'scripts', 'dashboard', 'data/raw', 'data/processed',
    'models', 'results/figures'
]

missing_dirs = []
for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"  ✅ {dir_path}/")
    else:
        print(f"  ❌ {dir_path}/ (存在しません)")
        missing_dirs.append(dir_path)

if missing_dirs:
    print(f"\n⚠️ 以下のディレクトリが存在しません: {', '.join(missing_dirs)}")
    print("  これらは実行時に自動的に作成されます")
else:
    print("\n✅ ディレクトリ構造は問題ありません")

# 4. 設定ファイルのチェック
print("\n[4/6] 設定ファイルのチェック...")
if Path("config.yaml").exists():
    print("  ✅ config.yaml")
    try:
        import yaml
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("  ✅ config.yamlの読み込み成功")
    except Exception as e:
        print(f"  ❌ config.yamlの読み込みエラー: {e}")
else:
    print("  ❌ config.yaml (存在しません)")

# 5. モジュールのインポートテスト
print("\n[5/6] モジュールのインポートテスト...")
try:
    from src.data.load_data import load_config, generate_sample_data
    print("  ✅ src.data.load_data")
except Exception as e:
    print(f"  ❌ src.data.load_data: {e}")

try:
    from src.models.train import train_xgboost
    print("  ✅ src.models.train")
except Exception as e:
    print(f"  ❌ src.models.train: {e}")

try:
    from src.interpretability.shap_analysis import calculate_shap_values
    print("  ✅ src.interpretability.shap_analysis")
except Exception as e:
    print(f"  ❌ src.interpretability.shap_analysis: {e}")

# 6. 簡単な動作テスト
print("\n[6/6] 簡単な動作テスト...")
try:
    import pandas as pd
    import numpy as np
    
    # サンプルデータ生成テスト
    from src.data.load_data import generate_sample_data
    test_data = generate_sample_data(
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    print(f"  ✅ サンプルデータ生成: {test_data.shape}")
    
    # 特徴量エンジニアリングテスト
    from src.data.preprocess import create_time_features
    from src.data.feature_engineering import create_all_features
    
    test_data = create_time_features(test_data)
    test_data = create_all_features(test_data, target_col="demand")
    print(f"  ✅ 特徴量エンジニアリング: {test_data.shape}")
    
except Exception as e:
    print(f"  ❌ 動作テストエラー: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("動作確認が完了しました")
print("=" * 60)
print("\n次のステップ:")
print("1. モデル訓練: python scripts/train_model.py")
print("2. SHAP分析: python scripts/interpretability_analysis.py")
print("3. ダッシュボード: streamlit run dashboard/app.py")
print("=" * 60)
