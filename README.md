# ガス需要予測モデルと解釈可能性分析プロジェクト

## 📋 プロジェクト概要

気象データ、経済指標、時間要因などを用いてガス需要を高精度に予測し、SHAP/LIMEで需要変動の要因を解釈するプロジェクトです。

## 🔗 GitHubリポジトリ

**リポジトリURL**: https://github.com/keisuke58/gas_demand_prediction

## 🗂️ プロジェクト構造

```
gas-demand-forecast/
├── README.md                    # このファイル
├── requirements.txt             # Python依存パッケージ
├── config.yaml                  # 設定ファイル
├── data/                        # データディレクトリ
│   ├── raw/                     # 生データ
│   ├── processed/               # 処理済みデータ
│   └── external/                # 外部データ（気象データ等）
├── notebooks/                   # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_interpretability_analysis.ipynb
├── src/                         # ソースコード
│   ├── __init__.py
│   ├── data/                    # データ処理モジュール
│   │   ├── __init__.py
│   │   ├── load_data.py         # データ読み込み
│   │   ├── preprocess.py        # 前処理
│   │   └── feature_engineering.py  # 特徴量エンジニアリング
│   ├── models/                  # モデルモジュール
│   │   ├── __init__.py
│   │   ├── train.py             # モデル訓練
│   │   ├── predict.py           # 予測
│   │   └── evaluate.py          # 評価
│   ├── interpretability/        # 解釈可能性モジュール
│   │   ├── __init__.py
│   │   ├── shap_analysis.py     # SHAP分析
│   │   └── visualization.py    # 可視化
│   └── utils/                   # ユーティリティ
│       ├── __init__.py
│       └── helpers.py           # ヘルパー関数
├── models/                      # 訓練済みモデル保存先
├── results/                     # 結果保存先
│   ├── figures/                 # 図表
│   └── reports/                 # レポート
├── dashboard/                   # Streamlitダッシュボード
│   └── app.py
└── scripts/                     # 実行スクリプト
    ├── train_model.py
    ├── predict.py
    └── generate_report.py
```

## 🚀 セットアップ

### クイックスタート（Windows - 推奨）

```bash
# 自動セットアップスクリプトを実行
setup_and_run.bat
```

### 手動セットアップ

```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化（Windows）
venv\Scripts\activate

# 仮想環境の有効化（Mac/Linux）
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

**注意**: CPU環境で完全に動作します。GPUは不要です。詳細は`CPU_SETUP.md`を参照してください。

### 2. データの準備

`data/raw/` ディレクトリに以下を配置：
- ガス需要データ（CSV形式）
- 気象データ（CSV形式、またはAPIから取得）

### 3. 設定ファイルの編集

`config.yaml` を編集して、データパスやパラメータを設定

## 📊 使用方法

### 1. データ探索と前処理

```bash
# Jupyter Notebookで実行
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. モデル訓練

```bash
python scripts/train_model.py
```

### 3. 予測実行

```bash
python scripts/predict.py
```

### 4. ダッシュボード起動

```bash
streamlit run dashboard/app.py
```

## 📈 主要機能

1. **データ処理**
   - 時系列データの前処理
   - 気象データとの統合
   - 特徴量エンジニアリング

2. **モデル訓練**
   - XGBoost
   - Random Forest
   - Prophet
   - アンサンブル

3. **解釈可能性分析**
   - SHAP値の計算と可視化
   - Partial Dependence Plots
   - 特徴量重要度分析

4. **ダッシュボード**
   - 需要予測の可視化
   - 解釈可能性の可視化
   - インタラクティブな分析

## 📝 ライセンス

このプロジェクトはデータサイエンティスト向けの実装例です。
