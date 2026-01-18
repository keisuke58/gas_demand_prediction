# クイックスタートガイド

## 🚀 セットアップ（5分で開始）

### 1. 環境構築

```bash
# プロジェクトディレクトリに移動
cd gas-demand-forecast

# 仮想環境の作成（Windows）
python -m venv venv
venv\Scripts\activate

# 仮想環境の作成（Mac/Linux）
python3 -m venv venv
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. サンプルデータの生成とモデル訓練

```bash
# モデル訓練スクリプトを実行（サンプルデータを自動生成）
python scripts/train_model.py
```

このスクリプトは以下を実行します：
- サンプルデータの自動生成（実際のデータがない場合）
- データの前処理と特徴量エンジニアリング
- XGBoostとRandom Forestモデルの訓練
- モデルの評価と保存

### 3. 解釈可能性分析

```bash
# SHAP分析を実行
python scripts/interpretability_analysis.py
```

このスクリプトは以下を実行します：
- SHAP値の計算
- 特徴量重要度の可視化
- 予測結果の可視化

### 4. ダッシュボードの起動

```bash
# Streamlitダッシュボードを起動
streamlit run dashboard/app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

## 📁 プロジェクト構造

```
gas-demand-forecast/
├── README.md                    # プロジェクト説明
├── QUICKSTART.md                # このファイル
├── requirements.txt             # 依存パッケージ
├── config.yaml                  # 設定ファイル
├── data/                        # データディレクトリ
│   ├── raw/                     # 生データ
│   ├── processed/               # 処理済みデータ
│   └── external/                # 外部データ
├── src/                         # ソースコード
│   ├── data/                    # データ処理
│   ├── models/                  # モデル
│   ├── interpretability/        # 解釈可能性
│   └── utils/                   # ユーティリティ
├── scripts/                     # 実行スクリプト
│   ├── train_model.py          # モデル訓練
│   └── interpretability_analysis.py  # SHAP分析
├── dashboard/                   # Streamlitダッシュボード
│   └── app.py
├── models/                      # 訓練済みモデル
└── results/                     # 結果
    ├── figures/                 # 図表
    └── reports/                 # レポート
```

## 🔧 カスタマイズ

### 実際のデータを使用する場合

1. `data/raw/` ディレクトリに以下のCSVファイルを配置：
   - `gas_demand.csv`: 日付と需要量の列を含む
   - `weather_data.csv`: 日付、気温、降水量などの列を含む

2. `config.yaml` を編集してデータファイル名を指定

3. `scripts/train_model.py` を実行

### モデルパラメータの調整

`config.yaml` の `model` セクションを編集：

```yaml
model:
  xgboost:
    n_estimators: 200  # 増やすと精度向上（時間もかかる）
    max_depth: 8
    learning_rate: 0.05
```

## 📊 出力ファイル

### モデル訓練後

- `models/xgboost_model.pkl`: 訓練済みXGBoostモデル
- `models/random_forest_model.pkl`: 訓練済みRandom Forestモデル
- `data/processed/features.csv`: 特徴量エンジニアリング後のデータ

### SHAP分析後

- `results/figures/shap_summary.png`: SHAP Summary Plot
- `results/figures/shap_dependence_temperature.png`: 気温のDependence Plot
- `results/figures/feature_importance.png`: 特徴量重要度
- `results/feature_importance.csv`: 特徴量重要度（CSV）

## 🐛 トラブルシューティング

### エラー: モジュールが見つからない

```bash
# プロジェクトルートで実行しているか確認
# 仮想環境が有効化されているか確認
which python  # Mac/Linux
where python  # Windows
```

### エラー: データが見つからない

- `scripts/train_model.py` を実行すると自動的にサンプルデータが生成されます
- 実際のデータを使用する場合は、`data/raw/` に配置してください

### メモリエラー

- `config.yaml` の `interpretability.shap.sample_size` を小さくする（例: 50）
- データの期間を短くする

## 📚 次のステップ

1. **データの理解**: `notebooks/01_data_exploration.ipynb` でデータを探索
2. **特徴量の追加**: `src/data/feature_engineering.py` を編集
3. **モデルの改善**: `config.yaml` でパラメータを調整
4. **ダッシュボードのカスタマイズ**: `dashboard/app.py` を編集

## 💡 ヒント

- 最初はサンプルデータで動作確認
- 小規模なデータセットで試してから本番データに適用
- 特徴量エンジニアリングが最も重要（ドメイン知識を活用）

## 📞 サポート

問題が発生した場合：
1. エラーメッセージを確認
2. `config.yaml` の設定を確認
3. データの形式を確認（日付がインデックス、必要な列が存在）

---

**Happy Coding! 🚀**
