# CPU環境でのセットアップガイド

## ✅ CPU環境でも完全に動作します！

このプロジェクトは**CPU環境で完全に動作**するように設計されています。

## 🚀 クイックセットアップ（Windows）

### 方法1: 自動セットアップスクリプト（推奨）

```bash
# プロジェクトディレクトリで実行
setup_and_run.bat
```

このスクリプトが以下を自動実行します：
1. 仮想環境の作成
2. 依存パッケージのインストール
3. モデル訓練の実行

### 方法2: 手動セットアップ

```bash
# 1. 仮想環境の作成
python -m venv venv
venv\Scripts\activate

# 2. 依存パッケージのインストール
pip install -r requirements.txt

# 3. モデル訓練
python scripts/train_model.py
```

## 📦 インストールされるパッケージ

### 必須パッケージ（CPU環境で動作）
- **XGBoost**: 高速な勾配ブースティング（CPU最適化済み）
- **scikit-learn**: 機械学習ライブラリ（CPU最適化済み）
- **pandas, numpy**: データ処理（CPU最適化済み）
- **SHAP**: 解釈可能性分析（CPU環境で動作）
- **Streamlit**: ダッシュボード（CPU環境で動作）

### オプショナルパッケージ
- **TensorFlow**: LSTM用（現在は使用していないため、requirements.txtでコメントアウト済み）

## ⚡ パフォーマンス

### CPU環境での実行時間（目安）

- **データ生成**: 数秒
- **特徴量エンジニアリング**: 10-30秒
- **XGBoost訓練**: 1-3分（データサイズによる）
- **Random Forest訓練**: 30秒-2分
- **SHAP分析**: 1-5分（サンプルサイズによる）

### メモリ使用量

- **最小**: 2GB RAM
- **推奨**: 4GB RAM以上
- **大規模データ**: 8GB RAM以上

## 🔧 最適化のヒント

### 1. SHAP分析の高速化

`config.yaml`でサンプルサイズを調整：

```yaml
interpretability:
  shap:
    sample_size: 50  # デフォルト100、小さくすると高速化
```

### 2. モデル訓練の高速化

`config.yaml`でパラメータを調整：

```yaml
model:
  xgboost:
    n_estimators: 50  # デフォルト100、小さくすると高速化
    max_depth: 4      # デフォルト6、小さくすると高速化
```

### 3. データサイズの調整

`config.yaml`で期間を短縮：

```yaml
model:
  train_start_date: "2022-01-01"  # 期間を短くすると高速化
```

## 🐛 トラブルシューティング

### エラー: メモリ不足

**解決策**:
1. `config.yaml`の`sample_size`を小さくする
2. データ期間を短くする
3. バッチサイズを小さくする

### エラー: インストールが遅い

**解決策**:
- 初回インストールは時間がかかります（特にXGBoost、SHAP）
- 10-15分程度かかる場合があります

### エラー: SHAPが遅い

**解決策**:
- `sample_size`を50以下に設定
- TreeSHAPが使用されているか確認（KernelSHAPより高速）

## ✅ 動作確認

セットアップ後、以下で動作確認：

```bash
# 1. モデル訓練（正常に完了するか確認）
python scripts/train_model.py

# 2. SHAP分析（正常に完了するか確認）
python scripts/interpretability_analysis.py

# 3. ダッシュボード（正常に起動するか確認）
streamlit run dashboard/app.py
```

## 💡 CPU環境での利点

1. **GPU不要**: 追加ハードウェア不要
2. **軽量**: メモリ使用量が少ない
3. **高速**: XGBoostはCPUでも十分高速
4. **互換性**: どのPCでも動作

## 📊 推奨システム要件

- **OS**: Windows 10/11, macOS, Linux
- **CPU**: 2コア以上（4コア推奨）
- **RAM**: 4GB以上（8GB推奨）
- **ストレージ**: 2GB以上の空き容量
- **Python**: 3.9以上

---

**CPU環境で完全に動作します！安心して使用してください。** 🚀
