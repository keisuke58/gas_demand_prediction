# 🌐 Streamlit Cloud セットアップ手順

## ✅ GitHubリポジトリ
**リポジトリ**: https://github.com/keisuke58/gas_demand_prediction

## 🚀 デプロイ手順（5分で完了）

### ステップ1: Streamlit Cloudにアクセス
https://share.streamlit.io/ にアクセス

### ステップ2: GitHubでログイン
「Sign in with GitHub」をクリックしてGitHubアカウントでログイン

### ステップ3: アプリをデプロイ
1. 右上の「New app」ボタンをクリック
2. 以下の情報を入力：
   - **Repository**: `keisuke58/gas_demand_prediction`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.10以上（自動検出）
3. 「Deploy!」ボタンをクリック

### ステップ4: デプロイ完了を待つ
- 初回デプロイは3-5分かかります
- 依存パッケージのインストールが自動的に行われます

### ステップ5: 公開URLを取得
デプロイが完了すると、以下のようなURLが生成されます：
```
https://gas-demand-prediction-xxxxx.streamlit.app
```

このURLを共有すれば、誰でもアクセスできます！

## 📋 デプロイ後の動作

### 自動生成機能
`streamlit_app.py` は `app_deploy.py` を使用しており、以下の機能があります：

1. **データの自動生成**: `data/processed/features.csv` が存在しない場合、自動的にサンプルデータを生成
2. **モデルの自動訓練**: `models/*.pkl` が存在しない場合、自動的にモデルを訓練
3. **初回アクセス**: 少し時間がかかりますが、その後は高速に動作

### 利用可能な機能
- ✅ データ概要の表示
- ✅ 時系列プロット（インタラクティブ）
- ✅ 需要予測と評価指標
- ✅ SHAP分析（ボタンクリックで実行）
- ✅ 特徴量重要度の可視化

## 🔄 更新方法

コードを更新した場合：

1. GitHubにプッシュ
   ```bash
   git add .
   git commit -m "Update code"
   git push
   ```

2. Streamlit Cloudが自動的に再デプロイ
   - 通常、数分で反映されます
   - 「Manage app」から手動で再デプロイも可能

## ⚙️ 設定のカスタマイズ

### 環境変数の設定
Streamlit Cloudの「Settings」から環境変数を設定できます。

### リソースの調整
- デフォルトで十分なリソースが割り当てられます
- 必要に応じて「Settings」から調整可能

## 🐛 トラブルシューティング

### デプロイエラー: モジュールが見つからない
- `requirements.txt` にすべての依存パッケージが含まれているか確認
- Streamlit Cloudのログを確認

### デプロイエラー: メモリ不足
- `config.yaml` の `sample_size` を小さくする
- データ期間を短くする

### アプリが起動しない
- Streamlit Cloudのログを確認
- `streamlit_app.py` が正しく設定されているか確認

## 📊 デプロイ状態の確認

Streamlit Cloudのダッシュボードで以下を確認できます：
- デプロイ状態（Running / Error）
- アクセスログ
- エラーログ
- リソース使用状況

---

**デプロイが完了したら、URLを共有して誰でもアクセスできるようになります！** 🎉
