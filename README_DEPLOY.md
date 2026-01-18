# 🚀 Streamlit Cloud デプロイガイド

## クイックデプロイ（3ステップ）

### ステップ1: GitHubリポジトリを確認
- リポジトリ: https://github.com/keisuke58/gas_demand_prediction
- ブランチ: `main`

### ステップ2: Streamlit Cloudにアクセス
https://share.streamlit.io/ にアクセスしてGitHubアカウントでログイン

### ステップ3: アプリをデプロイ
1. "New app" をクリック
2. **Repository**: `keisuke58/gas_demand_prediction`
3. **Branch**: `main`
4. **Main file path**: `streamlit_app.py` または `dashboard/app_deploy.py`
5. **Python version**: 3.10以上
6. "Deploy!" をクリック

## 📝 デプロイ設定

### Main file path の選択肢
- `streamlit_app.py` (推奨) - エントリーポイント
- `dashboard/app_deploy.py` - デプロイ用アプリ（自動生成機能付き）

### 自動生成機能
`app_deploy.py` を使用すると：
- モデルが存在しない場合は自動的に訓練
- データが存在しない場合は自動的に生成
- 初回アクセス時に少し時間がかかりますが、その後は高速

## 🌐 公開URL

デプロイが完了すると、以下のようなURLが生成されます：
```
https://gas-demand-prediction-xxxxx.streamlit.app
```

このURLを共有すれば、誰でもアクセスできます！

## 🔧 トラブルシューティング

### エラー: モジュールが見つからない
- `requirements.txt` にすべての依存パッケージが含まれているか確認

### エラー: モデルが見つからない
- `app_deploy.py` を使用すると自動的に生成されます
- または、`models/` ディレクトリのファイルをGitHubに追加

### エラー: データが見つからない
- `app_deploy.py` を使用すると自動的に生成されます

## 📊 デプロイ後の機能

- ✅ データ概要の表示
- ✅ 時系列プロット
- ✅ 需要予測と評価指標
- ✅ SHAP分析（ボタンクリックで実行）
- ✅ インタラクティブな可視化

---

**デプロイが完了したら、URLを共有して誰でもアクセスできるようになります！** 🎉
