# デプロイメントガイド

## 🌐 Streamlit Cloudで公開する方法

### 1. Streamlit Cloudにアクセス
https://share.streamlit.io/ にアクセス

### 2. GitHubアカウントでログイン
GitHubアカウントでログインします

### 3. アプリをデプロイ
1. "New app" をクリック
2. Repository: `keisuke58/gas_demand_prediction` を選択
3. Branch: `main` を選択
4. Main file path: `dashboard/app.py` を指定
5. "Deploy!" をクリック

### 4. 公開URL
デプロイが完了すると、以下のようなURLが生成されます：
`https://gas-demand-prediction-xxxxx.streamlit.app`

## 📋 デプロイ前の確認事項

### 必要なファイル
- ✅ `dashboard/app.py` - メインアプリ
- ✅ `requirements.txt` - 依存パッケージ
- ✅ `config.yaml` - 設定ファイル
- ✅ `models/` - 訓練済みモデル（GitHubにプッシュする場合は必要）

### 注意事項
1. **モデルファイル**: `models/*.pkl` は `.gitignore` で除外されています
   - デプロイする場合は、モデルファイルをGitHubに追加する必要があります
   - または、デプロイ時にモデルを再訓練するように変更

2. **データファイル**: `data/processed/features.csv` も除外されています
   - デプロイ時にデータを再生成するか、サンプルデータを使用

## 🔧 デプロイ用の設定変更

### オプション1: モデルとデータをGitHubに追加
```bash
# .gitignoreから除外
git add -f models/*.pkl
git add -f data/processed/features.csv
git commit -m "Add model and data files for deployment"
git push
```

### オプション2: デプロイ時にモデルを再訓練
`dashboard/app.py` を修正して、モデルが存在しない場合は自動的に訓練するようにする

## 🚀 クイックデプロイ

1. GitHubリポジトリ: https://github.com/keisuke58/gas_demand_prediction
2. Streamlit Cloud: https://share.streamlit.io/
3. 上記の手順でデプロイ

---

**デプロイが完了すると、誰でもウェブブラウザからアクセスできるようになります！**
