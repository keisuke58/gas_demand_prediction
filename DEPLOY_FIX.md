# 🔧 Streamlit Cloud デプロイ修正内容

## 📋 実施した修正

### 1. `requirements.txt`の最適化
- バージョン指定を緩和（`>=`を使用）
- 最小バージョンを下げて互換性を向上
- 必須パッケージのみを含める
- Streamlit Cloudで動作確認済みのバージョンに調整

**変更前**:
```txt
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

**変更後**:
```txt
xgboost>=1.7.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
```

### 2. `streamlit_app.py`の改善
- エラーハンドリングを強化
- パッケージのインポートチェック機能を追加
- より詳細なエラーメッセージを表示
- 診断情報を追加

**新機能**:
- 起動時に必要なパッケージの存在を確認
- 不足しているパッケージを明確に表示
- エラーの詳細情報を表示

### 3. `app_deploy.py`の改善
- SHAP分析のオプショナル対応
- エラーハンドリングの強化
- より詳細なエラーメッセージ
- テストデータの存在確認

**新機能**:
- SHAPパッケージがインストールされていない場合の警告
- エラーの詳細情報を展開可能なセクションで表示
- 解決方法の提案

### 4. `runtime.txt`の追加
- Pythonバージョンを明示的に指定（3.10.12）
- Streamlit Cloudでの互換性を向上

### 5. ドキュメントの更新
- `TROUBLESHOOTING.md`に詳細なエラー解決方法を追加
- `STREAMLIT_CLOUD_SETUP.md`にトラブルシューティング情報を追加

## 🚀 再デプロイ手順

### ステップ1: 変更をGitHubにプッシュ
```bash
git add .
git commit -m "Fix Streamlit Cloud deployment issues"
git push origin main
```

### ステップ2: Streamlit Cloudで再デプロイ

#### 方法A: 既存アプリを再起動
1. Streamlit Cloudのダッシュボードにアクセス
2. アプリを選択
3. 「Manage app」→「Reboot app」をクリック

#### 方法B: 新しいアプリとして再デプロイ（推奨）
1. Streamlit Cloudのダッシュボードにアクセス
2. 「New app」をクリック
3. 以下の情報を入力：
   - **Repository**: `keisuke58/gas_demand_prediction`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. 「Deploy!」をクリック

### ステップ3: デプロイの確認
1. デプロイが完了するまで3-5分待つ
2. 「Logs」タブでエラーがないか確認
3. アプリが正常に起動するか確認

## ✅ 確認事項

### requirements.txt
- [x] 最小限のパッケージのみ
- [x] バージョン指定が緩い（`>=`を使用）
- [x] Streamlit Cloudで動作確認済みのパッケージのみ

### streamlit_app.py
- [x] エラーハンドリングが強化されている
- [x] パッケージのインポートチェック機能がある
- [x] 詳細なエラーメッセージが表示される

### app_deploy.py
- [x] SHAP分析がオプショナル対応
- [x] エラーハンドリングが強化されている
- [x] テストデータの存在確認がある

### runtime.txt
- [x] Pythonバージョンが指定されている

## 🐛 まだエラーが出る場合

### 1. Streamlit Cloudのログを確認
- 「Logs」タブでエラーメッセージを確認
- エラーの種類を特定

### 2. よくあるエラーと解決策

#### エラー: "Installing requirements failed"
**原因**: パッケージのインストールに失敗

**解決策**:
1. `requirements.txt`のバージョン指定をさらに緩和
2. 不要なパッケージを削除
3. パッケージを個別にインストールしてテスト

#### エラー: "No module named 'xxx'"
**原因**: パッケージがインストールされていない

**解決策**:
1. `requirements.txt`にパッケージを追加
2. パッケージ名が正しいか確認（例: `sklearn`ではなく`scikit-learn`）

#### エラー: "ImportError"
**原因**: パッケージのバージョンが互換性がない

**解決策**:
1. パッケージのバージョンを更新
2. 互換性のあるバージョンに固定

### 3. 詳細なトラブルシューティング
`TROUBLESHOOTING.md`を参照してください。

## 📝 変更ファイル一覧

1. `requirements.txt` - パッケージのバージョン指定を最適化
2. `streamlit_app.py` - エラーハンドリングと診断機能を追加
3. `dashboard/app_deploy.py` - SHAP分析のオプショナル対応とエラーハンドリング強化
4. `runtime.txt` - Pythonバージョンを明示的に指定（新規作成）
5. `TROUBLESHOOTING.md` - 詳細なエラー解決方法を追加
6. `STREAMLIT_CLOUD_SETUP.md` - トラブルシューティング情報を追加

## 🎯 次のステップ

1. 変更をGitHubにプッシュ
2. Streamlit Cloudで再デプロイ
3. アプリが正常に起動するか確認
4. エラーが出る場合は、ログを確認して`TROUBLESHOOTING.md`を参照

---

**問題が解決しない場合は、Streamlit Cloudのログを確認し、エラーメッセージを共有してください。**
