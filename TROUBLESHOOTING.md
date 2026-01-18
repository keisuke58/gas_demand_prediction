# 🔧 Streamlit Cloud トラブルシューティング

## ❌ エラー: Installing requirements failed

### 原因
- パッケージのバージョン指定が厳しすぎる
- 互換性のないパッケージの組み合わせ
- インストールに時間がかかりすぎるパッケージ

### 解決策

#### 1. requirements.txtを簡素化
現在の`requirements.txt`は最小限のパッケージのみを含んでいます。

#### 2. バージョン指定を緩和
厳密なバージョン指定（`==`）を緩い指定（`>=`）に変更しました。

#### 3. オプショナルパッケージを除外
以下のパッケージはオプショナルとして除外しました：
- `prophet` - インストールに時間がかかる
- `lightgbm` - 現在使用していない
- `lime` - 現在使用していない
- `tsfresh` - 現在使用していない
- `optuna` - 現在使用していない

### デプロイ時の設定

#### Main file path
- `streamlit_app.py` を使用（推奨）

#### Python version
- 3.10以上（自動検出）

#### Advanced settings
- 必要に応じて環境変数を設定

## 🔄 再デプロイ手順

1. **requirements.txtを確認**
   - 最小限のパッケージのみが含まれているか確認

2. **Streamlit Cloudで再デプロイ**
   - 「Manage app」→「Reboot app」をクリック
   - または、新しいアプリとして再デプロイ

3. **ログを確認**
   - Streamlit Cloudの「Logs」タブでエラーを確認

## ✅ 動作確認済みパッケージ

以下のパッケージはStreamlit Cloudで正常に動作します：
- ✅ pandas
- ✅ numpy
- ✅ xgboost
- ✅ scikit-learn
- ✅ shap
- ✅ matplotlib
- ✅ seaborn
- ✅ plotly
- ✅ streamlit
- ✅ pyyaml

## 🐛 よくあるエラーと解決策

### エラー: "No module named 'xxx'"
**原因**: 必要なパッケージが`requirements.txt`に含まれていない、またはインストールに失敗した

**解決策**:
1. `requirements.txt`にパッケージを追加
2. バージョン指定を緩和（`>=`を使用）
3. Streamlit Cloudのログでインストールエラーを確認

### エラー: "Version conflict"
**原因**: パッケージのバージョン指定が厳しすぎる、または互換性のない組み合わせ

**解決策**:
1. バージョン指定を`>=`に変更（`==`は避ける）
2. 最小バージョンを下げる
3. 互換性のあるバージョンに調整

### エラー: "Installation timeout"
**原因**: パッケージのインストールに時間がかかりすぎる

**解決策**:
1. 不要なパッケージを削除
2. オプショナルなパッケージ（prophet、lightgbmなど）を除外
3. 軽量な代替パッケージを使用

### エラー: "Memory error"
**原因**: メモリ不足

**解決策**:
1. `config.yaml`の`sample_size`を小さくする
2. データ期間を短くする
3. SHAP分析のサンプルサイズを減らす

### エラー: "ImportError: cannot import name 'xxx'"
**原因**: パッケージのバージョンが古い、または新しすぎる

**解決策**:
1. パッケージを最新バージョンに更新
2. 互換性のあるバージョンに固定
3. 代替パッケージを使用

### エラー: "FileNotFoundError"
**原因**: 必要なファイルがリポジトリに含まれていない

**解決策**:
1. `.gitignore`を確認し、必要なファイルが除外されていないか確認
2. ファイルをGitHubにコミット・プッシュ
3. ファイルパスが正しいか確認

## 🔍 デバッグ手順

### 1. Streamlit Cloudのログを確認
1. Streamlit Cloudのダッシュボードにアクセス
2. アプリを選択
3. 「Logs」タブを開く
4. エラーメッセージを確認

### 2. ローカルでテスト
```bash
# 仮想環境を作成
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 依存パッケージをインストール
pip install -r requirements.txt

# アプリを起動
streamlit run streamlit_app.py
```

### 3. パッケージのインストールを確認
```bash
# インストール済みパッケージを確認
pip list

# 特定のパッケージのバージョンを確認
pip show streamlit pandas numpy xgboost shap
```

## 📝 最新の修正内容

### requirements.txtの最適化（最新）
- バージョン指定を緩和（`>=`を使用）
- 最小バージョンを下げて互換性を向上
- 必須パッケージのみを含める

### streamlit_app.pyの改善（最新）
- エラーハンドリングを強化
- 診断情報を追加
- より詳細なエラーメッセージを表示

---

**問題が解決しない場合は、Streamlit Cloudのログを確認し、エラーメッセージを共有してください。**
