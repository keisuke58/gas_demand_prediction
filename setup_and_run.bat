@echo off
REM 完全セットアップと実行スクリプト

echo ========================================
echo ガス需要予測プロジェクト - 完全セットアップ
echo ========================================
echo.

REM 仮想環境の確認と作成
if not exist venv (
    echo [1/4] 仮想環境を作成中...
    python -m venv venv
    echo 仮想環境を作成しました
) else (
    echo 仮想環境は既に存在します
)

REM 仮想環境の有効化
echo.
echo [2/4] 仮想環境を有効化中...
call venv\Scripts\activate.bat

REM 依存パッケージのインストール
echo.
echo [3/4] 依存パッケージをインストール中...
echo （初回は数分かかる場合があります）
pip install --upgrade pip
pip install -r requirements.txt

REM モデル訓練の実行
echo.
echo [4/4] モデル訓練を実行中...
python scripts/train_model.py

echo.
echo ========================================
echo セットアップが完了しました！
echo ========================================
echo.
echo 次のステップ:
echo 1. SHAP分析を実行: python scripts/interpretability_analysis.py
echo 2. ダッシュボード起動: streamlit run dashboard/app.py
echo.
pause
