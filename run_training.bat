@echo off
REM モデル訓練スクリプト実行用バッチファイル

echo ========================================
echo ガス需要予測モデル訓練スクリプト
echo ========================================

REM 仮想環境の有効化（存在する場合）
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM モデル訓練スクリプトの実行
python scripts/train_model.py

pause
