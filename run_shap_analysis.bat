@echo off
REM SHAP分析スクリプト実行用バッチファイル

echo ========================================
echo 解釈可能性分析スクリプト
echo ========================================

REM 仮想環境の有効化（存在する場合）
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM SHAP分析スクリプトの実行
python scripts/interpretability_analysis.py

pause
