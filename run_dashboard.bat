@echo off
REM Streamlitダッシュボード起動用バッチファイル

echo ========================================
echo Streamlitダッシュボード起動
echo ========================================

REM 仮想環境の有効化（存在する場合）
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Streamlitダッシュボードの起動
streamlit run dashboard/app.py

pause
