@echo off
title Lancement Interface Web GraphRAG (Streamlit)
cd /d "%~dp0"

echo ======================================================================
echo   DEMARRAGE DE L'APPLICATION WEB GRAPHRAG (5IADA - EMSI)
echo ======================================================================
echo.

if exist "venv\Scripts\streamlit.exe" (
    venv\Scripts\streamlit.exe run app.py
) else (
    streamlit run app.py
)

pause

