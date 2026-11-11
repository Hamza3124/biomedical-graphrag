@echo off
title Demonstration PFA - GraphRAG Biomedical Pipeline
cd /d "%~dp0"

echo ======================================================================
echo   LANCEMENT DU DEMONSTRATEUR PFA (5IADA - EMSI)
echo   Diseases Cause Oriented Analysis using RAG and Knowledge Graphs
echo ======================================================================
echo.

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe demo.py
) else (
    python demo.py
)

pause

