@echo off
echo プログラム開始


SET VIRTUAL_ENV_NAME="similar_search"

REM 仮想環境をactivate
@echo on
call activate %VIRTUAL_ENV_NAME%

call python similar_search.py