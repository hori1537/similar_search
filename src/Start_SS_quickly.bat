@echo off
echo �v���O�����J�n


SET VIRTUAL_ENV_NAME="similar_search"

REM ���z����activate
@echo on
call activate %VIRTUAL_ENV_NAME%

call python similar_search.py