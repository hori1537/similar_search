@echo off
echo �v���O�����J�n

REM start_program�̂���t�H���_�ɖ߂�
cd /d %~dp0

REM conda�����s�ł��邩�m�F

call conda -V

if %errorlevel% neq 0 ( 
echo conda��PATH���ʂ��Ă��܂���
echo Anaconda �v�����v�g���N�����Astart.bat�����s���Ă�������
echo 10�b��Ƀv���O�������I�����܂�
TIMEOUT /T 10
exit /B
) else (
echo conda�R�}���h�͐���Ɏ��s����܂���
)

SET VIRTUAL_ENV_NAME="similar_search"

REM ���z����activate
@echo on
call activate %VIRTUAL_ENV_NAME%
@echo off

if %errorlevel% neq 0 (
echo ���z����activate���s
echo ���z���̍쐬�J�n
@echo on
echo Y | call conda create -n %VIRTUAL_ENV_NAME% python=3.6
call activate %VIRTUAL_ENV_NAME%
) else (
@echo off
echo ���z����activate���܂���
)

@echo on
REM �K�v�ȃ��C�u������pip install
call pip install -r requirements.txt
call python similar_search.py