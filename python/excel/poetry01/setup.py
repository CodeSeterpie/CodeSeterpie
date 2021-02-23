import sys
from cx_Freeze import setup, Executable
 
base = None
 
if sys.platform == 'win32':
    base = 'Win32GUI'
# Winデスクトップ出ない場合「CUI」の場合はif文をコメントアウト
 
exe = Executable(script = "test.py", base= base)
# "test.py"にはexe化するファイルの名前を記載。
 
setup(name = 'your_filename',
    version = '0.1',
    description = 'converter',
    executables = [exe])