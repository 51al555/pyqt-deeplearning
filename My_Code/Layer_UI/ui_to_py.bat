@echo off
goto start
echo 无后缀名： %~n1
echo 有后缀名： %~nx1
echo 绝对路径： %1
echo 短路径名的绝对路径： %~s1
echo 驱动器和路径： %~dp1
echo 驱动器： %~d1
echo 路径： %~p1
echo 文件属性： %~a1
echo 日期/时间： %~t1
echo 文件大小： %~z1
:start

CALL F:\anaconda\Scripts\activate.bat F:\anaconda
python -m PyQt5.uic.pyuic %~nx1 -o %~dp1%~n1.py