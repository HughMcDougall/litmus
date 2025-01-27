@echo off
set /p "commitname=Enter commit message: "

git add .
git commit -m "%commitname%"
git push