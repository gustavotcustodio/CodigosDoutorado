@echo off
setlocal enabledelayedexpansion

REM DATASETS=australian_credit german_credit heart pima wdbc contraceptive iris wine
set DATASETS=blood
set BASE_CLASSIFIERS=svm dt lr

for %%D in (%DATASETS%) do (
    REM for %%M in (75 50) do (
    for %%C in (%BASE_CLASSIFIERS%) do (
        python supervised_clustering.py -d %%D -b %%C
        REM python supervised_clustering.py -d %%D -b %%C -m %%M
    )
    REM )
)
