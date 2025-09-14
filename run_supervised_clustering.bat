@echo off
setlocal enabledelayedexpansion

REM DATASETS=australian_credit german_credit heart pima wdbc contraceptive iris wine
set DATASETS=normal_2_class normal_3_class electricity
set BASE_CLASSIFIERS=svm dt lr

for %%D in (%DATASETS%) do (
    for %%C in (%BASE_CLASSIFIERS%) do (
        python supervised_clustering.py -d %%D -b %%C
    )
)
