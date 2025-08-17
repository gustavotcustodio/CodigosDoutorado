@echo off
setlocal enabledelayedexpansion

set DATASETS=blood
REM australian_credit contraceptive german_credit heart iris pima wdbc wine
set BASE_CLASSIFIERS=nb svm knn5 knn7 lr dt rf gb xb

for %%D in (%DATASETS%) do (
    for %%M in (100 75 50) do (
        for %%C in (%BASE_CLASSIFIERS%) do (
            echo Running %%D %%M %%C...
            python base_classifier.py -d %%D -c %%C -m %%M
        )
    )
)

endlocal
