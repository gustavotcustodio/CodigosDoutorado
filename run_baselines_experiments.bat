@echo off
setlocal enabledelayedexpansion

set DATASETS=normal_2_class normal_3_class electricity
REM australian_credit contraceptive german_credit heart iris pima wdbc wine rectangles elipses
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
