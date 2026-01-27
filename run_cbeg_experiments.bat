@echo off
setlocal enabledelayedexpansion

REM Define variable lists
set DATASETS=rectangles elipses pima wdbc heart normal_2_class normal_3_class contraceptive
set EVALUATION_METRIC=dbc dbc_rand rand
set COMBINATION_METHODS=meta_classifier weighted_membership majority_voting
set POSSIBLE_N_CLUSTERS=2 3
REM set CLASSIFIERS_SELECTION=crossval default pso
set CLASSIFIERS_SELECTION=pso

REM Loop through datasets
for %%D in (%DATASETS%) do (
    for %%M in (100 75 50) do (
        for %%S in (%CLASSIFIERS_SELECTION%) do (
            for %%C in (%COMBINATION_METHODS%) do (

                REM First loop: using evaluation metrics with -n compare
                for %%E in (%EVALUATION_METRIC%) do (
                    echo ======================================================
                    echo Running %%D %%M %%E %%C %%S...
                    echo ======================================================
                    python cbeg.py -d %%D -n compare -m %%M -e %%E -c %%C -p %%S
                )

                REM Second loop: trying different number of clusters
                REM for %%N in (%POSSIBLE_N_CLUSTERS%) do (
                REM     echo ======================================================
                REM     echo Running %%D %%M %%N %%C %%S
                REM     echo ======================================================
                REM     python cbeg.py -d %%D -n %%N -m %%M -c %%C -p %%S
                REM )
            )
        )
    )
)
