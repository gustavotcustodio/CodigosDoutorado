@echo off
setlocal enabledelayedexpansion

REM Define variable lists
set DATASETS=rectangles elipses australian_credit german_credit contraceptive wine pima wdbc iris heart
set EVALUATION_METRIC=dbc dbc_rand rand
set COMBINATION_METHODS=meta_classifier weighted_membership majority_voting
set POSSIBLE_N_CLUSTERS=2 3
set CLASSIFIERS_SELECTION=crossval default pso

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
                for %%N in (%POSSIBLE_N_CLUSTERS%) do (
                    echo ======================================================
                    echo Running %%D %%M %%N %%C %%S
                    echo ======================================================
                    python cbeg.py -d %%D -n %%N -m %%M -c %%C -p %%S
                )
            )
        )
    )
)
