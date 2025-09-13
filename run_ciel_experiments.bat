@echo off
setlocal enabledelayedexpansion
REM Preciso rodar o contraceptive do zero e descobrir qual é o problema dele Começar do fold 5

REM set DATASETS=rectangles elipses contraceptive german_credit heart pima wdbc australian_credit wine iris
set DATASETS=contraceptive
set N_PARTICLES=30
set N_ITERS=10

for %%D in (%DATASETS%) do (
    echo Running for %%D...
    python ciel.py -d %%D -n %N_ITERS% -p %N_PARTICLES%
)

endlocal
