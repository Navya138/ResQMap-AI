@echo off
cd /d %~dp0
echo ===== ResQMap AI : one-click run =====
python -m pip install -r requirements.txt
if errorlevel 1 goto err

if exist xbd\ (
  set ROOT=xbd
  echo Using real xBD dataset from the xbd folder
) else (
  set ROOT=synthetic_xbd
  echo No xbd folder found - generating SYNTHETIC demo data
  python make_synthetic.py --out synthetic_xbd
  if errorlevel 1 goto err
)

python prepare_data.py --root %ROOT% --max_per_class 1500
if errorlevel 1 goto err
python train_damage.py --epochs 4 --workers 0
if errorlevel 1 goto err
python infer.py --n 4 --use_gt
if errorlevel 1 goto err

echo ===== Done. Opening the app (Ctrl+C to stop) =====
streamlit run app.py
goto :eof

:err
echo.
echo Something failed. Copy the error above and send it to Claude.
pause
