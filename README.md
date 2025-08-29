# MaintAI

MaintAI is a predictive maintenance project that estimates the **Remaining Useful Life (RUL)** of engines using NASA's CMAPSS dataset.

## Project Structure
- `data/CMAPSS/` → dataset files (not in repo)
- `notebooks/` → exploratory analysis (EDA) 
- `src/` → reusable code (data loader, features, models)
- `main_train.py` → end-to-end pipeline script

## Setup
```bash
# create virtual environment (first time only)
python -m venv .venv

# activate it (PowerShell)
.venv\Scripts\Activate

# install dependencies
pip install -r requirements.txt
