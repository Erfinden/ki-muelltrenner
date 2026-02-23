# Ki-Muelltrenner: Fast Labeling (Windows)

## 1) Install Git on Windows
1. Open your web browser and go to https://git-scm.com/downloads/win
2. Download Git for Windows.
3. Open the installer you downloaded.
4. Click Next until it finishes (the default options are fine).
5. When it is done, open the Start Menu and search for "Git Bash".

## 2) Get the project via Git
1. Open Git Bash.
2. Choose a folder where you want the project, for example your Desktop:
   - Type: `cd ~/Desktop`
3. Download (clone) the project:
   - Type: `git clone https://github.com/Erfinden/ki-muelltrenner.git`
4. Go into the project folder:
   - Type: `cd ki-muelltrenner/fast-labeling`

## 3) Install Python on Windows
1. Open your web browser and go to https://www.python.org/downloads/windows/
2. Download the latest Python 3 installer.
3. Open the installer.
4. IMPORTANT: Check the box "Add Python to PATH".
5. Click "Install Now" and wait until it finishes.
6. Close the installer.

## 4) Create a virtual environment (venv) and install packages
1. Open Git Bash (or re-open it if it was closed).
2. Go to the project folder (if needed):
   - Type: `cd ~/Desktop/ki-muelltrenner/fast-labeling`
3. Create the venv:
   - Type: `python -m venv .venv`
4. Turn on the venv:
   - Type: `source .venv/Scripts/activate`
   - You should see `(.venv)` at the start of the line.
5. Install the packages:
   - Type: `pip install -r requirements.txt`

## 5) Start fast-labeling.py
1. Make sure the venv is on (you see `(.venv)`).
2. Run the program:
   - Type: `python fast-labeling.py`

## Troubleshooting
- If `python` does not work, try `py` instead.
- If you see an error about missing packages, run:
  - `pip install -r requirements.txt`
