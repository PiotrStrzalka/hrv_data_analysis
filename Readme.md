# Python cheatsheet:

**Create virtual env:**
```bash 
python -m venv .venv
```

**Activare virtual env:**
```
Windows:        .venv\Scripts\activate.bat
(to create an alias: doskey va=.venv\Scripts\activate.bat)

Linux MacOS:    source .venv/Scripts/activate
```

**Deactivate virtual env:**
```
deactivate
```

**Install packages:**
```python
python -m pip install -r requirements.txt
or
pip install -r requirements.txt
```