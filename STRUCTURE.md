**STRUCTURE**

project-name/
│
├── .venv/                       # virtual environment (gitignored)
├── .gitignore
├── README.md
├── requirements.txt
│
├── analysis.ipynb               # THE orchestrator (the only notebook)
│
├── data/
│   ├── raw/                     # original, untouched data
│   ├── interim/                 # partially cleaned (optional)
│   └── processed/               # final, analysis-ready data
│
├── src/                         # all the actual logic lives here
│   ├── __init__.py
│   ├── config.py                # paths, constants, settings
│   ├── data_loader.py           # API calls, file loading, scraping
│   ├── cleaning.py              # cleaning and transformation functions
│   ├── analysis.py              # EDA, stats, modeling functions
│   └── plots.py                 # reusable visualization functions
│
└── reports/
    └── figures/                 # exported charts