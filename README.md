# Prompt Tuning for CLIP-Based Action Recognition

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Research on prompt tuning techniques for video action recognition using CLIP-based vision-language models.

## 📊 Results

**[View Complete Results →](RESULTS.md)**

Comprehensive results document with all experimental tables, figures, and key findings including:
- UCF101, Kinetics400, and HMDB51 performance metrics
- Robustness analysis and augmentation effects
- Computational cost analysis
- Model calibration and explainability visualizations

All experimental data is available as Python dictionaries in [src/tables/results_data.py](src/tables/results_data.py) for easy programmatic access and custom analyses. See [docs/RESULTS_DATA.md](docs/RESULTS_DATA.md) for the complete API reference and [example_usage.py](example_usage.py) for usage examples.

To regenerate the results document:
```bash
python generate_results_readme.py
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         vlms_initial_testing and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── vlms_initial_testing   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes vlms_initial_testing a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

