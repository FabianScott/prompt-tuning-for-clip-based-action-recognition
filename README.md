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
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers
├── RESULTS.md         <- Comprehensive results document (auto-generated)
├── generate_results_readme.py  <- Script to regenerate RESULTS.md
├── example_usage.py   <- Examples of using results data programmatically
├── requirements.txt   <- Requirements file for reproducing the environment
│
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- Final, canonical data sets for modeling
│   │   ├── prompts    <- Generated prompts
│   │   └── results    <- Model predictions and evaluations
│   └── raw            <- Original, immutable data dump
│
├── docs               <- Documentation (mkdocs format)
│   ├── mkdocs.yml     <- MkDocs configuration
│   ├── RESULTS_DATA.md <- API reference for results data
│   └── docs/          <- Documentation pages
│
├── figures            <- Generated visualizations
│   ├── calibration    <- Model calibration plots
│   ├── explainer      <- Attention rollout visualizations
│   └── gflops         <- Computational cost analysis
│
├── hpc_submit         <- HPC job submission scripts
│   ├── data           <- Data preparation jobs
│   ├── evaluation     <- Model evaluation jobs
│   ├── explainability <- Explainability analysis jobs
│   ├── train_models   <- Model training jobs
│   └── utilities      <- Utility scripts
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks and analysis scripts
│   ├── data           <- Data exploration notebooks
│   ├── evaluation     <- Model evaluation notebooks
│   ├── explainability <- Explainability analysis notebooks
│   ├── tables         <- Table generation scripts for results
│   └── train_models   <- Training notebooks
│
├── references         <- Data dictionaries, manuals, and explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX
│   └── figures        <- Generated graphics for reporting
│
├── src                <- Source code for this project
│   ├── __init__.py    <- Makes src a Python module
│   ├── plots.py       <- Visualization utilities
│   ├── configs        <- Configuration files
│   ├── data           <- Data loading and processing scripts
│   ├── eval           <- Evaluation utilities
│   ├── modeling       <- Model architectures and training code
│   └── tables         <- Table generation utilities
│       └── results_data.py  <- Central data store for all experimental results
│
├── tests              <- Unit tests
│
└── tokens             <- API tokens and credentials
```

--------

