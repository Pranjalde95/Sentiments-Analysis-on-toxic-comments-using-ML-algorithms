# Sentiment Analysis on Toxic Comments Using Machine Learning Algorithms

**MSc Business Analytics dissertation project — University of Kent (2023)**

This repository preserves code and a Jupyter notebook from a dissertation investigating machine-learning methods for identifying toxic online comments. Despite *sentiment analysis* in the dissertation title, the main predictive task shown in the code is **binary toxicity classification** (`IsToxic`), not a conventional positive/negative sentiment classifier. The notebook also includes exploratory analysis of the `IsRacist` label.

## What the original project contains

The archived work uses the **YouTube toxic-comment dataset**, loaded from a local file named `youtoxic_english_1000.csv`, and explores TF-IDF text features with classical classifiers including logistic regression, support vector classification (SVC), k-nearest neighbours, decision trees, random forests, and (in a later comparison) Gaussian Naive Bayes. It uses cross-validation, accuracy, ROC curves and ROC-AUC to compare approaches.

| File | Purpose |
| --- | --- |
| `pranjal_Assignment.ipynb` | **Historical dissertation notebook.** Retained as originally uploaded, including its saved outputs. |
| `Code_File` | Historical code extract, retained unchanged; **not an executable standalone script as-is**. |
| `src/toxicity_baseline.py` | **New, independently written baseline pipeline** for reproducible future experiments. It does not recreate or verify the original thesis results. |
| `tests/test_toxicity_baseline.py` | Lightweight tests using **synthetic examples only**; no dissertation dataset or model retraining is required. |

## Historical notebook results (not independently reproduced)

The saved notebook outputs include:

| Reported experiment in the notebook | Reported value |
| --- | ---: |
| Tuned SVC five-fold cross-validation accuracy | 65.57% |
| SVC ROC-AUC from a later ROC-curve section | 0.7850 |

**Interpret these as archived outputs, not validated benchmark claims.** The sections use different modeling/evaluation steps; these two numbers should not be presented as results of a single independently verified run. The notebook includes methodological limitations, including fitting a TF-IDF vectorizer before the initial train/test split and reusing `y_train`/`y_test` variable names across later splits. These problems can affect comparability and evaluation validity. No reported result was recalculated or changed during this repository cleanup.

## Reproducible baseline (separate from the original dissertation)

The optional new script builds a TF-IDF-plus-classifier `Pipeline` for each model, **splits comments into training and held-out testing sets before fitting TF-IDF**, and uses stratified cross-validation on the training set. It reports cross-validation accuracy (mean/standard deviation), held-out accuracy, F1, and ROC-AUC. Hyperparameters are simple fixed defaults rather than the full historical parameter searches.

Requirements: Python 3.11 recommended; a CSV with `Text` and binary `IsToxic` columns (0 = not toxic, 1 = toxic).

```bash
python -m pip install -r requirements.txt
python src/toxicity_baseline.py --data /path/to/youtoxic_english_1000.csv --output outputs/baseline_metrics.json
```

**No dataset is included** in this repository. Obtain the original dataset through an appropriate source and check its terms before use. The repository ignores local CSV files and generated `outputs/` by default. The baseline cannot produce actual study-specific scores until a suitable dataset is supplied and the command is run; it has **not** been run on the dissertation dataset as part of this cleanup.

For a quick code sanity check using **synthetic text only**:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

## Limitations and responsible use

- The original notebook is retained as an archival research artifact, **not** an assured end-to-end reproducible or leakage-free evaluation. Its separate `Code_File` contains a transcription typo, and the original data are unavailable in this repo.
- Toxicity labels reflect a particular dataset and labeling context. Classification mistakes, demographic biases, quotation/context errors, and language variation matter when considering real moderation systems.
- The dataset, label definitions, representativeness, annotations, and possible duplicate-text overlap require separate inspection before any new quantitative result is asserted.
- The new baseline is **not** a fairness audit, a deployed moderation product, or a reimplementation of the original thesis experiments. Numerical outputs should be cited only after real data provenance and a fresh evaluation are documented.
- The notebook includes examples of offensive language in historical testing cells. Review it before presenting it to general audiences.

## Research context

This coursework/dissertation project represents an early investigation of applied NLP classification and model evaluation. Its enduring relevance to responsible AI lies in the **question** of how to assess systems that flag harmful content—not in claiming that the archived notebook establishes production-grade performance or fairness.
