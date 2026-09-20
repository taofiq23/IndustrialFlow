# Paper

**How Often Does a Drift Monitor Cry Wolf? Multiple Testing, Reference Sampling, and Autocorrelation in Kolmogorov-Smirnov Monitoring of Sensor Streams**

A 13-page paper (single-column, 11pt) reporting a simulation study of the drift monitor in this repository. [Read the compiled PDF](paper.pdf).

- Every number, figure and table is generated from the result files in [`../study_results/`](../study_results) by [`src/industrialflow/study_figures.py`](../src/industrialflow/study_figures.py); the experiments are in [`src/industrialflow/drift_study.py`](../src/industrialflow/drift_study.py) and are seeded.
- All 17 references were checked against primary sources (title, authors, venue, year). Spot-check them before reusing.
- It is a simulation study with Gaussian marginals and step-change drift, not an evaluation on real plant data, and it has not been peer reviewed.

## Recompiling

Needs a LaTeX distribution (MiKTeX, TeX Live) or [Overleaf](https://overleaf.com) (upload this folder).

```bash
pdflatex paper.tex
pdflatex paper.tex   # twice: the first pass resolves citations and cross-references
```

## Regenerating figures and tables

```bash
pip install -r ../requirements-study.txt
python -m src.industrialflow.study_figures   # run from the repository root
```
