# Draft GitHub issue for petrobras/3W (not posted — review first)

**Title:** Two pitfalls in 3W 2.0.0: instance file names repeat across class folders, and hand-drawn
pressures use a different unit than real instances

**Body:**

Hello 3W maintainers, and thank you for the dataset.

While building a forecasting benchmark on 3W 2.0.0 (commit `826fec4`), we hit two issues that silently
changed our results. They may not be bugs in the dataset itself, but they are easy to miss, so we
suggest documenting them (and possibly adding a check to the toolkit loader).

### 1. Instance file names are not unique across class folders

225 file names appear in more than one class folder, covering 885 of the 2,228 instances (865 simulated,
20 hand-drawn). For example, `DRAWN_00001.parquet` exists in both `dataset/1/` and `dataset/7/`, and
several `SIMULATED_000xx.parquet` names exist in more than one folder.

Any code that identifies an instance by its file name alone (a dictionary or cache keyed by file name,
for example) silently mixes instances of different classes. In our case about half of the synthetic
training windows received another class's signal; after keying by `<class>/<file>` the results of every
sequence model changed substantially. Real-instance names (`WELL-xxxxx_<timestamp>`) are unique, so
real-only evaluations are not affected.

**Suggestion:** state in the README / dataset description that an instance is identified by its class
folder *and* file name, or make the synthetic names unique (e.g. include the class code).

### 2. Hand-drawn pressures appear to be in bar, real instances in Pa

The median of `P-MON-CKP` in a real class-7 instance is about 2.4 × 10⁶ (Pa), while in
`dataset/7/DRAWN_00001.parquet` it is about 14.7, i.e. about 10⁵ times smaller, consistent with bar. Used
together without conversion, models learn "low pressure = event": in our case an LSTM ranked the real
scaling well 24 almost perfectly inverted (AUROC 0.10) until we multiplied hand-drawn pressures by 10⁵.

**Suggestion:** document the unit of each source (real / simulated / hand-drawn), or convert the
hand-drawn pressures to Pa.

We searched the open and closed issues and did not find these reported; apologies if we missed an
existing discussion. Happy to open a PR for a loader check if that is useful.

— Muhammad Haseeb, FAST-NUCES Islamabad (MS AI)

---

**Before posting, check:** (1) the 3W README / Scientific Data paper does not already document units per
source; (2) the numbers above against `src/data/preprocess.py` and `trainer.ikey`. Posting is public and
under your name, so decide whether to include your affiliation.
