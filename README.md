# Late Delivery Risk at DataCo: what drives it, and what it is worth

MSDS692 Practicum, Regis University. Data: [DataCo Smart Supply Chain (Kaggle)](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis), January 2015 to January 2018.

**Question:** which order and shipping factors go with late delivery, is the pattern stable, and how many dollars sit on late orders?
**Stakeholder:** a supply chain / logistics operations manager at DataCo, the person who chooses how an order ships and whether to step in.
**Action:** use the findings to decide which shipping tiers to fix first.

## Bottom line

Fix First and Second Class's delivery promises first, then reduce Standard Class's late orders. Standard Class is the largest single tier by dollars; First and Second Class together hold more late revenue ($10.9M against $8.4M).

- **First and Second Class take about twice as long as promised.** First Class promises 1 day and takes exactly 2 on every one of its 27,812 order items. Second Class promises 2 and takes about 4.
- **54.8% of orders are late** (36,043 of 65,741 orders; 98,972 of 180,508 order items). By class: First 95.3%, Second 76.6%, Same Day 45.7%, Standard 38.1% (by item).
- **A one-line rule ("First or Second Class = late") matches both models on accuracy:** 69.5% against 69.4% (logistic regression) and 69.2% (random forest) on the held-out test set. Differences of a few tenths of a point are within split-to-split noise (notebook Section 16). The class carries the signal.
- **The recommendation keeps the 1-day and 2-day promises** and looks for where the extra day is lost.

## Found in the data and assumed

| Found in the data | Assumed (not confirmed by the data) |
|---|---|
| Promised days by class (0 / 1 / 2 / 4) and actual days | Day counts were recorded correctly |
| Late rates, counts, and sales on late items | "Late" = actual days > scheduled days. The dataset documents no rule; I deduced it. It matches `Delivery Status` for every row except canceled and suspected-fraud rows (7,754), which are always marked not late |
| Rule vs models; same result when the split is grouped by order | Handling cost of $15 / $20 / $40 per late order comes from outside industry sources (HubBox, Locus.sh, OneRail), not DataCo data. $20 is the middle case I chose |
| Shipping Mode is the only strong factor (Cramér's V 0.46; Payment Type 0.08; all others 0.02 or lower) | 80/20 split, seed 42 and model settings were my choices (see Section 16 for what changed when I varied them) |
| Late items earn $0.78 less profit on average, but the gap is not statistically significant (p = 0.11) | The data reflects real DataCo operations. The customer fields look synthetic: 2 countries, and one city (Caguas) holds 37% of rows |

$20.1M is sales on late items. It is a ceiling on exposure, **not** a confirmed loss. Handling cost is per unique late order: 36,043 x $15 / $20 / $40 = $0.54M / $0.72M / $1.44M.

## Where the data goes

The CSV is not in this repository (96 MB). Download `DataCoSupplyChainDataset.csv` from the Kaggle link above and place it in `data/` (see `data/README.md`). It is Latin-1 encoded; expected shape 180,519 rows x 53 columns.

## How to run

1. Install the packages (versions I ran, Python 3.14.7):
```bash
   pip install -r requirements.txt
```
2. Put the CSV in `data/` (see "Where the data goes" above).
3. Open `notebooks/Late_Delivery_Analysis.ipynb` in Jupyter or VS Code and run all cells.
4. Rebuild the slide charts into `charts/`, from the repository root:
```bash
   python scripts/charts.py      # takes a few minutes
```

Alternative to step 3, from the repository root, for a machine without Jupyter:
```bash
python scripts/exec_notebook.py   # runs every cell and writes the outputs back into the notebook file
```
Note that this overwrites the outputs stored in the notebook.

Stored outputs come from running all cells in VS Code (Python 3.14.7). Random forest figures can differ in the fourth decimal place across library versions.


## Slide-to-notebook map (deck: `deck/Late_Delivery_Final_Presentation.pdf`)

| Slide | Claim | Notebook section |
|---|---|---|
| 1 | Recommendation | Summary of Sections 5, 12, 14 |
| 2 | What is assumed vs worked out | 3 (canceled/fraud), 13 cell "13a-0" (late rule), 14 |
| 3 | First and Second Class take about twice as long as promised | 5 |
| 4 | Late rate by class, 38% to 95% | 4, 14 (14a-2 by order, 14c without canceled/fraud) |
| 5 | A one-line rule matches the models on accuracy | 10b, 13, 15, 16 |
| 6 | Standard Class: lowest rate, most dollars as a single tier | 12, 14a-2 |
| 7 | $20.1M ceiling; handling cost $0.5M to $1.4M | 12, 14a, 14a-2 |
| 8 | Recommendation by tier | 12, 14a-2 |
| 9 | What the data cannot tell us | 3, 6 (circularity note), 11, 14b, 14c |
| 10 | Next step | None (a proposal, not an analysis) |
| 11 | Shipping Mode is the only strong factor | 4b, 6 |
| 12 | How "late" is defined | 3, 13 ("13a-0"), 14e |
| 13 | Numbers I chose and where they come from | 3, 12, 14a; loss scenarios (illustrative, not findings) in 14d |
| 14 | Model settings held up | 9, 13c, 15, 16 |

## Limits

- **Correlation, not cause.** A real test would apply a fix to one First Class group and compare late rates against a control.
- **Part of the Shipping Mode result is true by definition.** "Late" is built from actual vs scheduled days, and scheduled days are fixed by mode.
- **The data cannot show lost revenue.** Canceled and suspected-fraud rows are never flagged late, so it cannot show whether lateness causes cancellation.
- **It cannot explain why First Class takes 2 days**, only that it does.
- **The "late" rule is deduced**, and the data may be rule-generated.
- **No 14-day forecast.** Actual shipping time never exceeds 6 days and the longest promise is 4. The Week 1 plan (forecast, per-order loss model, SHAP, fraud flag) was not built.
- **No model is saved or deployed.** The models are checks, not products.
- Each row is an order item (2.75 per order). Late rates are essentially the same by order and by item (Same Day is 46.1% by order, 45.7% by item). 2018 is a partial year.
- Settings were searched with cross-validation on the training part only. The held-out test part was used to report results.

## Layout

```
notebooks/Late_Delivery_Analysis.ipynb   the analysis, Sections 1-16, with outputs
scripts/charts.py                        rebuilds all slide charts from the CSV into charts/
scripts/exec_notebook.py                 equivalent to nbconvert --execute, for machines without Jupyter
scratch/                                 old helper scripts, kept for the record, not needed to reproduce anything
charts/                                  chart images used on the slides
deck/                                    the final deck (PDF)
data/README.md                           where to download the CSV
requirements.txt                         package versions I ran
```
