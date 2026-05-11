# IC50 Tools

Python pipeline for IC50 determination from 96-well microplate growth assays.  
Fits 4-parameter logistic (4PL) dose-response curves and produces publication-ready figures and Excel reports.

**Author:** Felix Knote

---

## Scripts

| Script | Purpose |
|---|---|
| `01_IC_50.py` | Single-plate IC50 analysis — fits two antibiotics from one 96-well plate, generates per-antibiotic dose-response plots |
| `02_IC_50_Batch.py` | Batch pipeline — processes multiple plates from a directory using a plate map CSV, includes pre-normalization QC figures, DMSO growth curve, IC50 summary grid, and Excel export |
| `03_IC50_Replicate_Comparison.py` | Biological replicate reproducibility — compares IC50 values across multiple experimental days, computes CV%, ICC, variance components (technical vs biological), and flags outlier days |
| `03_IC50_Strain_Comparison.py` | Strain comparison — compares IC50 values between two bacterial strains (e.g. MG1655 vs ACE-1), generates grouped bar charts and log2 fold-change plots with propagated technical error |

---

## Plate Layout (96-well, 8×12)

```
Col:  1    2   3   4   5   6   7   8   9  10  11   12
A:   [B]  [G] [G] [G] [G] [G] [G] [G] [G] [G] [G] [B]
B:        [1] [1] [1] [1] [1] [1] [1] [1] [1] [1]
C:        [1] [1] [1] [1] [1] [1] [1] [1] [1] [1]
D:        [1] [1] [1] [1] [1] [1] [1] [1] [1] [1]
E:        [2] [2] [2] [2] [2] [2] [2] [2] [2] [2]
F:        [2] [2] [2] [2] [2] [2] [2] [2] [2] [2]
G:        [2] [2] [2] [2] [2] [2] [2] [2] [2] [2]
H:   [B]  [D] [D] [D] [D] [D] [D] [D] [D] [D] [D] [B]
```

- **[B]** Blank wells (A1, A12, H1, H12)  
- **[G]** Growth control — cells only, 100% growth reference (A2–A11)  
- **[D]** DMSO carrier control — 2-fold dilution series, max 5% DMSO (H2–H11)  
- **[1]** Antibiotic 1 — 3 technical row replicates (rows B, C, D), 10 concentrations (2-fold dilution series, columns 2–11)  
- **[2]** Antibiotic 2 — 3 technical row replicates (rows E, F, G)

---

## Input Format

**Plate reader files** — vertical semicolon-delimited format (one well per line):
```
A01;  0.118
A02;  0.543
...
```

**Plate map CSV** — pairs antibiotics to plates:
```
Index, Antibiotics, Highest concentration [ug/mL]
1, AmpicillinR, 128
2, Kanamycin, 64
3, Ciprofloxacin, 4
4, Tetracycline, 32
...
```
Rows are paired: rows 1–2 → Plate 1, rows 3–4 → Plate 2, etc.

---

## Dependencies

```
numpy
pandas
scipy
matplotlib
openpyxl
```

Install with:
```bash
pip install numpy pandas scipy matplotlib openpyxl
```

---

## Usage

Set the `INPUT_DIR` / `PLATE_MAP` / `BASE_DIR` paths at the top of each script, then run:

```bash
python 01_IC_50.py
python 02_IC_50_Batch.py
python 03_IC50_Replicate_Comparison.py
python 03_IC50_Strain_Comparison.py
```

Outputs are saved to an `IC50_output/` or `Strain Comparison/` subfolder next to the input data.
