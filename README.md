# Photon buildup factors — web app

A Streamlit front end for Part II of `ANS643_physics_features_and_regression_PREDICTIONS_revised.ipynb`.
It gives the buildup factor B(E, x) for point-isotropic gamma sources at 0.015–15 MeV and 1–40 mfp, for any
element from Be (Z = 4) to U (Z = 92) plus water, air and concrete, as EBF or EABF.

Every answer follows the notebook's `predict_buildup()` exactly:

| Request | How the value is produced |
|---|---|
| Tabulated material at a tabulated energy | the ANSI/ANS‑6.4.3 G‑P formula of that coefficient set (exact) |
| Tabulated material, any other energy | the method that won the notebook's held‑out test for energy gaps |
| Element not in the tables | the method that won the held‑out test for unlisted elements |

Each result carries its route, a depth‑dependent typical error band, a K‑edge low‑confidence flag and the size of any
physical guard‑rail correction.

## What's in the folder

| File | Purpose |
|---|---|
| `app.py` | the Streamlit app (single point, batch, compare materials, identify material, model and data) |
| `buildup_core.py` | prediction logic, no Streamlit dependency — import it in your own scripts too |
| `buildup_bundle.joblib` | the trained predictor exported by section 25 of the notebook |
| `sample_batch.csv` | a batch file to try, including two rows that are deliberately out of range |
| `identify_selftest.py` | measures how reliably materials are identified; writes `identify_selftest.json` |
| `.streamlit/config.toml` | theme and server settings |
| `.streamlit/secrets.toml` | the app password (kept out of Git by `.gitignore`) |
| `static/fonts/` | the Archivo typeface (SIL Open Font License), served locally |

## 1 · Build the model bundle from your data

The bundle shipped here is a **demo** built from a stand‑in dataset; the app shows a banner while it is in use.
Replace it with one trained on your own source workbook:

1. Put `ANS643_ML_dataset_extraction_final.xlsx` next to the notebook.
2. Run the whole notebook (about 10–15 minutes). Section 25 writes `buildup_bundle.joblib` (≈ 17 MB).
3. Copy that file into this folder, replacing the demo one.

The full 400‑tree Extra Trees model is too large to deploy, so section 25 stores its output as a dense lookup table
(every Z, both responses, a fine log grid in energy and depth, with extra points either side of each K‑edge and of
1.022 MeV). Section 25 prints how closely the lookup reproduces the model; the app shows the same figures on its
"Model and data" tab.

## Password

The app asks for a password before showing anything. It is read from `.streamlit/secrets.toml`:

```toml
app_password = "your password"
```

A random password is already set in that file; change it to your own. The file is listed in `.gitignore`, so it is not
pushed to GitHub. On Streamlit Community Cloud, open the app's **Settings → Secrets** and paste the same line there.
On other hosts you can set the `APP_PASSWORD` environment variable instead. If no password is configured anywhere,
the app stays locked and says how to set one. The "Lock the app" button at the bottom asks for the password again.

Everyone shares this one password. For individual accounts, use Streamlit's built-in sign-in (`st.login`) or a
private app on Community Cloud.

## 2 · Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3 · Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (the bundle is well under GitHub's 100 MB file limit).
2. At share.streamlit.io choose **New app**, pick the repository and set the main file to `app.py`.
3. Deploy. `requirements.txt` and `.streamlit/config.toml` are picked up automatically.

Any host that runs Python works the same way (Render, Railway, a VM, Docker):
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
To keep the bundle elsewhere, set the `BUILDUP_BUNDLE` environment variable to its path.

## Troubleshooting

`AttributeError: 'BuildupModel' object has no attribute ...` means `app.py` and `buildup_core.py` come from different
versions, or Streamlit is still holding the old code. Copy the whole folder from one download, then stop Streamlit
(Ctrl+C) and start it again.

## Batch files

CSV or Excel, one point per row:

| column | required | example |
|---|---|---|
| `material` | yes | `Lead`, `Pb`, `82`, `Water` |
| `energy_MeV` | yes | `0.662` |
| `depth_mfp` | yes | `10` |
| `response` | no (EBF if blank) | `EBF` or `EABF` |

Rows that cannot be predicted (unknown material, out‑of‑range energy, depth or Z) are kept in the output with the
reason in an `error` column, so one bad row never stops the batch. Results download as CSV or Excel.

## Identify material

The reverse question: given buildup factors you know (energy, depth in mfp, B), which material and response do they
belong to? Every element from Be to U plus water, air and concrete is tried in EBF and EABF. Each candidate's
predicted curve is compared with your points, weighting each residual by your stated uncertainty plus the model's
typical error at that depth, and candidates are ranked by Gaussian likelihood. The tab reports:

* an **effective atomic number** with a 90 % range — the most dependable output, because neighbouring elements often
  have near-identical curves;
* the ranked candidates with their share of the match weight;
* whether the values behave like EBF or EABF, or that the points cannot tell them apart;
* a warning when nothing fits (usually a unit problem — depth in cm instead of mfp — or a mixture).

Several samples can be identified at once from a file with a `sample` column.

After exporting a new model file, run `python identify_selftest.py` once. It hides real tabulated curves behind a few
noisy points and measures how often the answer is recovered; the results appear on the tab. With the demo model,
4 points and 5 % scatter: the true Z fell inside the 90 % range 99 % of the time (typical error about 1 in Z), the exact
material and response ranked first 60 % of the time and in the top five 87 %, and the response was called right 44 %,
wrong 2 % and left undecided 54 % of the time.

## Using the core without the app

```python
from buildup_core import BuildupModel
m = BuildupModel.load("buildup_bundle.joblib")
m.predict("Nickel", [0.35, 0.9, 7], [6.3, 17.5, 33], "EBF")      # long table with route, band and flags
m.predict_rows(pandas_dataframe)                                 # batch, one output row per input row
m.identify(points_df)          # points_df: energy_MeV, depth_mfp, B -> ranking, effective Z range, response verdict
```
