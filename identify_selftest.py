"""How reliable is material identification?  Run after exporting a new buildup_bundle.joblib:

    python identify_selftest.py            # writes identify_selftest.json, shown on the app's Identify tab

Each trial takes a clean tabulated coefficient set (a real material, response and energy), reads B at a few random
depths from its G-P curve, adds random scatter of the stated size, hides the answer, and asks identify() to find it.
Truth always comes from the tables, so this measures how well tabulated materials are recognised; for an unlisted
element the effective-Z range is the meaningful output.
"""
import json
import sys

import numpy as np
import pandas as pd

from buildup_core import BuildupModel, gp_buildup


def run(bundle="buildup_bundle.joblib", trials=300, n_points=4, scatter=0.05, seed=7):
    m = BuildupModel.load(bundle)
    S = m.table_points
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(trials):
        s = S.iloc[rng.integers(len(S))]
        xs = np.sort(rng.choice([1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40], n_points, replace=False)).astype(float)
        B = gp_buildup(xs, s.b, s.c, s.a, s.d, s.Xk_mfp) * np.exp(rng.normal(0, np.log(1 + scatter), n_points))
        B = np.maximum(B, 1.0)
        r = m.identify(pd.DataFrame({"energy_MeV": s.energy_MeV, "depth_mfp": xs, "B": B}), meas_unc=scatter)
        R = r["ranking"]
        top = R[["material", "response"]].head(5).apply(tuple, axis=1).tolist()
        z_true = float(s.atomic_number)
        rows.append({"top1": top[0] == (s.material, s.response),
                     "top5": (s.material, s.response) in top,
                     "material_top1": R.material.iloc[0] == s.material,
                     "z_in_range": r["z_range"][0] <= z_true <= r["z_range"][1],
                     "z_err": abs(r["z_mean"] - z_true),
                     "z_err_rel": abs(r["z_mean"] - z_true) / z_true,
                     "resp_right": r["response_verdict"] == s.response,
                     "resp_wrong": r["response_verdict"] not in ("undecided", s.response),
                     "resp_undecided": r["response_verdict"] == "undecided"})
    D = pd.DataFrame(rows)
    out = {"bundle_created": m.meta.get("created"), "trials": trials, "points_per_trial": n_points,
           "scatter": scatter, "exact_match_top1": D.top1.mean(), "exact_match_top5": D.top5.mean(),
           "material_top1": D.material_top1.mean(), "true_Z_inside_90pct_range": D.z_in_range.mean(),
           "median_abs_Z_error": D.z_err.median(), "median_rel_Z_error": D.z_err_rel.median(),
           "response_correct": D.resp_right.mean(), "response_wrong": D.resp_wrong.mean(),
           "response_undecided": D.resp_undecided.mean()}
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


if __name__ == "__main__":
    res = run(*(sys.argv[1:2] or ["buildup_bundle.joblib"]))
    with open("identify_selftest.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
