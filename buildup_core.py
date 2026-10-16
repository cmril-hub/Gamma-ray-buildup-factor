"""Prediction core for the buildup-factor web app.

Mirrors ``predict_buildup()`` from Part II of the notebook (section 19):

* listed material + listed energy with a clean ANS-6.4.3 coefficient set  -> the G-P formula itself (exact)
* listed material, any other energy                                       -> route chosen for "energy gaps" in section 18
* unlisted element                                                        -> route chosen for "unlisted elements"

A route is one of: interpolation ("interp"), the Extra Trees model ("ml") or their average in log B ("blend").
The Extra Trees output is read from the lookup table exported in section 25 (bilinear in log E and log depth).
Everything the module needs comes from ``buildup_bundle.joblib``; it has no Streamlit dependency.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

ELEMENTS = [
    ("H", "Hydrogen"), ("He", "Helium"), ("Li", "Lithium"), ("Be", "Beryllium"), ("B", "Boron"), ("C", "Carbon"),
    ("N", "Nitrogen"), ("O", "Oxygen"), ("F", "Fluorine"), ("Ne", "Neon"), ("Na", "Sodium"), ("Mg", "Magnesium"),
    ("Al", "Aluminum"), ("Si", "Silicon"), ("P", "Phosphorus"), ("S", "Sulphur"), ("Cl", "Chlorine"), ("Ar", "Argon"),
    ("K", "Potassium"), ("Ca", "Calcium"), ("Sc", "Scandium"), ("Ti", "Titanium"), ("V", "Vanadium"),
    ("Cr", "Chromium"), ("Mn", "Manganese"), ("Fe", "Iron"), ("Co", "Cobalt"), ("Ni", "Nickel"), ("Cu", "Copper"),
    ("Zn", "Zinc"), ("Ga", "Gallium"), ("Ge", "Germanium"), ("As", "Arsenic"), ("Se", "Selenium"), ("Br", "Bromine"),
    ("Kr", "Krypton"), ("Rb", "Rubidium"), ("Sr", "Strontium"), ("Y", "Yttrium"), ("Zr", "Zirconium"),
    ("Nb", "Niobium"), ("Mo", "Molybdenum"), ("Tc", "Technetium"), ("Ru", "Ruthenium"), ("Rh", "Rhodium"),
    ("Pd", "Palladium"), ("Ag", "Silver"), ("Cd", "Cadmium"), ("In", "Indium"), ("Sn", "Tin"), ("Sb", "Antimony"),
    ("Te", "Tellurium"), ("I", "Iodine"), ("Xe", "Xenon"), ("Cs", "Caesium"), ("Ba", "Barium"), ("La", "Lanthanum"),
    ("Ce", "Cerium"), ("Pr", "Praseodymium"), ("Nd", "Neodymium"), ("Pm", "Promethium"), ("Sm", "Samarium"),
    ("Eu", "Europium"), ("Gd", "Gadolinium"), ("Tb", "Terbium"), ("Dy", "Dysprosium"), ("Ho", "Holmium"),
    ("Er", "Erbium"), ("Tm", "Thulium"), ("Yb", "Ytterbium"), ("Lu", "Lutetium"), ("Hf", "Hafnium"),
    ("Ta", "Tantalum"), ("W", "Tungsten"), ("Re", "Rhenium"), ("Os", "Osmium"), ("Ir", "Iridium"),
    ("Pt", "Platinum"), ("Au", "Gold"), ("Hg", "Mercury"), ("Tl", "Thallium"), ("Pb", "Lead"), ("Bi", "Bismuth"),
    ("Po", "Polonium"), ("At", "Astatine"), ("Rn", "Radon"), ("Fr", "Francium"), ("Ra", "Radium"),
    ("Ac", "Actinium"), ("Th", "Thorium"), ("Pa", "Protactinium"), ("U", "Uranium"),
]
SYMBOL2Z = {s: i + 1 for i, (s, _) in enumerate(ELEMENTS)}
NAME2Z = {n.lower(): i + 1 for i, (_, n) in enumerate(ELEMENTS)}
NAME2Z.update({"aluminium": 13, "sulfur": 16, "cesium": 55, "wolfram": 74})

ROUTE_LABEL = {
    ("table", None): "G-P table",
    ("interp", "energy"): "interpolated in energy", ("interp", "element"): "interpolated in Z",
    ("ml", "energy"): "Extra Trees", ("ml", "element"): "Extra Trees",
    ("blend", "energy"): "blend: Extra Trees + interpolation in energy",
    ("blend", "element"): "blend: Extra Trees + interpolation in Z",
}
METHODS = ("auto", "interp", "ml", "blend")
CORE_VERSION = "2026-10-16.2"   # bump when the code changes; the app uses it to refresh its cached model
_TANH_M2 = np.tanh(-2.0)


def gp_buildup(x, b, c, a, d, Xk):
    """ANSI/ANS-6.4.3 geometric-progression buildup factor at depth x (mfp)."""
    x = np.asarray(x, dtype=float)
    K = c * np.power(x, a) + d * (np.tanh(x / Xk - 2.0) - _TANH_M2) / (1.0 - _TANH_M2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        general = 1.0 + (b - 1.0) * (np.power(K, x) - 1.0) / (K - 1.0)
    return np.where(np.isclose(K, 1.0), 1.0 + (b - 1.0) * x, general)


def _interp_rows(q, xs, L):
    """Row of matrix L at coordinate q, linear in xs (sorted 1-D), clamped to the end rows."""
    if len(xs) == 1:
        return L[0]
    j = int(np.clip(np.searchsorted(xs, q), 1, len(xs) - 1))
    t = float(np.clip((q - xs[j - 1]) / (xs[j] - xs[j - 1]), 0.0, 1.0))
    return L[j - 1] * (1 - t) + L[j] * t


def _zkey(Z, comp) -> tuple:
    return (round(float(Z), 2), int(comp))


class BuildupModel:
    """Loads ``buildup_bundle.joblib`` and answers buildup-factor queries."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.meta = bundle["meta"]
        self.E_MIN, self.E_MAX = self.meta["E_range"]
        self.X_MIN, self.X_MAX = self.meta["X_range"]
        self.Z_MIN, self.Z_MAX = self.meta["Z_range"]
        self.KEDGE_LO, self.KEDGE_HI = self.meta["kedge_window"]
        self.routing = dict(self.meta["routing"])
        self.compound_z = {k.lower(): float(v) for k, v in bundle["compound_z"].items()}

        ke = bundle["k_edge_MeV"]
        self._kz = np.array(sorted(ke), float)
        self._ke = np.log(np.array([ke[int(z)] for z in self._kz]))

        lst = bundle["listed"]
        self.listed_name = {_zkey(r.Z, r.is_compound): r.material for r in lst.itertuples()}
        self.listed = lst.sort_values("Z").reset_index(drop=True)

        self.sets = {}
        S = bundle["sets"]
        for (z, c, resp), g in S.groupby(["atomic_number", "is_compound", "response"]):
            g = g.sort_values("energy_MeV")
            self.sets[(_zkey(z, c), resp)] = (g.energy_MeV.values.astype(float),
                                              g[["b", "c", "a", "d", "Xk_mfp"]].values.astype(float))
        self.elem_z = {resp: np.array(sorted(k[0][0] for k in self.sets if k[1] == resp and k[0][1] == 0))
                       for resp in ("EBF", "EABF")}
        self.table_points = S  # for plotting tabulated points

        G = bundle["ml_grid"]
        self._grid = G["log10B"]
        E = np.asarray(G["E"], float)            # (nE,) shared axis, or (nZ, nE) one axis per Z row
        self._lE = np.log(E if E.ndim == 2 else np.broadcast_to(E, (len(G["Z"]), len(E))))
        self._lX = np.log(np.asarray(G["X"], float))
        self._zrow = {_zkey(z, c): k for k, (z, c) in enumerate(G["Z"])}

        self._err = {g: (np.log(bundle[f"err_{g}"]["depth"]), np.asarray(bundle[f"err_{g}"]["mae"]))
                     for g in ("energy", "element")}
        self.missing = bundle["tables"]["missing"]

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str) -> "BuildupModel":
        return cls(joblib.load(path))

    # ------------------------------------------------------------------ physics helpers
    def k_edge(self, Z):
        """K-edge energy (MeV), log-log interpolated in Z."""
        return np.exp(np.interp(Z, self._kz, self._ke))

    def near_k_edge(self, Z, E):
        ek = self.k_edge(np.asarray(Z, float))
        E = np.asarray(E, float)
        return (ek >= self.E_MIN) & (E > self.KEDGE_LO * ek) & (E <= self.KEDGE_HI * ek)

    # ------------------------------------------------------------------ material lookup
    def resolve(self, material):
        """-> dict(name, Z, comp, listed).  Accepts a name, chemical symbol or atomic number."""
        if isinstance(material, (int, float, np.integer, np.floating)) and not isinstance(material, bool):
            z = float(material)
            for nm, zc in self.compound_z.items():
                if np.isclose(z, zc):
                    return self._entry(nm.capitalize(), zc, 1)
            if not float(z).is_integer():
                raise ValueError(f"Atomic number must be a whole number (got {material}).")
            return self._entry(None, z, 0)
        text = str(material).strip()
        if not text:
            raise ValueError("Material is empty.")
        try:
            number = float(text)
        except ValueError:
            number = None
        if number is not None:
            return self.resolve(number)
        key = text.lower()
        if key in self.compound_z:
            return self._entry(key.capitalize(), self.compound_z[key], 1)
        if key in NAME2Z:
            return self._entry(None, float(NAME2Z[key]), 0)
        sym = text[:1].upper() + text[1:].lower()
        if sym in SYMBOL2Z:
            return self._entry(None, float(SYMBOL2Z[sym]), 0)
        raise ValueError(f"Unknown material '{material}'. Use an element name, symbol or atomic number, "
                         f"or Water / Air / Concrete.")

    def _entry(self, name, Z, comp):
        key = _zkey(Z, comp)
        listed = key in self.listed_name
        if name is None:
            name = self.listed_name.get(key) or (ELEMENTS[int(Z) - 1][1] if 1 <= Z <= len(ELEMENTS) else f"Z={Z:g}")
        elif listed:
            name = self.listed_name[key]
        return {"name": name, "Z": float(Z), "comp": int(comp), "listed": listed}

    # ------------------------------------------------------------------ building blocks
    @lru_cache(maxsize=4096)
    def _set_matrix(self, key, resp, xs_t):
        hit = self.sets.get((key, resp))
        if hit is None:
            return None, None
        Es, C = hit
        xs = np.array(xs_t)
        L = np.log10(np.clip(np.array([gp_buildup(xs, *c) for c in C]), 1, None))
        return Es, L

    def _listed_curve(self, key, resp, E, xs_t):
        Es, L = self._set_matrix(key, resp, xs_t)
        if Es is None:
            return None, None
        hit = np.isclose(Es, E, rtol=1e-6)
        if hit.any():
            return L[int(np.argmax(hit))], "table"
        ek = float(self.k_edge(key[0]))
        same = (Es > ek) == (E > ek)
        Ese, Le = (Es[same], L[same]) if same.any() else (Es, L)
        return _interp_rows(np.log(E), np.log(Ese), Le), "energy"

    def _z_curve(self, Z, resp, E, xs_t):
        zs = self.elem_z[resp]
        lo, hi = zs[zs <= Z], zs[zs >= Z]
        pick = sorted(set(([lo[-1]] if len(lo) else []) + ([hi[0]] if len(hi) else [])))
        L = np.vstack([self._listed_curve(_zkey(z, 0), resp, E, xs_t)[0] for z in pick])
        return _interp_rows(np.log(Z), np.log(np.array(pick)), L)

    def _ml(self, key, resp, Es, xs):
        """Extra Trees log10 B from the lookup table, bilinear in (log E, log x)."""
        k = self._zrow[key]
        G = self._grid[k, 0 if resp == "EBF" else 1]
        lE, lX = self._lE[k], self._lX
        qE = np.log(np.asarray(Es, float))[:, None]
        qX = np.log(np.asarray(xs, float))[None, :]
        i = np.clip(np.searchsorted(lE, qE), 1, len(lE) - 1)
        j = np.clip(np.searchsorted(lX, qX), 1, len(lX) - 1)
        u = np.clip((qE - lE[i - 1]) / (lE[i] - lE[i - 1]), 0, 1)
        v = np.clip((qX - lX[j - 1]) / (lX[j] - lX[j - 1]), 0, 1)
        return ((1 - u) * (1 - v) * G[i - 1, j - 1] + u * (1 - v) * G[i, j - 1]
                + (1 - u) * v * G[i - 1, j] + u * v * G[i, j]).astype(float)

    def expected_factor(self, gap, depths):
        """Typical error factor (10^MAE) for a route and depth, from the held-out tests of section 18/20."""
        lx, mae = self._err[gap]
        return 10 ** np.interp(np.log(np.asarray(depths, float)), lx, mae)

    # ------------------------------------------------------------------ main entry point
    def predict(self, material, energy_MeV, depth_mfp, response="EBF", method="auto") -> pd.DataFrame:
        """Long table (energy x depth) of predictions with route, error band and flags."""
        m = self.resolve(material)
        Z, comp = m["Z"], m["comp"]
        resp = str(response).strip().upper()
        if resp not in ("EBF", "EABF"):
            raise ValueError(f"Response must be EBF or EABF (got '{response}').")
        if method not in METHODS:
            raise ValueError(f"Method must be one of {METHODS}.")
        if not (self.Z_MIN <= Z <= self.Z_MAX):
            raise ValueError(f"{m['name']} (Z={Z:g}) is outside the supported range Z={self.Z_MIN}-{self.Z_MAX}.")
        Es = np.atleast_1d(np.asarray(energy_MeV, float))
        xs = np.atleast_1d(np.asarray(depth_mfp, float))
        if not np.all(np.isfinite(Es)) or Es.min() < self.E_MIN - 1e-12 or Es.max() > self.E_MAX + 1e-12:
            raise ValueError(f"Energy must be between {self.E_MIN} and {self.E_MAX} MeV.")
        if not np.all(np.isfinite(xs)) or xs.min() < self.X_MIN - 1e-12 or xs.max() > self.X_MAX + 1e-12:
            raise ValueError(f"Depth must be between {self.X_MIN:g} and {self.X_MAX:g} mfp.")

        lg, corr, src, gaps = self._compute(Z, comp, resp, Es, xs, method)
        n = len(xs)
        out = pd.DataFrame({
            "material": m["name"], "Z": Z, "element_status": "listed" if m["listed"] else "unlisted",
            "response": resp, "energy_MeV": np.repeat(Es, n), "depth_mfp": np.tile(xs, len(Es)),
            "predicted_B": 10 ** lg.ravel(), "predicted_log10B": lg.ravel(),
            "source": np.repeat(src, n), "gap": np.repeat(gaps, n),
            "near_K_edge": np.repeat(self.near_k_edge(Z, Es), n) & (comp == 0),
            "guard_correction_log10": np.round(corr.ravel(), 6),
        })
        f = np.full(len(out), np.nan)
        for g in ("energy", "element"):
            sel = (out.gap == g).values
            if sel.any():
                f[sel] = self.expected_factor(g, out.depth_mfp.values[sel])
        out["expected_error_factor"] = f
        ff = np.where(np.isnan(f), 1.0, f)
        out["B_low"] = out.predicted_B / ff
        out["B_high"] = out.predicted_B * ff
        return out

    def _compute(self, Z, comp, resp, Es, xs, method="auto"):
        """Core of predict(): log10 B matrix (energies x depths), guard correction, route labels and gaps."""
        order = np.argsort(xs)
        xs_t = tuple(np.round(xs[order], 9))
        key = _zkey(Z, comp)
        has_sets = (key, resp) in self.sets
        raw = np.zeros((len(Es), len(xs)))
        src, gaps, need_ml = [], [], []
        for i, E in enumerate(Es):
            curve, how = self._listed_curve(key, resp, E, xs_t) if has_sets else (None, None)
            if method != "ml" and how == "table":
                raw[i], s, g = curve, ROUTE_LABEL[("table", None)], "table"
            else:
                g = "energy" if has_sets else "element"
                mth = self.routing[g] if method == "auto" else method
                if mth != "ml":
                    raw[i] = curve if has_sets else self._z_curve(Z, resp, E, xs_t)
                if mth != "interp":
                    need_ml.append((i, mth))
                s = ROUTE_LABEL[(mth, g)]
            src.append(s)
            gaps.append(g)
        if need_ml:
            idx = [i for i, _ in need_ml]
            ml = self._ml(key, resp, Es[idx], np.array(xs_t))
            for k, (i, mth) in enumerate(need_ml):
                raw[i] = ml[k] if mth == "ml" else 0.5 * (ml[k] + raw[i])
        lg = np.maximum.accumulate(np.clip(raw, 0, None), axis=1)   # B >= 1, non-decreasing in depth
        corr = lg - raw
        back = np.argsort(order)
        lg, corr = lg[:, back], corr[:, back]
        return lg, corr, src, gaps

    # ------------------------------------------------------------------ batch
    def predict_rows(self, rows: pd.DataFrame, method="auto") -> pd.DataFrame:
        """Predict a table with columns material, energy_MeV, depth_mfp[, response].

        Every input row gets exactly one output row, in the original order; rows that cannot be predicted carry
        a plain-language reason in the ``error`` column instead of failing the whole batch."""
        df = rows.copy().reset_index(drop=True)
        if "response" not in df:
            df["response"] = "EBF"
        df["response"] = df["response"].fillna("EBF").astype(str).str.strip().str.upper()
        df["_row"] = np.arange(len(df))
        parts, errs = [], []
        for (mat, resp, e), g in df.groupby(["material", "response", "energy_MeV"], sort=False, dropna=False):
            try:
                if pd.isna(mat) or pd.isna(e):
                    raise ValueError("Material or energy is missing.")
                depths = pd.to_numeric(g.depth_mfp, errors="coerce")
                if depths.isna().any():
                    raise ValueError("Depth is missing or not a number.")
                p = self.predict(mat, float(e), depths.values, resp, method)
                p.insert(0, "_row", g._row.values)
                parts.append(p)
            except (ValueError, TypeError) as exc:
                errs.append(pd.DataFrame({"_row": g._row.values, "error": str(exc)}))
        res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["_row"])
        if errs:
            res = pd.concat([res, pd.concat(errs, ignore_index=True)], ignore_index=True)
        if "error" not in res:
            res["error"] = ""
        res["error"] = res["error"].fillna("")
        inp = df.rename(columns={"material": "input_material", "energy_MeV": "input_energy_MeV",
                                 "depth_mfp": "input_depth_mfp", "response": "input_response"})
        res = inp.merge(res, on="_row", how="left").drop(columns="_row")
        return res

    # ------------------------------------------------------------------ identification (inverse problem)
    def candidates(self, scope="all") -> list[dict]:
        """Every material the predictor covers: elements Z_MIN..Z_MAX plus the compounds ('listed' = tabulated only)."""
        out = [self._entry(None, float(z), 0) for z in range(int(self.Z_MIN), int(self.Z_MAX) + 1)]
        out += [self._entry(nm.capitalize(), zc, 1) for nm, zc in self.compound_z.items()]
        return [c for c in out if c["listed"]] if scope == "listed" else out

    def identify(self, points: pd.DataFrame, meas_unc=0.05, responses=("EBF", "EABF"), scope="all") -> dict:
        """Rank materials and responses by how well their buildup curves match measured points.

        ``points`` needs columns energy_MeV, depth_mfp, B (one row per measured value).  Each point's log10 B residual is
        weighted by the combined measurement and model uncertainty (the model part is the route's typical error at that
        depth), giving a chi-square per candidate; relative probabilities follow from exp(-chi2/2) with a flat prior.
        Model errors at neighbouring depths are correlated, so treat the probabilities as a ranking, not exact odds."""
        p = self.check_points(points)
        Es = np.sort(p.energy_MeV.unique())
        y = np.log10(p.B.values)
        s_meas = np.log10(1 + float(meas_unc))
        rows = []
        for c in self.candidates(scope):
            for resp in responses:
                pred = np.empty(len(p)); s_mod = np.empty(len(p))
                for E in Es:
                    sel = np.isclose(p.energy_MeV.values, E)
                    xs = p.depth_mfp.values[sel]
                    lg, _, _, gaps = self._compute(c["Z"], c["comp"], resp, np.array([E]), xs)
                    pred[sel] = lg[0]
                    s_mod[sel] = (0.005 if gaps[0] == "table"
                                  else 1.25 * np.log10(self.expected_factor(gaps[0], xs)))   # MAE -> ~1 sigma
                r = y - pred
                s2 = s_meas ** 2 + s_mod ** 2
                chi2 = float(np.sum(r ** 2 / s2))
                # full Gaussian deviance: the log(s2) term stops candidates with large model uncertainty from winning
                # just because their residuals are divided by a bigger number
                dev = chi2 + float(np.sum(np.log(s2)))
                rows.append({"material": c["name"], "Z": c["Z"], "compound": bool(c["comp"]),
                             "tabulated": c["listed"], "response": resp, "chi2": chi2, "deviance": dev,
                             "typical_deviation": float(10 ** np.sqrt(np.mean(r ** 2)) - 1),
                             "max_deviation": float(10 ** np.max(np.abs(r)) - 1)})
        R = pd.DataFrame(rows)
        w = np.exp(-(R.deviance - R.deviance.min()) / 2)
        R["probability"] = w / w.sum()
        R = R.sort_values(["deviance", "Z"]).reset_index(drop=True)
        R["rank"] = np.arange(1, len(R) + 1)

        best = R.iloc[0]
        pz = R.groupby("Z").probability.sum().sort_index()
        cdf = pz.cumsum().values
        z_lo = float(pz.index[np.searchsorted(cdf, 0.05)])
        z_hi = float(pz.index[min(np.searchsorted(cdf, 0.95), len(cdf) - 1)])
        z_mean = float(np.sum(pz.index.values * pz.values))
        p_resp = R.groupby("response").probability.sum().to_dict()
        if len(responses) == 1:
            verdict_resp = f"{responses[0]} (given)"
        elif p_resp.get("EBF", 0) >= 0.8:
            verdict_resp = "EBF"
        elif p_resp.get("EABF", 0) >= 0.8:
            verdict_resp = "EABF"
        else:
            verdict_resp = "undecided"
        n = len(p)
        poor_fit = best.chi2 / n > 9          # residuals ~3x the stated uncertainty on average
        return {"ranking": R, "best": best.to_dict(), "p_z": pz, "z_mean": z_mean, "z_range": (z_lo, z_hi),
                "p_response": p_resp, "response_verdict": verdict_resp, "n_points": n, "energies": Es,
                "poor_fit": bool(poor_fit), "points": p, "meas_unc": meas_unc}

    def check_points(self, points: pd.DataFrame) -> pd.DataFrame:
        """Validate measured points; raises ValueError with a plain-language reason."""
        p = points.copy()
        p.columns = [str(c).strip() for c in p.columns]
        low = {c.lower(): c for c in p.columns}
        ren = {}
        for want, alts in {"energy_MeV": ["energy_mev", "energy", "e_mev", "e"],
                           "depth_mfp": ["depth_mfp", "depth", "x_mfp", "x", "mfp"],
                           "B": ["b", "buildup", "buildup_factor", "measured_b"]}.items():
            for a in alts:
                if a in low:
                    ren[low[a]] = want
                    break
        p = p.rename(columns=ren)
        miss = [c for c in ("energy_MeV", "depth_mfp", "B") if c not in p]
        if miss:
            raise ValueError(f"Missing column(s): {', '.join(miss)}.")
        for c in ("energy_MeV", "depth_mfp", "B"):
            p[c] = pd.to_numeric(p[c], errors="coerce")
        p = p.dropna(subset=["energy_MeV", "depth_mfp", "B"])
        if len(p) < 2:
            raise ValueError("Enter at least two points (depth and B) to identify a material.")
        if (p.B < 1).any():
            raise ValueError("Buildup factors are at least 1; check the B column.")
        if p.energy_MeV.min() < self.E_MIN or p.energy_MeV.max() > self.E_MAX:
            raise ValueError(f"Energies must be between {self.E_MIN} and {self.E_MAX} MeV.")
        if p.depth_mfp.min() < self.X_MIN or p.depth_mfp.max() > self.X_MAX:
            raise ValueError(f"Depths must be between {self.X_MIN:g} and {self.X_MAX:g} mean free paths "
                             "(not cm - divide thickness by the mean free path).")
        return p.reset_index(drop=True)

    def identify_groups(self, points: pd.DataFrame, group_col="sample", **kw) -> pd.DataFrame:
        """One identification per sample; failures are reported in an ``error`` column."""
        out = []
        for g, sub in points.groupby(group_col, sort=False):
            try:
                r = self.identify(sub, **kw)
                b = r["best"]
                second = r["ranking"].iloc[1]
                out.append({group_col: g, "points": r["n_points"], "best_material": b["material"], "Z": b["Z"],
                            "response": b["response"], "probability": b["probability"],
                            "runner_up": f'{second.material} {second.response}',
                            "runner_up_probability": second.probability,
                            "effective_Z": round(r["z_mean"], 1),
                            "Z_range_90pct": f'{r["z_range"][0]:g} - {r["z_range"][1]:g}',
                            "response_verdict": r["response_verdict"], "P_EBF": r["p_response"].get("EBF", np.nan),
                            "typical_deviation": b["typical_deviation"], "poor_fit": r["poor_fit"], "error": ""})
            except ValueError as exc:
                out.append({group_col: g, "error": str(exc)})
        return pd.DataFrame(out)
