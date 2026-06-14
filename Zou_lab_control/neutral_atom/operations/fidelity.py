"""Per-site readout-fidelity characterization from labelled data (Rb87 flow).

``calibrate_threshold_from_images`` sets one threshold per site from a single
distribution.  This module models the full experimental fidelity flow with
GROUND-TRUTH labels, ported from the Rb87 qCMOS analysis as pure, general
functions (any number of sites; no session, no frontend):

1. :func:`reference_labels` -- high-SNR reference shots (long exposure), shape
   ``(n_groups, n_reference_shots, n_sites)`` -> per-site bimodal fit -> classify
   every reference shot -> strict consensus across a group's reference shots ->
   a bright / dark / ambiguous label per ``(group, site)``.
2. :func:`train_test_split` -- stratified-by-site random split of the labelled
   groups into a training set (fit the threshold) and a held-out test set
   (score it honestly).
3. :func:`characterize_readout` -- fit each site's threshold on the training
   split of the SHORT (readout) signal, then report held-out per-site fidelity,
   one global threshold for comparison, and a drop-worst-site ablation.

The per-site bimodal/Gaussian primitives come from :mod:`..core.bimodal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.bimodal import (
    BimodalFit,
    classify_threshold,
    fit_bimodal,
    gaussian_fidelity,
    optimal_gaussian_threshold,
)


# --------------------------------------------------------------------------- #
# Reference labels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReferenceLabels:
    """Ground-truth occupancy labels per ``(group, site)`` from reference shots."""

    occupied: np.ndarray   # (n_groups, n_sites) bool, valid entries only meaningful
    dark: np.ndarray       # (n_groups, n_sites) bool
    valid: np.ndarray      # (n_groups, n_sites) bool (False = ambiguous / failed-fit site)
    fits: list[BimodalFit]
    n_reference_shots: int

    @property
    def n_groups(self) -> int:
        return int(self.occupied.shape[0])

    @property
    def n_sites(self) -> int:
        return int(self.occupied.shape[1])

    @property
    def ambiguous(self) -> np.ndarray:
        return ~self.valid


def reference_labels(reference_signals) -> ReferenceLabels:
    """Strict-consensus bright/dark labels from reference shots.

    ``reference_signals`` is ``(n_groups, n_reference_shots, n_sites)``.  For each
    site a bimodal threshold is fit on all reference samples; every reference shot
    of a group is classified, and the group is labelled bright only if ALL its
    reference shots are bright (dark if all dark), else ambiguous.  Sites whose
    bimodal fit fails are marked invalid for every group.
    """

    ref = np.asarray(reference_signals, dtype=float)
    if ref.ndim != 3:
        raise ValueError("reference_signals must have shape (n_groups, n_reference_shots, n_sites).")
    n_groups, n_ref, n_sites = ref.shape
    bright = np.zeros((n_groups, n_ref, n_sites), dtype=bool)
    fits: list[BimodalFit] = []
    fit_ok = np.zeros(n_sites, dtype=bool)
    for j in range(n_sites):
        fit = fit_bimodal(ref[:, :, j].ravel())
        fits.append(fit)
        if fit.ok and np.isfinite(fit.threshold):
            bright[:, :, j] = classify_threshold(ref[:, :, j], fit.threshold, fit.bright_above)
            fit_ok[j] = True
    all_bright = np.all(bright, axis=1)
    all_dark = np.all(~bright, axis=1)
    occupied = all_bright
    dark = all_dark
    valid = (all_bright | all_dark) & fit_ok[np.newaxis, :]
    occupied = occupied & valid
    dark = dark & valid
    return ReferenceLabels(occupied=occupied, dark=dark, valid=valid, fits=fits, n_reference_shots=int(n_ref))


# --------------------------------------------------------------------------- #
# Train / test split
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainTestSplit:
    train: np.ndarray  # (n_groups, n_sites) bool
    test: np.ndarray   # (n_groups, n_sites) bool
    seed: int
    train_fraction: float


def train_test_split(labels: ReferenceLabels, *, train_fraction: float = 0.9, seed: int = 0) -> TrainTestSplit:
    """Stratified-by-site, by-class random split of the valid labelled groups.

    Each site's bright and dark labelled groups are split independently so both
    classes appear in train and test when there are at least two of each.
    """

    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    rng = np.random.default_rng(int(seed))
    valid = np.asarray(labels.valid, dtype=bool)
    occ = np.asarray(labels.occupied, dtype=bool)
    train = np.zeros_like(valid)
    test = np.zeros_like(valid)
    frac = float(train_fraction)
    for j in range(valid.shape[1]):
        for state in (False, True):
            idx = np.where(valid[:, j] & (occ[:, j] == state))[0]
            if idx.size == 0:
                continue
            perm = rng.permutation(idx)
            n_train = int(round(frac * idx.size))
            n_train = min(max(n_train, 1), idx.size - 1) if idx.size >= 2 else 1
            train[perm[:n_train], j] = True
            if n_train < idx.size:
                test[perm[n_train:], j] = True
    return TrainTestSplit(train=train, test=test, seed=int(seed), train_fraction=frac)


# --------------------------------------------------------------------------- #
# Threshold training + held-out scoring
# --------------------------------------------------------------------------- #
def _common_bin_edges(values, mask, *, bins: int = 120, quantile=(0.001, 0.999), margin: float = 0.04) -> np.ndarray:
    vals = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.linspace(-1.0, 1.0, int(bins) + 1)
    lo, hi = np.quantile(vals, quantile)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(float(hi - lo), 1.0)
    return np.linspace(float(lo) - margin * span, float(hi) + margin * span, int(bins) + 1)


def _empirical_threshold(dark, bright, edges, *, bright_above: bool, tie_target: float) -> float:
    """Balanced-fidelity-maximizing threshold among histogram bin edges (O(N+B))."""

    dark = np.asarray(dark, dtype=float)
    bright = np.asarray(bright, dtype=float)
    dark = dark[np.isfinite(dark)]
    bright = bright[np.isfinite(bright)]
    if dark.size == 0 or bright.size == 0:
        return float("nan")
    hd, _ = np.histogram(dark, bins=edges)
    hb, _ = np.histogram(bright, bins=edges)
    cd = np.concatenate([[0], np.cumsum(hd)])
    cb = np.concatenate([[0], np.cumsum(hb)])
    if bright_above:
        fd = cd / dark.size
        fb = (bright.size - cb) / bright.size
    else:
        fd = (dark.size - cd) / dark.size
        fb = cb / bright.size
    fid = 0.5 * (fd + fb)
    best_val = float(np.nanmax(fid))
    best = np.flatnonzero(np.isclose(fid, best_val, rtol=0, atol=1e-12))
    if best.size > 1:
        idx = int(best[np.argmin(np.abs(edges[best] - tie_target))])
    else:
        idx = int(best[0])
    return float(edges[idx])


@dataclass
class SiteFidelity:
    """Held-out fidelity of one site after training its threshold."""

    site: int
    threshold: float
    bright_above: bool
    fidelity: float           # held-out balanced 0.5*(F_dark+F_bright)
    fidelity_dark: float
    fidelity_bright: float
    model_fidelity: float     # Gaussian-model fidelity from the trained fit
    mu_dark: float
    sigma_dark: float
    mu_bright: float
    sigma_bright: float
    n_test: int
    n_train_dark: int
    n_train_bright: int


def _confusion(pred, occupied, valid) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool)[np.asarray(valid, dtype=bool)]
    occ = np.asarray(occupied, dtype=bool)[np.asarray(valid, dtype=bool)]
    tp = int(np.sum(pred & occ))
    tn = int(np.sum((~pred) & (~occ)))
    fp = int(np.sum(pred & (~occ)))
    fn = int(np.sum((~pred) & occ))
    n_bright, n_dark = int(np.sum(occ)), int(np.sum(~occ))
    f_dark = tn / n_dark if n_dark else float("nan")
    f_bright = tp / n_bright if n_bright else float("nan")
    fidelity = 0.5 * (f_dark + f_bright) if np.isfinite([f_dark, f_bright]).all() else float("nan")
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, n_bright=n_bright, n_dark=n_dark,
                n_valid=int(occ.size), F_dark=f_dark, F_bright=f_bright, F=fidelity,
                errors=fp + fn, correct=tp + tn)


def _fit_site_threshold(dark_train, bright_train, edges):
    """Train a threshold from dark/bright training samples (empirical, Gaussian tie)."""

    d = np.asarray(dark_train, dtype=float)
    b = np.asarray(bright_train, dtype=float)
    d = d[np.isfinite(d)]
    b = b[np.isfinite(b)]
    if d.size < 2 or b.size < 2:
        return dict(threshold=float("nan"), bright_above=True, mu_dark=float("nan"), sigma_dark=float("nan"),
                    mu_bright=float("nan"), sigma_bright=float("nan"), model_fidelity=float("nan"),
                    n_train_dark=int(d.size), n_train_bright=int(b.size))
    mu_d, mu_b = float(np.mean(d)), float(np.mean(b))
    s_d = max(float(np.std(d, ddof=1)), 1e-12)
    s_b = max(float(np.std(b, ddof=1)), 1e-12)
    t_gauss, bright_above = optimal_gaussian_threshold(mu_d, s_d, mu_b, s_b)
    threshold = _empirical_threshold(d, b, edges, bright_above=bright_above, tie_target=t_gauss)
    if not np.isfinite(threshold):
        threshold = t_gauss
    _, _, model_f = gaussian_fidelity(mu_d, s_d, mu_b, s_b, threshold, bright_above)
    return dict(threshold=float(threshold), bright_above=bool(bright_above), mu_dark=mu_d, sigma_dark=s_d,
                mu_bright=mu_b, sigma_bright=s_b, model_fidelity=float(model_f),
                n_train_dark=int(d.size), n_train_bright=int(b.size))


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #
@dataclass
class FidelityReport:
    """Held-out per-site and aggregate readout fidelity with a drop-worst ablation."""

    per_site: list[SiteFidelity]
    thresholds: np.ndarray         # (n_sites,) trained per-site thresholds
    pred: np.ndarray               # (n_groups, n_sites) bool, per-site predictions
    aggregate_fidelity: float      # held-out balanced fidelity, per-site thresholds
    global_threshold: float
    global_fidelity: float         # held-out balanced fidelity, one shared threshold
    ablation: list[dict[str, Any]]  # drop-worst-k rows
    labels: ReferenceLabels
    split: TrainTestSplit
    short_signals: np.ndarray      # (n_groups, n_sites) the readout signal that was scored

    @property
    def n_sites(self) -> int:
        return len(self.per_site)

    def per_site_arrays(self, *, valid_only: bool = True) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per-site ``(values, occupied)`` lists for plotting (ragged; valid groups only)."""

        values: list[np.ndarray] = []
        occupied: list[np.ndarray] = []
        for j in range(self.n_sites):
            mask = self.labels.valid[:, j] if valid_only else np.ones(self.labels.n_groups, dtype=bool)
            values.append(self.short_signals[mask, j])
            occupied.append(self.labels.occupied[mask, j])
        return values, occupied

    @property
    def site_fidelities(self) -> np.ndarray:
        return np.asarray([s.fidelity for s in self.per_site], dtype=float)

    @property
    def worst_sites(self) -> list[int]:
        order = np.argsort(np.nan_to_num(self.site_fidelities, nan=-np.inf))
        return [int(i) for i in order]

    def summary(self) -> dict[str, Any]:
        fids = self.site_fidelities
        return {
            "n_sites": self.n_sites,
            "n_groups": self.labels.n_groups,
            "aggregate_fidelity": float(self.aggregate_fidelity),
            "global_fidelity": float(self.global_fidelity),
            "mean_site_fidelity": float(np.nanmean(fids)) if fids.size else float("nan"),
            "min_site_fidelity": float(np.nanmin(fids)) if fids.size else float("nan"),
            "ambiguous_fraction": float(np.mean(self.labels.ambiguous)),
        }


def _drop_worst(per_site, pred, labels, split, max_drop) -> list[dict[str, Any]]:
    fids = [s.fidelity for s in per_site]
    order = list(np.argsort([f if np.isfinite(f) else -np.inf for f in fids]))  # worst first
    rows: list[dict[str, Any]] = []
    base_valid = split.test & labels.valid
    for k in range(int(max_drop) + 1):
        excluded = set(int(order[i]) for i in range(min(k, len(order))))
        keep = np.ones(labels.n_sites, dtype=bool)
        for j in excluded:
            keep[j] = False
        c = _confusion(pred, labels.occupied, base_valid & keep[np.newaxis, :])
        rows.append(dict(drop_worst_k=int(k), excluded_sites=sorted(excluded),
                         kept_sites=int(keep.sum()), fidelity=c["F"], errors=c["errors"],
                         n_valid=c["n_valid"]))
    return rows


def characterize_readout(
    short_signals,
    reference_signals,
    *,
    train_fraction: float = 0.9,
    seed: int = 0,
    bins: int = 120,
    max_drop: int = 5,
) -> FidelityReport:
    """Full per-site readout-fidelity characterization.

    ``short_signals`` is ``(n_groups, n_sites)`` (the readout being characterized);
    ``reference_signals`` is ``(n_groups, n_reference_shots, n_sites)`` (high-SNR
    ground-truth shots).  Returns a :class:`FidelityReport` with trained per-site
    thresholds, held-out per-site and aggregate fidelity, a single global
    threshold for comparison, and a drop-worst-site ablation.
    """

    short = np.asarray(short_signals, dtype=float)
    if short.ndim != 2:
        raise ValueError("short_signals must have shape (n_groups, n_sites).")
    labels = reference_labels(reference_signals)
    if short.shape != labels.occupied.shape:
        raise ValueError("short_signals and reference_signals must agree on (n_groups, n_sites).")
    split = train_test_split(labels, train_fraction=train_fraction, seed=seed)
    edges = _common_bin_edges(short, labels.valid, bins=bins)

    per_site: list[SiteFidelity] = []
    thresholds = np.full(labels.n_sites, np.nan, dtype=float)
    pred = np.zeros_like(labels.occupied, dtype=bool)
    for j in range(labels.n_sites):
        finite = np.isfinite(short[:, j])
        train_mask = split.train[:, j] & labels.valid[:, j] & finite
        test_mask = split.test[:, j] & labels.valid[:, j] & finite
        fit = _fit_site_threshold(short[train_mask & (~labels.occupied[:, j]), j],
                                  short[train_mask & labels.occupied[:, j], j], edges)
        thresholds[j] = fit["threshold"]
        if np.isfinite(fit["threshold"]):
            pred[:, j] = classify_threshold(short[:, j], fit["threshold"], fit["bright_above"])
        c = _confusion(pred[:, j], labels.occupied[:, j], test_mask)
        per_site.append(SiteFidelity(
            site=j, threshold=fit["threshold"], bright_above=fit["bright_above"],
            fidelity=c["F"], fidelity_dark=c["F_dark"], fidelity_bright=c["F_bright"],
            model_fidelity=fit["model_fidelity"], mu_dark=fit["mu_dark"], sigma_dark=fit["sigma_dark"],
            mu_bright=fit["mu_bright"], sigma_bright=fit["sigma_bright"], n_test=c["n_valid"],
            n_train_dark=fit["n_train_dark"], n_train_bright=fit["n_train_bright"],
        ))

    aggregate = _confusion(pred, labels.occupied, split.test & labels.valid)["F"]

    # One global threshold trained on the pooled valid training samples.
    train_all = split.train & labels.valid & np.isfinite(short)
    g_fit = _fit_site_threshold(short[train_all & (~labels.occupied)], short[train_all & labels.occupied], edges)
    if np.isfinite(g_fit["threshold"]):
        g_pred = classify_threshold(short, g_fit["threshold"], g_fit["bright_above"])
    else:
        g_pred = np.zeros_like(labels.occupied, dtype=bool)
    global_fidelity = _confusion(g_pred, labels.occupied, split.test & labels.valid)["F"]

    ablation = _drop_worst(per_site, pred, labels, split, max_drop)
    return FidelityReport(
        per_site=per_site, thresholds=thresholds, pred=pred, aggregate_fidelity=float(aggregate),
        global_threshold=float(g_fit["threshold"]), global_fidelity=float(global_fidelity),
        ablation=ablation, labels=labels, split=split, short_signals=short,
    )


__all__ = [
    "ReferenceLabels",
    "TrainTestSplit",
    "SiteFidelity",
    "FidelityReport",
    "reference_labels",
    "train_test_split",
    "characterize_readout",
]
