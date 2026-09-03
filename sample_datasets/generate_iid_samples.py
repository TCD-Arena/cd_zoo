"""
Generate i.i.d. (non-time-series) linear-SCM test sets for cd_zoo.

Each test set is a folder under ``sample_datasets/`` containing:
    X.npy        : float array (n_samples, n_vars, n_obs)
                   — n_samples independent datasets; each is n_vars variables
                     observed n_obs times (i.i.d. draws, NO temporal order).
    Y.npy        : float array (n_samples, n_vars, n_vars)
                   — ground-truth summary adjacency in [effect, cause]
                     orientation (Y[e, c] = weight of edge c -> e), matching
                     cd_zoo's convention.  0 = no edge.
    config.yaml  — generation parameters (provenance / reproducibility).

The data are linear structural causal models (SCMs)

        x = B x + e   =>   x = (I - B)^{-1} e

where B is the weighted adjacency of a random DAG (acyclic by construction:
nodes are topologically ordered and only earlier nodes may point to later
ones).  Two noise regimes are produced:

    * iid_gaussian    : e ~ N(0, 1)
      -> constraint/score methods (PC, GES, DAGMA) recover the skeleton;
         LiNGAM methods cannot orient edges from Gaussian noise alone
         (expected behaviour — a useful contrast test).

    * iid_nongaussian : e ~ Uniform(-sqrt(3), sqrt(3))   (variance 1)
      -> non-Gaussianity lets DirectLiNGAM / ICALiNGAM orient the full DAG;
         all methods should do well here.

Orientation note
----------------
B is built directly as [effect, cause]: the equation for variable x_j is
x_j = sum_i B[j, i] * x_i + e_j, so B[j, i] is the coefficient of parent
i in child j's equation.  This matches the [effect, cause] convention used
by the cd_zoo i.i.d. method wrappers and the ``score`` function — NO
transpose is needed.  (Verified empirically: DirectLiNGAM / DAGMA reach
AUROC 1.0 against these labels.)

Usage
-----
    python generate_iid_samples.py            # writes both folders
    python generate_iid_samples.py --help     # override parameters

The generated folders are loadable with the existing ``benchmark.py`` via
``load_numpy_ds`` (``benchmark.py ds_name=iid_gaussian method=lingam``).
"""
import argparse
import numpy as np
import yaml
from pathlib import Path


def sample_random_dag(n_vars, edge_prob, rng, weight_range=(0.5, 1.0)):
    """
    Return a weighted adjacency B ([effect, cause]) of a random DAG.

    A random topological order is drawn; each earlier node may point to each
    later node with probability ``edge_prob``.  Edge weights are drawn
    uniformly from ``weight_range`` (positive).
    """
    order = rng.permutation(n_vars)          # topological order of node indices
    B = np.zeros((n_vars, n_vars))
    lo, hi = weight_range
    for pos_j in range(n_vars):
        child = order[pos_j]                 # effect
        for pos_i in range(pos_j):
            parent = order[pos_i]            # cause
            if rng.random() < edge_prob:
                B[child, parent] = rng.uniform(lo, hi)   # B[effect, cause]
    return B


def simulate_linear_scm(B, n_obs, rng, noise):
    """
    Simulate n_obs i.i.d. draws from x = (I - B)^{-1} e.

    Parameters
    ----------
    B : (n_vars, n_vars) weighted DAG adjacency [effect, cause]
    n_obs : int
    rng : np.random.Generator
    noise : str  -- "gaussian" or "nongaussian"
    """
    n_vars = B.shape[0]
    if noise == "gaussian":
        e = rng.standard_normal((n_vars, n_obs))
    elif noise == "nongaussian":
        # Uniform(-sqrt(3), sqrt(3)) has variance 1 and is clearly non-Gaussian
        # (sub-Gaussian, excess kurtosis -1.2) -> LiNGAM-identifiable.
        e = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(n_vars, n_obs))
    else:
        raise ValueError(f"unknown noise type: {noise!r}")
    # x = (I - B)^{-1} e
    x = np.linalg.solve(np.eye(n_vars) - B, e)
    return x  # (n_vars, n_obs)


def generate_dataset(n_vars, n_samples, n_obs, edge_prob, seed, noise,
                     weight_range=(0.5, 1.0)):
    """
    Generate n_samples independent random-DAG linear SCMs.
    Returns X (n_samples, n_vars, n_obs) and Y (n_samples, n_vars, n_vars).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_vars, n_obs))
    Y = np.zeros((n_samples, n_vars, n_vars))
    for s in range(n_samples):
        B = sample_random_dag(n_vars, edge_prob, rng, weight_range)
        X[s] = simulate_linear_scm(B, n_obs, rng, noise)
        Y[s] = B
    return X, Y


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-vars", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=50,
                   help="number of independent datasets (different random DAGs)")
    p.add_argument("--n-obs", type=int, default=1000,
                   help="number of i.i.d. observations per dataset")
    p.add_argument("--edge-prob", type=float, default=0.3,
                   help="probability of each possible edge in the DAG")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weight-range", type=float, nargs=2, default=[0.5, 1.0],
                   metavar=("LO", "HI"))
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).parent))
    args = p.parse_args()

    cfg = dict(
        n_vars=args.n_vars,
        n_samples=args.n_samples,
        n_obs=args.n_obs,
        edge_prob=args.edge_prob,
        weight_range=list(args.weight_range),
        seed=args.seed,
        noise_type=None,          # filled per dataset below
        model="linear SCM  x = (I - B)^{-1} e",
        orientation="[effect, cause]  (Y[e,c] = weight of edge c->e)",
        note="i.i.d. samples; NO temporal ordering. For i.i.d. methods "
             "(lingam, pc, ges, ica_lingam, dagma, avici, cdfm, arrow, tabcausal).",
    )

    for noise, folder in [("gaussian", "iid_gaussian"),
                          ("nongaussian", "iid_nongaussian")]:
        X, Y = generate_dataset(
            n_vars=args.n_vars,
            n_samples=args.n_samples,
            n_obs=args.n_obs,
            edge_prob=args.edge_prob,
            seed=args.seed,
            noise=noise,
            weight_range=tuple(args.weight_range),
        )
        # benchmark.py expects data_path/<regime_index>/{X,Y,instant_links}.npy
        # (it listdir's the dataset folder and picks the regime via
        #  ``which_dataset``).  One regime (index 0) per noise type here.
        out = Path(args.out_dir) / folder / "0"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "X.npy", X)
        np.save(out / "Y.npy", Y)
        cfg_noise = dict(cfg)
        cfg_noise["noise_type"] = noise
        with open(out / "config.yaml", "w") as f:
            yaml.safe_dump(cfg_noise, f, sort_keys=False)
        n_edges = int((Y != 0).sum() / args.n_samples)
        print(f"{folder:16s}: X={X.shape} Y={Y.shape}  "
              f"avg {n_edges:.1f} edges/dataset  noise={noise}")


if __name__ == "__main__":
    main()
