import numpy as np
import lingam


def run_varma_lingam(data_sample, cfg):
    """
    VARMA-LiNGAM for time series with non-Gaussian noise (Hyvärinen, Zhang,
    Shimizu & Hoyer, JMLR 2010).

    A structural vector autoregressive moving-average (SVARMA) model estimated
    using non-Gaussianity. It combines instantaneous LiNGAM effects with both
    lagged autoregressive (AR) and moving-average (MA) effects, extending
    VAR-LiNGAM (which has no MA component). The model is identifiable without
    prior knowledge of the network structure (assuming acyclicity of the
    instantaneous effects and no latent variables).

    Implementation: wraps ``lingam.VARMALiNGAM`` (the ``lingam`` package, already
    a dependency of cd_zoo via the DirectLiNGAM / ICALiNGAM / VAR-LiNGAM
    methods). The API mirrors the VAR-LiNGAM wrapper: ``VARMALiNGAM(order=
    (p, q), criterion=..., prune=...).fit(X)`` → ``.adjacency_matrices_``.

    ``adjacency_matrices_`` is a **tuple of two** arrays ``(psis, omegas)``:
      * ``psis`` (shape ``(p, N, N)``): the AR + instantaneous structural
        coefficients. ``psis[0]`` is the instantaneous effect matrix ``B``
        (lag 0); ``psis[1:]`` are the lagged AR coefficients.
      * ``omegas`` (shape ``(q, N, N)``): the MA coefficients (effects of the
        noise terms, **not** edges between observed variables — ignored for
        structure scoring).

    Only ``psis`` is used (the AR part), transposed to ``[effect, cause, lag]``
    — the same convention as VAR-LiNGAM. The absolute value ``|psi|`` is the
    per-edge score. (Verified empirically: the orientation matches VAR-LiNGAM
    on a known VAR — no additional transpose needed.)

    Note: VARMA coefficient estimation (via statsmodels) is considerably slower
    than plain VAR; for large ``T`` or high orders expect longer runtimes. The
    noise must be non-Gaussian for full identifiability of the instantaneous
    effects.

    Parameters
    ----------
    data_sample : pandas.DataFrame or np.ndarray
        Time series. DataFrame is ``(T, n_vars)``; ndarray is ``(n_vars, T)``
        (cd_zoo convention).
    cfg : OmegaConf
        ``order`` (default ``(3, 1)`` = AR order 3, MA order 1), ``criterion``
        (default ``None``; one of ``'aic'``/``'fpe'``/``'hqic'``/``'bic'``/
        ``None`` — ``None`` searches for the best lag order), ``prune``
        (default ``True``, prunes weak edges via LASSO).
    """
    if isinstance(data_sample, np.ndarray):
        data_sample = data_sample.T

    order = tuple(cfg.order) if hasattr(cfg.order, "__iter__") else (cfg.order, 1)
    try:
        model = lingam.VARMALiNGAM(
            order=order, criterion=cfg.criterion, prune=cfg.prune
        )
        model.fit(data_sample)

        psis = np.asarray(model.adjacency_matrices_[0])  # (p, N, N)
        # transpose to [effect, cause, lag], same as VAR-LiNGAM
        pred = np.transpose(psis, axes=[1, 2, 0])
    except Exception as e:
        print(f"VARMA-LiNGAM fit failed, predicting zeros: {e}")
        n = data_sample.shape[1]
        pred = np.zeros((n, n, order[0]))

    if np.isnan(np.array(pred)).sum() > 0:
        print("ISSUE detected, prediction set to 0")
        pred = np.zeros(pred.shape)

    pred = np.abs(pred)
    # psis[0] = instantaneous (lag 0), psis[1:] = lagged AR
    if pred.shape[2] > 1:
        return pred[:, :, 1:], pred[:, :, 0]
    else:
        # only instantaneous, no lags
        return np.zeros((pred.shape[0], pred.shape[1], 1)), pred[:, :, 0]
