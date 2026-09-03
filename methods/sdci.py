"""
SDCI — Causal Discovery from Conditionally Stationary Time Series
(Balsells-Rodas et al., ICML 2025).

This module vendors the *minimal* model code required to run the pretrained
MLP-encoder checkpoint in a single forward pass.  The encoder maps a
multivariate time series directly to a matrix of per-edge posterior
probabilities (a summary causal graph).  The state-encoder / decoder are only
needed for (variational) reconstruction during training and are **not** loaded
here — inference is encoder-only, following the official ``measure_performance``
script.

Only the code paths exercised by the MLP encoder are vendored, so the
``torch_scatter`` dependency (required only by the ``att``/``rnn`` edge
encoders, which are not used) is avoided entirely.
"""

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Vendored model components (adapted from charlio23/SDCI, MIT-licensed parts
# originating from ethanfetaya/NRI and loeweX/AmortizedCausalDiscovery).
# ---------------------------------------------------------------------------


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    return np.array(list(map(classes_dict.get, labels)), dtype=np.float32)


def create_rel_rec_send(num_atoms):
    """Receiver / sender one-hot matrices for all off-diagonal directed edges."""
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
    rel_rec = torch.FloatTensor(encode_onehot(np.where(off_diag)[0]))
    rel_send = torch.FloatTensor(encode_onehot(np.where(off_diag)[1]))
    return rel_rec, rel_send


def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.0, use_batch_norm=True,
                 final_linear=False, activation='elu'):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.use_batch_norm = use_batch_norm
        self.final_linear = final_linear
        if self.final_linear:
            self.fc_final = nn.Linear(n_out, n_out)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        if self.final_linear:
            x = self.fc_final(x)
        return self.batch_norm(x) if self.use_batch_norm else x


class Encoder(nn.Module):
    def __init__(self, factor=True):
        super().__init__()
        self.factor = factor

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        return torch.cat([senders, receivers], dim=-1)


class MLPEncoder(Encoder):
    """NRI-style MLP edge encoder; weights are independent of ``num_atoms``."""

    def __init__(self, n_in, n_hid, n_edge_types, n_states, do_prob=0.0, factor=True):
        super().__init__(factor)
        self.n_edge_types = n_edge_types
        self.n_states = n_states
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_edge_types * n_states)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, rel_rec, rel_send):
        # inputs: [B, num_atoms, T, num_dims]
        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)  # [B, atoms, T*num_dims]
        x = self.mlp1(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)
            x = self.mlp4(x)
        batch, num_rel, _ = x.size()
        return self.fc_out(x).reshape(batch, num_rel, self.n_states, self.n_edge_types)


# ---------------------------------------------------------------------------
# Pretrained-checkpoint handling.
# ---------------------------------------------------------------------------

# Architecture of the released springs *region* checkpoint (args.json):
#   n_balls=5, n_edge_types=2, n_states=2, length=8100, sample_freq=100.
# The encoder's mlp1.fc1 expects n_in = num_dims * seq_len = 4 * 80 = 320,
# i.e. 4 features per node per step over 80 steps.  The MLP weights are
# *independent of the node count*, so the encoder generalises to any number of
# variables (only rel_rec/rel_send, rebuilt at runtime, depend on num_atoms).
_NUM_DIMS = 4
_SEQ_LEN = 80
_N_HID = 256
_N_EDGE_TYPES = 2
_N_STATES = 2

_GDRIVE_ID = "1TpHG9nmK2VDXplZGoGxLpFOpITwMhAOd"
_CKPT_NAME = "springs_region_5_2_states_latest.pth.tar"

_MODEL_CACHE = {}


def _checkpoint_path(cfg):
    cache_dir = getattr(cfg, "cache_dir", "cd_zoo/methods/sdci_weights")
    return os.path.join(cache_dir, _CKPT_NAME), cache_dir


def _get_model(cfg):
    """Load (once) and cache the pretrained SDCI MLP encoder."""
    dest, cache_dir = _checkpoint_path(cfg)
    key = os.path.abspath(dest)
    if key not in _MODEL_CACHE:
        if not os.path.exists(dest):
            print(f"[SDCI] downloading checkpoint -> {dest}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            import gdown
            gdown.download(id=_GDRIVE_ID, output=dest, quiet=False)
        ckpt = torch.load(dest, map_location="cpu", weights_only=False)
        enc = MLPEncoder(
            n_in=_NUM_DIMS * _SEQ_LEN, n_hid=_N_HID,
            n_edge_types=_N_EDGE_TYPES, n_states=_N_STATES, do_prob=0.0,
        )
        enc.load_state_dict(ckpt["state_dict_encoder"])
        enc.to("cpu").eval()
        _MODEL_CACHE[key] = enc
    return _MODEL_CACHE[key]


def _to_features(X):
    """
    Expand a 1-D-per-variable time series ``(T, N)`` into the 4 features per
    node per step the springs encoder expects (``[x, y, vx, vy]``).  We use the
    value plus its first three discrete derivatives
    (position / velocity / acceleration / jerk), each normalised to ``[-1, 1]``.

    This is an *out-of-distribution* application: the model was trained on 2-D
    particle-simulation trajectories, so absolute accuracy on arbitrary series
    is expected to be modest.  The integration nonetheless exercises the real
    amortized model end-to-end.
    """
    T = X.shape[0]
    feats = [X]
    for _ in range(_NUM_DIMS - 1):
        feats.append(np.diff(feats[-1], axis=0, prepend=feats[-1][:1]))
    feats = np.stack(feats, axis=-1)  # [T, N, 4]
    # normalise each feature channel to [-1, 1]
    out = np.zeros_like(feats, dtype=np.float32)
    for c in range(_NUM_DIMS):
        lo, hi = feats[..., c].min(), feats[..., c].max()
        out[..., c] = (feats[..., c] - lo) * 2.0 / (hi - lo + 1e-8) - 1.0
    # truncate / pad time to the model's seq_len
    if T > _SEQ_LEN:
        out = out[-_SEQ_LEN:]
    elif T < _SEQ_LEN:
        out = np.concatenate(
            [out, np.zeros((_SEQ_LEN - T, out.shape[1], _NUM_DIMS), dtype=np.float32)],
            axis=0,
        )
    return out  # [_SEQ_LEN, N, 4]


def run_sdci(data_sample, cfg):
    """
    SDCI (Balsells-Rodas et al., ICML 2025) for causal discovery from
    conditionally-stationary time series.

    A pretrained NRI-style MLP encoder ingests a multivariate time series and
    outputs, in one forward pass, a per-edge probability matrix giving the
    *summary* causal graph (no lag information).  Edge probabilities are
    aggregated to ``[effect, cause]`` orientation to match cd_zoo (receiver =
    effect, sender = cause), with no transpose.  Returns a 2-D summary
    prediction ``(pred, None)`` like the i.i.d. methods; cd_zoo's ``score``
    sums lagged ground truth over the lag axis when comparing.

    .. note::
       The released checkpoint is the springs *region* model trained on 2-D
       particle simulations (5 particles, 80 steps, 4 features).  1-D time
       series are expanded to 4 features via discrete derivatives and truncated
       to 80 steps — an out-of-distribution setting, so results on arbitrary
       benchmarks are expected to be modest.

    Parameters
    ----------
    data_sample : pandas.DataFrame or np.ndarray
        Time series.  DataFrame is ``(T, n_vars)``; ndarray is
        ``(n_vars, T)`` (cd_zoo convention).
    cfg : OmegaConf
        ``cache_dir`` (default ``cd_zoo/methods/sdci_weights``).
    """
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values
    X = X.astype(np.float32)
    n_vars = X.shape[1]

    try:
        enc = _get_model(cfg)
        feats = _to_features(X)  # [80, N, 4]
        var = torch.from_numpy(feats).permute(1, 0, 2).unsqueeze(0)  # [1, N, 80, 4]

        rel_rec, rel_send = create_rel_rec_send(n_vars)

        with torch.no_grad():
            logits = enc(var, rel_rec, rel_send)  # [1, E, n_states, n_edge_types]
            prob = my_softmax(logits, -1)          # softmax over edge types
            # P(edge exists) per state, then aggregate over states (max)
            edge_prob = prob[..., 1].max(dim=-1)[0].squeeze(0)  # [E]

        # Map directed edges back to an [effect, cause] adjacency matrix.
        off_diag = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        receivers, senders = np.where(off_diag)  # receiver=effect, sender=cause
        pred = np.zeros((n_vars, n_vars), dtype=np.float32)
        pred[receivers, senders] = edge_prob.numpy()

    except Exception as e:
        print(f"SDCI failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted by SDCI. replacing with 0")
        pred = np.zeros((n_vars, n_vars))

    return pred, None  # Summary graph only; no lags / no instant channel.
