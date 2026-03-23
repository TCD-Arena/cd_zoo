import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import pickle
from os import listdir
import yaml
from sklearn.metrics import roc_auc_score
import numpy as np
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.common import ListDataset
from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
import pickle
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
import cvxpy as cvx
from scipy import linalg
import warnings

# this is a custom implementation of: https://github.com/wasimahmadpk/gCause


def cov2cor(Sigma):
    """
    Converts a covariance matrix to a correlation matrix
    :param Sigma : A covariance matrix (p x p)
    :return: A correlation matrix (p x p)
    """
    sqrtDiagSigma = np.sqrt(np.diag(Sigma))
    scalingFactors = np.outer(sqrtDiagSigma,sqrtDiagSigma)
    return np.divide(Sigma, scalingFactors)

def solve_sdp(Sigma, tol=1e-3):
    """
    Computes s for sdp-correlated Gaussian knockoffs
    :param Sigma : A covariance matrix (p x p)
    :param mu    : An array of means (p x 1)
    :return: A matrix of knockoff variables (n x p)
    """

    # Convert the covariance matrix to a correlation matrix
    # Check whether Sigma is positive definite
    if(np.min(np.linalg.eigvals(Sigma))<0):
        corrMatrix = cov2cor(Sigma + (1e-8)*np.eye(Sigma.shape[0]))
    else:
        corrMatrix = cov2cor(Sigma)
    p,_ = corrMatrix.shape
    s = cvx.Variable(p)
    objective = cvx.Maximize(sum(s))
    constraints = [ 2.0*corrMatrix >> cvx.diag(s) + cvx.diag([tol]*p), 0<=s, s<=1]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver='CVXOPT')
    assert prob.status == cvx.OPTIMAL
    s = np.clip(np.asarray(s.value).flatten(), 0, 1)
    # Scale back the results for a covariance matrix
    return np.multiply(s, np.diag(Sigma))
    
class GaussianKnockoffs:
    """
    Class GaussianKnockoffs
    Knockoffs for a multivariate Gaussian model
    """
    def __init__(self, Sigma, method="equi", mu=[], tol=1e-3):
        """
        Constructor
        :param model  : A multivariate Gaussian model object containing the covariance matrix
        :param method : Specifies how to determine the free parameters of Gaussian knockoffs.
                        Allowed values: "equi", "sdp" (default "equi")
        :return:
        """
        if len(mu)==0:
            self.mu = np.zeros((Sigma.shape[0],))
        else:
            self.mu = mu
        self.p = len(self.mu)
        self.Sigma = Sigma
        self.method = method

        # Initialize Gaussian knockoffs by computing either SDP or min(Eigs)

        if self.method=="equi":
            lambda_min = linalg.eigh(self.Sigma, eigvals_only=True, subset_by_index=(0,0))[0]
            s = min(1,2*(lambda_min-tol))
            self.Ds = np.diag([s]*self.Sigma.shape[0])
        elif self.method=="sdp":
            self.Ds = np.diag(solve_sdp(self.Sigma,tol=tol))
        else:
            raise ValueError('Invalid Gaussian knockoff type: '+self.method)
        self.SigmaInvDs = linalg.lstsq(self.Sigma,self.Ds)[0]
        self.V = 2.0*self.Ds - np.dot(self.Ds, self.SigmaInvDs)
        self.LV = np.linalg.cholesky(self.V+1e-10*np.eye(self.p))
        if linalg.eigh(self.V, eigvals_only=True, subset_by_index=(0,0))[0] <= tol:
            warnings.warn("Warning...........\
            The conditional covariance matrix for knockoffs is not positive definite. \
            Knockoffs will not have any power.")

    def generate(self, X):
        """
        Generate knockoffs for the multivariate Gaussian model
        :param X: A matrix of observations (n x p)
        :return: A matrix of knockoff variables (n x p)
        """
        n, p = X.shape
        muTilde = X - np.dot(X-np.tile(self.mu,(n,1)), self.SigmaInvDs)
        N = np.random.normal(size=muTilde.shape)
        return muTilde + np.dot(N,self.LV.T)

def generate_knockoff(datax):

        X_train = datax[0:round(len(datax)*1.0), :]
        # Compute the empirical covariance matrix of the training data
        SigmaHat = np.cov(X_train, rowvar=False)
        # TODO : Gaussian Knockoffs are standardly used. If something else is necessary check original library.
        second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp")

        # Measure pairwise second-order knockoff correlations
        # This might be useful to asses quality of the fit if necessary
        #corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

        # Generate second-order knockoffs
        Xk = second_order.generate(X_train)
        return Xk




def train_deep_ar_estimator(df, cfg):
    """
    df : pandas.DataFrame
    training_length : int
    cfg: Hydra config.

    """
    train_ds = ListDataset(
        [{"target": df.values.T.tolist(), "start": df.index[0]}],
        freq=cfg.freq,  # Complete artifact (Not needed.)
        one_dim_target=False,
    )

    # Training model if necessary
    if not cfg.use_cached_model:
        # create estimator
        estimator = DeepAREstimator(
            prediction_length=cfg.prediction_length,
            context_length=cfg.context_length,
            freq=cfg.freq,
            num_layers=cfg.num_layers,
            num_cells=cfg.num_cells,
            dropout_rate=cfg.dropout_rate,
            trainer=Trainer(
                ctx=cfg.device,
                epochs=cfg.epochs,
                hybridize=False,
                learning_rate=cfg.learning_rate,
                batch_size=cfg.batch_size,
            ),
            distr_output=MultivariateGaussianOutput(dim=df.shape[1]),
        )

        print("Training forecasting model....")
        M = estimator.train(train_ds)
        if cfg.save_intermediate:
            pickle.dump(M, open(cfg.save_path + "/model.p", "wb"))
        return M

    else:  # load alternatively
        print("Loading forecasting model....")
        M = pickle.load(open(cfg.save_path + "/model.p", "rb"))
        return M


def eval_deep_ar_estimator(predictor, data, cfg):
    """
    predictor: model
    data: data that is required to make a prediction for the next window.
    cfg: Hydra config file
    return prediction for the next forecast window.
    """
    test_ds = ListDataset(
        [{"target": data.values.T.tolist(), "start": data.index[0]}],
        freq=cfg.freq,
        one_dim_target=False,
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=cfg.num_samples,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    y_pred = []
    for i in range(cfg.num_samples):
        y_pred.append(forecasts[0].samples[i].transpose().tolist())
    y_pred = np.array(y_pred)
    return y_pred.mean(axis=0).T


def remove_diagonal(T):
    # Takes in 3 dim tensor and removes diagonal of 2/3 dim.
    out = []
    for x in T:
        out.append(x[~np.eye(x.shape[0], dtype=bool)].reshape(x.shape[0], -1))
    return np.stack(out)


def eval_with_custom_data(eval_func, predictor, data, test_window, cfg):
    pred = eval_func(
        predictor,
        data=data,
        cfg=cfg,
    )
    errors = [
        calc_error(test_window.values[:, x], pred[:, x], cfg)
        for x in range(test_window.shape[1])
    ]
    return pred, errors


def cdmi(original_data, predictor, eval_func, training_length, num_windows, cfg):
    """
    CDMI Wrapper to estimate the causal relationship
    via a trained model and data intervention
    """
    start = 0
    residual_stack = []
    residual_intervention_stack = []
    window_preds = []
    data_samples = []
    knockoffs = []
    # multiple prediction windows to construct the residual distribution.
    for iteration in range(num_windows):
        # print("Window:", iteration)
        # Generate an intervention for the window that includes ONLY the training interval and the testing interval.
        end = start + training_length + cfg.prediction_length
        test_window = original_data.iloc[end - cfg.prediction_length : end].copy()
        int_data = get_intervention(original_data.iloc[start:end].copy(), cfg)
        knockoffs.append(int_data)
        data_samples.append([int_data, original_data.iloc[start:end]])
        # We here perform an intervention on the variable i and observe residual changes.
        # Accuracy without the intervention.
        no_intervention = original_data.iloc[start:end].copy()
        pred, errors = eval_with_custom_data(
            eval_func, predictor, no_intervention, test_window, cfg
        )
        # Now we perform an intervention on the variable i and observe residual changes.
        intervention_error_stack = []
        intervention_prediction_stack = []
        for i in original_data.columns:
            # ONLY replace the current ts with the intervention.
            single_intervention = original_data.iloc[start:end].copy()
            single_intervention[i] = int_data[i]
            int_pred, int_errors = eval_with_custom_data(
                eval_func, predictor, single_intervention, test_window, cfg
            )
            intervention_error_stack.append(int_errors)
            intervention_prediction_stack.append(int_pred)

        # save changes
        window_preds.append([pred, intervention_prediction_stack, test_window.values])
        residual_intervention_stack.append(intervention_error_stack)
        residual_stack.append(errors)
        # step forward for new window
        start += cfg.step_size

    if cfg.save_intermediate:
        pickle.dump(knockoffs, open(cfg.save_path + "/knockoffs.p", "wb"))
        pickle.dump(residual_stack, open(cfg.save_path + "/default_resids.p", "wb"))
        pickle.dump(
            residual_intervention_stack,
            open(cfg.save_path + "/intervention_resids.p", "wb"),
        )
        pickle.dump(window_preds, open(cfg.save_path + "/pred.p", "wb"))
        pickle.dump(data_samples, open(cfg.save_path + "/intervention.p", "wb"))

    return construct_statistics(
        np.array(residual_stack), np.array(residual_intervention_stack), cfg
    )


def construct_statistics(residual_stack, residual_intervention_stack, cfg):
    # run the specified significance test for all combos.
    decision_matrix = np.zeros(residual_intervention_stack.shape[1:])

    for intervention in range(residual_intervention_stack.shape[1]):
        for effect in range(residual_intervention_stack.shape[1]):
            stat = test_significance(
                residual_stack[:, intervention],
                residual_intervention_stack[:, intervention, effect],
                cfg,
            )
            decision_matrix[effect, intervention] = stat
    if cfg.normalize_effect_strength:
        print("Normalizing effect strengths")
        decision_matrix = (decision_matrix - decision_matrix.min()) / (
            decision_matrix.max() - decision_matrix.min()
        )
    return decision_matrix


def calc_error(y_true, y_pred, cfg):
    if cfg.error_metric == "mape":
        error = mean_absolute_percentage_error(y_true, y_pred)
    elif cfg.error_metric == "mse":
        error = mean_squared_error(y_true, y_pred)
    elif cfg.error_metric == "mae":
        error = mean_absolute_error(y_true, y_pred)
    return error


def get_intervention(data, cfg):
    """
    int_data: pd.DataFrame
    i: int (knockoff variable)
    cfg: Hydra config
    """
    int_data = data.copy()
    # Generate sample specific intervention.
    if cfg.intervention_type == "knockoff":
        intervene = np.array(generate_knockoff(int_data.values))
        return pd.DataFrame(intervene, columns=int_data.columns, index=data.index)
    else:
        for col in int_data.columns:
            if cfg.intervention_type == "mean":
                int_data[col] = np.random.normal(
                    int_data[col].mean(), int_data[col].std(), len(int_data)
                )
            elif cfg.intervention_type == "normal":
                int_data[col] = np.random.normal(0, cfg.mean_std, len(int_data))
            elif cfg.intervention_type == "uniform":
                int_data[col] = np.random.uniform(
                    int_data[col].min(), int_data[col].max(), len(int_data)
                )
            elif cfg.intervention_type == "extreme":
                int_data[col] = np.random.normal(100, cfg.mean_std, len(int_data))
            else:
                raise ValueError("Unknown intervention type")
        return int_data


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def abs_residual_increase(y_true, y_pred):
    # The lower the value the higher the "significance" (as for p values)
    return np.abs(y_true).sum() - np.abs(y_pred).sum()


def test_significance(normal, inter, cfg):
    # The lower the higher the chance for a link
    if cfg.significance_test == "kolmo":
        _, stat = ks_2samp(normal, inter)
    elif cfg.significance_test == "kl_div":
        stat = kl_divergence(normal, inter)
    elif cfg.significance_test == "abs_error":
        stat = abs_residual_increase(normal, inter)

    else:
        print("STAT TEST UNKNOWN")
        stat = None
    return stat
