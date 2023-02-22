import torch
import numpy as np

from sklearn.neighbors import BallTree
from sklearn.utils import check_random_state

from aqua.models.modules.cincer.negsup.fisher import *


def get_margin(model, X, y, i):
    """Computes the margin of an example wrt a model."""
    x = X[i:i+1]
    phat = model.predict_proba(x)
    yhat = np.argmax(phat)
    return phat[0, yhat] - phat[0, y[i]]

def get_suspiciousness(model,
                       X_tr, y_tr,
                       kn,
                       i,
                       n_labels,
                       inspector,
                       **kwargs):
    """Computes the ``suspiciousness'' of an example.
    Arguments
    ---------
    model :
        The Model.
    X_tr, y_tr : ndarrays
        The dataset used for training the model.
    kn : list
        Indices of examples that the model knows within the training set.
    i : int
        The index of the target example within the target set.
    inspector : str, one of ['always', 'never', 'margin', 'influence']
        Method to be used.
    Return
    ------
    suspiciousness : float
        Measure of suspiciousness of the target example.
    """
    if inspector == 'always':
        return -np.inf
    elif inspector == 'never':
        return np.inf
    elif inspector == 'margin':
        return get_margin(model, X_tr, y_tr, i)
    elif inspector == 'gradient':
        raise NotImplementedError
        #return get_expected_gradient_len(model, X_tr, y_tr, i, n_labels)
    elif inspector == 'influence':
        raise NotImplementedError
        #return get_influence_on_params(model, X_tr, y_tr, kn, i, **kwargs)
    elif inspector == 'fisher':
        raise NotImplementedError
        #fi = _get_fi_vector(model, X_tr, y_tr, i)
        #return np.dot(fi, fi)
    else:
        raise ValueError(f'invalid suspiciousness inspector "{inspector}"')



def find_counterexample(model,
                        dataset,
                        kn, i,
                        negotiator,
                        if_config,
                        radius=None,
                        rng=None):
    """Computes a counter-example to a given example."""
    if rng is None:
        rng = check_random_state(rng)
    in_shape = (1,) + dataset.X_tr.shape[1:]

    # Select (indices of) candidates with the same annotated label as the predicted label of example i
    xi = dataset.X_tr[i].reshape(in_shape)
    phati = model.predict(xi)
    yhati = np.argmax(phati, axis=1)[0]

    candidates = []
    for k in sorted(set(kn) - {i}):
        yk = np.argmax(dataset.y_tr[k])
        if yk == yhati:
            candidates.append(k)
    n_candidates = len(candidates)
    assert n_candidates > 0, 'this is annoying'

    if negotiator == 'random':
        return rng.choice(candidates), None, candidates

    elif negotiator == 'nearest':
        X_candidates = dataset.X_tr[candidates].reshape(n_candidates, -1)

        tree = BallTree(X_candidates)
        indices_in_candidates = tree.query(dataset.X_tr[i].reshape(1, -1),
                                           return_distance=False,
                                           k=min(len(X_candidates), 25))[0]
        return candidates[indices_in_candidates[0]], None, indices_in_candidates

    elif negotiator == 'if':
        inf_model = InfluenceWithSTest(model,
                                       dataset.X_tr,
                                       dataset.y_tr,
                                       dataset.X_tr,
                                       dataset.y_tr,
                                       model.loss,
                                       **if_config)
        influences = np.array([
            inf_model.get_influence_on_prediction(k, i, known=kn)
            for k in candidates
        ])

        if False:
            print('influences =', influences)
        assert np.isfinite(influences).all(), \
            'IF is inf/nan! Use --bits 64, increase epochs or if-damping, decrease lissa-depth'
        argsort = np.argsort(influences)[::-1]
        ordered_candidates = np.array(candidates)[argsort]
        return candidates[np.argmax(influences)], None, ordered_candidates

    elif negotiator == 'nearest-if':
        X_candidates = dataset.X_tr[candidates].reshape(n_candidates, -1)

        # Identify the 100 closes examples
        tree = BallTree(X_candidates)
        n_neighbors = min(100, len(candidates))
        indices_in_candidates = tree.query(dataset.X_tr[i].reshape(1, -1),
                                           return_distance=False,
                                           k=n_neighbors)[0]
        closest_candidates = [candidates[l] for l in indices_in_candidates]

        # Sort them by influence
        inf_model = InfluenceWithSTest(model,
                                       dataset.X_tr,
                                       dataset.y_tr,
                                       dataset.X_tr,
                                       dataset.y_tr,
                                       model.loss,
                                       **if_config)
        influences = np.array([
            inf_model.get_influence_on_prediction(k, i, known=kn)
            for k in closest_candidates
        ])
        if False:
            print('influences =', influences)
        assert np.isfinite(influences).all(), \
            'IF is inf/nan! Use --bits 64, increase epochs or if-damping, decrease lissa-depth'
        return closest_candidates[np.argmax(influences)], None

    elif negotiator == 'nearest_fisher':
        # Identify the neighbours within a radius r
        tree = BallTree(dataset.X_tr[candidates].reshape(len(candidates), -1))
        indices_in_candidates = tree.query_radius(dataset.X_tr[i].reshape(1, -1),
                                                  radius,
                                                  return_distance=False)[0]
        if len(indices_in_candidates)==0:
            indices_in_candidates = tree.query(dataset.X_tr[i].reshape(1, -1),
                                               return_distance=False,
                                               k=1)[0]

        closest_candidates = [candidates[i] for i in indices_in_candidates]

        score = score_counterexamples_with_fisher_kernel(
            model,
            dataset,
            kn,
            i,
            closest_candidates,
            'top_fisher',
            damping=if_config['damping'],
            rng=rng
        )

        return closest_candidates[np.argmax(score)], None, None

    elif 'fisher' in negotiator or 'ce_removal' == negotiator:
        neg = 'top_fisher' if 'ce_removal' == negotiator else negotiator

        score = score_counterexamples_with_fisher_kernel(
            model,
            dataset,
            kn,
            i,
            candidates,
            neg,
            damping=if_config['damping'],
            rng=rng
        )
        argsort = np.argsort(score)[::-1]
        ordered_candidates = np.array(candidates)[argsort]
        return candidates[np.argmax(score)], np.amax(score), ordered_candidates

    else:
        raise ValueError(f'invalid negotiator {negotiator}')