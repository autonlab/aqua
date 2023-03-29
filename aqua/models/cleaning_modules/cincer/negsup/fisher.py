import torch
import numpy as np

def _get_fi_vector(model, X, y, i, label=None, return_prob=False, flatten=True):
    """Computes ∇θ log P(y=label else y_i | x_i)."""
    x = X[i]

    if label is None:
        label = torch.argmax(y[i])

    phat_label = model(x[None, :])[0, label]
    conditional_log_likelihood = torch.log(phat_label)

    print(conditional_log_likelihood.grad)
    print(phat_label.grad)
    
    conditional_log_likelihood.backward()

    print(conditional_log_likelihood.grad)
    print(phat_label.grad)
    raise KeyboardInterrupt

    # if flatten:
    #     gradient = np.concatenate(
    #         [torch.reshape(t, [-1]) for t in gradient]
    #     )

    if return_prob:
        return gradient, phat_label
    else:
        return gradient



def score_counterexamples_with_fisher_kernel(
        model,
        dataset,
        kn,
        i,
        candidates,
        negotiator,
        damping=0.0,
        rng=None
):
    X, y, n_classes = dataset.X_tr, dataset.y_tr, dataset.n_classes

    if negotiator == 'practical_fisher':
        fi = _get_fi_vector(model, X, y, i)

        kernel = [
            np.dot(fi, _get_fi_vector(model, X, y, j))
            for j in candidates
        ]

    elif negotiator == 'approx_fisher':
        fi = _get_fi_vector(model, X, y, i)

        inv_diag_fim = _get_inv_diagonal_fim(model, kn, X, y, n_classes)
        fi_times_inv_fim = inv_diag_fim * fi

        kernel = [
            np.dot(fi_times_inv_fim, _get_fi_vector(model, X, y, j))
            for j in candidates
        ]

    elif negotiator == 'top_fisher':
        top_fi = _get_top_fi_vector(model, X, y, i)

        # NOTE use Moore-Penrose pseudo-inverse to avoid issues with singular matrices
        top_fim = _get_top_fim(model, kn, X, y, n_classes, rng=rng)
        inv_top_fim = np.linalg.pinv(top_fim, hermitian=True)

        top_fi_times_inv_top_fim = np.dot(inv_top_fim, top_fi)

        kernel = [
            np.dot(
                top_fi_times_inv_top_fim,
                _get_top_fi_vector(model, X, y, j)
            )
            for j in candidates
        ]

    elif negotiator == 'full_fisher':
        fi = _get_fi_vector(model, X, y, i)

        fim = _get_fim(model, kn, X, y, n_classes)
        inv_fim = np.linalg.pinv(fim, hermitian=True)

        fi_times_inv_fim = np.dot(inv_fim, fi)

        kernel = [
            np.dot(fi_times_inv_fim, _get_fi_vector(model, X, y, j))
            for j in candidates
        ]

    elif negotiator == 'block_fisher':
        fi = _get_fi_vector(model, X, y, i, flatten=False)
        fi = _tl_tonp(fi, dtype=np.float32)

        print('computing block FIM')
        block_fim = _get_block_fim(model, kn, X, y, n_classes, rng=rng)

        preconditioned_block_fim = [
            np.array(t) + damping * np.eye(len(t), dtype=t.dtype)
            for t in block_fim
        ]

        print('inverting block FIM')
        if False:
            fi_times_inv_fim = _cg(
                lambda v: _tl_bmvp(preconditioned_block_fim, v),
                fi,
                tol=1e-5,
                atol=1e-8
            )

        else:
            inv_block_fim = [
                np.linalg.inv(t).astype(np.float32)
                for t in preconditioned_block_fim
            ]
            fi_times_inv_fim = _tl_bmvp(inv_block_fim, fi)

        # Computes the (block-wise) Fisher kernel
        print('scoring candidates')
        kernel = [
            _tl_dot(
                fi_times_inv_fim,
                _tl_tonp(
                    _get_fi_vector(model, X, y, j, flatten=False),
                    dtype=np.float32
                )
            )
            for j in candidates
        ]
    else:
        raise ValueError(negotiator)
    return kernel