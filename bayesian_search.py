# System modules
import time
import traceback

# Maths modules
import numpy as np

# Optimization module
from hyperopt import fmin, tpe, Trials, space_eval, STATUS_OK, STATUS_FAIL, hp
from hyperopt.pyll import scope

# OD modules
from MM.init_param import init
from MM.error import error_a_opti as error
#from MM.error import error_b_opti as error

# Figures modules
from MM.display import main_plot_history, main_plot_histogram, main_plot_vars, main_plot_Bode_a
from MM.display import main_plot_Bode_a as main_plot_Bode


def get_search_space(f_min=1e-4, f_max=1e4, Ns=10):
    """Create the bayesian research's space.

    Parameters
    ----------
    f_min : float
        Lower bound of frequence to use.
    fmaw : float
        Upper bound of frequence to use.
    Np : int
        Pole's number.

    Returns
    -------
    space : dict
        All arguments with their reseach's spaces.

    Examples
    --------
    >>> # Choose in a finite list
    >>> 'arg1': hp.choice('arg1', [1, 2, 3])
    >>>
    >>> # Choose in uniform integer distribution
    >>> 'arg2': scope.int(hp.uniform('arg2', 1e1, 1e3))
    >>>
    >>> # Choose in log normal distribution
    >>> 'arg3': hp.lognormal('arg3', 1e-2, 1)
    >>>
    >>> # Choose in log uniform distribution
    >>> 'arg4': hp.loguniform('arg4', np.log(1e-5), np.log(1e-1))
    """

    space_rk = {'r_{}'.format(i):
                0.1*hp.loguniform('r_{}'.format(i),
                                  np.log(f_min), np.log(f_max))
                for i in range(Ns)}

    space_sk = {'s_{}'.format(i):
                -hp.loguniform('s_{}'.format(i),
                               np.log(f_min), np.log(f_max))
                for i in range(Ns)}

    # Concatenate this spaces
    space = {**space_rk, **space_sk}

    return space



def optimize(f_opti, space, max_evals=100, case=0, f_min=1e-4, f_max=1e4,
             Nf=1000, Ns=10):

    # Get model parameters
    M, N, L, Mp, Np, Lp, ainf, gam = init(case=case)
    param_th = [ainf, M, N, L]

    # range of frequencies
    freq = np.logspace(np.log10(f_min), np.log10(f_max), Nf)

    # Get approximation function
    f = f_opti(param_th=param_th, freq=freq)

    def objective(args, f=f, Ns=Ns):

        # Rebuild arrays
        rk = [args['r_{}'.format(i)] for i in range(Ns)]
        sk = [args['s_{}'.format(i)] for i in range(Ns)]

        # Main part - evaluation
        try:
            loss = f(mm_param=[rk, sk])
            status = STATUS_OK
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            loss = np.inf
            status = STATUS_FAIL

        return {'loss': loss, 'status': status}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals,
                trials=trials, rstate=np.random.RandomState(0))

    return space_eval(space, best), trials



if __name__ == "__main__":
    # Weight and pole number
    Ns = 3
    Nf = 1000
    f_min, f_max = [1e1,1e7]

    # Number of iterrations
    max_evals = 5e3

    # Logs
    print("", '#'*80, '#{:^78}#'.format("Starting"), '#'*80, "", sep='\n')
    start = time.time()

    # Run
    args, trials = optimize(
        f_opti    = error,
        space     = get_search_space(f_min=f_min, f_max=f_max, Ns=Ns),
        max_evals = max_evals,
        case      = 0,
        f_min     = f_min,
        f_max     = f_max,
        Nf        = Nf,
        Ns        = Ns,
    )

    # mm_param
    rk = []
    sk = []

    # Logs
    print("", '#'*80, '#{:^78}#'.format("End"), '#'+'='*78+"#", sep='\n')
    msg = "Elapsed time: {:.3f} s".format(time.time()-start)
    print('#{:^78}#'.format(msg), '#'+'='*78+"#", sep='\n')
    print('#{:^78}#'.format("Best parameters"))
    for key, value in args.items():
        msg = "{}: {}".format(key, value)
        if value > 0: rk.append(value)
        if value < 0: sk.append(value)
        print('#  - {:<74} #'.format(msg))
    msg = "{}: {:.3E}".format("Best error", trials.average_best_error())
    print('#'+'='*78+"#", '#{:^78}#'.format(msg), '#'*80, sep='\n')

    # Display results
    main_plot_history(trials=trials)
    main_plot_histogram(trials=trials)
    main_plot_vars(
        trials          = trials,
        columns         = Ns,
        arrange_by_loss = False,
        space           = get_search_space(f_min=f_min, f_max=f_max, Ns=Ns)
    )

    # Additional plot
    M, N, L, Mp, Np, Lp, ainf, gam = init(case=0)
    phy_param_a = [ainf, M, N, L]
    phy_param_b = [gam, Mp, Np, Lp]
    mm_param = [rk,sk]
    fr_param = [f_min,f_max, Nf]
    main_plot_Bode(mm_param, phy_param_a, fr_param)