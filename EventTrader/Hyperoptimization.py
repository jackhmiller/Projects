import xgboost as xgb
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK, STATUS_FAIL
from sklearn.metrics import f1_score
import numpy as np

xgb_reg_params = {
    'learning_rate':    hp.choice('learning_rate',    [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1]),
    'reg_alpha':        hp.choice('reg_alpha',        [0, 0.001, 0.005, 0.01, 0.05]),
    'reg_lambda':       hp.choice('reg_lambda',       np.arange(1, 2, .2, dtype=int)),
    'gamma':            hp.choice('gamma',            [i/10.0 for i in range(0, 5)]),
    'max_depth':        hp.choice('max_depth',        np.arange(3, 10, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample',       0.5, 1),
    'n_estimators':     100,
}


xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params


class HPOpt(object):

    def __init__(self, X, y, max_evals=500):
        self.X = X
        self.y = y
        self.xgb_para = xgb_para
        self.max_evals = max_evals

    def process(self):
        space = self.xgb_para

        try:
            result = fmin(fn=self.score, space=space, algo=tpe.suggest, max_evals=self.max_evals, trials=Trials())
        except Exception as e:
            raise Exception({'status': STATUS_FAIL, 'exception': str(e)})
        return result

    def score(self, para):
        model = xgb.XGBClassifier(**para['reg_params'])
        model.fit(self.X, self.y)

        loss = 1 - f1_score(self.y, model.predict(self.X))

        return {'loss': loss, 'status': STATUS_OK}

# fn = getattr(self, fn_name)
# fmin(fn=fn)
# rename score to- f1_score, pass it as a string to the process() call, then add to hyperparams