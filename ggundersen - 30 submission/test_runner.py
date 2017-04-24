"""Predict eviction response variable.
"""

import numpy as np
from sklearn import metrics


def predict(models, dataset, fname, n_tests=5, oversample=False):
    """Run predictions on all models in models.
    """
    with open(fname, 'w+') as f:
        f.write('Model, Non-defaults, MSE, Accuracy, Precision, Recall\n')
        for m in models:
            Constructor = m['constructor']
            if 'args' in m:
                model = Constructor(**m['args'])
                args_desc = str(m['args'])
            else:
                model = Constructor()
                args_desc = ''
            name = Constructor.__name__
            print('*' * 80)
            print('%s\n%s' % (name, args_desc))
            print('*' * 80)

            loss_avg, accs_avg, precs_avg, recs_avg = _predict(model, dataset,
                                                               n_tests, oversample)
            output = '%s, %s, %s, %s, %s, %s\n' % (name, args_desc, loss_avg,
                                                   accs_avg, precs_avg,
                                                   recs_avg)
            f.write(output)


def _predict(model, dataset, n_tests, oversample):
    """Return eviction predictions.
    """
    losses = []
    accs   = []
    precs  = []
    recs   = []
    for i in range(n_tests):
        if oversample:
            dataset.split_then_oversample()
        else:
            dataset.split()
        fitted = model.fit(dataset.X_train, dataset.y_train)
        predictions = fitted.predict(dataset.X_test)
        print('Evaluating %s (%s / %s)' % (fitted.__class__.__name__, i + 1, n_tests))
        loss, acc, prec, rec = evaluate(dataset.y_test, predictions)
        print('=' * 80)
        losses.append(loss)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)

    loss_avg = np.array(losses).mean()
    accs_avg = np.array(accs).mean()
    precs_avg = np.array(precs).mean()
    recs_avg = np.array(recs).mean()

    print('Loss average: %s' % loss_avg)
    print('Accuracy average: %s' % accs_avg)
    print('Precision average: %s' % precs_avg)
    print('Recall average: %s' % recs_avg)

    print('*' * 80)

    return loss_avg, accs_avg, precs_avg, recs_avg


def evaluate(targs, preds):
    loss = metrics.mean_squared_error(targs, preds)
    acc  = metrics.accuracy_score(targs, preds)
    prec = metrics.precision_score(targs, preds)
    rec  = metrics.recall_score(targs, preds)
    print('Loss: %s' % loss)
    print('Accuracy: %s' % acc)
    print('Precision: %s' % prec)
    print('Recall: %s' % rec)
    return loss, acc, prec, rec