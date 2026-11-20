'''
online_generative_forecasting.py
================================

Reference paper of RAM-ESN (please cite):
L. Pedrelli, S. Dettori, V. Colla, "Random Attention Memory for Online Generative
Reservoir Computing", 2026.

Online generative forecasting task of the paper, defined on top of the DeepESN Python Library
(a subclass of deepesn.tasks.Task)

----

Luca Pedrelli
luca.pedrelli@santannapisa.it
lucapedrelli@gmail.com

The Telecommunications, Computer Engineering, and Photonics Institute (TeCIP)
Scuola Superiore Sant'Anna in Pisa
ICT-COISP (ICT for Complex Industrial Systems and Processes)
https://www.santannapisa.it/it/istituto/tecip

----
'''

import numpy as np

# DeepESN Python Library (https://github.com/lucapedrelli/DeepESN). Please cite:
#   C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical
#   Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99.
from deepesn.tasks import Task, as_series


class OnlineGenerativeForecasting(Task):
    '''
    Online generative forecasting. After the training segment, the
    evaluation segment is read as a stream, one step at a time; at each step t:

    1. the model observes u(t): x(t) = F(x(t-1), u(t));
    2. if the readout is online (iNE, RLS), it is updated with the pair
       (x(t-1), u(t)), i.e. for one-step-ahead prediction;
    3. starting from x(t), the model performs a generative rollout of horizon
       steps, feeding back its predictions, which generates
       u(t+1), ..., u(t+horizon); the context of the model (state and memory
       buffers) is saved before and restored after the rollout.

    Rollouts start at t = 0, stride, 2 stride, ... < T_eval - horizon (all the
    inputs are observed and, with online readouts, used for the updates); the
    metrics are averaged over all the windows. online_update=False keeps an
    online readout fixed on the stream; None (default) updates only online
    readouts.

    'pred' and 'true' are arrays (windows, horizon, F).
    '''

    def __init__(self, train, validation=None, test=None, washout=0, metrics=('mse', 'mae'),
                 horizon=1, stride=1, online_update=None):
        super(OnlineGenerativeForecasting, self).__init__(train, validation, test, washout, metrics)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.online_update = online_update

    def _repr_params(self):
        return ', horizon=%d, stride=%d' % (self.horizon, self.stride)

    def run(self, model, train, evaluation):
        train, ev = as_series(train), as_series(evaluation)
        H, stride = self.horizon, self.stride
        states = model.compute_states(train.T)
        self._fit_readout(model, states, train)
        readout = model.readout
        update = self._online_update(model)

        preds, trues = [], []
        x = model.state
        for t in range(0, ev.shape[0] - H):
            # 1. observe u(t)
            x_prev = x
            x = model.step(ev[t])
            # 2. online update with (x(t-1), u(t))
            if update:
                readout.train([x_prev.reshape(-1, 1)], [ev[t:t+1].T])
            # 3. generative rollout from x(t): u(t+1), ..., u(t+H)
            if t % stride == 0:
                context = model.get_context()
                preds.append(model.generate(H))
                model.set_context(context)
                trues.append(ev[t+1:t+1+H])
        if not preds:
            raise ValueError('the evaluation segment is shorter than horizon + 1')
        pred = np.stack(preds, 0)
        true = np.stack(trues, 0)
        return self._result(pred, true)
