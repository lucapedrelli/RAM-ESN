'''
Reference paper of RAM-ESN (please cite):
L. Pedrelli, S. Dettori, V. Colla, "Random Attention Memory for Online Generative
Reservoir Computing", 2026.

RAM-ESN layer: Echo State Network with Random Attention Memory, implemented as a
layer of the DeepESN Python Library (a subclass of deepesn.layers.Layer), so that it can
be used in a DeepESN model like the layers of the DeepESN Python Library:

    from deepesn import DeepESN, Readout
    from ramesn_layer import RAMESNLayer
    model = DeepESN(1, layers=[RAMESNLayer(100, memory=151)], readout=Readout('iNE'), seed=0)

Each neuron attends, through fixed random attention weights, to a window of
its own m past states, kept in a ring buffer of m+1 slots:

    c_i(t) = sum_{k=1..m} a_{i,k} x_i(t-k)              memory trace
    v_i(t) = w_i c_i(t) + (1 - w_i) x_i(t-1)            gated memory

    x(t) = tanh(W v(t) + Win [u(t); 1])                 RAM-ESN (variant IN-v)

with a_{i,k} = softmax_k(z_{i,k} / tau), z_{i,k} ~ N(0, 1), and gates
w_i ~ U(0, 1), all fixed and untrained.

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
from deepesn.layers import Layer


class RAMESNLayer(Layer):
    '''
    Random Attention Memory Echo State Network (RAM-ESN) layer.

    State update of the variants (w: vector of the gates, c(t) memory trace,
    v(t) gated memory, see the module docstring):

        IN-v  (RAM-ESN)  x(t) = tanh(W v(t) + Win u(t))
        IN-c             x(t) = tanh(W c(t) + Win u(t))
        OUT-v            x(t) = w * tanh(W x(t-1) + Win u(t)) + (1 - w) * v(t)
        OUT-c            x(t) = w * tanh(W x(t-1) + Win u(t)) + (1 - w) * c(t)
        LI-v             x(t) = w * tanh(W v(t) + Win u(t)) + (1 - w) * x(t-1)
        LI-c             x(t) = w * tanh(W c(t) + Win u(t)) + (1 - w) * x(t-1)

    (Win u(t) includes the bias). The variants can also be given by their
    number 1..6 (in the order above, as in the RAM-ESN paper and code).

    Initialization: W is drawn uniformly in (-1, 1) and rescaled to spectral
    radius rho(W) = spectral_radius, Win uniformly in
    (-input_scaling, input_scaling); then max_memory attention logits per
    neuron are drawn from N(0, 1) (the window of m steps uses the first m, so
    that, for a given seed, models with different m share all the random
    weights) and the gates from U(0, 1). All the variants share the same
    random weights.

    Sparse attention: only the attention weights larger than threshold are
    kept (a_{i,k} > threshold, threshold = 0: full attention), stored as the
    arrays sparse_index (column k-1 of the window for the delay k) and
    sparse_weights, zero padded to the maximum number of kept delays.

    States before the beginning of a sequence are equal to the initial state.
    Online (step), the past states are read from a ring buffer of m+1 slots,
    aligned with a batch sequence by set_history and saved/restored, together
    with the state, by get_context/set_context.

    Parameters
    ----------
    units : number of recurrent units
    spectral_radius : spectral radius of W
    input_scaling : scale of the input weights (and bias)
    memory : window length m of the attention
    temperature : softmax temperature tau of the attention
    variant : 'IN-v' (default, RAM-ESN), 'IN-c', 'OUT-v', 'OUT-c', 'LI-v', 'LI-c'
    threshold : attention weights kept by the sparse implementation (> threshold)
    max_memory : number of attention logits drawn per neuron (>= memory)
    connectivity : fraction of non-zero recurrent weights per row (1 = dense)
    '''

    VARIANTS = ('IN-v', 'IN-c', 'OUT-v', 'OUT-c', 'LI-v', 'LI-c')

    def __init__(self, units=100, spectral_radius=0.9, input_scaling=1.0, memory=200,
                 temperature=1e-4, variant='IN-v', threshold=1e-4, max_memory=500,
                 connectivity=1.0):
        super(RAMESNLayer, self).__init__(units, spectral_radius, input_scaling, connectivity)
        self.memory = int(memory)
        self.temperature = float(temperature)
        self.variant = self.variant_name(variant)
        self.threshold = float(threshold)
        self.max_memory = int(max_memory)
        if not 1 <= self.memory <= self.max_memory:
            raise ValueError('memory must be in [1, max_memory] (memory=%d, max_memory=%d)'
                             % (self.memory, self.max_memory))
        self._code = self.VARIANTS.index(self.variant) + 1
        self.logits = None            # (units, max_memory) attention logits
        self.attention = None         # (units, memory) dense attention weights a_{i,k}
        self.sparse_index = None      # (units, s_max) kept delays (column k-1 for delay k)
        self.sparse_weights = None    # (units, s_max) kept attention weights (zero padded)
        self._gates = None            # (units, 1) memory gates w_i
        self._ring = None             # (memory+1, units) ring buffer of the past states
        self._ptr = 0                 # write pointer of the ring buffer

    @classmethod
    def variant_name(cls, variant):
        '''Canonical name of a variant given by name or number (1..6).'''
        if isinstance(variant, (int, np.integer)) and not isinstance(variant, bool):
            if 1 <= variant <= len(cls.VARIANTS):
                return cls.VARIANTS[variant - 1]
        else:
            key = str(variant).strip().upper().replace('_', '-')
            for name in cls.VARIANTS:
                if name.upper() == key:
                    return name
        raise ValueError('unknown RAM-ESN variant %r: use one of %s or 1..6'
                         % (variant, ', '.join(cls.VARIANTS)))

    def _repr_params(self):
        p = super(RAMESNLayer, self)._repr_params()
        p.update(memory=self.memory, temperature=self.temperature, variant=repr(self.variant))
        return p

    # ------------------------------------------------------------------
    # attention and gates
    # ------------------------------------------------------------------
    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self, value):
        self._gates = value
        self._w = np.asarray(value).ravel()          # (units,) used by step()

    @property
    def buffer(self):
        '''Ring buffer (memory+1, units) of the online context and its write pointer.'''
        return self._ring, self._ptr

    @property
    def max_kept_delays(self):
        '''s_max: maximum number of attention weights kept per neuron.'''
        return self.sparse_index.shape[1]

    def _build_attention(self):
        # fixed attention: softmax of the first m logits at temperature tau
        Nr, H = self.units, self.memory
        logit = self.logits[:, :H]                                 # (Nr, H)
        logit = logit - logit.max(axis=1, keepdims=True)
        e = np.exp(logit / self.temperature)
        A = e / e.sum(axis=1, keepdims=True)                       # (Nr, H)
        self.attention = A
        # sparse attention: weights above the threshold, zero padded
        idx_list = []
        for i in range(Nr):
            idx = np.where(A[i] > self.threshold)[0].astype(np.int64)
            idx_list.append((idx, A[i, idx]))
        k_max = max(len(x[0]) for x in idx_list)
        sp_idx = np.zeros((Nr, k_max), dtype=np.int64)
        sp_w = np.zeros((Nr, k_max), dtype=np.float64)
        for i, (idx, w) in enumerate(idx_list):
            sp_idx[i, :len(idx)] = idx
            sp_w[i, :len(idx)] = w
        self.sparse_index = sp_idx
        self.sparse_weights = sp_w
        # cached for step()
        self._ar = np.arange(Nr)
        self._ar_col = self._ar[:, None]
        self._sidx_1d = sp_idx[:, 0] if k_max == 1 else None

    def set_attention(self, memory=None, temperature=None):
        '''
        Change the memory window m and/or the temperature tau of a built layer,
        keeping all its random weights (W, Win, attention logits, gates): the
        attention of the window m is the softmax of the first m logits, so the
        layer is the same as one built with that window and temperature and the
        same seed (e.g. to compare several m on the same network without
        rebuilding it). The online context is reset (the ring buffer has m+1 slots).
        '''
        memory = self.memory if memory is None else int(memory)
        if not 1 <= memory <= self.max_memory:
            raise ValueError('memory must be in [1, max_memory] (memory=%d, max_memory=%d)'
                             % (memory, self.max_memory))
        self.memory = memory
        if temperature is not None:
            self.temperature = float(temperature)
        if self.logits is not None:
            self._build_attention()
            if self.built:
                self.reset()
        return self

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def _rescale(self):
        # rho(W) = spectral_radius
        actual_rho = self._max_abs_eig(self.W)
        self.W = (self.W * self.spectral_radius) / actual_rho

    def _init_extra(self, rng):
        # same random sequence for all the variants and all m <= max_memory
        self.logits = rng.np.standard_normal(size=(self.units, self.max_memory))
        self._build_attention()
        self.gates = rng.np.uniform(0.0, 1.0, size=(self.units, 1))

    # ------------------------------------------------------------------
    # effective spectral radius
    # ------------------------------------------------------------------
    def lag_profile(self):
        '''Attention profile (memory, units) of the dynamics: element [k, i] is
        the weight of the delay k+1 for the neuron i (kept sparse weights).'''
        m, Nr = self.memory, self.units
        A = np.zeros((m, Nr))
        for i in range(Nr):
            np.add.at(A[:, i], self.sparse_index[i], self.sparse_weights[i])   # robust to the zero padding
        return A

    def companion_matvec(self):
        '''(mv, n): matrix-free product mv(z) = M0 z with the companion matrix M0
        (n = memory * units) of the dynamics linearized at the origin, on the
        stacked past states z = [x(t-1); ...; x(t-m)]. With d the delay profile
        of the memory read inside the variant (v: d_1 = (1-w) + w a_1,
        d_k = w a_k; c: d_k = a_k):
            IN   x(t) = W sum_k diag(d_k) x(t-k)
            OUT  x(t) = diag(w) W x(t-1) + sum_k diag((1-w) d_k) x(t-k)
            LI   x(t) = diag(w) W sum_k diag(d_k) x(t-k) + diag(1-w) x(t-1)'''
        Nr, m = self.units, self.memory
        W = self.W
        A = self.lag_profile()                                     # (m, Nr)
        w = np.asarray(self._gates, dtype=float).ravel()
        family, read = self.variant.split('-')
        if read == 'v':
            d = A * w[None, :]
            d[0] += (1.0 - w)
        else:
            d = A
        if family == 'IN':
            def mv(z):
                Z = z.reshape(m, Nr)
                top = W @ (d * Z).sum(axis=0)
                return np.concatenate([top, Z[:-1].ravel()])
        elif family == 'OUT':
            Dw = (1.0 - w)[None, :] * d
            def mv(z):
                Z = z.reshape(m, Nr)
                top = w * (W @ Z[0]) + (Dw * Z).sum(axis=0)
                return np.concatenate([top, Z[:-1].ravel()])
        else:   # LI
            def mv(z):
                Z = z.reshape(m, Nr)
                top = w * (W @ (d * Z).sum(axis=0)) + (1.0 - w) * Z[0]
                return np.concatenate([top, Z[:-1].ravel()])
        return mv, m * Nr

    def effective_spectral_radius(self, dense_max=2500, iters=(3000, 9000), seed=1, x0=None,
                                  return_vector=False):
        '''rho_eff = rho(M0), spectral radius of the companion matrix of the
        linearized dynamics (see companion_matvec), on the actual weights of the
        layer (W, kept attention weights, gates). Dense eigenvalues for
        memory * units <= dense_max; otherwise power iteration with the
        estimator of the log growth rate (iters = (warm-up, averaged)
        iterations; x0: starting vector, for warm starts; return_vector: also
        return the last vector).'''
        mv, n = self.companion_matvec()
        if n <= dense_max:
            M = np.empty((n, n))
            e = np.zeros(n)
            for j in range(n):
                e[j] = 1.0
                M[:, j] = mv(e)
                e[j] = 0.0
            r = float(np.max(np.abs(np.linalg.eigvals(M))))
            return (r, None) if return_vector else r
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n) if x0 is None else np.array(x0, dtype=float)
        x /= np.linalg.norm(x)
        K_warm, K_avg = iters
        logs = 0.0
        for i in range(K_warm + K_avg):
            y = mv(x)
            nrm = np.linalg.norm(y)
            if nrm == 0.0:
                return (0.0, x) if return_vector else 0.0
            x = y / nrm
            if i >= K_warm:
                logs += np.log(nrm)
        r = float(np.exp(logs / K_avg))
        return (r, x) if return_vector else r

    # ------------------------------------------------------------------
    # dynamics: batch
    # ------------------------------------------------------------------
    def compute_states(self, inputs, initial_state=None):
        x0 = self._initial_column(initial_state)
        input = self.Win[:, 0:-1].dot(inputs) + np.expand_dims(self.Win[:, -1], 1)
        T = input.shape[1]
        W = self.W
        wm = self._gates                                           # (Nr, 1)
        sp_idx, sp_w = self.sparse_index, self.sparse_weights
        _ar = self._ar
        init_val = x0[:, 0:1].repeat(sp_idx.shape[1], axis=1)      # states before the start
        state = np.zeros((self.units, T))

        if self._code == 1:
            # IN-v (RAM-ESN); first step: x(0) = tanh(W x_init + Win u(0))
            state[:, 0:1] = np.tanh(W.dot(x0) + input[:, 0:1])
            for t in range(1, T):
                ring_idx = t - 1 - sp_idx                          # (Nr, s_max)
                valid_mask = ring_idx >= 0
                clipped = np.clip(ring_idx, 0, t - 1)
                past = state[_ar[:, None], clipped]                # (Nr, s_max)
                past = np.where(valid_mask, past, init_val)
                M = (past * sp_w).sum(axis=1, keepdims=True)       # c(t)
                X_prev = state[:, t-1:t]
                V = wm * M + (1.0 - wm) * X_prev                   # v(t)
                state[:, t:t+1] = np.tanh(W.dot(V) + input[:, t:t+1])
        else:
            code = self._code
            for t in range(T):
                if t == 0:
                    past = init_val
                    X_prev = x0[:, 0:1]
                else:
                    ring_idx = t - 1 - sp_idx
                    past = np.where(ring_idx >= 0,
                                    state[_ar[:, None], np.clip(ring_idx, 0, t - 1)],
                                    init_val)
                    X_prev = state[:, t-1:t]
                M = (past * sp_w).sum(axis=1, keepdims=True)       # c(t)
                u_t = input[:, t:t+1]
                if code == 2:      # IN-c
                    x_t = np.tanh(W.dot(M) + u_t)
                elif code == 3:    # OUT-v
                    V = wm * M + (1.0 - wm) * X_prev
                    x_t = wm * np.tanh(W.dot(X_prev) + u_t) + (1.0 - wm) * V
                elif code == 4:    # OUT-c
                    x_t = wm * np.tanh(W.dot(X_prev) + u_t) + (1.0 - wm) * M
                elif code == 5:    # LI-v
                    V = wm * M + (1.0 - wm) * X_prev
                    x_t = wm * np.tanh(W.dot(V) + u_t) + (1.0 - wm) * X_prev
                else:              # LI-c
                    x_t = wm * np.tanh(W.dot(M) + u_t) + (1.0 - wm) * X_prev
                state[:, t:t+1] = x_t
        return state

    # ------------------------------------------------------------------
    # dynamics: online, with the ring buffer
    # ------------------------------------------------------------------
    def reset(self, initial_state=None):
        x0 = self._initial_column(initial_state)[:, 0]
        self._ring = np.tile(x0, (self.memory + 1, 1))             # all past states = x_init
        self._ptr = 0
        self._x = x0.copy()

    def set_history(self, states):
        # last m+1 states of the sequence in the ring buffer, oldest first
        Hbuf = self.memory + 1
        seg = states[:, max(0, states.shape[1] - Hbuf):]
        n = seg.shape[1]
        ring = np.zeros((Hbuf, self.units))
        ring[:n] = seg.T
        self._ring = ring
        self._ptr = n % Hbuf
        self._x = states[:, -1].copy()

    def get_context(self):
        return (self._x.copy(), self._ring.copy(), self._ptr)

    def set_context(self, context):
        x, ring, ptr = context
        self._x = np.array(x, dtype=np.float64)
        if self._ring is not None and self._ring.shape == ring.shape:
            np.copyto(self._ring, ring)
        else:
            self._ring = np.array(ring, dtype=np.float64)
        self._ptr = int(ptr)

    def step(self, u):
        Hbuf = self.memory + 1
        p = self._ptr
        ring = self._ring
        x_prev = self._x
        wm = self._w
        code = self._code
        if code == 1:
            # IN-v (RAM-ESN): one gather of the attended past states
            if self._sidx_1d is not None:
                M = ring[(p - 1 - self._sidx_1d) % Hbuf, self._ar]
            else:
                vals = ring[(p - 1 - self.sparse_index) % Hbuf, self._ar_col]
                M = (vals * self.sparse_weights).sum(axis=1)
            V = wm * M + (1.0 - wm) * x_prev
            x = np.tanh(self.W.dot(V) + self._Win_u.dot(u) + self._Win_b)
        else:
            M = (ring[(p - 1 - self.sparse_index) % Hbuf, self._ar_col] * self.sparse_weights).sum(axis=1)
            u_in = self._Win_u.dot(u) + self._Win_b
            W = self.W
            if code == 2:      # IN-c
                x = np.tanh(W.dot(M) + u_in)
            elif code == 3:    # OUT-v
                V = wm * M + (1.0 - wm) * x_prev
                x = wm * np.tanh(W.dot(x_prev) + u_in) + (1.0 - wm) * V
            elif code == 4:    # OUT-c
                x = wm * np.tanh(W.dot(x_prev) + u_in) + (1.0 - wm) * M
            elif code == 5:    # LI-v
                V = wm * M + (1.0 - wm) * x_prev
                x = wm * np.tanh(W.dot(V) + u_in) + (1.0 - wm) * x_prev
            else:              # LI-c
                x = wm * np.tanh(W.dot(M) + u_in) + (1.0 - wm) * x_prev
        ring[p] = x
        self._ptr = (p + 1) % Hbuf
        self._x = x
        return x
