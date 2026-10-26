"""Lowest-Landau-level engine for monodromy defects on the fuzzy sphere.

A monodromy line through the origin pierces each sphere of radial quantization at its two poles. Giving a fermion
flavour of charge q an Aharonov-Bohm flux q*alpha through the poles shifts its angular momenta to
m in (-s + {q alpha} + Z) cap [-s, s], which keeps only the orbitals regular at both poles. Every interaction matrix
element is then a closed-form Beta-function integral of four orbitals.

Contents, in order:
  the store, sections and the command line;
  Wigner symbols, Haldane pseudopotentials and fast Fock bases (fs/);
  the Ising model of Zhu et al. with the Z2 twist (twist.py), its matrix elements and mirror parity;
  shared estimator helpers (defect_corr.py);
  sections: the O(M) models of Guo et al. (onmodel), the spin-one O(2) model of Dey et al. (o2twist), bulk tuning and
  defect levels of the vector models (run_on_defect), the regulator's local potential (core_strip), the anisotropy
  flows (o3flow, gflow), and the self-tests (test_twist, test_o2).

usage: python fuzzy.py test        runs the self-tests
"""

# ====================================================================================================
# Infrastructure: the data store, sections and the command line
# ====================================================================================================
import os
import sys
import json
import types
import fnmatch
import glob
import functools
import tempfile
import fcntl

ROOT = os.path.dirname(os.path.abspath(__file__))


def _jsonable(x):
    import numpy as np
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


class Store:
    """All stored results, one JSON file per physics question (01_known_results/*.json, 02_rg_flow/*.json).

    Each file maps keys to results. A key is the path the result had in the research tree where the code was
    written (twist/... for the fixed-point defects, flow/... for the flows), so the code reads and writes the same
    names as before. A .json key holds a JSON object, a .log or .txt key holds text. New keys go to the file of the
    question being run."""

    PREFIXES = ('twist/', 'flow/', 'fun/', 'analytic/', 'largeN/', 'junction_eps2/', 's4d/')

    def __init__(self):
        self.target = os.environ.get('FUZZY_TARGET')
        self._data, self._where = None, None
        self.base = None        # a question whose scripts used paths relative to one research folder sets it

    def _load(self):
        if self._data is None:
            self._data, self._where = {}, {}
            for f in sorted(glob.glob(os.path.join(ROOT, '0*', '*.json'))):
                self._data[f] = json.load(open(f))
                for k in self._data[f]:
                    self._where[k] = f

    def key(self, k):
        k = os.path.normpath(str(k)).replace(os.sep, '/')
        if self.base and not k.startswith(self.PREFIXES):
            k = self.base + k
        return k

    def get(self, k):
        self._load()
        k = self.key(k)
        if k not in self._where:
            raise FileNotFoundError(k)
        return self._data[self._where[k]][k]

    text = get

    def exists(self, k):
        self._load()
        return self.key(k) in self._where

    def glob(self, pattern):
        self._load()
        p = self.key(pattern)
        return sorted(k for k in self._where if fnmatch.fnmatchcase(k, p))

    def put(self, k, value):
        self._load()
        k = self.key(k)
        assert k.startswith(self.PREFIXES), \
            f'key {k} is not a research path'
        f = self._where.get(k) or self.target
        if f is None:
            raise RuntimeError(f'no question file for new key {k}; run this step from its question file')
        lock = os.path.join(tempfile.gettempdir(), 'fuzzy_store_' + os.path.basename(f) + '.lock')
        with open(lock, 'w') as fl:
            fcntl.flock(fl, fcntl.LOCK_EX)
            d = json.load(open(f)) if os.path.exists(f) else {}
            d[k] = json.loads(json.dumps(value, default=_jsonable))
            tmp = f + '.tmp'
            with open(tmp, 'w') as fh:
                json.dump(d, fh, indent=1, sort_keys=True)
            os.replace(tmp, f)
            fcntl.flock(fl, fcntl.LOCK_UN)
        self._data[f] = d
        self._where[k] = f


store = Store()


class Namespace:
    """The names of a section. Like a module: setting an attribute (M.Flavours = Patched) also rebinds the name for
    the section's own functions, and names the section imported from its file are reachable as attributes."""

    def __init__(self, names, file_globals):
        self.__dict__.update(names)
        cells = {}
        for v in names.values():
            fns = [v] if isinstance(v, types.FunctionType) else []
            if isinstance(v, type):
                for x in vars(v).values():
                    x = getattr(x, '__func__', x)
                    if isinstance(x, types.FunctionType):
                        fns.append(x)
            for f in fns:
                for name, c in zip(f.__code__.co_freevars, f.__closure__ or ()):
                    cells.setdefault(name, []).append(c)
        self.__dict__['_cells'], self.__dict__['_globals'] = cells, file_globals

    def __setattr__(self, k, v):
        self.__dict__[k] = v
        for c in self._cells.get(k, ()):
            c.cell_contents = v

    def __getattr__(self, k):
        if k in self.__dict__['_globals']:
            return self.__dict__['_globals'][k]
        raise AttributeError(k)


def namespace(names, file_globals):
    return Namespace(names, file_globals)


def section(fn):
    """A section holds one original script. Its body runs once, on first use, and its names are returned as a
    namespace, so sections never share global names."""
    cache = {}

    @functools.wraps(fn)
    def get():
        if 'ns' not in cache:
            cache['ns'] = fn()
        return cache['ns']
    return get


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)

    def flush(self):
        for s in self.streams:
            s.flush()


class logged:
    """with logged(key): ... prints as usual and stores everything printed under the text key (a .log path), which is
    how the analytic scripts kept their results."""

    def __init__(self, key):
        self.key = key

    def __enter__(self):
        import io
        self.buf, self.out = io.StringIO(), sys.stdout
        sys.stdout = _Tee(self.out, self.buf)
        return self

    def __exit__(self, *exc):
        sys.stdout = self.out
        if exc[0] is None:
            store.put(self.key, self.buf.getvalue())


_questions = {}


def question(rel):
    """The sections of another question file, for example question('04_pinned_lines/junctions').theta_on()."""
    import importlib.util
    if rel not in _questions:
        spec = importlib.util.spec_from_file_location('q_' + rel.replace('/', '_'), os.path.join(ROOT, rel + '.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _questions[rel] = mod
    return _questions[rel]


def npz_put(key, **arrays):
    """np.savez(file, **arrays), into the store: a .npz key holds a dict of lists."""
    import numpy as np
    store.put(key, {k: np.asarray(v).tolist() for k, v in arrays.items()})


def npz_load(key):
    """np.load(file) of a stored .npz key: a dict of arrays."""
    import numpy as np
    return {k: np.asarray(v) for k, v in store.get(key).items()}


def figpath(path):
    """Figures go to figs/ next to the question file."""
    d = os.path.join(os.path.dirname(store.target or ROOT), 'figs')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, os.path.basename(path))


def cli(file, steps, report=()):
    """python <question>.py <step> [args]: run one step; 'report' prints the paper's numbers from the stored data."""
    store.target = os.path.splitext(os.path.abspath(file))[0] + '.json'
    os.environ['FUZZY_TARGET'] = store.target
    argv = sys.argv[1:]
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print((sys.modules['__main__'].__doc__ or '').strip() + '\n\nsteps:')
        for name, s in steps.items():
            doc = (s.__wrapped__.__doc__ or '').strip().splitlines()
            print(f'  {name:22s} {doc[0] if doc else ""}')
        print(f'  {"report":22s} runs ' + ', '.join(' '.join(r) for r in report))
        return
    if argv[0] == 'report':
        for r in report:
            print(f'\n======== {" ".join(r)}', flush=True)
            sys.argv = list(r)
            steps[r[0]]().cli(list(r))
        return
    sys.argv = argv
    steps[argv[0]]().cli(argv)


# ----------------------------------------------------------------------------------------------------
# from fs/wigner.py
"""Exact Wigner 3j / 6j symbols, memoized, returned as floats.

Half-integer arguments are passed around as ``Fraction`` or float; internally we use
sympy's exact rationals so that no precision is lost in the 3j recursion, which matters
because ``haldane_V`` sums many near-cancelling terms.
"""
from fractions import Fraction
from functools import lru_cache

from sympy import Rational
from sympy.physics.wigner import wigner_3j as _w3j, wigner_6j as _w6j


def _rat(x):
    """Coerce a half-integer to an exact sympy Rational."""
    fr = Fraction(x).limit_denominator(2)
    return Rational(fr.numerator, fr.denominator)


@lru_cache(maxsize=None)
def _w3j_cached(j1, j2, j3, m1, m2, m3):
    return float(_w3j(*(_rat(v) for v in (j1, j2, j3, m1, m2, m3))))


@lru_cache(maxsize=None)
def _w6j_cached(j1, j2, j3, j4, j5, j6):
    return float(_w6j(*(_rat(v) for v in (j1, j2, j3, j4, j5, j6))))


def w3j(j1, j2, j3, m1, m2, m3):
    """Wigner 3j symbol as a float, with the usual selection rules applied first."""
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if abs(m1 + m2 + m3) > 1e-12:
        return 0.0
    if not (abs(j1 - j2) - 1e-12 <= j3 <= j1 + j2 + 1e-12):
        return 0.0
    return _w3j_cached(j1, j2, j3, m1, m2, m3)


def w6j(j1, j2, j3, j4, j5, j6):
    """Wigner 6j symbol as a float."""
    return _w6j_cached(j1, j2, j3, j4, j5, j6)


# ----------------------------------------------------------------------------------------------------
# from fs/pseudo.py
"""Haldane pseudopotentials on the sphere and the four-index interaction tensor.

Conventions follow 2210.13482 Eq. (15) and Appendix A:

    V_{m1 m2 m3 m4} = sum_l V_l (2L+1) 3j(s,s,L; m1,m2,-m1-m2) 3j(s,s,L; m4,m3,-m3-m4)

with ``L = 2s - l`` the *total* pair angular momentum and ``l`` the Haldane relative
index, so ``V_0`` is the closest-approach channel.
"""
import numpy as np



def haldane_V(s, Vl):
    """Dense interaction tensor ``V[a1,a2,a3,a4]`` in orbital indices ``a = m + s``.

    Parameters
    ----------
    s : float
        Monopole strength; the LLL holds ``2s+1`` orbitals.
    Vl : sequence of float
        Haldane pseudopotentials ``V_0, V_1, ...`` (relative-angular-momentum index).

    Returns
    -------
    ndarray, shape (2s+1,)*4
        Zero wherever ``m1 + m2 != m3 + m4``.
    """
    norb = int(round(2 * s)) + 1
    ms = np.array([a - s for a in range(norb)])
    V = np.zeros((norb,) * 4)

    # Pair-channel form factors  F_L[a1,a2] = 3j(s,s,L; m1,m2,-m1-m2)
    chans = []
    for l, vl in enumerate(Vl):
        if vl == 0.0:
            continue
        L = 2 * s - l
        if L < 0:
            continue
        F = np.zeros((norb, norb))
        for a1 in range(norb):
            for a2 in range(norb):
                F[a1, a2] = w3j(s, s, L, ms[a1], ms[a2], -(ms[a1] + ms[a2]))
        chans.append((vl * (2 * L + 1), F))

    for weight, F in chans:
        # V += weight * F[a1,a2] * F[a4,a3]; the delta on m-sums is automatic because
        # both 3j's carry the same third magnetic index only when m1+m2 == m3+m4.
        V += weight * np.einsum("ij,lk->ijkl", F, F)

    # Enforce the momentum delta explicitly (kills numerically tiny off-shell leakage).
    tot_in = ms[:, None] + ms[None, :]
    mask = np.abs(tot_in[:, :, None, None] - tot_in[None, None, :, :].transpose(0, 1, 2, 3)) < 1e-9
    # careful: axes are (a1,a2,a3,a4) with in-pair (a3,a4) -> m3+m4
    m34 = ms[:, None] + ms[None, :]
    mask = np.abs(tot_in[:, :, None, None] - m34[None, None, :, :]) < 1e-9
    V *= mask
    return V


def pseudopotentials_from_realspace(s, g0, g1):
    """``(V_0, V_1)`` for the ultralocal real-space interaction ``g0 d(W) + g1 grad^2 d(W)``.

    2210.13482 Eqs. (A9), (A11).  Note both terms feed ``V_0`` while only the
    ``grad^2 delta`` term feeds ``V_1``.
    """
    V0 = g0 * (2 * s + 1) ** 2 / (4 * s + 1) - g1 * s * (2 * s + 1) ** 2 / (4 * s + 1)
    V1 = g1 * s * (2 * s + 1) ** 2 / (4 * s - 1)
    return V0, V1


# ----------------------------------------------------------------------------------------------------
# from fs/fockfast.py
"""Vectorized Fock-space enumeration and operator application.

Two things make the difference between N <= 12 and N = 14:

1. **Enumeration by sector.**  Iterating ``combinations(2N, N)`` and filtering on ``M``
   costs C(32,16) = 6e8 Python steps at N=16.  Here the basis is built orbital by
   orbital with pruning on both particle number and residual ``M``, so only states in
   the target sector are ever materialized.

2. **Vectorized ladder operators.**  All bit twiddling is done on the whole ``states``
   array at once (``np.bitwise_count`` for the Jordan-Wigner sign, ``np.searchsorted``
   for the index lookup), which removes the per-state Python loop entirely.
"""
import numpy as np

ONE = np.int64(1)


def enumerate_sector(norb, ne, M, fixed=None):
    """All bitmasks over ``2*norb`` modes with ``ne`` particles and ``sum_a a*n_a == M``.

    Mode ``k = 2*a + t``; the two bits of orbital ``a`` are ``2a`` (up) and ``2a+1`` (down).

    ``fixed`` is an optional ``{mode: 0 or 1}`` constraint, used for the ``h_d -> infinity``
    magnetic line defect: pinning the ``m = +-s`` orbitals to spin down restricts the
    Hilbert space *before* enumeration, so the defect problem is strictly smaller than the
    bulk one rather than larger (2308.01903 Supp. Note 6).
    """
    fixed = fixed or {}
    # cumulative bounds on what orbitals a..norb-1 can still contribute
    suffix_min = np.zeros(norb + 1, dtype=np.int64)   # min M from placing p particles
    cur = {(0, 0): np.zeros(1, dtype=np.int64)}
    for a in range(norb):
        rem = norb - 1 - a                            # orbitals strictly after a
        # which of the four occupation patterns of orbital `a` are allowed?
        allowed = []
        for bits, dn, dm in ((0, 0, 0), (1, 1, a), (2, 1, a), (3, 2, 2 * a)):
            up_occ, dn_occ = bits & 1, (bits >> 1) & 1
            if fixed.get(2 * a) is not None and up_occ != fixed[2 * a]:
                continue
            if fixed.get(2 * a + 1) is not None and dn_occ != fixed[2 * a + 1]:
                continue
            allowed.append((bits, dn, dm))
        nxt = {}
        for (n, m), arr in cur.items():
            for bits, dn, dm in allowed:
                n2, m2 = n + dn, m + dm
                if n2 > ne or m2 > M:
                    continue
                need = ne - n2
                cap = sum(2 - sum(1 for t in (0, 1) if fixed.get(2 * b + t) == 0)
                          for b in range(a + 1, norb)) if fixed else 2 * rem
                if need > cap:
                    continue
                # cheapest / dearest way to place `need` particles in orbitals a+1..norb-1
                lo, hi, left = 0, 0, need
                for b in range(a + 1, norb):
                    take = min(2, left)
                    lo += take * b
                    left -= take
                    if left == 0:
                        break
                left = need
                for b in range(norb - 1, a, -1):
                    take = min(2, left)
                    hi += take * b
                    left -= take
                    if left == 0:
                        break
                if m2 + lo > M or m2 + hi < M:
                    continue
                key = (n2, m2)
                new = arr + np.int64(bits << (2 * a))
                if key in nxt:
                    nxt[key].append(new)
                else:
                    nxt[key] = [new]
        cur = {k: (v[0] if len(v) == 1 else np.concatenate(v)) for k, v in nxt.items()}
    out = cur.get((ne, M), np.zeros(0, dtype=np.int64))
    out.sort()
    return out


class FastBasis:
    """Sorted bitmask array for one ``(ne, M)`` sector, with vectorized lookup."""

    def __init__(self, s, ne, M, fixed=None):
        self.s = s
        self.norb = int(round(2 * s)) + 1
        self.nmode = 2 * self.norb
        self.ne = ne
        self.M = M
        self.fixed = fixed or {}
        self.states = enumerate_sector(self.norb, ne, M, fixed=self.fixed)
        self.dim = len(self.states)

    @property
    def lz(self):
        return self.M - self.s * self.ne

    @staticmethod
    def mode(a, t):
        return 2 * a + t

    def lookup(self, masks, valid=None):
        """Vectorized mask -> index.  Returns ``(idx, found)``."""
        pos = np.searchsorted(self.states, masks)
        pos_c = np.clip(pos, 0, max(self.dim - 1, 0))
        found = (pos < self.dim) & (self.states[pos_c] == masks)
        if valid is not None:
            found &= valid
        return pos_c, found

    def __repr__(self):
        return (f"FastBasis(s={self.s}, norb={self.norb}, ne={self.ne}, M={self.M}, "
                f"Lz={self.lz:g}, dim={self.dim})")


if hasattr(np, "bitwise_count"):
    def popcount(x):
        """Population count as **int64**.

        ``np.bitwise_count`` returns ``uint8``; doing signed arithmetic on that result
        silently wraps (``1 - 2*1 == 255``), so every caller must widen first.
        """
        return np.bitwise_count(x).astype(np.int64)
else:
    # numpy < 2.0 has no bitwise_count.  SWAR popcount on int64, same int64 contract.
    _M1 = np.int64(0x5555555555555555)
    _M2 = np.int64(0x3333333333333333)
    _M4 = np.int64(0x0F0F0F0F0F0F0F0F)

    def popcount(x):
        """Population count as **int64** (SWAR fallback for numpy < 2.0)."""
        v = np.asarray(x, dtype=np.int64).copy()
        v -= (v >> np.int64(1)) & _M1
        v = (v & _M2) + ((v >> np.int64(2)) & _M2)
        v = (v + (v >> np.int64(4))) & _M4
        return (v * np.int64(0x0101010101010101)) >> np.int64(56)


def _sign_below(masks, k):
    """(-1)^{# occupied modes below k}, vectorized."""
    if k == 0:
        return np.ones(len(masks), dtype=np.int64)
    low = masks & np.int64((1 << k) - 1)
    return 1 - 2 * (popcount(low) & 1)


def apply_ladder_vec(masks, ops):
    """Apply ``ops = [(kind, k), ...]`` right-to-left to every mask at once.

    ``kind`` is ``'c'`` (annihilate) or ``'d'`` (create).  Returns
    ``(new_masks, sign, alive)``.
    """
    m = masks.copy()
    sgn = np.ones(len(masks), dtype=np.int64)
    alive = np.ones(len(masks), dtype=bool)
    for kind, k in reversed(ops):
        bit = ((m >> np.int64(k)) & ONE).astype(bool)
        alive &= bit if kind == "c" else ~bit
        if not alive.any():
            return m, sgn, alive
        s = _sign_below(m, k)
        sgn = np.where(alive, sgn * s, sgn)
        flip = np.where(alive, m ^ (ONE << np.int64(k)), m)
        m = flip
    return m, sgn, alive


# ----------------------------------------------------------------------------------------------------
# from fs/opsfast.py
"""Vectorized operator builders on a :class:`~fs.fockfast.FastBasis`.

Same physics as ``fs.operators``, but every operator is assembled by looping over
*orbital quartets* (O(norb^3) of them after momentum conservation) with a single
vectorized pass over the whole basis inside, instead of looping over basis states in
Python.  At N=12 this is ~50x faster; it is what makes N=14 reachable.

The two discrete symmetries have closed-form Jordan-Wigner signs, verified against the
reference implementation in ``fs.operators`` (see ``tests/test_fast.py``):

    sign(Z2) = (-1)^{# doubly occupied orbitals}
    sign(P)  = (-1)^{sum_{k in occ} k + n_up + # doubly occupied orbitals}

The Z2 result follows because the spin flip ``k -> k^1`` is order preserving except on
doubly occupied orbitals, each of which contributes exactly one inversion.  For P one
additionally uses that ``c_{k}`` acting on the filled state sees exactly ``k`` occupied
modes below it.
"""
import numpy as np
import scipy.sparse as sp


UP, DN = 0, 1
_MASK_UP = np.int64(int("01" * 31, 2))     # even bit positions = up modes


def _momentum_quartets(norb, V, tol=1e-14):
    """Yield ``(a1, a2, a3, a4, V)`` with ``|V| > tol`` (momentum already enforced)."""
    out = []
    for a1 in range(norb):
        for a2 in range(norb):
            tot = a1 + a2
            for a3 in range(norb):
                a4 = tot - a3
                if not (0 <= a4 < norb):
                    continue
                v = V[a1, a2, a3, a4]
                if abs(v) > tol:
                    out.append((a1, a2, a3, a4, float(v)))
    return out


class _Accum:
    """COO triplet accumulator.

    ``flush_every=None`` (the default for the Hamiltonian) collects every triplet and
    does a **single** COO->CSR conversion at the end.  That matters at large N: each
    intermediate ``csr + csr`` allocates a fresh matrix of the combined size, so
    flushing k times costs ~2x peak memory on a multi-GB matrix.
    """

    def __init__(self, shape, flush_every=None):
        self.shape = shape
        self.flush_every = flush_every
        self._r, self._c, self._v = [], [], []
        self._n = 0
        self.mat = sp.csr_matrix(shape)

    def add(self, rows, cols, vals):
        if len(rows) == 0:
            return
        self._r.append(rows.astype(np.int32, copy=False))
        self._c.append(cols.astype(np.int32, copy=False))
        self._v.append(vals.astype(np.float64, copy=False))
        self._n += len(rows)
        if self.flush_every is not None and self._n >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._r:
            return
        r = np.concatenate(self._r); self._r = []
        c = np.concatenate(self._c); self._c = []
        v = np.concatenate(self._v); self._v = []
        self._n = 0
        block = sp.coo_matrix((v, (r, c)), shape=self.shape).tocsr()
        del r, c, v
        self.mat = block if self.mat.nnz == 0 else (self.mat + block)

    def result(self):
        self.flush()
        self.mat.sum_duplicates()
        return self.mat


def build_hamiltonian_fast(basis, Vl, h, V=None, verbose=False, symmetrize=None):
    """``H_int + H_t`` as an explicit CSR matrix, vectorized.

    ``symmetrize`` averages with the transpose to scrub round-off asymmetry.  It doubles
    peak memory, so it defaults to on only below 300k basis states; the construction is
    symmetric by exact arithmetic anyway (asserted in the tests), the averaging just
    removes ~1e-15 noise.
    """
    if symmetrize is None:
        symmetrize = basis.dim < 300_000
    if V is None:
        V = haldane_V(basis.s, Vl)
    quartets = _momentum_quartets(basis.norb, V)
    if verbose:
        print(f"    {len(quartets)} momentum-conserving quartets")
    acc = _Accum((basis.dim, basis.dim))
    st = basis.states
    col = np.arange(basis.dim, dtype=np.int32)

    for (a1, a2, a3, a4, v) in quartets:
        ops = [("d", FastBasis.mode(a1, UP)), ("d", FastBasis.mode(a2, DN)),
               ("c", FastBasis.mode(a3, DN)), ("c", FastBasis.mode(a4, UP))]
        new, sgn, alive = apply_ladder_vec(st, ops)
        if not alive.any():
            continue
        idx, found = basis.lookup(new, alive)
        if not found.any():
            continue
        acc.add(idx[found], col[found], 2.0 * v * sgn[found])

    for a in range(basis.norb):
        for (kd, ka) in ((FastBasis.mode(a, UP), FastBasis.mode(a, DN)),
                         (FastBasis.mode(a, DN), FastBasis.mode(a, UP))):
            new, sgn, alive = apply_ladder_vec(st, [("d", kd), ("c", ka)])
            if not alive.any():
                continue
            idx, found = basis.lookup(new, alive)
            if not found.any():
                continue
            acc.add(idx[found], col[found], -h * sgn[found].astype(np.float64))

    H = acc.result()
    if symmetrize:
        H = ((H + H.T) * 0.5).tocsr()
    return H


def build_bilinear_fast(basis_from, basis_to, pairs):
    """``sum over (a1, t1, a2, t2, coef) of coef * c^dag_{a1,t1} c_{a2,t2}``, vectorized."""
    acc = _Accum((basis_to.dim, basis_from.dim))
    st = basis_from.states
    col = np.arange(basis_from.dim, dtype=np.int32)
    for (a1, t1, a2, t2, coef) in pairs:
        if coef == 0.0:
            continue
        new, sgn, alive = apply_ladder_vec(
            st, [("d", FastBasis.mode(a1, t1)), ("c", FastBasis.mode(a2, t2))])
        if not alive.any():
            continue
        idx, found = basis_to.lookup(new, alive)
        if not found.any():
            continue
        acc.add(idx[found], col[found], coef * sgn[found].astype(np.float64))
    return acc.result()


def build_jplus_fast(basis_from, basis_to):
    s = basis_from.s
    pairs = []
    for a in range(basis_from.norb - 1):
        m = a - s
        amp = float(np.sqrt((s - m) * (s + m + 1)))
        for t in (UP, DN):
            pairs.append((a + 1, t, a, t, amp))
    return build_bilinear_fast(basis_from, basis_to, pairs)


def _n_double(states):
    return popcount((states & (states >> ONE)) & _MASK_UP)


def _sum_k(states):
    """sum of occupied mode indices, vectorized over a 64-bit mask array."""
    tot = np.zeros(len(states), dtype=np.int64)
    for k in range(64):
        bit = (states >> np.int64(k)) & ONE
        if not bit.any():
            continue
        tot += np.int64(k) * bit
    return tot


def build_spinflip_fast(basis):
    st = basis.states
    up = st & _MASK_UP
    dn = st & ~_MASK_UP
    new = (up << ONE) | (dn >> ONE)
    sgn = (1 - 2 * (_n_double(st) & 1)).astype(np.float64)
    idx, found = basis.lookup(new)
    assert found.all(), "Z2 image left the sector"
    return sp.csr_matrix((sgn, (idx, np.arange(basis.dim))), shape=(basis.dim,) * 2)


def build_parity_fast(basis):
    st = basis.states
    full = np.int64((1 << basis.nmode) - 1)
    holes = (~st) & full
    up = holes & _MASK_UP
    dn = holes & ~_MASK_UP
    new = (up << ONE) | (dn >> ONE)          # particle-hole, then spin flip
    expo = (_sum_k(st) + popcount(st & _MASK_UP) + _n_double(st)) & 1
    sgn = (1 - 2 * expo).astype(np.float64)
    idx, found = basis.lookup(new)
    assert found.all(), "parity image left the sector"
    return sp.csr_matrix((sgn, (idx, np.arange(basis.dim))), shape=(basis.dim,) * 2)


def build_density_mode_fast(basis_from, basis_to, l, mm, spin_matrix):
    """``n^A_{l,m}`` between two sectors (see ``fs.operators.build_density_mode``)."""
    pass
    s = basis_from.s
    norb = basis_from.norb
    Nsl = (2 * s + 1) * w3j(s, s, l, s, -s, 0)
    A = np.asarray(spin_matrix, dtype=float)
    pairs = []
    for a1 in range(norb):
        m1 = a1 - s
        for a2 in range(norb):
            m2 = a2 - s
            if abs((m1 - m2) - mm) > 1e-9:
                continue
            c = Nsl * ((-1.0) ** (s - m1)) * np.sqrt(2 * l + 1) * w3j(s, s, l, m1, -m2, -mm)
            if abs(c) < 1e-14:
                continue
            for t in (UP, DN):
                for tp in (UP, DN):
                    if A[t, tp] != 0.0:
                        pairs.append((a1, t, a2, tp, c * A[t, tp]))
    return build_bilinear_fast(basis_from, basis_to, pairs)


# ----------------------------------------------------------------------------------------------------
# from twist/twist.py
"""Z2 monodromy (twist) line defect of the 3d Ising CFT on the fuzzy sphere.

Construction.  Work in the x-basis psi_pm = (psi_up +- psi_dn)/sqrt2, where the Ising Z2
acts as psi_- -> -psi_-.  A straight line defect through the origin pierces the sphere at the
two poles.  Give psi_- (only) half a unit of Aharonov-Bohm flux through the poles: its LLL
orbitals become

    Phi~_m(theta, phi) = N_m e^{i m phi} cos^{s+m}(theta/2) sin^{s-m}(theta/2),
    m in {-s+1/2, ..., s-1/2}          (N-1 orbitals, regular at both poles)

while psi_+ keeps the usual m in {-s, ..., s}.  Every operator with an odd number of psi_-,
i.e. every Z2-odd operator, is then antiperiodic around the z axis: that is the twist.
The Hamiltonian is Z2-even, hence single-valued, and SO(3) breaks to U(1)_z (transverse spin).

Interaction.  The Ising interaction of 2210.13482 is ultralocal,
U = G0 delta + G1 grad^2 delta, so every matrix element is a single sphere integral of four
orbitals, done in closed form with Beta functions (valid for half-integer offsets).  In the
x-basis

    H = 1/2 sum W_{aa'bb'} V^{(aa'bb')}_{m1 m2 m3 m4} c+_{m1 a} c+_{m2 b} c_{m3 b'} c_{m4 a'}
        - h sum (n_+ - n_-),          W = 1 x 1 - X x X,  X = [[0,1],[1,0]]

(normal ordered, exactly as fs/on.py).  With the twist off this is the pipeline's Ising model
in a rotated basis; tests assert the spectra agree.
"""
import sys
from math import lgamma, log, pi, exp
import numpy as np
import scipy.sparse as sp



# ----------------------------------------------------------------------------------------------
# orbitals and closed-form integrals
# ----------------------------------------------------------------------------------------------
def orbital_ms(N, twisted):
    s = (N - 1) / 2
    if twisted:
        return [-s + 0.5 + j for j in range(N - 1)]
    return [-s + j for j in range(N)]


def _lognorm(s, m):
    return 0.5 * (lgamma(2 * s + 2) - log(4 * pi) - lgamma(s + m + 1) - lgamma(s - m + 1))


def _beta(x, y):
    return exp(lgamma(x) + lgamma(y) - lgamma(x + y))


def four_orbital(s, m1, m2, m3, m4):
    """(I_delta, I_lap) for A = Phi*_{m1} Phi_{m4} (point a), B = Phi*_{m2} Phi_{m3} (point b).

    I_delta = int A B dOmega,  I_lap = int A lap(B) dOmega = -int grad A . grad B dOmega.
    Requires m1 + m2 = m3 + m4 (the phi integral); unit sphere.
    """
    if abs((m1 + m2) - (m3 + m4)) > 1e-9:
        return 0.0, 0.0
    lnN = _lognorm(s, m1) + _lognorm(s, m2) + _lognorm(s, m3) + _lognorm(s, m4)
    NN = exp(lnN)
    tot = m1 + m2 + m3 + m4
    P, Qq = 4 * s + tot, 4 * s - tot
    Idel = NN * 2 * pi * 2 * _beta(Qq / 2 + 1, P / 2 + 1)
    pA, qA, muA = 2 * s + m1 + m4, 2 * s - m1 - m4, m4 - m1
    pB, qB = 2 * s + m2 + m3, 2 * s - m2 - m3
    terms = 0.0
    c1 = qA * qB / 4
    if c1 != 0:
        terms += c1 * _beta(Qq / 2, P / 2 + 2)
    c2 = -(qA * pB + pA * qB) / 4
    if c2 != 0:
        terms += c2 * _beta(Qq / 2 + 1, P / 2 + 1)
    c3 = pA * pB / 4
    if c3 != 0:
        terms += c3 * _beta(Qq / 2 + 2, P / 2)
    c4 = muA * muA / 4
    if c4 != 0:
        terms += c4 * _beta(Qq / 2, P / 2)
    Ilap = -NN * 2 * pi * 2 * terms
    return Idel, Ilap


def pair_pseudopotentials(N, G0, G1):
    """(V0, V1) of the untwisted interaction, from the 2x2 pair problem at M = 2s - 1."""
    s = (N - 1) / 2
    ms = orbital_ms(N, False)
    a, b = ms[-1], ms[-2]                      # m = s and s-1
    basis = [(a, b), (b, a)]
    M = np.zeros((2, 2))
    for i, (m1, m2) in enumerate(basis):
        for j, (m4, m3) in enumerate(basis):
            Id, Il = four_orbital(s, m1, m2, m3, m4)
            M[i, j] = G0 * Id + G1 * Il
    w = np.linalg.eigvalsh((M + M.T) / 2)
    return M, w


def couplings_for(N, V0, V1):
    """(G0, G1) such that the untwisted pseudopotentials are (V0, V1).

    The delta term lives only in the l=0 (pair-symmetric-in-orbital, L = 2s) channel.  Channel
    identification: at M = 2s-1 the L = 2s pair state is (|s,s-1> + (-1)^? |s-1,s>)/sqrt2; we
    read V_l off the eigenvector overlap with the delta-only eigenvector.
    """
    Md, wd = pair_pseudopotentials(N, 1.0, 0.0)
    Ml, wl = pair_pseudopotentials(N, 0.0, 1.0)
    # eigenvector of the delta interaction with nonzero eigenvalue = the l=0 channel
    ev, U = np.linalg.eigh((Md + Md.T) / 2)
    u0 = U[:, np.argmax(np.abs(ev))]
    u1 = U[:, np.argmin(np.abs(ev))]
    V0d, V1d = u0 @ Md @ u0, u1 @ Md @ u1
    V0l, V1l = u0 @ Ml @ u0, u1 @ Ml @ u1
    A = np.array([[V0d, V0l], [V1d, V1l]])
    G0, G1 = np.linalg.solve(A, [V0, V1])
    return G0, G1, A


# ----------------------------------------------------------------------------------------------
# Fock space: + orbitals are modes 0..N-1, - orbitals are modes N..N+Nm-1
# ----------------------------------------------------------------------------------------------
class TwistBasis:
    def __init__(self, N, twisted, twoLz, z2):
        self.N, self.twisted = N, twisted
        self.s = (N - 1) / 2
        self.mp = orbital_ms(N, False)
        self.mm = orbital_ms(N, twisted)
        self.Np, self.Nm = len(self.mp), len(self.mm)
        self.twoLz, self.z2 = twoLz, z2
        # group subsets of each flavour by (count, 2*sum m)
        def groups(ms):
            g = {}
            n = len(ms)
            two = [int(round(2 * m)) for m in ms]
            for S in range(1 << n):
                c = bin(S).count('1')
                t = sum(two[i] for i in range(n) if S >> i & 1)
                g.setdefault((c, t), []).append(S)
            return {k: np.array(v, dtype=np.int64) for k, v in g.items()}
        gp, gm = groups(self.mp), groups(self.mm)
        out = []
        for (cm, tm), Sm in gm.items():
            if (cm % 2 == 0) != (z2 == +1):
                continue
            key = (N - cm, twoLz - tm)
            if key not in gp:
                continue
            Sp = gp[key]
            out.append((Sp[None, :] | (Sm[:, None] << np.int64(self.Np))).ravel())
        self.states = np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)
        self.dim = len(self.states)

    def lookup(self, masks, valid):
        pos = np.searchsorted(self.states, masks)
        pc = np.clip(pos, 0, max(self.dim - 1, 0))
        found = (pos < self.dim) & (self.states[pc] == masks) & valid
        return pc, found


W_TABLE = {  # (a, a', b, b') -> W,  flavours: 0 = '+', 1 = '-'
    (0, 0, 0, 0): 1.0, (0, 0, 1, 1): 1.0, (1, 1, 0, 0): 1.0, (1, 1, 1, 1): 1.0,
    (0, 1, 0, 1): -1.0, (0, 1, 1, 0): -1.0, (1, 0, 0, 1): -1.0, (1, 0, 1, 0): -1.0,
}


def interaction_terms(N, twisted, G0, G1, tol=1e-13):
    """List of (mode1, mode2, mode3, mode4, coeff) for c+_1 c+_2 c_3 c_4, coeff includes 1/2 W."""
    s = (N - 1) / 2
    ms = {0: orbital_ms(N, False), 1: orbital_ms(N, twisted)}
    off = {0: 0, 1: N}
    cache = {}
    terms = {}
    for (al, alp, be, bep), Wv in W_TABLE.items():
        for i1, m1 in enumerate(ms[al]):
            for i2, m2 in enumerate(ms[be]):
                for i3, m3 in enumerate(ms[bep]):
                    m4 = m1 + m2 - m3
                    # find orbital index of m4 in flavour alp
                    j = m4 - ms[alp][0]
                    i4 = int(round(j))
                    if abs(j - i4) > 1e-9 or not (0 <= i4 < len(ms[alp])):
                        continue
                    key = (m1, m2, m3, m4)
                    if key not in cache:
                        cache[key] = four_orbital(s, m1, m2, m3, m4)
                    Id, Il = cache[key]
                    v = 0.5 * Wv * (G0 * Id + G1 * Il)
                    if abs(v) < tol:
                        continue
                    k = (off[al] + i1, off[be] + i2, off[bep] + i3, off[alp] + i4)
                    terms[k] = terms.get(k, 0.0) + v
    return [(k[0], k[1], k[2], k[3], v) for k, v in terms.items() if abs(v) > tol]


def build_H(basis, terms, h):
    st = basis.states
    col = np.arange(basis.dim, dtype=np.int32)
    acc = _Accum((basis.dim, basis.dim))
    for (k1, k2, k3, k4, v) in terms:
        new, sgn, alive = apply_ladder_vec(st, [("d", k1), ("d", k2), ("c", k3), ("c", k4)])
        if not alive.any():
            continue
        idx, found = basis.lookup(new, alive)
        if found.any():
            acc.add(idx[found], col[found], v * sgn[found])
    # field: -h (n_+ - n_-): diagonal
    occ_p = np.zeros(basis.dim)
    occ_m = np.zeros(basis.dim)
    for a in range(basis.Np):
        occ_p += (st >> np.int64(a)) & 1
    for b in range(basis.Nm):
        occ_m += (st >> np.int64(basis.Np + b)) & 1
    acc.add(col, col, -h * (occ_p - occ_m))
    return acc.result().tocsr()


# ----------------------------------------------------------------------------------------------
# numba builder (column by column, duplicates merged in place -> CSC with no duplicate entries)
# ----------------------------------------------------------------------------------------------
import numba as nb


@nb.njit(cache=True)
def _popc(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@nb.njit(cache=True)
def _bsearch(arr, x):
    a, b = 0, len(arr)
    while a < b:
        m = (a + b) >> 1
        if arr[m] < x:
            a = m + 1
        else:
            b = m
    if a < len(arr) and arr[a] == x:
        return a
    return -1


@nb.njit(cache=True)
def _build_csc(states, nmodes, start, K1, K2, K3, K4, VV, diag):
    n = len(states)
    indptr = np.zeros(n + 1, np.int64)
    cap = n * 64
    rows = np.empty(cap, np.int32)
    vals = np.empty(cap, np.float64)
    nn = 0
    occ = np.empty(nmodes, np.int64)
    bufr = np.empty(1 << 16, np.int64)
    bufv = np.empty(1 << 16, np.float64)
    for i in range(n):
        st = states[i]
        no = 0
        for k in range(nmodes):
            if (st >> k) & 1:
                occ[no] = k
                no += 1
        nb_ = 0
        bufr[nb_] = i
        bufv[nb_] = diag[i]
        nb_ += 1
        for a in range(no):
            k3 = occ[a]
            for b in range(no):
                if a == b:
                    continue
                k4 = occ[b]
                p = k3 * nmodes + k4
                for t in range(start[p], start[p + 1]):
                    k1, k2 = K1[t], K2[t]
                    s1 = st
                    sg = 1
                    if _popc(s1 & ((np.int64(1) << k4) - 1)) & 1:
                        sg = -sg
                    s1 ^= np.int64(1) << k4
                    if _popc(s1 & ((np.int64(1) << k3) - 1)) & 1:
                        sg = -sg
                    s1 ^= np.int64(1) << k3
                    if (s1 >> k2) & 1:
                        continue
                    if _popc(s1 & ((np.int64(1) << k2) - 1)) & 1:
                        sg = -sg
                    s1 |= np.int64(1) << k2
                    if (s1 >> k1) & 1:
                        continue
                    if _popc(s1 & ((np.int64(1) << k1) - 1)) & 1:
                        sg = -sg
                    s1 |= np.int64(1) << k1
                    j = _bsearch(states, s1)
                    if j < 0:
                        continue
                    bufr[nb_] = j
                    bufv[nb_] = VV[t] * sg
                    nb_ += 1
        # merge duplicates: sort buffer by row
        order = np.argsort(bufr[:nb_])
        last = -1
        for q in range(nb_):
            r = bufr[order[q]]
            v = bufv[order[q]]
            if r == last:
                vals[nn - 1] += v
            else:
                if nn == len(rows):
                    rows = np.concatenate((rows, np.empty(len(rows), np.int32)))
                    vals = np.concatenate((vals, np.empty(len(vals), np.float64)))
                rows[nn] = r
                vals[nn] = v
                nn += 1
                last = r
        indptr[i + 1] = nn
    return indptr, rows[:nn], vals[:nn]


def build_H_fast(basis, terms, h):
    nmodes = basis.Np + basis.Nm
    T = np.array([(t[2], t[3], t[0], t[1]) for t in terms], dtype=np.int64)   # (k3, k4, k1, k2)
    V = np.array([t[4] for t in terms])
    order = np.lexsort((T[:, 3], T[:, 2], T[:, 1], T[:, 0]))
    T, V = T[order], V[order]
    key = T[:, 0] * nmodes + T[:, 1]
    start = np.searchsorted(key, np.arange(nmodes * nmodes + 1)).astype(np.int64)
    st = basis.states
    occ_p = np.zeros(basis.dim)
    occ_m = np.zeros(basis.dim)
    for a in range(basis.Np):
        occ_p += (st >> np.int64(a)) & 1
    for b in range(basis.Nm):
        occ_m += (st >> np.int64(basis.Np + b)) & 1
    diag = -h * (occ_p - occ_m)
    indptr, rows, vals = _build_csc(st, nmodes, start, T[:, 2].copy(), T[:, 3].copy(),
                                    T[:, 0].copy(), T[:, 1].copy(), V, diag)
    return sp.csc_matrix((vals, rows, indptr), shape=(basis.dim, basis.dim))


# ----------------------------------------------------------------------------------------------------
# from twist/ising_matrix_elements.py
"""Vacuum-to-defect-primary matrix elements of the fuzzy-sphere densities in the Ising half-twist.

In the x-basis (psi_+, psi_-) with psi_- twisted, n^x = n_+ - n_- (Z2 even, leading content eps) and
n^z = psi_+^dag psi_- + h.c. (Z2 odd, leading content sigma).  For a defect primary |O_s> and the defect vacuum,
conformal symmetry gives |<0_D| O(theta, phi) |O_s>| = b_{O O_s} (sin theta)^{Dhat - Delta} on the unit sphere, so
the theta profile is fixed by the dimensions and its prefactor is the bulk-to-defect coefficient.
Channels: n^z -> psi_1/2 (odd, 2Lz=1), n^z -> psi_3/2 (odd, 2Lz=3), n^x -> S (even, 2Lz=0, 2nd level),
n^x -> D (even, 2Lz=2).  (V0, V1, h) = (4.75, 1, 3.16).
usage: python ising_matrix_elements.py N   -> zgh_results/ising_me_N{N}.json"""
import sys


import json


from math import lgamma, log, pi, exp


import numpy as np


import scipy.sparse.linalg as sla


def lognorm(s, m):
    return 0.5 * (lgamma(2 * s + 2) - log(4 * pi) - lgamma(s + m + 1) - lgamma(s - m + 1))


def orb(s, m, th):
    return exp(lognorm(s, m)) * np.cos(th / 2) ** (s + m) * np.sin(th / 2) ** (s - m)


def lowest(b, H, k):
    if b.dim <= 2500:
        w, v = np.linalg.eigh(H.toarray()); return w[:k], v[:, :k]
    w, v = sla.eigsh(H, k=k, which='SA', tol=1e-12); o = np.argsort(w); return w[o], v[:, o]


def onebody(ba, va, bb, vb, k1, k2):
    """<a| c+_{k1} c_{k2} |b> for single vectors va (basis ba), vb (basis bb); modes are bit positions."""
    st = bb.states
    has = ((st >> np.int64(k2)) & 1).astype(bool)
    if k1 != k2:
        has &= ~((st >> np.int64(k1)) & 1).astype(bool)
    src = st[has]; amp = vb[has]
    t = src & ~(np.int64(1) << np.int64(k2))
    lo, hi = min(k1, k2), max(k1, k2)
    mask = ((np.int64(1) << np.int64(hi)) - 1) & ~((np.int64(1) << np.int64(lo + 1)) - 1)
    between = t & mask
    sign = 1 - 2 * (np.array([bin(int(x)).count('1') for x in between]) & 1)
    t = t | (np.int64(1) << np.int64(k1))
    pos = np.searchsorted(ba.states, t)
    ok = (pos < ba.dim) & (ba.states[np.clip(pos, 0, ba.dim - 1)] == t)
    return float(np.sum(va[pos[ok]] * amp[ok] * sign[ok]))



# ----------------------------------------------------------------------------------------------------
# from twist/qian_tune.py
"""Tune our Ising model with Qian's defect-only cost function (arXiv:2610.27926, Eq. 4.8).

P(V1, h) = sum_i |dE_i(inf)| / <dE_i>_N, where for each state i the defect gap obeys the fit
dE_i(N) = dE_i(inf) + c_i / sqrt(N) over the sizes used.  States: first excited defect level in each of the sectors
(0+), (0-), (1/2), (1), (3/2), with 0+- the eigenvalue of R_yz: psi_m -> psi_{-m} in each flavour (relative to the
defect ground state).  V0 = 1 (our pseudopotential normalization; only V1/V0 and the h/V0 scale matter).
usage: python qian_tune.py eval V1 h  |  python qian_tune.py opt V1 h  |  python qian_tune.py check N V1 h"""
import sys


import time


import json


import numpy as np


import scipy.sparse.linalg as sla


from scipy.optimize import minimize


def mirror(basis):
    """Signed permutation for psi_m -> psi_{-m} within each flavour on the basis states."""
    Np, Nm = basis.Np, basis.Nm
    st = basis.states
    p = st & ((1 << Np) - 1); q = st >> Np
    def rev(x, n):
        y = np.zeros_like(x)
        for i in range(n):
            y |= ((x >> i) & 1) << (n - 1 - i)
        return y
    pr, qr = rev(p, Np), rev(q, Nm)
    kp = np.array([bin(int(v)).count('1') for v in p]); kq = np.array([bin(int(v)).count('1') for v in q])
    sign = (-1.0) ** ((kp * (kp - 1) // 2 + kq * (kq - 1) // 2) % 2)
    idx = np.searchsorted(st, pr | (qr << Np))
    assert np.all(st[idx] == (pr | (qr << Np)))
    return idx, sign



# ----------------------------------------------------------------------------------------------------
# from twist/defect_corr.py (helpers shared by the estimators)
def popc(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def apply_cdc(st_in, vec, a, b, st_out):
    """c+_a c_b (a != b) on vec (basis st_in) -> (indices into st_out, values)."""
    ok = (((st_in >> np.int64(b)) & 1) == 1) & (((st_in >> np.int64(a)) & 1) == 0)
    lo, hi = min(a, b), max(a, b)
    between = np.int64(((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
    sign = 1 - 2 * (popc(st_in & between) & 1)
    new = st_in ^ (np.int64(1) << np.int64(a)) ^ (np.int64(1) << np.int64(b))
    pos = np.searchsorted(st_out, new)
    pc = np.clip(pos, 0, max(len(st_out) - 1, 0))
    hit = ok & (pos < len(st_out)) & (st_out[pc] == new)
    return pc[hit], sign[hit] * vec[hit]


def wave_abs(s, m, Th):
    pass
    return np.exp(_lognorm(s, m)) * np.cos(Th / 2) ** (s + m) * np.sin(Th / 2) ** (s - m)



@section
def onmodel():
    """O(N) fuzzy-sphere models with a monodromy flux tube (Wilson-Fisher and free scalar).

Bulk models: Guo, Zhou, Wei, He, arXiv:2512.02234.  Flavours psi_0 (O(N) singlet) and psi_1..psi_N
(O(N) vector), filling N_e = N_orb = 2s+1, sphere radius R^2 = N_orb:

  WF     H = int d^2x [ n_tot^2 - U sum_a V_a lap V_a + h n_00 ],          V_a = psi_0^+ psi_a + h.c.
  Gauss  H = int d^2x [ n_tot^2 + U sum_{a<b} w_ab lap w_ab + (h/R^2) n_00 ], w_ab = i(psi_a^+ psi_b - h.c.)

with n_tot = sum_a psi_a^+ psi_a.  Here h multiplies psi_0^+ psi_0.  The source defines n_00 = 2 psi_0^+ psi_0
(its Eq. 3), but its published (U_1, h_1) are numerically our (U, h) with ordering=True: at (0.264, -0.096) this
code reproduces every O(3) entry of Table IV of Yang et al. arXiv:2609.03289 (which uses that bulk) to four
decimals at N=7, 8, with the pole field V_a(pole) = (c^+_{s,a} c_{s,0} + h.c.)/(4 pi N).  Normal ordering or
h = 2 h_1 do not (spectator_pinning/YANG_CONTROL.txt, 2026-10-22).  On the unit sphere a delta interaction g0
becomes G0 = g0/R^2 and a Laplacian one g1 becomes G1 = g1/R^4.  Products of bilinears are kept as
written (not normal ordered), so the ordering produces a one-body term; it is m-independent in the bulk
but not with the flux, and is kept exactly, as for the spin-1 model in o2twist.py.

Defect: a list of `pairs` (a, b) of vector components is diagonalised into psi_pm = (psi_a -+ i psi_b)/sqrt2
with charges +-1 under the rotation in the (a, b) plane.  `twist[p]` = 1 puts that rotation's charge into
the monodromy, so a flavour of total twisted charge q gets AB flux q*alpha and orbitals
m in (-s + {q alpha} + Z) cap [-s, s].  Every operator of twisted charge Q picks up exp(2 pi i Q alpha).
Conserved: N_e, each pair charge Q_p, L_z = K + alpha*Q_tw, and Z2 parities of unpaired vector components.
"""
    HERE = 'twist'
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla


    # ------------------------------------------------------------------------------------------ flavours
    class Flavours:
        """Charge basis psi'_k with psi_alpha = sum_k U[alpha, k] psi'_k (alpha = 0 singlet, 1..Nv vector)."""

        def __init__(self, Nv, pairs=(), twist=None):
            self.Nv, self.pairs = Nv, list(pairs)
            self.twist = list(twist) if twist is not None else [1] * len(self.pairs)
            F = Nv + 1
            U = np.zeros((F, F), complex)
            names, qpair, unpaired = ['0'], [[0] * len(self.pairs)], []
            U[0, 0] = 1
            col = 1
            used = set(x for p in self.pairs for x in p)
            for ip, (a, b) in enumerate(self.pairs):
                for sgn in (+1, -1):                      # psi'_{+-} = (psi_a -+ i psi_b)/sqrt2
                    U[a, col] = 1 / np.sqrt(2)
                    U[b, col] = sgn * 1j / np.sqrt(2)
                    q = [0] * len(self.pairs); q[ip] = sgn
                    qpair.append(q); names.append(f"{a}{b}{'+' if sgn > 0 else '-'}")
                    col += 1
            for a in range(1, Nv + 1):
                if a not in used:
                    U[a, col] = 1; qpair.append([0] * len(self.pairs)); names.append(str(a))
                    unpaired.append(col); col += 1
            self.U, self.F, self.names = U, F, names
            self.qpair = np.array(qpair, int).reshape(F, len(self.pairs))
            self.qtw = self.qpair @ np.array(self.twist, int) if self.pairs else np.zeros(F, int)
            self.unpaired = unpaired                      # flavours carrying a Z2 parity label

        def rot(self, A):
            return self.U.conj().T @ A @ self.U


    def orbitals(nm, alpha, q):
        s = (nm - 1) / 2
        fr = (q * alpha) % 1.0
        if abs(fr) < 1e-12 or abs(fr - 1) < 1e-12:
            return [-s + j for j in range(nm)], 0
        return [-s + fr + j for j in range(nm - 1)], (1 if q < 0 else 0)   # second: K offset (see Basis)


    def channels(model, Nv, nm, U, h, G=None):
        """[(A, G0, G1)] for int int (psi^+ A psi)(x) [G0 delta + G1 lap delta] (psi^+ A psi)(x') on the unit
    sphere (Cartesian flavour basis), and the one-body coefficient mu0 of psi_0^+ psi_0.

    'wf', 'gauss': Guo et al. (U, h as in their Table 1).  'dey': the spin-1 XY model of 2604.18705 with
    psi_0 = S^z = 0 state, (G0, G1) = G from twist.couplings_for(nm, 4, 1) and h = D; used only as a
    regression test against o2twist.py (its D (n_1 + n_2) = D (N_e - n_0) is applied as -D on psi_0, so
    absolute energies differ from o2twist by the constant D*N_e)."""
        F = Nv + 1
        R2 = float(nm)
        E = lambda a, b: np.eye(F)[:, [a]] @ np.eye(F)[[b], :]
        ch = [(np.eye(F), 1.0 / R2, 0.0)]
        if model == 'wf':
            ch += [(E(0, a) + E(a, 0), 0.0, -U / R2**2) for a in range(1, F)]
            mu0 = h          # h on psi_0^+ psi_0; the source's h_1 n_00 is literally 2 h_1 psi_0^+ psi_0 (unresolved, see docstring)
        elif model == 'gauss':
            ch += [(1j * (E(a, b) - E(b, a)), 0.0, U / R2**2) for a in range(F) for b in range(a + 1, F)]
            mu0 = h / R2
        elif model == 'rotor':
            # O(3) quantum-rotor model of Dey, Herviou, Mudry, Lauchli (arXiv:2510.09755): u H_Hub - v H_Heis + h H_trans.
            # Both two-body terms use the same pseudopotentials (V0, V1) = (6.5, 1.0) (G = couplings_for(nm, 6.5, 1)).
            # Their rotor matrix M_R = (3/4pi) n_i.n_j with n^a = A_a/sqrt3, A_a = E(0,a) + E(a,0), i.e. (1/4pi) sum A_a A_a.
            # U = v/u (1.4 there); h multiplies psi_0^+ psi_0, equal to -h_their n_p up to a constant and a normalization.
            # NOT VALIDATED (2026-10-20): M_R matches the paper's eq. (6) exactly, but the spectrum does not. At N = 6-7,
            # tuned to Delta_phi, T comes out at 2.35-2.63 (theirs 2.97 at N = 10-12) for every H_Heis/H_Hub
            # normalization tried (x 4pi, 8pi, 12pi), so the relative normalization or the pseudopotential convention
            # (couplings_for vs their eq. 7) still differs. Do not use for physics until it reproduces their Table I.
            assert Nv == 3
            if G is None:
                pass
                G = couplings_for(nm, 6.5, 1.0)[:2]
            G0, G1 = G
            ch = [(np.eye(F), G0, G1)] + [(E(0, a) + E(a, 0), -U * G0 / (4 * np.pi), -U * G1 / (4 * np.pi))
                                           for a in range(1, F)]
            mu0 = h
        elif model == 'dey':
            assert Nv == 2
            G0, G1 = G
            # spin-1 in the Cartesian basis (x, y, z) = (psi_1, psi_2, psi_0): (S^k)_{ij} = -i eps_{kij}
            idx = {'x': 1, 'y': 2, 'z': 0}
            def S(k):
                M = np.zeros((3, 3), complex)
                for i in 'xyz':
                    for j in 'xyz':
                        e = {('x','y','z'): 1, ('y','z','x'): 1, ('z','x','y'): 1,
                             ('x','z','y'): -1, ('z','y','x'): -1, ('y','x','z'): -1}.get((k, i, j), 0)
                        M[idx[i], idx[j]] = -1j * e
                return M
            ch = [(np.eye(F), G0, G1), (S('x'), -0.5 * G0, -0.5 * G1), (S('y'), -0.5 * G0, -0.5 * G1)]
            mu0 = -h
        else:
            raise ValueError(model)
        return ch, mu0


    # ------------------------------------------------------------------------------------------ terms
    def build_terms(model, Nv, nm, alpha, U, h, pairs=(), twist=None, G=None, ordering=True, tol=1e-13):
        """Two-body terms (k1,k2,k3,k4,v) for c+_k1 c+_k2 c_k3 c_k4 and one-body diagonal mu[k]; modes are
    numbered flavour by flavour in the charge basis.  Returns (fl, orbs, koff, two, mu)."""
        fl = Flavours(Nv, pairs, twist)
        s = (nm - 1) / 2
        orbs, koff = zip(*[orbitals(nm, alpha, fl.qtw[k]) for k in range(fl.F)])
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])])
        chans, mu0 = channels(model, Nv, nm, U, h, G)
        cache = {}

        def I(m1, m2, m3, m4):
            key = (round(m1, 9), round(m2, 9), round(m3, 9), round(m4, 9))
            if key not in cache:
                cache[key] = four_orbital(s, m1, m2, m3, m4)
            return cache[key]

        index = [{round(m, 9): i for i, m in enumerate(o)} for o in orbs]
        two, one = {}, {}
        for A0, G0, G1 in chans:
            A = fl.rot(A0)
            A[np.abs(A) < 1e-14] = 0
            nz = [(a, b) for a in range(fl.F) for b in range(fl.F) if A[a, b] != 0]
            for (a, b) in nz:
                for (c, d) in nz:
                    w = A[a, b] * A[c, d]
                    for i1, m1 in enumerate(orbs[a]):
                        for i2, m2 in enumerate(orbs[c]):
                            for i3, m3 in enumerate(orbs[d]):
                                i4 = index[b].get(round(m1 + m2 - m3, 9))
                                if i4 is None:
                                    continue
                                Id, Il = I(m1, m2, m3, m1 + m2 - m3)
                                v = w * (G0 * Id + G1 * Il)
                                if abs(v) < tol:
                                    continue
                                key = (off[a] + i1, off[c] + i2, off[d] + i3, off[b] + i4)
                                two[key] = two.get(key, 0) + v
                # ordering term: (c+_{m1 a} c_{m4 b})(c+_{m2 b} c_{m3 d}) -> delta_{m4 m2} c+_{m1 a} c_{m1 d}
                for (c, d) in nz:
                    if c != b or not ordering:
                        continue
                    w = A[a, b] * A[b, d]
                    for i1, m1 in enumerate(orbs[a]):
                        i3 = index[d].get(round(m1, 9))
                        if i3 is None:
                            continue
                        v = w * sum(G0 * I(m1, m2, m1, m2)[0] + G1 * I(m1, m2, m1, m2)[1] for m2 in orbs[b])
                        key = (off[a] + i1, off[d] + i3)
                        one[key] = one.get(key, 0) + v
        for i in range(len(orbs[0])):                    # psi_0 is never rotated: mode i of flavour 0
            one[(i, i)] = one.get((i, i), 0) + mu0
        vals = np.array(list(two.values()) + list(one.values()))
        assert np.abs(vals.imag).max() < 1e-10, "complex Hamiltonian"
        offd = [abs(v) for (k, l), v in one.items() if k != l]
        assert not offd or max(offd) < 1e-10, "off-diagonal one-body term"
        two = [(k[0], k[1], k[2], k[3], v.real) for k, v in two.items() if abs(v) > tol]
        mu = np.zeros(off[-1])
        for (k, l), v in one.items():
            if k == l:
                mu[k] += v.real
        return fl, orbs, koff, two, mu


    # ------------------------------------------------------------------------------------------ ED basis
    class Basis:
        """Fock states with N_e electrons, pair charges Qp, K = L_z - alpha*Q_tw, and Z2 parities of the
    unpaired flavours (+1/-1 each).  Modes ordered flavour by flavour as in build_terms."""

        def __init__(self, fl, orbs, koff, Ne, Qp, K, par=(), nm=None, mods=None, Lz=None, mscale=None):
            """With Lz and mscale given, sectors are labelled by the exact L_z (needed when twisted flavours have
        different fractional orbital offsets, e.g. Z_3 fluxes 1/3, 2/3): t sums round(mscale*(m+s))."""
            nf = [len(o) for o in orbs]
            off = np.concatenate([[0], np.cumsum(nf)]).astype(int)
            nm = nm or max(nf)                            # orbitals of the untwisted problem (sets s)
            s = (nm - 1) / 2
            if Lz is not None:
                target = int(round(mscale * (Lz + s * Ne)))
            else:
                target = K + int(round(s * Ne))           # sum over occupied modes of (j + koff_f)
            npair = len(fl.pairs)
            # groups per flavour: (count, sum j + koff*count) -> states (shifted into place)
            groups = []
            for f in range(fl.F):
                g = {}
                n = nf[f]
                mvals = [int(round(mscale * (orbs[f][j] + s))) for j in range(n)] if Lz is not None else None
                for S in range(1 << n):
                    c = bin(S).count('1')
                    if Lz is not None:
                        t = sum(mvals[j] for j in range(n) if S >> j & 1)
                    else:
                        t = sum(j for j in range(n) if S >> j & 1) + koff[f] * c
                    g.setdefault((c, t), []).append(S << int(off[f]))
                groups.append({k: np.array(v, np.int64) for k, v in g.items()})
            # merge flavour by flavour; key = (count, t, pair charges..., parities...)
            par_idx = {f: i for i, f in enumerate(fl.unpaired)}
            cur = {(0, 0) + (0,) * npair + (0,) * len(fl.unpaired): np.zeros(1, np.int64)}
            remaining = [sum(nf[g] for g in range(f + 1, fl.F)) for f in range(fl.F)]
            for f in range(fl.F):
                new = {}
                for key, arr in cur.items():
                    c0, t0 = key[0], key[1]
                    for (c, t), S in groups[f].items():
                        cc = c0 + c
                        if cc > Ne or Ne - cc > remaining[f]:
                            continue
                        q = list(key[2:2 + npair])
                        for p in range(npair):
                            q[p] += c * fl.qpair[f, p]
                            if mods and mods[p]:
                                q[p] %= mods[p]
                        pr = list(key[2 + npair:])
                        if f in par_idx:
                            pr[par_idx[f]] = (pr[par_idx[f]] + c) % 2
                        nk = (cc, t0 + t) + tuple(q) + tuple(pr)
                        blk = (arr[:, None] | S[None, :]).ravel()
                        new.setdefault(nk, []).append(blk)
                cur = {k: np.concatenate(v) for k, v in new.items()}
            Qw = [q % mods[i] if (mods and mods[i]) else q for i, q in enumerate(Qp)]
            want = (Ne, target) + tuple(Qw) + tuple((0 if p > 0 else 1) for p in par)
            self.states = np.sort(cur.get(want, np.zeros(0, np.int64)))
            self.dim = len(self.states)
            self.nmodes = int(off[-1])


    def build_H(basis, two, mu):
        nm = basis.nmodes
        T = np.array([(t[2], t[3], t[0], t[1]) for t in two], dtype=np.int64)     # (k3, k4, k1, k2)
        Vv = np.array([t[4] for t in two])
        order = np.lexsort((T[:, 3], T[:, 2], T[:, 1], T[:, 0]))
        T, Vv = T[order], Vv[order]
        key = T[:, 0] * nm + T[:, 1]
        start = np.searchsorted(key, np.arange(nm * nm + 1)).astype(np.int64)
        st = basis.states
        diag = np.zeros(basis.dim)
        for k in range(nm):
            if mu[k] != 0:
                diag += mu[k] * ((st >> np.int64(k)) & 1)
        indptr, rows, vals = _build_csc(st, nm, start, T[:, 2].copy(), T[:, 3].copy(),
                                        T[:, 0].copy(), T[:, 1].copy(), Vv, diag)
        return sp.csc_matrix((vals, rows, indptr), shape=(basis.dim, basis.dim))


    def lowest(H, k=4):
        if H.shape[0] <= 2500:
            return np.linalg.eigvalsh(H.toarray())[:k]
        return np.sort(sla.eigsh(H, k=min(k, H.shape[0] - 2), which='SA', tol=1e-10)[0])
    return namespace(locals(), globals())


@section
def o2twist():
    """O(2) monodromy line defect with arbitrary angle alpha on the fuzzy sphere.

Bulk model: the spin-1 XY fuzzy sphere of arXiv:2604.18705 (O(2) Wilson-Fisher), exactly as written
there (products of bilinears, NOT normal ordered):

    H = sum V (c+_m1 . c_m1+m)(c+_m2 . c_m2-m)
        - 1/2 sum V [(c+ S^x c)(c+ S^x c) + (c+ S^y c)(c+ S^y c)]
        + D sum_m c+_m (S^z)^2 c_m ,        flavours sigma = +1, 0, -1;  N_e = N;  V0 = 4, V1 = 1.

U(1) charge Q = N_{+1} - N_{-1}.  Defect: flavour sigma = +1 gets Aharonov-Bohm flux +alpha through
the poles and sigma = -1 gets -alpha (regular orbitals, N-1 of each for alpha not in Z):

    sigma = +1 : m in {-s+alpha, ..., s-1+alpha}       sigma = 0 : m in {-s, ..., s}
    sigma = -1 : m in {-s+1-alpha, ..., s-alpha}

so every operator of charge q picks up exp(2 pi i q alpha) around the z axis, and a state of charge Q has
transverse spin L_z in Z + alpha Q.  alpha = 0 is the bulk model (validated against the paper's D_c).

Normal ordering the products of bilinears gives the two-body term plus a one-body term
U_a(m) n_{m a}, U_a(m) = sum_b K_abba sum_{m2 in orb(b)} V^{(abba)}_{m m2 m m2}; it is m-independent in
the bulk (Schur) but not with the flux, so it is kept exactly.
"""
    HERE = 'twist'
    import numpy as np
    import scipy.sparse as sp

    FL = (+1, 0, -1)                       # flavour order (their spinor: c_{m,1}, c_{m,0}, c_{m,-1})
    SX = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    SY = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
    SZ2 = np.diag([1.0, 0.0, 1.0])
    # K_{abcd} for (c+_a c_b)(c+_c c_d):  delta_ab delta_cd - 1/2 (Sx_ab Sx_cd + Sy_ab Sy_cd)
    K = np.zeros((3, 3, 3, 3))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    w = (SX[a, b] * SX[c, d] + SY[a, b] * SY[c, d]).real
                    K[a, b, c, d] = (1.0 if (a == b and c == d) else 0.0) - 0.5 * w
    K[np.abs(K) < 1e-14] = 0.0


    def orbitals(N, alpha):
        s = (N - 1) / 2
        if alpha == 0:
            return [[-s + j for j in range(N)] for _ in range(3)]
        return [[-s + alpha + j for j in range(N - 1)],
                [-s + j for j in range(N)],
                [-s + 1 - alpha + j for j in range(N - 1)]]


    class O2Basis:
        """Sector (Q, K) with L_z = K + alpha Q; K integer."""

        def __init__(self, N, alpha, Q, Kint):
            self.N, self.alpha, self.Q, self.K = N, alpha, Q, Kint
            self.orb = orbitals(N, alpha)
            self.nf = [len(o) for o in self.orb]
            self.off = [0, self.nf[0], self.nf[0] + self.nf[1]]
            self.nmodes = sum(self.nf)
            s = (N - 1) / 2
            tw = 1 if alpha != 0 else 0

            def groups(n):
                g = {}
                for S in range(1 << n):
                    c = bin(S).count('1')
                    t = sum(j for j in range(n) if S >> j & 1)
                    g.setdefault((c, t), []).append(S)
                return {k: np.array(v, dtype=np.int64) for k, v in g.items()}
            G = [groups(n) for n in self.nf]
            target = Kint + int(round(s * N))              # = sum_j j + tw * n_{-1}
            out = []
            for (c0, t0), S0 in G[0].items():
                for (c2, t2), S2 in G[2].items():
                    if c0 - c2 != Q:
                        continue
                    c1 = N - c0 - c2
                    if c1 < 0:
                        continue
                    t1 = target - t0 - t2 - tw * c2
                    if (c1, t1) not in G[1]:
                        continue
                    S1 = G[1][(c1, t1)]
                    blk = (S0[:, None, None] | (S1[None, :, None] << np.int64(self.off[1]))
                           | (S2[None, None, :] << np.int64(self.off[2])))
                    out.append(blk.ravel())
            self.states = np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)
            self.dim = len(self.states)


    def build_terms(N, alpha, G0, G1, tol=1e-13):
        """Two-body terms (k1,k2,k3,k4,v) for c+_k1 c+_k2 c_k3 c_k4, and one-body U[f][j]."""
        s = (N - 1) / 2
        orb = orbitals(N, alpha)
        off = [0, len(orb[0]), len(orb[0]) + len(orb[1])]
        cache = {}

        def V(m1, m2, m3, m4):
            key = (m1, m2, m3, m4)
            if key not in cache:
                Id, Il = four_orbital(s, m1, m2, m3, m4)
                cache[key] = G0 * Id + G1 * Il
            return cache[key]

        terms = {}
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        kk = K[a, b, c, d]
                        if kk == 0:
                            continue
                        # (c+_{m1 a} c_{m4 b})(c+_{m2 c} c_{m3 d}) -> c+_{m1 a} c+_{m2 c} c_{m3 d} c_{m4 b}
                        for i1, m1 in enumerate(orb[a]):
                            for i2, m2 in enumerate(orb[c]):
                                for i3, m3 in enumerate(orb[d]):
                                    m4 = m1 + m2 - m3
                                    j = m4 - orb[b][0]
                                    i4 = int(round(j))
                                    if abs(j - i4) > 1e-9 or not (0 <= i4 < len(orb[b])):
                                        continue
                                    v = kk * V(m1, m2, m3, m4)
                                    if abs(v) < tol:
                                        continue
                                    key = (off[a] + i1, off[c] + i2, off[d] + i3, off[b] + i4)
                                    terms[key] = terms.get(key, 0.0) + v
        two = [(k[0], k[1], k[2], k[3], v) for k, v in terms.items() if abs(v) > tol]
        # one-body from normal ordering: delta_{bc} delta_{m4 m2} c+_{m1 a} c_{m3 d}, with m1 = m3, a = d
        U = [np.zeros(len(o)) for o in orb]
        for a in range(3):
            for b in range(3):
                kk = K[a, b, b, a]
                if kk == 0:
                    continue
                for i1, m1 in enumerate(orb[a]):
                    U[a][i1] += kk * sum(V(m1, m2, m1, m2) for m2 in orb[b])
        return two, U


    def build_H(basis, two, U, D):
        nm = basis.nmodes
        T = np.array([(t[2], t[3], t[0], t[1]) for t in two], dtype=np.int64)     # (k3, k4, k1, k2)
        Vv = np.array([t[4] for t in two])
        order = np.lexsort((T[:, 3], T[:, 2], T[:, 1], T[:, 0]))
        T, Vv = T[order], Vv[order]
        key = T[:, 0] * nm + T[:, 1]
        start = np.searchsorted(key, np.arange(nm * nm + 1)).astype(np.int64)
        st = basis.states
        diag = np.zeros(basis.dim)
        for f in range(3):
            for j in range(basis.nf[f]):
                k = basis.off[f] + j
                occ = ((st >> np.int64(k)) & 1).astype(float)
                diag += (U[f][j] + D * SZ2[f, f]) * occ
        indptr, rows, vals = _build_csc(st, nm, start, T[:, 2].copy(), T[:, 3].copy(),
                                        T[:, 0].copy(), T[:, 1].copy(), Vv, diag)
        return sp.csc_matrix((vals, rows, indptr), shape=(basis.dim, basis.dim))
    return namespace(locals(), globals())


@section
def run_on_defect():
    """Monodromy defects in the O(N) fuzzy-sphere models of Guo et al. (onmodel.py), by exact diagonalization.

usage: python run_on_defect.py <config> <N> <alphas> [nlev]
  config: o2gauss | o2wf | o3wf | o4wf_diag | o4wf_pair
Tunes h at alpha=0 so that Delta_phi/Delta_dphi equals the target (bulk), then stores the lowest `nlev`
energies of every listed sector for each alpha, at full precision, in on_results/<config>_N<N>.json.
"""
    import sys
    import os
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import brentq
    M = onmodel()

    HERE = 'twist'
    # Delta_phi targets: free scalar 1/2; O(2) 0.519088 (Chester et al. 2019); O(3) 0.518942 (Chester et al.
    # 2020); O(4) 0.51812 from eta = 0.03624(8) (Hasenbusch, arXiv:2112.03783, Monte Carlo).
    CONFIGS = {
        'o2gauss':   dict(model='gauss', Nv=2, U=0.148, pairs=[(1, 2)], twist=[1], dphi=0.5),
        'o2wf':      dict(model='wf', Nv=2, U=0.270, pairs=[(1, 2)], twist=[1], dphi=0.519088),
        'o3wf':      dict(model='wf', Nv=3, U=0.264, pairs=[(1, 2)], twist=[1], dphi=0.518942),
        'o4wf_diag': dict(model='wf', Nv=4, U=0.255, pairs=[(1, 2), (3, 4)], twist=[1, 1], dphi=0.51812),
        'o4wf_pair': dict(model='wf', Nv=4, U=0.255, pairs=[(1, 2), (3, 4)], twist=[1, 0], dphi=0.51812),
    }


    def sectors_for(cfg):
        """(pair charges, K, parities) to compute.  Bulk tuning uses vac, phi (charge 1, K=0), dphi (K=1)."""
        npair, npar = len(cfg['pairs']), cfg['Nv'] - 2 * len(cfg['pairs'])
        z, p = (0,) * npair, (+1,) * npar
        q1 = (1,) + (0,) * (npair - 1)
        out = [(z, 0, p), (z, 1, p), (z, 2, p), (q1, 0, p), (q1, -1, p), (q1, 1, p),
               ((2,) + (0,) * (npair - 1), 0, p), ((2,) + (0,) * (npair - 1), -1, p)]
        if npar:                                       # untwisted component: its parity-odd sectors
            pm = (-1,) + (+1,) * (npar - 1)
            out += [(z, 0, pm), (z, 1, pm), (q1, 0, pm)]
        if npair == 2:                                 # second pair: (1,-1) separates SU(2) triplets
            out += [((1, -1), 0, p), ((1, -1), 1, p), ((0, 1), 0, p), ((0, 1), -1, p)]
        return out


    def main():
        name, N = sys.argv[1], int(sys.argv[2])
        alphas = [float(a) for a in sys.argv[3].split(',')]
        nlev = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        cfg = CONFIGS[name]
        out_path = os.path.join(HERE, 'on_results', f'{name}_N{N}.json')
        None
        res = store.get(out_path) if store.exists(out_path) else {'config': name, 'N': N, 'cfg': cfg}
        kw = dict(pairs=cfg['pairs'], twist=cfg['twist'])
        t0 = time.time()
        # ---- bulk tuning (alpha = 0): h enters only through psi_0's chemical potential
        if 'h_c' not in res:
            fl, orbs, koff, two, mu0 = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], 0.0, **kw)
            _, _, _, _, mu1 = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], 1.0, **kw)
            dmu = mu1 - mu0
            npair, npar = len(cfg['pairs']), cfg['Nv'] - 2 * len(cfg['pairs'])
            z, p, q1 = (0,) * npair, (+1,) * npar, (1,) + (0,) * (npair - 1)
            secs = {'vac': (z, 0), 'phi': (q1, 0), 'dphi': (q1, 1)}
            B = {k: M.Basis(fl, orbs, koff, N, list(q), K, p) for k, (q, K) in secs.items()}
            H0 = {k: M.build_H(B[k], two, mu0) for k in secs}
            nh = {k: sum(dmu[i] * ((B[k].states >> np.int64(i)) & 1) for i in range(len(dmu)) if dmu[i] != 0)
                  for k in secs}
            E = lambda h, k: M.lowest(H0[k] + sp.diags(h * nh[k]), 1)[0]
            tgt = cfg['dphi'] / (1 + cfg['dphi'])
            f = lambda h: (E(h, 'phi') - E(h, 'vac')) / (E(h, 'dphi') - E(h, 'vac')) - tgt
            hs = np.linspace(-0.2, 0.0, 21)
            vals = [f(h) for h in hs]
            br = [(hs[i], hs[i + 1]) for i in range(20) if vals[i] * vals[i + 1] < 0]
            hc = brentq(f, *br[-1], xtol=1e-8)            # the transition closest to h = 0
            res['h_c'] = hc
            print(f"{name} N={N}: h_c = {hc:.6f}  [{time.time() - t0:.0f}s]", flush=True)
        hc = res['h_c']
        for al in [0.0] + alphas:
            key = f"{al:.4f}"
            if key in res.get('alpha', {}):
                continue
            t1 = time.time()
            fl, orbs, koff, two, mu = M.build_terms(cfg['model'], cfg['Nv'], N, al, cfg['U'], hc, **kw)
            lev = {}
            for (q, K, p) in sectors_for(cfg):
                b = M.Basis(fl, orbs, koff, N, list(q), K, p)
                if b.dim == 0:
                    continue
                lev[f"{q}|{K}|{p}"] = [float(x) for x in M.lowest(M.build_H(b, two, mu), nlev)]
            res.setdefault('alpha', {})[key] = lev
            store.put(out_path, res)
            print(f"  alpha={al:.3f}: {len(lev)} sectors  [{time.time() - t1:.0f}s]", flush=True)


    def cli(argv):
        main()
    return namespace(locals(), globals())


@section
def core_strip():
    """Strip the twist-induced core potential of the onmodel (GZWH) monodromy defects.

The products in the GZWH Hamiltonian are kept as written (ordering=True), so their ordering gives a one-body term for
every orbital. In the bulk (alpha = 0) it is a constant per flavour; with the flux it becomes m-dependent near the
poles, for the twisted flavours (whose orbitals change) and for the untwisted ones (through the gradient term). This
deviation from the bulk constant is a defect-localized potential that the monodromy itself does not require.
    mu(lam) = mu_twisted - (1 - lam) * dev,   dev_f(m) = mu_f^{alpha}(m) - mu_f^{bulk}
lam = 1 is the model as used so far, lam = 0 strips the potential. (flow/core_residue.py stripped only psi_0's part.)
mode 'spectrum': the run_on_defect sectors at lam = 1 (checked against on_results) and lam = 0, in bulk units
    (d phi - phi = 1) and displacement units (D at 2).
usage: python core_strip.py spectrum <config> N [alpha=0.5]   -> core_strip_results/<config>_N{N}_a{alpha}.json
"""
    import os
    import sys
    import json
    import numpy as np
    M = onmodel()
    CONFIGS = run_on_defect().CONFIGS
    sectors_for = run_on_defect().sectors_for

    HERE = 'twist'


    def build(cfgname, N, alpha, lam=1.0, hc=None, only=None):
        """(fl, orbs, koff, two, mu(lam), dev, hc) for the twisted problem; only = flavour names whose deviation is scaled."""
        cfg = CONFIGS[cfgname]
        if hc is None:
            hc = store.get(os.path.join(HERE, 'on_results', f'{cfgname}_N{N}.json'))['h_c']
        kw = dict(pairs=cfg['pairs'], twist=cfg['twist'])
        fl, ob, _, _, mub = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], hc, **kw)
        offb = np.concatenate([[0], np.cumsum([len(o) for o in ob])]).astype(int)
        fl, ot, kt, two, mut = M.build_terms(cfg['model'], cfg['Nv'], N, alpha, cfg['U'], hc, **kw)
        offt = np.concatenate([[0], np.cumsum([len(o) for o in ot])]).astype(int)
        dev = np.zeros_like(mut)
        for f in range(fl.F):
            c = mub[offb[f]:offb[f + 1]]
            assert np.allclose(c, c[0]), f'bulk one-body term of flavour {fl.names[f]} is not constant'
            if only is None or fl.names[f] in only:
                dev[offt[f]:offt[f + 1]] = mut[offt[f]:offt[f + 1]] - c[0]
        return fl, ot, kt, two, mut - (1.0 - lam) * dev, dev, hc


    def wait_for_memory(N):
        """N >= 9 ED needs ~5-14 GB: wait until no other N = 9 ED job of the campaign is running."""
        import subprocess
        import time
        pat = r'python .*(joint_scan.py 9|amp_withheld.py measure 9|curv_families.py run 9|traj_validate.py [ABC] 9|si_tcheck.py stat 9)'
        while N >= 9 and subprocess.run(['pgrep', '-f', pat], capture_output=True).stdout.strip():
            time.sleep(60)


    def mode_spectrum(cfgname, N, alpha=0.5):
        wait_for_memory(N)
        cfg = CONFIGS[cfgname]
        R = store.get(os.path.join(HERE, 'on_results', f'{cfgname}_N{N}.json'))
        stored = R['alpha'].get(f'{alpha:.4f}', {})
        npair, npar = len(cfg['pairs']), cfg['Nv'] - 2 * len(cfg['pairs'])
        z, p, q1 = (0,) * npair, (+1,) * npar, (1,) + (0,) * (npair - 1)
        bulk = R['alpha']['0.0000']
        u = bulk[f'{q1}|1|{p}'][0] - bulk[f'{q1}|0|{p}'][0]
        out = dict(config=cfgname, N=N, alpha=alpha, u=u, lam={})
        for lam in (1.0, 0.0):
            fl, orbs, koff, two, mu, dev, hc = build(cfgname, N, alpha, lam)
            lev = {}
            for (q, K, pp) in sectors_for(cfg):
                b = M.Basis(fl, orbs, koff, N, list(q), K, pp)
                if b.dim == 0:
                    continue
                lev[f'{q}|{K}|{pp}'] = [float(x) for x in M.lowest(M.build_H(b, two, mu), 4)]
            E0, ED = lev[f'{z}|0|{p}'][0], lev[f'{z}|1|{p}'][0]
            rec = dict(bulk_units={k: [(e - E0) / u for e in v] for k, v in lev.items()},
                       D_units={k: [2 * (e - E0) / (ED - E0) for e in v] for k, v in lev.items()},
                       dev_sum={fl.names[f]: float(dev[sum(len(o) for o in orbs[:f]):sum(len(o) for o in orbs[:f + 1])].sum())
                                for f in range(fl.F)})
            if lam == 1.0 and stored:
                rec['gate'] = max(abs(lev[k][i] - stored[k][i]) for k in lev if k in stored for i in range(min(len(lev[k]), len(stored[k]))))
            out['lam'][f'{lam:g}'] = rec
        None
        store.put(os.path.join(HERE, 'core_strip_results', f'{cfgname}_N{N}_a{alpha:.3f}.json'), out)
        keys = [f'{z}|1|{p}', f'{q1}|0|{p}', f'{q1}|-1|{p}', f'{q1}|1|{p}', f'{z}|0|{p}']
        names = ['D', 'Psi_a', 'Psi_a-1', 'Psi_a+1', 'neutral s=0 (2nd)']
        print(f"{cfgname} N={N} alpha={alpha}: gate (lam=1 vs on_results) {out['lam']['1'].get('gate', 'n/a')}")
        for lam in ('1', '0'):
            r = out['lam'][lam]
            row = []
            for k, nm in zip(keys, names):
                i = 1 if nm.startswith('neutral') else 0
                if k in r['bulk_units']:
                    row.append(f"{nm} {r['bulk_units'][k][i]:.3f}/{r['D_units'][k][i]:.3f}")
            print(f"  lam={lam} (bulk units / D at 2): " + '  '.join(row))


    def cli(argv):
        if argv[1] == 'spectrum':
            mode_spectrum(argv[2], int(argv[3]), float(argv[4]) if len(argv) > 4 else 0.5)
    return namespace(locals(), globals())


@section
def o3flow():
    """RG flow out of the 3d O(3) Wilson-Fisher CFT on the fuzzy sphere, driven by a spin-2 anisotropy.

Model: Guo-Zhou-Wei-He O(3) (twist/onmodel.py, config 'o3wf') plus a traceless flavour field

    H(h, g) = H_GZWH(h) + g W,     W = sum_m ( n_{m,1} + n_{m,2} - 2 n_{m,3} ),

which is exactly the spin-2 (traceless symmetric) component c = (1, 1, -2) of psi_a^+ psi_b.  It
preserves the rotation in the (1, 2) plane (pair charge Q) and the parity P of component 3, and flows
to the O(2) (easy-plane) or Ising (easy-axis) CFT depending on the sign of g.  Since the operator is
traceless, the critical point moves only at O(g^2).

Sectors are labelled (Q, K, P): Q = (1,2)-plane charge, K = L_z, P = parity of component 3.
O(3) content at g = 0:  phi -> phi_+- (Q=+-1, P=+), phi_3 (Q=0, P=-);  t -> t_+-2 (2,+), t_+-1 (1,-),
t_0 (0,+);  s, T, J_12 in (0,+);  J_+-3 in (1,-).
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from scipy.optimize import brentq

    HERE = 'flow'
    M = onmodel()

    CFG = dict(model='wf', Nv=3, U=0.264, pairs=[(1, 2)], twist=[1])
    SECTORS = {                                              # name: (Q, K, P)
        'S0': (0, 0, +1), 'S1': (0, 1, +1), 'S2': (0, 2, +1),
        'P0': (0, 0, -1), 'P1': (0, 1, -1),
        'V0': (1, 0, +1), 'V1': (1, 1, +1),
        'A0': (1, 0, -1), 'A1': (1, 1, -1),
        'B0': (2, 0, +1), 'B1': (2, 1, +1),
    }
    TARGET = {'o3': 0.518942, 'xy': 0.519088, 'ising': 0.518149}   # order-parameter dimensions
    RCFG = dict(model='rotor', Nv=3, U=1.4, pairs=[(1, 2)], twist=[1])   # Dey et al. rotor model (U = v/u)


    class FlowModel:
        def __init__(self, N, cache=('S0', 'P0', 'P1', 'V0', 'V1'), verbose=True, cfg=None):
            self.N = N
            t0 = time.time()
            cfg = cfg or CFG
            self.cfg = cfg
            kw = dict(pairs=cfg['pairs'], twist=cfg['twist'])
            fl, orbs, koff, two, mu0 = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], 0.0, **kw)
            _, _, _, _, mu1 = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], 1.0, **kw)
            self.fl, self.orbs, self.koff, self.two, self.mu0 = fl, orbs, koff, two, mu0
            self.dmu = mu1 - mu0                                 # h direction (psi_0 chemical potential)
            nf = [len(o) for o in orbs]
            off = np.concatenate([[0], np.cumsum(nf)]).astype(int)
            w = np.zeros(off[-1])
            for f, name in enumerate(fl.names):
                c = {'0': 0.0, '3': -2.0}.get(name, 1.0)          # '12+', '12-' -> +1
                w[off[f]:off[f + 1]] = c
            self.wmode = w
            self.cache, self.cache_names = {}, set(cache)
            if verbose:
                print(f"[N={N}] terms built: {len(two)} two-body, flavours {fl.names}  [{time.time()-t0:.1f}s]",
                      flush=True)

        def sector(self, name):
            if name in self.cache:
                return self.cache[name]
            Q, K, P = SECTORS[name]
            t0 = time.time()
            b = M.Basis(self.fl, self.orbs, self.koff, self.N, [Q], K, (P,))
            H0 = M.build_H(b, self.two, self.mu0)
            st = b.states
            nh = np.zeros(b.dim)
            nw = np.zeros(b.dim)
            for k in range(len(self.dmu)):
                occ = ((st >> np.int64(k)) & 1).astype(float)
                if self.dmu[k] != 0:
                    nh += self.dmu[k] * occ
                if self.wmode[k] != 0:
                    nw += self.wmode[k] * occ
            out = dict(dim=b.dim, H0=H0, nh=nh, nw=nw, t=time.time() - t0, nnz=H0.nnz)
            if name in self.cache_names:
                self.cache[name] = out
            return out

        def H(self, name, h, g):
            s = self.sector(name)
            return s['H0'] + sp.diags(h * s['nh'] + g * s['nw'])

        def levels(self, name, h, g, k=1, vecs=False):
            Hm = self.H(name, h, g)
            k = min(k, Hm.shape[0] - 2) if Hm.shape[0] > 3 else Hm.shape[0]
            if Hm.shape[0] <= 600:
                E, U = np.linalg.eigh(Hm.toarray())
                return (E[:k], U[:, :k]) if vecs else E[:k]
            E, U = sla.eigsh(Hm, k=k, which='SA', tol=1e-11)
            o = np.argsort(E)
            return (E[o], U[:, o]) if vecs else E[o]

        # -------------------------------------------------------------- criticality along the flow
        def lightest(self, h, g):
            """which order-parameter component is lighter: 'P' (phi_3) or 'V' (phi_+-)."""
            e0 = self.levels('S0', h, g)[0]
            return 'P' if self.levels('P0', h, g)[0] - e0 <= self.levels('V0', h, g)[0] - e0 else 'V'

        def ratio(self, h, g, comp):
            e0 = self.levels('S0', h, g)[0]
            e1 = self.levels(comp + '0', h, g)[0]
            e2 = self.levels(comp + '1', h, g)[0]          # descendant d(phi) at l = 1
            return (e1 - e0) / (e2 - e0)

        def f_and_dfdh(self, h, g, comp, tgt):
            """tuning function f = (E1-E0)/(E2-E0) - tgt and its exact h-derivative (Hellmann-Feynman)."""
            out = []
            for nm in ('S0', comp + '0', comp + '1'):
                E, U = self.levels(nm, h, g, k=1, vecs=True)
                p = np.abs(U[:, 0]) ** 2
                out.append((E[0], float(p @ self.sector(nm)['nh'])))
            (e0, d0), (e1, d1), (e2, d2) = out
            a, b = e1 - e0, e2 - e0
            return a / b - tgt, ((d1 - d0) * b - a * (d2 - d0)) / b ** 2

        def tune_newton(self, g, h_guess, comp, target, tol=1e-10, maxit=10, max_step=0.02):
            """Newton iteration on the tuning condition; returns (h_c, iterations) or (None, it) on failure."""
            tgt = target / (1 + target)
            h = h_guess
            for it in range(1, maxit + 1):
                f, fp = self.f_and_dfdh(h, g, comp, tgt)
                if not np.isfinite(fp) or fp == 0:
                    return None, it
                step = -f / fp
                if abs(step) > max_step:
                    return None, it
                h += step
                if abs(f) < tol:
                    return h, it
            return None, maxit

        def tune(self, g, h_guess, comp=None, target=None, span=0.02, n=9):
            comp = comp or self.lightest(h_guess, g)
            target = target or TARGET['o3' if g == 0 else ('ising' if comp == 'P' else 'xy')]
            tgt = target / (1 + target)
            f = lambda h: self.ratio(h, g, comp) - tgt
            hs = np.linspace(h_guess - span, h_guess + span, n)
            vals = [f(x) for x in hs]
            br = [(hs[i], hs[i + 1]) for i in range(n - 1) if vals[i] * vals[i + 1] < 0]
            if not br:
                return None, comp, (hs, vals)
            i = min(range(len(br)), key=lambda j: abs(br[j][0] - h_guess))
            return brentq(f, *br[i], xtol=1e-9), comp, None


    def profile(N):
        fm = FlowModel(N, cache=())
        for nm in SECTORS:
            s = fm.sector(nm)
            print(f"  N={N} {nm} {SECTORS[nm]}: dim {s['dim']:>9,}  nnz {s['nnz']:>12,}  build {s['t']:.1f}s",
                  flush=True)


    def cli(argv):
        for N in map(int, argv[1:]):
            profile(N)
    return namespace(locals(), globals())


@section
def gflow():
    """General RG flows out of an O(M) Wilson-Fisher point on the fuzzy sphere, driven by a traceless flavour field.

Model: Guo-Zhou-Wei-He O(M) (twist/onmodel.py, model 'wf') plus g W with W = sum_m sum_a w_a n_{m,a}, a = 1..M,
sum_a w_a = 0, i.e. one component of the rank-two tensor t along diag(w).  The spec of each flow fixes the
flavour pairs kept as U(1) charges, the Z2 parities of the unpaired components, the sectors (Qs, K, parities),
which sector holds the light and the heavy order-parameter component for each sign of g, and the UV/IR
order-parameter dimensions used as tuning targets (they agree to 0.2% for all flows here).

Flows:
  o2   O(2) -> Ising   w = (1, -1)            (g > 0 only: g -> -g is the swap 1 <-> 2)
  o4a  O(4) -> Ising   w = (1, 1, 1, -3), g > 0;   O(4) -> O(3), g < 0
  o4b  O(4) -> XY      w = (1, 1, -1, -1)        (g > 0 only: g -> -g is the swap (12) <-> (34))
  o3   O(3) -> Ising / XY, w = (1, 1, -2): regression test against o3flow.py
Sector naming: S = singlet-like (all charges 0, all parities +); L/H = light/heavy phi component for g > 0.
usage: python gflow.py FLOW N [N ...]     -> sector dimensions
"""
    import os
    import sys
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from scipy.optimize import brentq

    HERE = 'flow'
    M = onmodel()

    DPHI = {'o2': 0.519088, 'o3': 0.518942, 'o4': 0.51812, 'ising': 0.518149}
    SPECS = {
        'o2': dict(cfg=dict(model='wf', Nv=2, U=0.270, pairs=[], twist=[]), w={1: 1.0, 2: -1.0},
                   sectors={'S0': ((), 0, (1, 1)), 'S1': ((), 1, (1, 1)), 'S2': ((), 2, (1, 1)),
                            'L0': ((), 0, (1, -1)), 'L1': ((), 1, (1, -1)),
                            'H0': ((), 0, (-1, 1)), 'H1': ((), 1, (-1, 1)),
                            'X0': ((), 0, (-1, -1)), 'X1': ((), 1, (-1, -1))},
                   light={+1: 'L', -1: 'H'}, uv='o2', ir={+1: 'ising', -1: 'ising'}, signs=(+1,)),
        'o4a': dict(cfg=dict(model='wf', Nv=4, U=0.255, pairs=[(1, 2)], twist=[0]), w={1: 1.0, 2: 1.0, 3: 1.0, 4: -3.0},
                    sectors={'S0': ((0,), 0, (1, 1)), 'S1': ((0,), 1, (1, 1)), 'S2': ((0,), 2, (1, 1)),
                             'L0': ((0,), 0, (1, -1)), 'L1': ((0,), 1, (1, -1)),        # phi_4
                             'H0': ((0,), 0, (-1, 1)), 'H1': ((0,), 1, (-1, 1)),        # phi_3
                             'V0': ((1,), 0, (1, 1)), 'V1': ((1,), 1, (1, 1)),          # phi_+- (same O(3) triplet)
                             'Y0': ((0,), 0, (-1, -1)), 'Y1': ((0,), 1, (-1, -1)),      # t_34 ...
                             'B0': ((2,), 0, (1, 1)), 'B1': ((2,), 1, (1, 1))},         # charge-2 t
                    light={+1: 'L', -1: 'H'}, uv='o4', ir={+1: 'ising', -1: 'o3'}, signs=(+1, -1)),
        'o4b': dict(cfg=dict(model='wf', Nv=4, U=0.255, pairs=[(1, 2), (3, 4)], twist=[0, 0]),
                    w={1: 1.0, 2: 1.0, 3: -1.0, 4: -1.0},
                    sectors={'S0': ((0, 0), 0, ()), 'S1': ((0, 0), 1, ()), 'S2': ((0, 0), 2, ()),
                             'L0': ((0, 1), 0, ()), 'L1': ((0, 1), 1, ()),              # phi of the (34) pair
                             'H0': ((1, 0), 0, ()), 'H1': ((1, 0), 1, ()),              # phi of the (12) pair
                             'M0': ((1, 1), 0, ()), 'M1': ((1, 1), 1, ()),              # t mixing the pairs
                             'B0': ((0, 2), 0, ()), 'B1': ((0, 2), 1, ())},             # charge-2 of the light pair
                    light={+1: 'L', -1: 'H'}, uv='o4', ir={+1: 'o2', -1: 'o2'}, signs=(+1,)),
        'o3': dict(cfg=dict(model='wf', Nv=3, U=0.264, pairs=[(1, 2)], twist=[1]), w={1: 1.0, 2: 1.0, 3: -2.0},
                   sectors={'S0': ((0,), 0, (1,)), 'S1': ((0,), 1, (1,)), 'S2': ((0,), 2, (1,)),
                            'L0': ((0,), 0, (-1,)), 'L1': ((0,), 1, (-1,)),
                            'H0': ((1,), 0, (1,)), 'H1': ((1,), 1, (1,))},
                   light={+1: 'L', -1: 'H'}, uv='o3', ir={+1: 'ising', -1: 'o2'}, signs=(+1, -1)),
    }


    class GFlow:
        def __init__(self, flow, N, cache=None, verbose=True):
            self.flow, self.N = flow, N
            self.spec = sp_ = SPECS[flow]
            cfg = sp_['cfg']
            t0 = time.time()
            kw = dict(pairs=cfg['pairs'], twist=cfg['twist'])
            fl, orbs, koff, two, mu0 = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], 0.0, **kw)
            _, _, _, _, mu1 = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], 1.0, **kw)
            self.fl, self.orbs, self.koff, self.two, self.mu0 = fl, orbs, koff, two, mu0
            self.dmu = mu1 - mu0                                 # h direction (psi_0 chemical potential)
            nf = [len(o) for o in orbs]
            off = np.concatenate([[0], np.cumsum(nf)]).astype(int)
            w = np.zeros(off[-1])
            for f, name in enumerate(fl.names):
                if name == '0':
                    c = 0.0
                elif name[-1] in '+-':                           # a pair 'ab+' / 'ab-': W must be pair-invariant
                    a, b = int(name[0]), int(name[1])
                    assert sp_['w'][a] == sp_['w'][b], (name, sp_['w'])
                    c = sp_['w'][a]
                else:
                    c = sp_['w'][int(name)]
                w[off[f]:off[f + 1]] = c
            assert abs(sum(sp_['w'].values())) < 1e-12
            self.wmode = w
            self.cache = {}
            self.cache_names = set(sp_['sectors'] if cache is None else cache)
            if verbose:
                print(f"[{flow} N={N}] terms built: {len(two)} two-body, flavours {fl.names}  [{time.time()-t0:.1f}s]",
                      flush=True)

        def sector(self, name):
            if name in self.cache:
                return self.cache[name]
            Qs, K, P = self.spec['sectors'][name]
            t0 = time.time()
            b = M.Basis(self.fl, self.orbs, self.koff, self.N, list(Qs), K, tuple(P))
            H0 = M.build_H(b, self.two, self.mu0)
            st = b.states
            nh, nw = np.zeros(b.dim), np.zeros(b.dim)
            for k in range(len(self.dmu)):
                if self.dmu[k] == 0 and self.wmode[k] == 0:
                    continue
                occ = ((st >> np.int64(k)) & 1).astype(float)
                if self.dmu[k] != 0:
                    nh += self.dmu[k] * occ
                if self.wmode[k] != 0:
                    nw += self.wmode[k] * occ
            out = dict(dim=b.dim, H0=H0, nh=nh, nw=nw, t=time.time() - t0, nnz=H0.nnz)
            if name in self.cache_names:
                self.cache[name] = out
            return out

        def H(self, name, h, g):
            s = self.sector(name)
            return s['H0'] + sp.diags(h * s['nh'] + g * s['nw'])

        def levels(self, name, h, g, k=1, vecs=False):
            Hm = self.H(name, h, g)
            if Hm.shape[0] <= 800:
                E, U = np.linalg.eigh(Hm.toarray())
                return (E[:k], U[:, :k]) if vecs else E[:k]
            k = min(k, Hm.shape[0] - 2)
            E, U = sla.eigsh(Hm, k=k, which='SA', tol=1e-11)
            o = np.argsort(E)
            return (E[o], U[:, o]) if vecs else E[o]

        def light(self, g):
            return self.spec['light'][+1 if g >= 0 else -1]

        def heavy(self, g):
            return self.spec['light'][-1 if g >= 0 else +1]

        def f_and_dfdh(self, h, g, comp, tgt):
            """tuning function f = (E_phi - E_0)/(E_dphi - E_0) - tgt and its exact h-derivative."""
            out = []
            for nm in ('S0', comp + '0', comp + '1'):
                E, U = self.levels(nm, h, g, k=1, vecs=True)
                p = np.abs(U[:, 0]) ** 2
                out.append((E[0], float(p @ self.sector(nm)['nh'])))
            (e0, d0), (e1, d1), (e2, d2) = out
            a, b = e1 - e0, e2 - e0
            return a / b - tgt, ((d1 - d0) * b - a * (d2 - d0)) / b ** 2

        def tune_newton(self, g, h_guess, comp, target, tol=1e-10, maxit=12, max_step=0.03):
            tgt = target / (1 + target)
            h = h_guess
            for it in range(1, maxit + 1):
                f, fp = self.f_and_dfdh(h, g, comp, tgt)
                if not np.isfinite(fp) or fp == 0:
                    return None, it
                step = -f / fp
                if abs(step) > max_step:
                    return None, it
                h += step
                if abs(f) < tol:
                    return h, it
            return None, maxit

        def ratio(self, h, g, comp):
            e0 = self.levels('S0', h, g)[0]
            return (self.levels(comp + '0', h, g)[0] - e0) / (self.levels(comp + '1', h, g)[0] - e0)

        def tune_scan(self, g, h_guess, comp, target, span=0.02, n=9):
            tgt = target / (1 + target)
            f = lambda h: self.ratio(h, g, comp) - tgt
            hs = np.linspace(h_guess - span, h_guess + span, n)
            vals = [f(x) for x in hs]
            br = [(hs[i], hs[i + 1]) for i in range(n - 1) if vals[i] * vals[i + 1] < 0]
            if not br:
                return None, (hs, vals)
            i = min(range(len(br)), key=lambda j: abs(br[j][0] - h_guess))
            return brentq(f, *br[i], xtol=1e-10), None


    def profile(flow, N):
        fm = GFlow(flow, N, cache=())
        for nm, sec in SPECS[flow]['sectors'].items():
            s = fm.sector(nm)
            print(f"  {flow} N={N} {nm} {sec}: dim {s['dim']:>10,}  nnz {s['nnz']:>13,}  build {s['t']:.1f}s", flush=True)


    def cli(argv):
        for N in map(int, argv[2:]):
            profile(argv[1], N)
    return namespace(locals(), globals())


@section
def test_twist():
    """test_twist"""
    HERE = 'twist'
    import sys
    import time
    import numpy as np
    import scipy.sparse.linalg as sla
    from mpmath import mp, quad, cos, sin, pi as mpi, gamma, sqrt, diff


    def cli(argv):

        ok = True
        def check(name, cond):
            nonlocal ok
            print(('PASS ' if cond else 'FAIL ') + name, flush=True); ok = ok and bool(cond)

        # 1. closed-form four-orbital integrals vs direct quadrature (incl. half-integer offsets)
        mp.dps = 30
        def F(s, m1, m4, th):          # real theta-part of Phi*_{m1} Phi_{m4}
            N = lambda m: sqrt(gamma(2*s+2) / (4*mpi*gamma(s+m+1)*gamma(s-m+1)))
            return N(m1)*N(m4) * cos(th/2)**(2*s+m1+m4) * sin(th/2)**(2*s-m1-m4)
        def direct(s, m1, m2, m3, m4):
            mu = m4 - m1
            Id = 2*mpi*quad(lambda t: F(s,m1,m4,t)*F(s,m2,m3,t)*sin(t), [0, mpi])
            Il = -2*mpi*quad(lambda t: (diff(lambda x: F(s,m1,m4,x), t)*diff(lambda x: F(s,m2,m3,x), t)
                                        + mu**2*F(s,m1,m4,t)*F(s,m2,m3,t)/sin(t)**2)*sin(t), [0, mpi])
            return float(Id), float(Il)
        worst = 0
        for (s, q) in [(2.5, (1.5, -0.5, 0.5, 0.5)), (2.5, (2.0, -1.0, 1.0, 0.0)),   # twisted (half-int offsets)
                       (2.5, (2.5, -1.5, 0.5, 0.5)), (3.0, (1, 0, 0, 1)), (3.0, (2, -2, -1, 1)),
                       (3.0, (1.5, 0.5, -0.5, 2.5)), (3.5, (0.0, 1.0, 2.0, -1.0))]:
            a = four_orbital(s, *q); b = direct(s, *q)
            worst = max(worst, abs(a[0]-b[0]), abs(a[1]-b[1]))
        check(f"closed-form integrals vs quadrature (7 cases, half-integer offsets incl.): max err {worst:.1e}", worst < 1e-10)

        # 2. couplings reproducing (V0,V1) = (4.75, 1); V2 = 0
        for N in (6, 8):
            G0, G1, A = couplings_for(N, 4.75, 1.0)
            s = (N-1)/2; ms = orbital_ms(N, False)
            trip = [(ms[-1], ms[-3]), (ms[-2], ms[-2]), (ms[-3], ms[-1])]
            M = np.array([[G0*four_orbital(s, m1, m2, m3, m4)[0] + G1*four_orbital(s, m1, m2, m3, m4)[1]
                           for (m4, m3) in trip] for (m1, m2) in trip])
            ev = np.sort(np.linalg.eigvalsh((M+M.T)/2))
            check(f"N={N}: G0={G0:.4f} G1={G1:.4f}; pair spectrum at M=2s-2 = {ev.round(6)} (want 0,1,4.75)",
                  np.allclose(ev, [0, 1, 4.75], atol=1e-9))

        # 3. twist OFF reproduces the pipeline Ising spectrum (x-basis vs z-basis)
        for N in (6, 8):
            s = (N-1)/2; h = 3.16
            G0, G1, _ = couplings_for(N, 4.75, 1.0)
            terms = interaction_terms(N, False, G0, G1)
            ev = []
            for z2 in (+1, -1):
                b = TwistBasis(N, False, 0, z2); H = build_H(b, terms, h)
                check(f"N={N} Z2={z2:+d}: Hermitian ||H-H^T||={abs(H-H.T).max():.1e}", abs(H-H.T).max() < 1e-12)
                ev += list(np.linalg.eigvalsh(H.toarray()) if b.dim < 3000 else sla.eigsh(H, k=20, which='SA')[0])
            ev = np.sort(ev)[:20]
            pb = FastBasis(s, N, int(round(N*s)))
            Hp = build_hamiltonian_fast(pb, [4.75, 1.0], h)
            ep = np.sort(np.linalg.eigvalsh(Hp.toarray()) if pb.dim < 3000 else sla.eigsh(Hp, k=20, which='SA')[0])[:20]
            check(f"N={N}: twist-off spectrum == pipeline Ising, max|dE| = {np.abs(ev-ep).max():.1e}", np.abs(ev-ep).max() < 1e-9)

        # 4. twist ON: Hermitian, and L_z -> -L_z degeneracy (R_y(pi) symmetry)
        for N in (6, 8):
            G0, G1, _ = couplings_for(N, 4.75, 1.0)
            terms = interaction_terms(N, True, G0, G1)
            for (z2, tl) in ((+1, 2), (-1, 1), (-1, 3)):
                bp, bm = TwistBasis(N, True, tl, z2), TwistBasis(N, True, -tl, z2)
                Hp, Hm = build_H(bp, terms, 3.16), build_H(bm, terms, 3.16)
                herm = max(abs(Hp-Hp.T).max(), abs(Hm-Hm.T).max())
                kk = min(12, bp.dim-2)
                ep = np.sort(np.linalg.eigvalsh(Hp.toarray()))[:kk] if bp.dim < 3000 else np.sort(sla.eigsh(Hp, k=kk, which='SA')[0])
                em = np.sort(np.linalg.eigvalsh(Hm.toarray()))[:kk] if bm.dim < 3000 else np.sort(sla.eigsh(Hm, k=kk, which='SA')[0])
                check(f"N={N} twist Z2={z2:+d} 2Lz=+-{tl}: dims {bp.dim}/{bm.dim}, herm {herm:.1e}, "
                      f"spectra(+Lz) == spectra(-Lz): {np.abs(ep-em).max():.1e}", herm < 1e-12 and np.abs(ep-em).max() < 1e-9)
        print('\nALL PASS' if ok else '\nFAILURES')
    return namespace(locals(), globals())


@section
def test_o2():
    """test_o2"""
    HERE = 'twist'
    import sys
    import time
    import numpy as np
    import scipy.sparse.linalg as sla
    from scipy.optimize import brentq


    def cli(argv):
        O2Basis = o2twist().O2Basis
        build_terms = o2twist().build_terms
        build_H = o2twist().build_H

        def low(b, H, k):
            if b.dim <= 2000:
                return np.linalg.eigvalsh(H.toarray())[:k]
            return np.sort(sla.eigsh(H, k=min(k, b.dim - 2), which='SA', tol=1e-12)[0])

        paper = {6: 2.9600, 7: 2.9075, 8: 2.8747, 9: 2.8516, 10: 2.8347}
        target = 0.519088 / 1.519088
        for N in [int(x) for x in argv[1:]] or [6, 7, 8]:
            t0 = time.time()
            G0, G1, _ = couplings_for(N, 4.0, 1.0)
            two, U = build_terms(N, 0, G0, G1)
            b00, b10, b11 = O2Basis(N, 0, 0, 0), O2Basis(N, 0, 1, 0), O2Basis(N, 0, 1, 1)
            # D enters only on the diagonal: build once at D=0, add D * (n_+ + n_-) later
            H0 = {b: build_H(b, two, U, 0.0) for b in (b00, b10, b11)}
            nD = {b: sum(((b.states >> np.int64(b.off[f] + j)) & 1) for f in (0, 2) for j in range(b.nf[f])).astype(float)
                  for b in (b00, b10, b11)}
            herm = max(abs(H - H.T).max() for H in H0.values())
            import scipy.sparse as sp
            def ratio(D):
                E = {b: low(b, H0[b] + sp.diags(D * nD[b]), 1)[0] for b in (b00, b10, b11)}
                return (E[b10] - E[b00]) / (E[b11] - E[b00]) - target
            Dc = brentq(ratio, 2.0, 4.0, xtol=1e-6)
            print(f"N={N}: dims {b00.dim}/{b10.dim}/{b11.dim}  herm {herm:.1e}  D_c = {Dc:.4f}   paper {paper.get(N,'-')}"
                  f"   ({time.time()-t0:.0f}s)", flush=True)
    return namespace(locals(), globals())


if __name__ == '__main__':
    if sys.argv[1:2] == ['test']:
        sys.argv = ['test_twist']
        test_twist().cli(sys.argv)
        sys.argv = ['test_o2'] + sys.argv[2:]
        test_o2().cli(sys.argv)
    else:
        print(__doc__)
