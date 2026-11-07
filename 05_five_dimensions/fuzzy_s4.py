"""Can a monodromy defect be built on the fuzzy four-sphere, and how close is it to the free five-dimensional defect?

A codimension-two defect in five dimensions meets the radial S^4 on a great S^2. The odd flavour is given holonomy -1
around it by trial sections w^(1/2) x^m and wbar^(1/2) x^m of the Yang-monopole bundle; a separated band of the angular
form in this twisted trial space is the orbital projector (6, 14 and 26 orbitals at I = 2, 3, 4), and local interactions
projected onto it give the many-body Hamiltonian (paper, Sec. 6 and App. F). The tests are against the free 5d half-twist:
the spectrum with the displacement fixed at four (Table 25), the parallel SO(3) spin of the descendants, the bulk-to-defect
overlaps, the equal-time defect correlator and the tilt Ward identity (App. F.4). The density weights of the projected
lowest Landau level test its spatial resolution (App. F.3, audit_s4d), and the first moment of the energy density along
the defect tests the descendant tower with a parallel conformal generator (App. F.5, Table 26, lambda5 and audit_s4d).

The bands (save_band) and interaction terms (build_I, build_I3) are large files rather than stored results; the steps
rebuild them into s4d_files/ next to this file, where the other steps read them. The DMRG steps take hours to days; their
outputs are stored, and the report prints the paper's numbers from them.

usage: python fuzzy_s4.py <step> [args]      (python fuzzy_s4.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401

# The research scripts of this question read and wrote results by paths relative to their folder s4d/; with this base
# the store finds them under s4d/... in fuzzy_s4.json. Their large intermediate files (twisted bands band_I*.npz,
# 12-56 MB; interaction terms terms_*.pkl, up to 13 MB; compressed MPOs mpo_cache/) live in s4d_files/.
store.base = 's4d/'
FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's4d_files')


@section
def s3lll():
    """Fuzzy three-sphere (Meng, Hu, Zhu arXiv:2607.11371) one-particle problem, with an optional
Aharonov-Bohm flux along a great circle of S^3.

Meron gauge.  One-particle states are C^{I+1}-valued functions on S^3 = SU(2) (isospin S = I/2, I even),
and the SO(4) Landau Hamiltonian is H_1 = 2J^2 + 2K^2 - S^2 with J = L (left translations) and
K = R + S (right translations plus isospin).  On functions L^2 = R^2, so H_1 = (2R + S)^2.  Its lowest
level is the SO(4) irrep (I/4, I/4) at H_1 = I/2 (degeneracy D = (I/2+1)^2).

Coordinates.  X = [[a, -b*], [b, a*]], a = sqrt(1-x) e^{i xi1}, b = sqrt(x) e^{i xi2}, x in [0, 1]; the
normalized Haar measure is dx dxi1 dxi2 / (2 pi)^2.  The defect circle is C = {b = 0} (x = 0).  Rotations
about C shift xi2 (transverse), rotations along C shift xi1 (parallel).  For f = e^{i(p xi1 + t xi2)} g(x):

    R^z = (p + t)/2,   L^z = (p - t)/2,
    R^+ : (p, t, g) -> (p+1, t+1, -sqrt(x(1-x)) g' + (t/2) sqrt((1-x)/x) g - (p/2) sqrt(x/(1-x)) g),
    R^- : (p, t, g) -> (p-1, t-1, +sqrt(x(1-x)) g' + (t/2) sqrt((1-x)/x) g - (p/2) sqrt(x/(1-x)) g).

Jacobi basis g = x^{|t|/2} (1-x)^{|p|/2} P_n^{(|t|,|p|)}(1-2x), with R^2 = r(r+1), r = n + (|p|+|t|)/2.

Flux.  Charge-q fields get AB flux q*alpha along C: t in Z + q*alpha, with the regular (Friedrichs) boundary
condition g ~ x^{|t|/2} at C.  A sector (P, T) = (K^z + L^z, K^z - L^z) (parallel momentum, transverse spin)
has components sigma = -S..S with (p, t) = (P - sigma, T - sigma).  H_1 is assembled from its quadratic form
||(2R^z + S^z) psi||^2 + (||(2R^+ + S^+) psi||^2 + ||(2R^- + S^-) psi||^2)/2, so the singular derivatives
at C never have to lie in the basis.

Quadrature.  x = u^2 and Gauss-Legendre in u.  For q*alpha in {0, 1/2} every integrand below is then a
polynomial in u, so the quadrature is exact.

From the fuzzy-S^3 code: inter.py takes its Clebsch-Gordan coefficients from here.
"""
    import numpy as np
    from functools import lru_cache
    from scipy.special import eval_jacobi, roots_legendre
    from sympy.physics.wigner import clebsch_gordan
    from sympy import Rational
    NQ = 96
    _u, _w = roots_legendre(NQ)
    U = 0.5 * (_u + 1.0)
    X = U * U
    WX = 0.5 * _w * 2.0 * U              # dx = 2u du on [0, 1]


    def jac(n, a, b, x):
        return eval_jacobi(n, a, b, 1.0 - 2.0 * x) if n >= 0 else np.zeros_like(x)


    def djac(n, a, b, x):
        """d/dx P_n^{(a,b)}(1 - 2x)."""
        if n == 0:
            return np.zeros_like(x)
        return -(n + a + b + 1.0) * eval_jacobi(n - 1, a + 1, b + 1, 1.0 - 2.0 * x)


    def basis_fn(p, t, n):
        """(g, h_plus, h_minus) on the grid: g normalized, h_pm = radial part of R^pm (e^{i(p xi1+t xi2)} g)."""
        a, b = abs(t) / 2.0, abs(p) / 2.0
        P = jac(n, abs(t), abs(p), X)
        dP = djac(n, abs(t), abs(p), X)
        g = X**a * (1 - X)**b * P
        nrm = np.sqrt(np.sum(WX * g * g))
        pre = X**(a - 0.5) * (1 - X)**(b - 0.5)
        Ap, Bp = (t - abs(t)) / 2.0, (abs(p) - p) / 2.0
        Am, Bm = (t + abs(t)) / 2.0, (p + abs(p)) / 2.0
        hp = pre * (Ap * (1 - X) * P + Bp * X * P - X * (1 - X) * dP)
        hm = pre * (Am * (1 - X) * P - Bm * X * P + X * (1 - X) * dP)
        return g / nrm, hp / nrm, hm / nrm


    @lru_cache(maxsize=None)
    def cg(j1, m1, j2, m2, J, M):
        """<j1 m1; j2 m2 | J M> (arguments as floats with exact half-integer values)."""
        R = lambda v: Rational(int(round(2 * v)), 2)
        if abs(m1 + m2 - M) > 1e-9 or abs(m1) > j1 + 1e-9 or abs(m2) > j2 + 1e-9 or abs(M) > J + 1e-9:
            return 0.0
        return float(clebsch_gordan(R(j1), R(j2), R(J), R(m1), R(m2), R(M)))


    def sector_H(I, P, T, nmax):
        """H_1 in sector (P, T): returns (H, comps, nb) with basis index (sigma_index, n)."""
        S = I / 2.0
        sig = [-S + k for k in range(int(round(2 * S)) + 1)]
        ns = nmax + 1
        dim = len(sig) * ns
        kap = (P + T) / 2.0
        G, HP, HM = {}, {}, {}
        for a, s in enumerate(sig):
            p, t = P - s, T - s
            for n in range(ns):
                G[a, n], HP[a, n], HM[a, n] = basis_fn(p, t, n)
        H = np.zeros((dim, dim))
        idx = lambda a, n: a * ns + n
        for a, s in enumerate(sig):                      # ||(2R^z + S^z) psi||^2, orthonormal basis
            for n in range(ns):
                H[idx(a, n), idx(a, n)] += (2 * kap - s) ** 2
        cp = lambda s: np.sqrt(S * (S + 1) - s * (s + 1))   # S^+ |s> = cp(s) |s+1>
        cm = lambda s: np.sqrt(S * (S + 1) - s * (s - 1))   # S^- |s> = cm(s) |s-1>
        for b, so in enumerate(sig):                      # output component sigma' = so
            for H_R, src, coef in ((HP, b - 1, cp), (HM, b + 1, cm)):
                M = np.zeros((NQ, dim))
                for n in range(ns):
                    M[:, idx(b, n)] += 2.0 * H_R[b, n]
                if 0 <= src < len(sig):
                    c = coef(sig[src])
                    for n in range(ns):
                        M[:, idx(src, n)] += c * G[src, n]
                H += 0.5 * (M.T * WX) @ M
        return H, sig, ns, G


    def solve_sector(I, P, T, nmax, k=None):
        H, sig, ns, G = sector_H(I, P, T, nmax)
        E, V = np.linalg.eigh((H + H.T) / 2)
        k = k or len(E)
        orbs = []
        for i in range(min(k, len(E))):
            comps = []
            for a, s in enumerate(sig):
                gv = sum(V[a * ns + n, i] * G[a, n] for n in range(ns))
                comps.append(gv)
            orbs.append(np.array(comps))           # shape (2S+1, NQ): radial parts, phases (P-s, T-s)
        return E, orbs


    def spectrum(I, qa, nmax=8, span=None, nlow=3):
        """Lowest `nlow` levels of every sector (P, T), P in Z, T in Z + qa."""
        j = I / 4.0
        span = span or int(np.ceil(2 * j)) + 2
        out = []
        for P in range(-span, span + 1):
            for k in range(-span - 1, span + 1):
                T = k + qa
                if abs(T) > span + 0.6:
                    continue
                E, orbs = solve_sector(I, P, T, nmax, k=nlow)
                for i in range(min(nlow, len(E))):
                    out.append((E[i], P, T, orbs[i]))
        out.sort(key=lambda r: r[0])
        return out
    return namespace(locals(), globals())


@section
def s3model():
    """Many-body fuzzy-S^3 Ising model (bulk and Z2 twist surface defect) from s3lll orbitals.

Interaction.  Fibre-channel contact terms on S^3: with psi_{up,dn}(X) in V_{I/2} (meron gauge),
    H_int = sum_L U_L int dX sum_M (psi_up psi_dn)_{L M}^dag (psi_up psi_dn)_{L M},
    (psi psi')_{L M} = sum CG(S s; S s' | L M) psi_s psi'_{s'},  S = I/2.
L = I is the symmetric top channel, i.e. the contact interaction on the total space S^3 x S^3_fibre; on
the even-I lowest level it is exactly the V_{00} pseudopotential of Meng-Hu-Zhu, and L = I-1 is exactly
V_{01} = V_{10} (tested in test_model.py against their Clebsch-Gordan construction).  This real-space form
is what lets the same Hamiltonian act on twisted orbitals.

Flavours psi_pm = (psi_up +- psi_dn)/sqrt2 (Ising Z2: psi_- -> -psi_-), transverse field
-h sum psi^dag sigma^x psi = -h (N_+ - N_-).  The Z2 twist gives psi_- half a unit of AB flux along the
great circle C = {b = 0}.  Conserved: N_e, P (parallel momentum along C), T (transverse spin about C),
and (-1)^{N_-}.

From the fuzzy-S^3 code: the s4d scripts use its Fock basis class Basis.
"""
    import numpy as np
    import scipy.sparse.linalg as sla
    L = globals()['s3lll']()   # was: import s3lll as L
    _s3lll = globals()['s3lll'](); WX, NQ, cg = _s3lll.WX, _s3lll.NQ, _s3lll.cg   # was: from s3lll import WX, NQ, cg
    import sys
    pass   # was: sys.path.insert(0, '<research tree>/twist')
    _onmodel = fuzzy.onmodel(); build_H = _onmodel.build_H   # was: from onmodel import build_H


    def sectors(I, qa, span=None):
        S = I / 2.0
        fr = S % 1.0
        span = span or int(np.ceil(I / 2)) + 2
        out = []
        for kP in range(-span - 1, span + 2):
            for kT in range(-span - 1, span + 2):
                P, T = kP + fr, kT + fr + qa
                if abs(P) <= span + 0.6 and abs(T) <= span + 0.6:
                    out.append((P, T))
        return out


    def band(I, qa, nband, nmax=8, nlow=3):
        """The lowest `nband` one-particle states: list of (E, P, T, comps)."""
        out = []
        for (P, T) in sectors(I, qa):
            E, orbs = L.solve_sector(I, P, T, nmax, k=nlow)
            out += [(E[i], P, T, orbs[i]) for i in range(min(nlow, len(E)))]
        out.sort(key=lambda r: r[0])
        return out[:nband], out


    def fibre_pairs(I, orbs, Ls):
        """G[L][(i,j)] = array (2L+1, NQ) of (Psi_i x Psi_j)_{L M}(x); keyed groups by (P_i+P_j, T_i+T_j)."""
        S = I / 2.0
        sig = [-S + a for a in range(int(round(2 * S)) + 1)]
        n = len(orbs)
        G = {Lc: {} for Lc in Ls}
        groups = {}
        for i in range(n):
            for j in range(n):
                key = (round(2 * (orbs[i]['P'] + orbs[j]['P'])), round(4 * (orbs[i]['T'] + orbs[j]['T'])))
                groups.setdefault(key, []).append((i, j))
                for Lc in Ls:
                    arr = np.zeros((2 * Lc + 1, NQ))
                    for iM, M in enumerate(np.arange(-Lc, Lc + 1)):
                        for a, s in enumerate(sig):
                            s2 = M - s
                            b = int(round(s2 + S))
                            if 0 <= b < len(sig):
                                c = cg(S, s, S, s2, Lc, M)
                                if c != 0.0:
                                    arr[iM] += c * orbs[i]['comps'][a] * orbs[j]['comps'][b]
                    G[Lc][i, j] = arr
        return G, groups


    def interaction_terms(I, orbs, U, tol=1e-12):
        """Two-body terms (k1, k2, k3, k4, V) for V c+_k1 c+_k2 c_k3 c_k4.  U = {L: U_L}."""
        Ls = list(U)
        G, groups = fibre_pairs(I, orbs, Ls)
        sgn = [1.0 if o['f'] == 0 else -1.0 for o in orbs]
        terms = {}
        for key, prs in groups.items():
            for Lc in Ls:
                A = np.array([G[Lc][p].ravel() for p in prs])          # (npairs, (2L+1)*NQ)
                Wt = np.tile(WX, 2 * Lc + 1)
                V = (A * Wt) @ A.T
                for a, (k, l) in enumerate(prs):
                    for b, (kk, ll) in enumerate(prs):
                        v = U[Lc] * sgn[l] * sgn[ll] / 4.0 * V[a, b]
                        if abs(v) > tol:
                            t = (l, k, kk, ll)
                            terms[t] = terms.get(t, 0.0) + v
        return [(a, b, c, d, v) for (a, b, c, d), v in terms.items() if abs(v) > tol]


    class Basis:
        """Fock states over modes ordered [+ orbitals, - orbitals]; sector (Ne, 2P, 4T, parity of N_-)."""

        def __init__(self, orbs, Ne, P, T, z2):
            fl = [o['f'] for o in orbs]
            plus = [k for k in range(len(orbs)) if fl[k] == 0]
            minus = [k for k in range(len(orbs)) if fl[k] == 1]
            assert plus == list(range(len(plus))) and minus == list(range(len(plus), len(orbs)))
            Pi = [int(round(2 * o['P'])) for o in orbs]
            Ti = [int(round(4 * o['T'])) for o in orbs]

            def groups(modes):
                g = {}
                n = len(modes)
                for Sb in range(1 << n):
                    c, p, t = 0, 0, 0
                    for q in range(n):
                        if Sb >> q & 1:
                            c += 1; p += Pi[modes[q]]; t += Ti[modes[q]]
                    g.setdefault((c, p, t), []).append(Sb)
                return {k: np.array(v, np.int64) for k, v in g.items()}
            gp, gm = groups(plus), groups(minus)
            tP, tT = int(round(2 * P)), int(round(4 * T))
            out = []
            for (cm, pm, tm), Sm in gm.items():
                if (cm % 2 == 0) != (z2 == +1):
                    continue
                key = (Ne - cm, tP - pm, tT - tm)
                if key in gp:
                    out.append((gp[key][None, :] | (Sm[:, None] << np.int64(len(plus)))).ravel())
            self.states = np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)
            self.dim = len(self.states)
            self.nmodes = len(orbs)


    def hamiltonian(orbs, terms, h, Ne, P, T, z2):
        B = Basis(orbs, Ne, P, T, z2)
        if B.dim == 0:
            return None, B
        mu = np.array([-h if o['f'] == 0 else h for o in orbs])
        return build_H(B, terms, mu), B


    def levels(Hm, k):
        n = Hm.shape[0]
        if n <= 3000:
            return np.linalg.eigvalsh(Hm.toarray())[:k]
        return np.sort(sla.eigsh(Hm, k=min(k, n - 2), which='SA', tol=1e-10)[0])


    def make_orbs(I, qa, nband_minus=None, nmax=8):
        """+ orbitals: untwisted lowest level; - orbitals: lowest nband_minus twisted states (qa = 0: same)."""
        D = (I // 2 + 1) ** 2 if I % 2 == 0 else (I + 1) * (I + 3) // 2
        bp, _ = band(I, 0.0, D, nmax=nmax)
        orbs = [dict(f=0, P=r[1], T=r[2], comps=r[3], E=r[0]) for r in bp]
        if qa == 0.0:
            bm = bp
        else:
            bm, _ = band(I, qa, nband_minus or D, nmax=nmax)
        orbs += [dict(f=1, P=r[1], T=r[2], comps=r[3], E=r[0]) for r in bm]
        return orbs, D
    return namespace(locals(), globals())


@section
def geom():
    """Fuzzy S^4 (SO(5) Landau problem, Yang monopole of isospin I/2) with an optional Z2 monodromy defect along
a great S^2, solved on the total space S^7 -> S^4.

Conventions.  q1 = z1 + j z2, q2 = z3 + j z4 (complex structure = right multiplication by i).  Hopf map
x5 = |z1|^2+|z2|^2-|z3|^2-|z4|^2,  u := x1+i x2 = 2(z1 z3* + z2* z4),  w := x3+i x4 = 2(z2 z3* - z1* z4),
i.e. x_a = z^dag Gamma_a z.  Spin(5) = Sp(2) acts C-linearly on z by m_ab = [Gamma_a, Gamma_b]/4, inducing
x_a -> x_b, x_b -> -x_a.  On functions on S^7, X_ab f = d/de f((1 + e m_ab) psi) (a derivation); on scalar
functions of x it is x_b d_a - x_a d_b.  The Landau Hamiltonian is the Casimir C2 = sum_{a<b} ||X_ab f||^2
(spinor 5/2, vector 4), up to a constant; its lowest level is Sym^I(C^4) at C2 = I^2/2 + 2I.

Sections.  f = sum_a z^a g_a(x) with z^a the degree-I monomials (charge I under the fibre U(1), annihilated
by the fibre raising operator) and g_a scalar functions on S^4; every section is of this form.  Fibre over x:
psi0(x) lambda, lambda in SU(2); z(psi0 lambda) = A(x) (alpha, beta)^T, so z^a restricted to the fibre is
sum_mu R_{a mu}(x) alpha^{I-mu} beta^mu, with fibre norms D_mu = (I-mu)! mu!/(I+1)!.  Orthonormal fibre
components of f: s(x) = D^{1/2} R(x)^T G(x).

Defect: the great S^2 = {w = 0}.  Torus charges (parallel P rotates u, transverse T rotates w):
z_i -> P = (+1/2, -1/2, -1/2, +1/2), T = (-1/2, +1/2, -1/2, +1/2).  Twisted g's carry w^{1/2} or wbar^{1/2}.
Torus-invariant integrands (matrix elements within a (P,T) block, contracted four-orbital products) depend
only on (chi, theta), with x = (sqrt(1-y) sin th, -, sqrt(y), -, sqrt(1-y) cos th), y = sin^2 chi, at
phi_parallel = phi_transverse = 0; measure cos^2 chi sin chi sin th dchi dth.
"""
    import numpy as np
    from math import factorial
    from itertools import product
    from scipy.special import roots_legendre
    # ---------------------------------------------------------------- Clifford algebra
    G = np.zeros((5, 4, 4), complex)
    def _set(g, entries):
        for (i, j), v in entries.items():
            g[i - 1, j - 1] = v
    _set(G[0], {(3, 1): 1, (1, 3): 1, (2, 4): 1, (4, 2): 1})                 # x1
    _set(G[1], {(3, 1): -1j, (1, 3): 1j, (2, 4): -1j, (4, 2): 1j})           # x2
    _set(G[2], {(3, 2): 1, (2, 3): 1, (1, 4): -1, (4, 1): -1})               # x3
    _set(G[3], {(3, 2): -1j, (2, 3): 1j, (1, 4): 1j, (4, 1): -1j})           # x4
    G[4] = np.diag([1, 1, -1, -1])                                            # x5
    PAIRS = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    MGEN = [0.25 * (G[a] @ G[b] - G[b] @ G[a]) for (a, b) in PAIRS]
    QP = np.array([0.5, -0.5, -0.5, 0.5]); QT = np.array([-0.5, 0.5, -0.5, 0.5])


    def check_clifford():
        err = max(np.abs(G[a] @ G[b] + G[b] @ G[a] - 2 * (a == b) * np.eye(4)).max() for a in range(5) for b in range(5))
        herm = max(np.abs(g - g.conj().T).max() for g in G)
        cas = -sum(m @ m for m in MGEN)
        return err, herm, np.abs(cas - 2.5 * np.eye(4)).max()


    # ---------------------------------------------------------------- monomials and lifts
    def monomials(I):
        return [a for a in product(range(I + 1), repeat=4) if sum(a) == I]


    def lift(m, mons):
        """Matrix of the derivation sum_ij m_ij z_j d/dz_i on degree-I monomials: (X z^a) = sum_b M[b,a] z^b."""
        idx = {a: k for k, a in enumerate(mons)}
        n = len(mons); M = np.zeros((n, n), complex)
        for k, a in enumerate(mons):
            for i in range(4):
                if a[i] == 0: continue
                for j in range(4):
                    if m[i, j] == 0: continue
                    b = list(a); b[i] -= 1; b[j] += 1
                    M[idx[tuple(b)], k] += m[i, j] * a[i]
        return M


    def charges(a):
        a = np.array(a, float)
        return float(QP @ a), float(QT @ a)


    # ---------------------------------------------------------------- grid and fibre frames
    class Grid:
        def __init__(self, I, nchi=48, nth=48):
            self.I = I
            xc, wc = roots_legendre(nchi); chi = (xc + 1) * np.pi / 4; wchi = wc * np.pi / 4
            xt, wt = roots_legendre(nth); th = (xt + 1) * np.pi / 2; wth = wt * np.pi / 2
            CH, TH = np.meshgrid(chi, th, indexing='ij')
            self.W = (np.outer(wchi, wth) * np.cos(CH) ** 2 * np.sin(CH) * np.sin(TH)).ravel()
            y = np.sin(CH.ravel()) ** 2
            self.y = y
            self.u = np.sqrt(1 - y) * np.sin(TH.ravel())         # x1 + i x2 at phi_par = 0
            self.w = np.sqrt(y)                                   # x3 + i x4 at phi_tr = 0
            self.x5 = np.sqrt(1 - y) * np.cos(TH.ravel())
            self.X = np.stack([self.u, 0 * self.u, self.w, 0 * self.w, self.x5])   # real Cartesian x
            self.npt = len(self.W)
            # gauge psi0(x) = (u, w, (1-x5), 0)/sqrt(2(1-x5))
            s = np.sqrt(2 * (1 - self.x5))
            z = np.stack([self.u / s, self.w / s, (1 - self.x5) / s, 0 * s]).astype(complex)
            self.z0 = z
            # A(x): z(psi0 lambda) = A (alpha, beta)^T with A = [[z1,-z2*],[z2,z1*],[z3,-z4*],[z4,z3*]]
            A = np.zeros((self.npt, 4, 2), complex)
            A[:, 0, 0], A[:, 0, 1] = z[0], -z[1].conj()
            A[:, 1, 0], A[:, 1, 1] = z[1], z[0].conj()
            A[:, 2, 0], A[:, 2, 1] = z[2], -z[3].conj()
            A[:, 3, 0], A[:, 3, 1] = z[3], z[2].conj()
            self.A = A
            self.mons = monomials(I)
            self.nmon = len(self.mons)
            self.D = np.array([factorial(I - mu) * factorial(mu) / factorial(I + 1) for mu in range(I + 1)])
            # R[p, a, mu]: coefficient of alpha^{I-mu} beta^mu in z^a(A (alpha,beta))
            R = np.zeros((self.npt, self.nmon, I + 1), complex)
            for k, a in enumerate(self.mons):
                poly = np.ones((self.npt, 1), complex)
                for i in range(4):
                    for _ in range(a[i]):
                        new = np.zeros((self.npt, poly.shape[1] + 1), complex)
                        new[:, :-1] += poly * A[:, i, 0][:, None]
                        new[:, 1:] += poly * A[:, i, 1][:, None]
                        poly = new
                R[:, k, :] = poly
            self.R = R
            self.F = R * np.sqrt(self.D)[None, None, :]          # fibre values of monomials: F[p, a, mu]
            self.Mlift = [lift(m, self.mons) for m in MGEN]
            self.FM = [np.einsum('pbm,ba->pam', self.F, M) for M in self.Mlift]   # fibre values of X_A z^a


    # ---------------------------------------------------------------- scalar functions on S^4
    class Scalar:
        """phi = u^c1 ubar^c2 w^c3 wbar^c4 x5^c5 (c3, c4 may be half-integers)."""
        def __init__(self, c):
            self.c = tuple(float(v) for v in c)

        @property
        def charges(self):
            c1, c2, c3, c4, c5 = self.c
            return c1 - c2, c3 - c4

        def value(self, g):
            c1, c2, c3, c4, c5 = self.c
            return (g.u ** c1) * (g.u ** c2) * (g.w ** c3) * (g.w ** c4) * (g.x5 ** c5) + 0j   # at phi = 0: u, w real

        def grad(self, g):
            """Cartesian gradient (5 components) at the grid points."""
            c1, c2, c3, c4, c5 = self.c
            v = self.value(g)
            with np.errstate(divide='ignore', invalid='ignore'):
                du = np.where(c1 != 0, c1 * v / g.u, 0); dub = np.where(c2 != 0, c2 * v / g.u, 0)
                dw = np.where(c3 != 0, c3 * v / g.w, 0); dwb = np.where(c4 != 0, c4 * v / g.w, 0)
                d5 = np.where(c5 != 0, c5 * v / g.x5, 0)
            return np.stack([du + dub, 1j * (du - dub), dw + dwb, 1j * (dw - dwb), d5])

        def L(self, g):
            """X_ab phi = x_b d_a phi - x_a d_b phi for the 10 pairs."""
            gr = self.grad(g)
            return [g.X[b] * gr[a] - g.X[a] * gr[b] for (a, b) in PAIRS]


    def scalar_basis(p, t, K):
        """Scalar functions of charges (p, t): u~^|p| w~^|t| x5^k1 y^k2, k1 + k2 <= K (t may be half-integer)."""
        out = []
        c1, c2 = (p, 0) if p >= 0 else (0, -p)
        c3, c4 = (t, 0) if t >= 0 else (0, -t)
        for k1 in range(K + 1):
            for k2 in range(K + 1 - k1):
                out.append(Scalar((c1, c2, c3 + k2, c4 + k2, k1)))
        return out


    # ---------------------------------------------------------------- block solver
    def block_matrices(g, P, T, K, twist):
        """Gram and Casimir quadratic form on basis f = z^a phi, total charges (P, T).  twist: scalar charges
        t in Z + 1/2 (True) or Z (False)."""
        basis = []
        for k, a in enumerate(g.mons):
            Pa, Ta = charges(a)
            p, t = P - Pa, T - Ta
            if abs(p - round(p)) > 1e-9: continue
            if twist != (abs((t % 1) - 0.5) < 1e-9): continue
            for s in scalar_basis(int(round(p)), t, K):
                basis.append((k, s))
        n = len(basis)
        if n == 0:
            return None
        S = np.zeros((n, g.npt, g.I + 1), complex)
        XS = np.zeros((10, n, g.npt, g.I + 1), complex)
        for i, (k, s) in enumerate(basis):
            ph = s.value(g); Lph = s.L(g)
            S[i] = g.F[:, k, :] * ph[:, None]
            for A in range(10):
                XS[A, i] = g.FM[A][:, k, :] * ph[:, None] + g.F[:, k, :] * Lph[A][:, None]
        Wt = g.W[None, :, None]
        Gram = np.einsum('ipm,jpm->ij', S.conj() * Wt, S)
        Q = sum(np.einsum('ipm,jpm->ij', XS[A].conj() * Wt, XS[A]) for A in range(10))
        return basis, Gram, Q, S, XS


    def solve_block(g, P, T, K, twist, nlow=None, tol=1e-9):
        out = block_matrices(g, P, T, K, twist)
        if out is None:
            return None
        basis, Gram, Q, S, XS = out
        Gram = (Gram + Gram.conj().T) / 2; Q = (Q + Q.conj().T) / 2
        ev, U = np.linalg.eigh(Gram)
        keep = ev > tol * ev.max()
        B = U[:, keep] / np.sqrt(ev[keep])
        E, C = np.linalg.eigh(B.conj().T @ Q @ B)
        C = B @ C
        nlow = nlow or len(E)
        return E[:nlow], C[:, :nlow], basis, S, XS
    return namespace(locals(), globals())


@section
def inter():
    """Local interactions on the fuzzy S^4, evaluated on the (chi, theta) grid from orbital fibre values.

Orbital record: dict(f=flavour (0:+, 1:-), P, T, s=(npt, I+1) fibre values, sA=(10, npt, I+1) fibre values of
X_A f).  Fibre index mu <-> isospin m = I/2 - mu (standard basis alpha^{I-mu} beta^mu).

Terms (all normal ordered, inter-flavour n0 n0 - nz nz = 4 n_up n_dn; psi_up/dn = (psi_+ +- psi_-)/sqrt2):
  contact[L]  : int sum_M |(psi_up psi_dn)_{L M}|^2
  cas[L]      : int sum_A sum_M |X_A (psi_up psi_dn)_{L M}|^2      (pair Casimir, one derivative squared)
  dens0       : int int delta [n0 n0 - nz nz]              = 4 sum_L contact[L]
  dens1       : int [n0 (-lap) n0 - nz (-lap) nz],  -lap = sum_A X_A^dag X_A on scalar functions
Integrands of charge-conserving matrix elements are torus invariant, so the 2d grid (phi = 0) is exact.
"""
    import sys
    import numpy as np
    pass   # was: sys.path.insert(0, '<research tree>/s3')
    _s3lll = globals()['s3lll'](); cg = _s3lll.cg   # was: from s3lll import cg


    def cg_tensor(I, L):
        """C[M, mu, nu] = <I/2 m1; I/2 m2 | L M> with m = I/2 - mu."""
        S = I / 2.0
        C = np.zeros((2 * L + 1, I + 1, I + 1))
        for iM, M in enumerate(np.arange(-L, L + 1)):
            for mu in range(I + 1):
                for nu in range(I + 1):
                    m1, m2 = S - mu, S - nu
                    if abs(m1 + m2 - M) < 1e-9:
                        C[iM, mu, nu] = cg(S, m1, S, m2, L, M)
        return C


    def _groups(orbs):
        """Ordered mode pairs grouped by total charge."""
        gr = {}
        n = len(orbs)
        for i in range(n):
            for j in range(n):
                key = (round(2 * (orbs[i]['P'] + orbs[j]['P'])), round(4 * (orbs[i]['T'] + orbs[j]['T'])))
                gr.setdefault(key, []).append((i, j))
        return gr


    def pair_terms(orbs, I, W, U, Ucas, tol=1e-12):
        """contact[L] with weights U[L] and cas[L] with weights Ucas[L] (dicts)."""
        sgn = [1.0 if o['f'] == 0 else -1.0 for o in orbs]
        terms = {}
        Ls = sorted(set(list(U) + list(Ucas)))
        gr = _groups(orbs)
        for L in Ls:
            C = cg_tensor(I, L)
            u0, u1 = U.get(L, 0.0), Ucas.get(L, 0.0)
            if u0 == 0.0 and u1 == 0.0: continue
            for key, prs in gr.items():
                # pair amplitudes A[pair, M, p] and derivatives
                A = np.array([np.einsum('Mmn,pm,pn->Mp', C, orbs[k]['s'], orbs[l]['s']) for (k, l) in prs])
                V = np.zeros((len(prs), len(prs)), complex)
                if u0 != 0.0:
                    Af = A.reshape(len(prs), -1)
                    V += u0 * (Af.conj() * np.tile(W, 2 * L + 1)) @ Af.T
                if u1 != 0.0:
                    for a in range(10):
                        dA = np.array([np.einsum('Mmn,pm,pn->Mp', C, orbs[k]['sA'][a], orbs[l]['s'])
                                       + np.einsum('Mmn,pm,pn->Mp', C, orbs[k]['s'], orbs[l]['sA'][a]) for (k, l) in prs])
                        dAf = dA.reshape(len(prs), -1)
                        V += u1 * (dAf.conj() * np.tile(W, 2 * L + 1)) @ dAf.T
                for x, (k, l) in enumerate(prs):
                    for y, (kk, ll) in enumerate(prs):
                        v = V[x, y] * sgn[l] * sgn[ll] / 4.0
                        if abs(v) > tol:
                            t = (l, k, kk, ll)
                            terms[t] = terms.get(t, 0.0) + v
        return terms


    def density_terms(orbs, W, g0, g1, tol=1e-12):
        """n0 n0 - nz nz with U = g0 delta + g1 (-lap) delta; fibre-traced densities rho_ij = s_i^dag s_j."""
        n = len(orbs)
        rho, drho = {}, {}
        for i in range(n):
            for j in range(n):
                rho[i, j] = np.sum(orbs[i]['s'].conj() * orbs[j]['s'], axis=1)
                if g1 != 0.0:
                    drho[i, j] = np.array([np.sum(orbs[i]['sA'][a].conj() * orbs[j]['s'] + orbs[i]['s'].conj() * orbs[j]['sA'][a], axis=1)
                                           for a in range(10)])
        Wfl = {}
        for a in range(2):
            for ap in range(2):
                for b in range(2):
                    for bp in range(2):
                        w = (a == ap) * (b == bp) - (a != ap) * (b != bp)
                        if w: Wfl[a, ap, b, bp] = w
        gr = {}
        for (i, j) in rho:
            key = (round(2 * (orbs[j]['P'] - orbs[i]['P'])), round(4 * (orbs[j]['T'] - orbs[i]['T'])))
            gr.setdefault(key, []).append((i, j))
        terms = {}
        for key, p14 in gr.items():
            k23 = (-key[0], -key[1])
            if k23 not in gr: continue
            p23 = gr[k23]
            R14 = np.array([rho[i, j] for (i, j) in p14]); R23 = np.array([rho[k, l] for (k, l) in p23])
            V = g0 * (R14 * W) @ R23.T
            if g1 != 0.0:
                # int rho_14 (-lap) rho_23 = sum_A <X_A rho_41, X_A rho_23>,  rho_41 = conj(rho_14)
                for a in range(10):
                    D41 = np.array([drho[j, i][a] for (i, j) in p14]); D23 = np.array([drho[k, l][a] for (k, l) in p23])
                    V = V + g1 * (D41.conj() * W) @ D23.T
            for x, (m1, m4) in enumerate(p14):
                for y, (m2, m3) in enumerate(p23):
                    v = V[x, y]
                    if abs(v) < tol: continue
                    w = Wfl.get((orbs[m1]['f'], orbs[m4]['f'], orbs[m2]['f'], orbs[m3]['f']), 0)
                    if w:
                        t = (m1, m2, m3, m4)
                        terms[t] = terms.get(t, 0.0) + v * w
        return terms


    def merge(*dicts, scale=None):
        out = {}
        for k, d in enumerate(dicts):
            c = 1.0 if scale is None else scale[k]
            for t, v in d.items():
                out[t] = out.get(t, 0.0) + c * v
        return out


    def to_list(terms, tol=1e-12):
        bad = max((abs(v.imag) for v in terms.values()), default=0.0)
        return [(a, b, c, d, float(v.real)) for (a, b, c, d), v in terms.items() if abs(v) > tol], bad


    def lll_orbitals(g, flavour):
        """Untwisted lowest level: normalized monomials."""
        out = []
        for k, a in enumerate(g.mons):
            s = g.F[:, k, :]
            nrm = np.sqrt(np.sum(g.W[:, None] * np.abs(s) ** 2))
            _geom = globals()['geom'](); charges = _geom.charges   # was: from geom import charges
            P, T = charges(a)
            out.append(dict(f=flavour, P=P, T=T, s=s / nrm, sA=np.array([g.FM[A][:, k, :] for A in range(10)]) / nrm, a=a))
        return out
    return namespace(locals(), globals())


@section
def mfree2():
    """Faster matrix-free Lanczos for large sectors (the I=3 defect, ~1.3e7 states).
Two changes from mfree.py: (1) terms are canonicalized to c+_k1 c+_k2 c_k3 c_k4 with k1<k2, k3<k4, which
merges the up-to-four equivalent orderings; (2) the output index comes from Lin's two-table lookup,
j = block[s >> nlo] + rank[s & mask], instead of a binary search.  The lookup is exact for any additive
conserved charges: states sharing a high part form one sorted block containing every complementary low part,
so the position inside the block is the rank of the low part among low parts with the same charges.
"""
    import sys, numpy as np, numba as nb
    import scipy.sparse.linalg as sla
    from collections import defaultdict
    pass   # was: sys.path.insert(0, '<research tree>/twist')
    _twist = fuzzy; _popc = _twist._popc   # was: from twist import _popc


    def canonical(terms, tol=1e-13):
        C = defaultdict(float)
        for (k1, k2, k3, k4, v) in terms:
            if k1 == k2 or k3 == k4: continue
            s = 1
            if k1 > k2: k1, k2, s = k2, k1, -s
            if k3 > k4: k3, k4, s = k4, k3, -s
            C[(k1, k2, k3, k4)] += s * float(np.real(v))
        return [(k[0], k[1], k[2], k[3], v) for k, v in C.items() if abs(v) > tol]


    def lin_tables(states, orbs, nlo):
        """rank[L]: rank of low part L among low parts with equal charges; block[H]: first index with high part H."""
        ch = np.array([[1, int(round(2 * o['P'])), int(round(4 * o['T'])), o['f']] for o in orbs[:nlo]], np.int64)
        L = np.arange(1 << nlo, dtype=np.int64)
        bits = ((L[:, None] >> np.arange(nlo)) & 1).astype(np.int64)
        c = bits @ ch
        c[:, 3] %= 2
        key = ((c[:, 0] * 1000 + c[:, 1] + 500) * 1000 + c[:, 2] + 500) * 2 + c[:, 3]
        order = np.lexsort((L, key))
        rank = np.empty(1 << nlo, np.int64)
        ks = key[order]
        first = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
        grp_start = np.repeat(first, np.diff(np.r_[first, len(ks)]))
        rank[order] = np.arange(len(ks)) - grp_start
        hi = states >> nlo
        nhi = 1 << (len(orbs) - nlo)
        block = np.full(nhi, -1, np.int64)
        u, idx = np.unique(hi, return_index=True)
        block[u] = idx
        return block, rank


    @nb.njit(parallel=True, cache=True)
    def _matvec2(states, nmodes, nlo, block, rank, pstart, K1, K2, VV, diag, x, y):
        n = len(states)
        mask = (np.int64(1) << nlo) - 1
        for i in nb.prange(n):
            st = states[i]
            acc = diag[i] * x[i]
            occ = np.empty(nmodes, np.int64)
            no = 0
            for k in range(nmodes):
                if (st >> k) & 1:
                    occ[no] = k; no += 1
            for a in range(no):
                k3 = occ[a]
                for b in range(a + 1, no):
                    k4 = occ[b]
                    p = k3 * nmodes + k4
                    t0 = pstart[p]; t1 = pstart[p + 1]
                    if t0 == t1: continue
                    sg0 = 1
                    if _popc(st & ((np.int64(1) << k4) - 1)) & 1: sg0 = -sg0
                    s0 = st ^ (np.int64(1) << k4)
                    if _popc(s0 & ((np.int64(1) << k3) - 1)) & 1: sg0 = -sg0
                    s0 ^= np.int64(1) << k3
                    for t in range(t0, t1):
                        k1 = K1[t]; k2 = K2[t]
                        if ((s0 >> k2) & 1) or ((s0 >> k1) & 1): continue
                        sg = sg0
                        if _popc(s0 & ((np.int64(1) << k2) - 1)) & 1: sg = -sg
                        s1 = s0 | (np.int64(1) << k2)
                        if _popc(s1 & ((np.int64(1) << k1) - 1)) & 1: sg = -sg
                        s1 |= np.int64(1) << k1
                        j = block[s1 >> nlo] + rank[s1 & mask]
                        acc += VV[t] * sg * x[j]
            y[i] = acc


    class Operator(sla.LinearOperator):
        def __init__(self, basis, orbs, terms, mu, nlo=None):
            nm = basis.nmodes
            nlo = nlo or min(nm // 2, 20)
            ct = canonical(terms)
            T = np.array([(t[2], t[3], t[0], t[1]) for t in ct], dtype=np.int64)     # (k3, k4, k1, k2)
            V = np.array([t[4] for t in ct], dtype=np.float64)
            o = np.lexsort((T[:, 3], T[:, 2], T[:, 1], T[:, 0])); T, V = T[o], V[o]
            key = T[:, 0] * nm + T[:, 1]
            self.pstart = np.searchsorted(key, np.arange(nm * nm + 1)).astype(np.int64)
            self.K1, self.K2, self.V = T[:, 2].copy(), T[:, 3].copy(), V
            st = basis.states
            self.diag = np.zeros(basis.dim)
            for k in range(nm):
                if mu[k] != 0: self.diag += mu[k] * ((st >> np.int64(k)) & 1)
            self.block, self.rank = lin_tables(st, orbs, nlo)
            self.states, self.nm, self.nlo, self.nterms = st, nm, nlo, len(ct)
            super().__init__(np.float64, (basis.dim, basis.dim))

        def _matvec(self, x):
            x = np.ascontiguousarray(x.ravel(), dtype=np.float64)
            y = np.empty_like(x)
            _matvec2(self.states, self.nm, self.nlo, self.block, self.rank, self.pstart, self.K1, self.K2, self.V,
                     self.diag, x, y)
            return y


    def lowest(basis, orbs, terms, mu, k=2, tol=1e-9, ncv=None):
        op = Operator(basis, orbs, terms, mu)
        e = sla.eigsh(op, k=k, which='SA', tol=tol, ncv=ncv, return_eigenvectors=False)
        return np.sort(e)
    return namespace(locals(), globals())


@section
def mb():
    """Many-body layer for the fuzzy S^4: sector Hamiltonians and SO(5) Casimir labels (bulk, I=2 by ED).
"""
    import sys, numpy as np
    from math import factorial
    from itertools import combinations
    import scipy.sparse.linalg as sla
    pass   # was: sys.path.insert(0, '<research tree>/s3')
    pass   # was: sys.path.insert(0, '<research tree>/twist')
    M3 = globals()['s3model']()   # was: import s3model as M3
    _onmodel = fuzzy.onmodel(); build_H = _onmodel.build_H   # was: from onmodel import build_H


    def sector(orbs, terms, h, Ne, P, T, z2):
        B = M3.Basis(orbs, Ne, P, T, z2)
        if B.dim == 0:
            return None, B
        mu = np.array([-h if o['f'] == 0 else h for o in orbs])
        return build_H(B, terms, mu), B


    def lowest(H, k):
        if H.shape[0] <= 2000:
            e, v = np.linalg.eigh(H.toarray())
            return e[:k], v[:, :k]
        e, v = sla.eigsh(H, k=min(k, H.shape[0] - 2), which='SA', tol=1e-10)
        o = np.argsort(e)
        return e[o], v[:, o]


    class Casimir:
        """<C2> = sum_A ||J_A v||^2 with J_A = sum_flavours sum_ab Mt_A[b,a] c+_b c_a on the full N_e space."""
        def __init__(self, Mt, nmon, Ne):
            nm = 2 * nmon
            st = []
            for occ in combinations(range(nm), Ne):
                s = 0
                for k in occ: s |= 1 << k
                st.append(s)
            self.states = np.array(sorted(st), dtype=np.int64)
            self.Mt, self.nmon, self.nm = Mt, nmon, nm

        def apply(self, Mb, v):
            out = np.zeros(len(self.states), complex)
            S = self.states
            for f in range(2):
                off = f * self.nmon
                for b in range(self.nmon):
                    for a in range(self.nmon):
                        c = Mb[b, a]
                        if c == 0: continue
                        ma, mbb = a + off, b + off
                        occa = (S >> ma) & 1
                        if ma == mbb:
                            out += c * occa * v; continue
                        ok = (occa == 1) & (((S >> mbb) & 1) == 0)
                        if not ok.any(): continue
                        s0 = S[ok]
                        sg1 = np.array([bin(int(x) & ((1 << ma) - 1)).count('1') for x in s0])
                        s1 = s0 & ~(np.int64(1) << ma)
                        sg2 = np.array([bin(int(x) & ((1 << mbb) - 1)).count('1') for x in s1])
                        s2 = s1 | (np.int64(1) << mbb)
                        pos = np.searchsorted(S, s2)
                        out[pos] += c * v[ok] * (-1.0) ** (sg1 + sg2)
            return out

        def c2(self, B, vec):
            v = np.zeros(len(self.states), complex)
            v[np.searchsorted(self.states, B.states)] = vec
            return sum(np.vdot(w, w).real for w in (self.apply(M, v) for M in self.Mt))


    def orthonormal_generators(g):
        fa = np.array([np.prod([factorial(x) for x in a]) for a in g.mons], float)
        return [M * np.sqrt(fa)[:, None] / np.sqrt(fa)[None, :] for M in g.Mlift]
    return namespace(locals(), globals())


@section
def s4dmrg():
    """DMRG (TeNPy, via fuzzy_sphere/twist/dmrg_twist.py) for fuzzy-S^4 Hamiltonians given as term lists.
Modes ordered by x5-latitude of the orbital (monomials: (a1+a2-a3-a4)/2; twisted: <x5> from fibre values),
flavours interleaved.  Conserved: N, 2P, 4T, parity of N_-.
"""
    import sys, time, numpy as np
    pass   # was: sys.path.insert(0, '<research tree>/twist')
    DT = __import__('dmrg')   # was: import dmrg_twist as DT

    def latitude(o, g=None):
        if 'lat' in o:                            # precomputed <x5> (term pickles from build_I3.py)
            return o['lat']
        if 'a' in o:
            a = o['a']; return (a[0] + a[1] - a[2] - a[3]) / 2.0
        # twisted: <x5> = int |s|^2 x5 / int |s|^2
        w = g.W[:, None] * np.abs(o['s']) ** 2
        return float(np.sum(w * g.x5[:, None]) / np.sum(w))

    ORDER = 'P-lat-T'                              # orbital ordering for the MPS chain (see order_keys); best of five at I=3

    def order_keys(orbs, g=None, scheme=None):
        """Sort keys for the MPS chain.  'lat-P-T': x5-latitude, then P, then T (flavours interleaved);
        'lat-T-P'; 'lat-snake' (P-T plane traversed boustrophedon within each latitude); 'P-lat-T'; 'random'."""
        scheme = scheme or ORDER
        lat = [round(latitude(o, g), 6) for o in orbs]
        if scheme == 'lat-P-T':
            return [(l, o['P'], o['T'], o['f']) for l, o in zip(lat, orbs)]
        if scheme == 'lat-T-P':
            return [(l, o['T'], o['P'], o['f']) for l, o in zip(lat, orbs)]
        if scheme == 'lat-snake':
            return [(l, o['P'], o['T'] * (1 if int(round(2 * o['P'])) % 4 < 2 else -1), o['f']) for l, o in zip(lat, orbs)]
        if scheme == 'P-lat-T':
            return [(o['P'], l, o['T'], o['f']) for l, o in zip(lat, orbs)]
        if scheme == 'random':
            r = np.random.default_rng(1).permutation(len(orbs))
            return [(int(r[k]),) for k in range(len(orbs))]
        raise ValueError(scheme)

    MPO_CACHE = 'mpo_cache'   # was the research folder's mpo_cache; steps run inside s4d_files/

    def build(orbs, terms, h, g=None, compress=True, scheme=None, cache=True):
        """MPO model for H = terms + onsite(h). With cache=True the compressed model is pickled under MPO_CACHE, keyed by
        a hash of the modes, the terms, h and the ordering, so repeated runs at the same field skip the (slow) build."""
        import hashlib, os, pickle
        if cache and compress:
            hsh = hashlib.sha1(repr(([(o['P'], o['T'], o['f'], o.get('lat')) for o in orbs], len(terms), terms[:50], terms[-50:],
                                     round(h, 8), scheme or ORDER)).encode()).hexdigest()[:16]
            os.makedirs(MPO_CACHE, exist_ok=True)
            fn = os.path.join(MPO_CACHE, f'{hsh}.pkl')
            if os.path.exists(fn):
                return pickle.load(open(fn, 'rb'))
            out = build(orbs, terms, h, g, compress, scheme, cache=False)
            pickle.dump(out, open(fn + '.tmp', 'wb')); os.replace(fn + '.tmp', fn)
            return out
        keys = order_keys(orbs, g, scheme)
        rank = {k: i for i, k in enumerate(sorted(set(keys)))}
        mode_m = [rank[k] for k in keys]
        charges = [[1, int(round(2 * o['P'])), int(round(4 * o['T'])), o['f']] for o in orbs]
        onsite = [-h if o['f'] == 0 else h for o in orbs]
        terms = [t for t in terms if t[0] != t[1] and t[2] != t[3]]
        model, order = DT.build_model(mode_m, charges, ['N', 'P2', 'T4', 'Zm'], terms, onsite, mods=[1, 1, 1, 2])
        if compress:                              # SVD compression (singular values < 1e-12 x max dropped; numerically lossless, not exact)
            model = DT.MPOModel(model.lat, DT.compress_mpo(model.H_MPO))
        return model, order

    def product_state_for(orbs, Ne, P, T, z2, seed=0):
        """A product state (set of occupied modes) with the requested charges, found by random search."""
        rng = np.random.default_rng(seed)
        n = len(orbs)
        Pk = np.array([o['P'] for o in orbs]); Tk = np.array([o['T'] for o in orbs]); Fk = np.array([o['f'] for o in orbs])
        for _ in range(200000):
            occ = rng.choice(n, Ne, replace=False)
            if abs(Pk[occ].sum() - P) < 1e-9 and abs(Tk[occ].sum() - T) < 1e-9 and (Fk[occ].sum() % 2 == (0 if z2 > 0 else 1)):
                return set(int(x) for x in occ)
        raise RuntimeError("no product state found")

    def best_product_state(orbs, terms, h, Ne, P, T, z2, maxflip=3):
        """Lowest-diagonal-energy occupation with the requested charges, among states reached from the
        f=0-filled reference by up to `maxflip` particle moves (exact diagonal energy of the term list)."""
        import itertools
        n = len(orbs)
        Pk = np.array([o['P'] for o in orbs]); Tk = np.array([o['T'] for o in orbs]); Fk = np.array([o['f'] for o in orbs])
        W = np.zeros((n, n))
        for (k1, k2, k3, k4, v) in terms:
            if k1 != k2 and {k1, k2} == {k3, k4}:
                sgn = 1.0 if (k1 == k4) else -1.0
                W[k1, k2] += np.real(v) * sgn
        W = W + W.T
        on = np.array([-h if f == 0 else h for f in Fk])
        def ediag(occ):
            occ = list(occ); return on[occ].sum() + 0.5 * W[np.ix_(occ, occ)].sum()
        ref = [k for k in range(n) if Fk[k] == 0][:Ne]
        if len(ref) < Ne:
            ref += [k for k in range(n) if Fk[k] == 1][:Ne - len(ref)]
        empty = [k for k in range(n) if k not in ref]
        best = None
        for m in range(maxflip + 1):
            for out in itertools.combinations(ref, m):
                for inn in itertools.combinations(empty, m):
                    occ = sorted(set(ref) - set(out) | set(inn))
                    if abs(Pk[occ].sum() - P) > 1e-9 or abs(Tk[occ].sum() - T) > 1e-9: continue
                    if Fk[occ].sum() % 2 != (0 if z2 > 0 else 1): continue
                    e = ediag(occ)
                    if best is None or e < best[0]: best = (e, occ)
            if best is not None:
                return set(best[1])
        return product_state_for(orbs, Ne, P, T, z2)

    def run(model, order, occ, chi, sweeps=14, orthogonal_to=None):
        """Two-site DMRG with a doubling chi ramp (64, 128, 256, 512, ... chi) and a fixed number of sweeps."""
        from tenpy.networks.mps import MPS
        from tenpy.algorithms import dmrg
        psi = MPS.from_product_state(model.lat.mps_sites(), DT.product_state(order, set(occ)), bc='finite')
        ramp = {0: 64, 2: 128, 4: 256}; k = 6
        while max(ramp.values()) < chi:
            ramp[k] = min(2 * max(ramp.values()), chi); k += 2
        ramp = {s: min(c, chi) for s, c in ramp.items()}
        params = {'mixer': True, 'max_E_err': 1e-12, 'max_S_err': 1e-9, 'min_sweeps': sweeps, 'max_sweeps': sweeps,
                  'trunc_params': {'chi_max': chi, 'svd_min': 1e-10}, 'chi_list': ramp, 'combine': True}
        eng = dmrg.TwoSiteDMRGEngine(psi, model, params, orthogonal_to=orthogonal_to or [])
        E, psi = eng.run()
        return E, psi

    def lowest(orbs, terms, h, Ne, P, T, z2, nstates=1, chi=400, sweeps=10, g=None, shift=2.0, verbose=0, init='best'):
        model, order = build(orbs, terms, h, g)
        occ = best_product_state(orbs, terms, h, Ne, P, T, z2) if init == 'best' else product_state_for(orbs, Ne, P, T, z2)
        Es, psis = [], []
        for k in range(nstates):
            E, psi = run(model, order, occ, chi, sweeps, orthogonal_to=psis or None)
            Es.append(E); psis.append(psi)
        return Es, psis
    return namespace(locals(), globals())


@section
def twisted_band():
    """Twisted SO(5) Landau spectrum (Z2 flux along a great S^2) by (P, T) block.
"""
    import numpy as np, time, sys
    geom = globals()['geom']()   # was: import geom

    def block_charges(I, twist):
        Pf = (I / 2.0) % 1.0
        Tf = ((I / 2.0) + (0.5 if twist else 0.0)) % 1.0
        Ps = [k + Pf for k in range(-I - 2, I + 3)]
        Ts = [k + Tf for k in range(-I - 3, I + 4)]
        return [(P, T) for P in Ps for T in Ts]

    def levels(I, K, twist, ngrid=40, nlow=5):
        g = geom.Grid(I, ngrid, ngrid)
        out = []
        for (P, T) in block_charges(I, twist):
            r = geom.solve_block(g, P, T, K, twist, nlow=nlow)
            if r is None: continue
            out += [(e, P, T) for e in r[0]]
        return sorted(out), g

    def main():
        I, K = int(sys.argv[1]), int(sys.argv[2])
        C0 = I * I / 2 + 2 * I
        for twist in (False, True):
            t0 = time.time()
            lv, g = levels(I, K, twist)
            E = np.array([r[0] for r in lv]) - C0
            vals, counts = np.unique(np.round(E, 4), return_counts=True)
            print(f"I={I} K={K} twist={twist}: " + ", ".join(f"{v:.4f}:{c}" for v, c in zip(vals[:7], counts[:7])) +
                  f"  [{time.time()-t0:.0f}s]", flush=True)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/twisted_band_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def save_band():
    """Compute and save the lowest twisted SO(5) Landau states (fibre values on the grid) for size I.

Run as save_band 2 4 14, save_band 3 4 26, save_band 4 4 40; the log is stored as s4d/band_I<I>.log.
"""
    import sys, time, numpy as np
    geom = globals()['geom'](); twisted_band = globals()['twisted_band']()   # was: import geom, twisted_band

    def main():
        I, K, nsave = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        g = geom.Grid(I, 40, 40)
        t0 = time.time(); cand = []
        for (P, T) in twisted_band.block_charges(I, True):
            r = geom.solve_block(g, P, T, K, True, nlow=3)
            if r is None: continue
            E, C, basis, S, XS = r
            for k in range(len(E)):
                cand.append((E[k], P, T, np.einsum('i,ipm->pm', C[:, k], S), np.einsum('i,Aipm->Apm', C[:, k], XS)))
        cand.sort(key=lambda r: r[0]); cand = cand[:nsave]
        np.savez(f'band_I{I}.npz', E=np.array([c[0] for c in cand]), P=np.array([c[1] for c in cand]), T=np.array([c[2] for c in cand]),
                 s=np.array([c[3] for c in cand]), sA=np.array([c[4] for c in cand]))
        C0 = I * I / 2 + 2 * I
        print(f"I={I}: saved {len(cand)} states [{time.time()-t0:.0f}s]:", [(round(c[0] - C0, 4), c[1], c[2]) for c in cand])

    def cli(argv):
        sys.argv = list(argv)
        sys.argv = list(argv)
        with fuzzy.logged(f's4d/band_I{argv[1]}.log'):
            main()
    return namespace(locals(), globals())


@section
def build_I():
    """Term lists for monopole charge I (bulk: N+N monomials; defect: N monomials + the nband regular twisted states).
usage: python build_I.py I config.json bulk|defect [nband]  -> terms_<tag>.pkl (each part merged into the same file)

Run as build_I 2 designB1c_I2.json defect 6, build_I 2 designB1c_I2_bulk.json bulk, build_I 4 i4A.json bulk, build_I 4 i4A.json defect 26.
"""
    import sys, os, time, json, pickle, numpy as np
    geom = globals()['geom'](); inter = globals()['inter']()   # was: import geom, inter

    def main():
        I = int(sys.argv[1]); cfg = store.get(sys.argv[2]); part = sys.argv[3]
        nband = int(sys.argv[4]) if len(sys.argv) > 4 else None
        tag = os.path.basename(sys.argv[2]).replace('.json', ''); out = f'terms_{tag}.pkl'
        res = pickle.load(open(out, 'rb')) if os.path.isfile(out) else {'h': cfg['h'], 'I': I}
        g = geom.Grid(I, 40, 40)
        plus = inter.lll_orbitals(g, 0)
        def terms_for(orbs):
            c = cfg['couplings']
            U = {int(n[1:]): v for n, v in c.items() if n.startswith('c') and abs(v) > 0}
            K = {int(n[1:]): v for n, v in c.items() if n.startswith('k') and abs(v) > 0}
            t = inter.pair_terms(orbs, I, g.W, U, K)
            if abs(c.get('d1', 0.0)) > 0:
                t = inter.merge(t, inter.density_terms(orbs, g.W, 0.0, c['d1']))
            tl, bad = inter.to_list(t)
            return [x for x in tl if x[0] != x[1] and x[2] != x[3]], bad
        strip = lambda orbs: [dict(f=o['f'], P=o['P'], T=o['T'], a=o.get('a')) for o in orbs]
        lat = lambda orbs: [float(np.sum(g.W[:, None] * np.abs(o['s']) ** 2 * g.x5[:, None]) / np.sum(g.W[:, None] * np.abs(o['s']) ** 2)) for o in orbs]
        t0 = time.time()
        if part == 'bulk':
            bulk = plus + inter.lll_orbitals(g, 1)
            tb, bad = terms_for(bulk)
            res.update(bulk_orbs=strip(bulk), bulk_terms=tb)
            print(f"I={I} bulk: {len(bulk)} modes, {len(tb)} terms (imag {bad:.1e}) [{time.time()-t0:.0f}s]", flush=True)
        else:
            b = np.load(f'band_I{I}.npz')
            tw = [dict(f=1, P=float(b['P'][k]), T=float(b['T'][k]), s=b['s'][k], sA=b['sA'][k]) for k in range(nband)]
            td, bad = terms_for(plus + tw)
            res.update(def_orbs=strip(plus + tw), def_terms=td, def_lat=lat(plus + tw), nband=nband)
            print(f"I={I} defect: {len(plus) + len(tw)} modes, {len(td)} terms (imag {bad:.1e}) [{time.time()-t0:.0f}s]", flush=True)
        pickle.dump(res, open(out, 'wb'))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/build_I_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def build_I3():
    """Term lists for the I=3 model (bulk: 20+20 monomials; defect: 20 monomials + 14 twisted), saved as npz.

Run as build_I3 i3A.json and build_I3 i3B.json.
"""
    import sys, time, json, pickle, numpy as np
    geom = globals()['geom'](); inter = globals()['inter']()   # was: import geom, inter

    def main():
        cfg = store.get(sys.argv[1])
        I = 3
        g = geom.Grid(I, 40, 40)
        plus = inter.lll_orbitals(g, 0)
        b = np.load('band_I3.npz')
        minus_tw = [dict(f=1, P=float(b['P'][k]), T=float(b['T'][k]), s=b['s'][k], sA=b['sA'][k]) for k in range(14)]
        def terms_for(orbs):
            c = cfg['couplings']
            U = {int(n[1:]): v for n, v in c.items() if n.startswith('c') and abs(v) > 0}
            K = {int(n[1:]): v for n, v in c.items() if n.startswith('k') and abs(v) > 0}
            t = inter.pair_terms(orbs, I, g.W, U, K)
            if abs(c.get('d1', 0.0)) > 0:
                t = inter.merge(t, inter.density_terms(orbs, g.W, 0.0, c['d1']))
            tl, bad = inter.to_list(t)
            return [x for x in tl if x[0] != x[1] and x[2] != x[3]], bad
        t0 = time.time()
        bulk = plus + inter.lll_orbitals(g, 1)
        tb, badb = terms_for(bulk)
        print(f"bulk: {len(tb)} terms (imag {badb:.1e}) [{time.time()-t0:.0f}s]", flush=True)
        td, badd = terms_for(plus + minus_tw)
        print(f"defect: {len(td)} terms (imag {badd:.1e}) [{time.time()-t0:.0f}s]", flush=True)
        strip = lambda orbs: [dict(f=o['f'], P=o['P'], T=o['T'], a=o.get('a')) for o in orbs]
        tag = sys.argv[1].replace('.json', '')
        pickle.dump({'bulk_orbs': strip(bulk), 'bulk_terms': tb, 'def_orbs': strip(plus + minus_tw), 'def_terms': td,
                     'def_lat': [float(np.sum(g.W[:, None] * np.abs(o['s']) ** 2 * g.x5[:, None]) / np.sum(g.W[:, None] * np.abs(o['s']) ** 2)) for o in plus + minus_tw],
                     'h': cfg['h']}, open(f'terms_{tag}.pkl', 'wb'))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/build_I3_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def design():
    """Bulk design at I=2: channel energies E_ch (two-particle up-down energies) -> local couplings -> spectrum.
Irreps from exact degeneracies across weight sectors (P,T): highest-weight sector of each multiplet
(0,0) scalar | (1,0) vector (0,1) | (1,1) adjoint (2,0) | (2,0) sym-2 (0,2) | (2,1) (2,1) | (2,2) (4,0).
"""
    import numpy as np, time, sys, json
    import scipy.sparse as sp
    geom = globals()['geom'](); inter = globals()['inter'](); mb = globals()['mb']()   # was: import geom, inter, mb

    I = 2
    g = geom.Grid(I, 40, 40)
    orbs = inter.lll_orbitals(g, 0) + inter.lll_orbitals(g, 1)
    D = g.nmon
    CH = ['00', '01', '02', '20', '21', '40']
    # channel energies per unit coupling (test_inter2), rows = channels in CH order
    OPS = {'c2': np.array([0, 0, 0, 0, 0, 2.380952381]), 'c1': np.array([0, 0, 0, 3.333333333, 0.476190476, 0]),
           'c0': np.array([5.0, 1.666666667, 0.238095238, 0, 0, 0]), 'k1': np.array([0, 0, 0, 20.0, 5.714285714, 0]),
           'k0': np.array([0, 6.666666667, 2.380952381, 0, 0, 0]),
           'd1': np.array([80.0, -13.33333333, -10.47619048, 33.33333333, -20.95238095, 15.23809524])}
    NAMES = ['c2', 'c1', 'c0', 'k1', 'k0', 'd1']
    MAT = np.array([OPS[n] for n in NAMES]).T          # 6 channels x 6 operators (square, invertible)

    def couplings(E):
        return dict(zip(NAMES, np.linalg.solve(MAT, E)))

    def op_terms(name):
        if name == 'd1':
            return inter.to_list(inter.density_terms(orbs, g.W, 0.0, 1.0))[0]
        L = int(name[1]); kind = name[0]
        t = inter.pair_terms(orbs, I, g.W, {L: 1.0} if kind == 'c' else {}, {L: 1.0} if kind == 'k' else {})
        return inter.to_list(t)[0]

    SECS = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    IRR = {(0, 0): '(0,0)', (1, 0): '(0,1)', (1, 1): '(2,0)', (2, 0): '(0,2)', (2, 1): '(2,1)', (2, 2): '(4,0)'}
    t0 = time.time()
    TERMS = {n: op_terms(n) for n in NAMES}
    MATS = {}
    for (P, T) in SECS:
        for z2 in (1, -1):
            Hs = {}
            for n in NAMES:
                H, B = mb.sector(orbs, TERMS[n], 0.0, D, P, T, z2)
                Hs[n] = H
            nh = np.zeros(B.dim)
            for k, o in enumerate(orbs):
                nh += (-1.0 if o['f'] == 0 else 1.0) * ((B.states >> np.int64(k)) & 1)
            Hs['h'] = sp.diags(nh)
            MATS[(P, T, z2)] = Hs
    print(f"setup {time.time()-t0:.0f}s; dims", {k: v['h'].shape[0] for k, v in MATS.items()}, flush=True)

    def spectrum(E, h, nlev=(30, 12, 10, 8, 6, 4)):
        c = couplings(np.asarray(E, float))
        lv = {}
        for (P, T, z2), Hs in MATS.items():
            H = sum(c[n] * Hs[n] for n in NAMES) + h * Hs['h']
            k = nlev[SECS.index((P, T))]
            lv[(P, T, z2)] = mb.lowest(H, k)[0]
        E0 = lv[(0, 0, 1)][0]
        out = []
        for z2 in (1, -1):
            for e in lv[(0, 0, z2)]:
                w = (0, 0)
                for s in SECS[1:]:
                    if np.any(np.abs(lv[(s[0], s[1], z2)] - e) < 1e-7 * max(1, abs(e))):
                        w = s
                out.append((e - E0, z2, IRR[w]))
        # unique multiplets: keep one entry per (energy, irrep)
        seen, uniq = set(), []
        for e, z2, ir in sorted(out):
            key = (round(e, 7), z2, ir)
            if key not in seen:
                seen.add(key); uniq.append((e, z2, ir))
        return uniq

    TARGET = [('-', '(0,0)', 0, 1.5), ('-', '(0,1)', 0, 2.5), ('-', '(0,2)', 0, 3.5), ('-', '(0,0)', 1, 4.5),
              ('+', '(0,0)', 0, 3.0), ('+', '(0,1)', 0, 4.0), ('+', '(0,2)', 0, 5.0), ('+', '(0,0)', 1, 5.0)]
    JUNK = ['(2,0)', '(2,1)', '(4,0)']

    def assign(spec):
        got = {}
        for sgn, ir, n, tgt in TARGET:
            z2 = 1 if sgn == '+' else -1
            cand = [e for e, zz, i in spec if zz == z2 and i == ir and e > 1e-9]
            got[(sgn, ir, n)] = cand[n] if len(cand) > n else np.nan
        junk = min([e for e, zz, i in spec if i in JUNK] + [np.inf])
        return got, junk

    def cost(params, verbose=False):
        E = params[:6]; h = 5.366
        try:
            spec = spectrum(E, h)
        except Exception:
            return 1e3
        got, junk = assign(spec)
        e = np.array([got[(s, i, n)] for s, i, n, t in TARGET]); t = np.array([t for *_, t in TARGET])
        if np.any(~np.isfinite(e)):
            return 1e2
        a = np.sum(e * t / t) / np.sum(e * e / t)
        c = np.sum((a * e - t) ** 2 / t) + 2.0 * max(0.0, 5.5 - a * junk) ** 2
        if verbose:
            print(f"  alpha={a:.4f}  " + "  ".join(f"{s}{i}#{n}:{a*v:.3f}" for (s, i, n, tt), v in zip(TARGET, e)) + f"  junk:{a*junk:.3f}  cost {c:.4f}")
        return c

    def main():
        E_qian = np.array([0, 0, 0, 3.2, 0.022, 2.0])
        print("Qian:"); cost(E_qian, True)
        from scipy.optimize import minimize
        best = None
        starts = [E_qian, np.array([2.0, 1.0, 0.5, 6.0, 6.0, 6.0]), np.array([4.0, 2.0, 1.0, 8.0, 8.0, 4.0]),
                  np.array([1.0, 0.5, 0.2, 4.0, 4.0, 2.0])]
        for x0 in starts:
            t0 = time.time()
            r = minimize(cost, x0, method='Nelder-Mead', options={'maxfev': 400, 'xatol': 1e-3, 'fatol': 1e-5})
            print(f"start {np.round(x0,3)} -> {np.round(r.x,4)} cost {r.fun:.4f} [{time.time()-t0:.0f}s]", flush=True)
            cost(r.x, True)
            if best is None or r.fun < best.fun: best = r
        store.put('design_I2.json', {'E': list(best.x), 'cost': best.fun, 'couplings': couplings(best.x[:6])})

    def cli(argv):
        with fuzzy.logged('s4d/logs/design_I2.log'):
            main()
    return namespace(locals(), globals())


@section
def bulk_scan():
    """I=3 bulk DMRG scan over h: ground state (0,0,+), phi (0,0,-), d-phi (0,1,-).
usage: python bulk_scan.py terms_<tag>.pkl chi sweeps h1 h2 ...
-> bulkscan/<tag>_chi<chi>_h<h>.json, one file per h (safe for parallel processes), banked per sector.
(Runs before 2026-10-10 11:00 wrote a single bulkscan_<tag>_chi<chi>.json instead.)

For example bulk_scan terms_i3A.pkl 800 14 3.0 4.5 6.0.
"""
    import sys, time, json, pickle, os
    s4dmrg = globals()['s4dmrg']()   # was: import s4dmrg

    def main():
        d = pickle.load(open(sys.argv[1], 'rb')); chi = int(sys.argv[2]); sweeps = int(sys.argv[3])
        hs = [float(x) for x in sys.argv[4:]]
        tag = os.path.basename(sys.argv[1])[len('terms_'):-len('.pkl')]
        pass   # was os.makedirs(...): the store needs no output directory
        orbs, terms = d['bulk_orbs'], d['bulk_terms']
        for h in hs:
            out = f"bulkscan/{tag}_chi{chi}_h{h:.4f}.json"
            res = store.get(out) if store.exists(out) else {}
            model = None
            Ne = len(orbs) // 2
            for (P, T, z2, name) in [(0, 0.0, 1, 'E0'), (0, 0.0, -1, 'phi'), (0, 1.0, -1, 'dphi')]:   # (0,1) copy of d-phi: least entangled with P-first ordering
                if name in res: continue
                t0 = time.time()
                if model is None:                         # one MPO per field (rebuilding it per sector costs minutes at I=4)
                    model, order = s4dmrg.build(orbs, terms, h)
                    print(f"h={h}: model built [{time.time()-t0:.0f}s]", flush=True); t0 = time.time()
                occ = s4dmrg.best_product_state(orbs, terms, h, Ne, P, T, z2)
                E, psi = s4dmrg.run(model, order, occ, chi, sweeps)
                E, psis = [E], [psi]
                res[name] = float(E[0]); res[name + '_S'] = float(max(psis[0].entanglement_entropy()))
                store.put(out, res)
                print(f"h={h} {name}: E={E[0]:.8f} Smax={res[name + '_S']:.3f} [{time.time()-t0:.0f}s]", flush=True)
            e0, p, dp = res['E0'], res['phi'], res['dphi']
            print(f"   h={h}: phi gap {p-e0:.5f}, dphi gap {dp-e0:.5f}, ratio phi/(dphi-phi) = {(p-e0)/(dp-p):.4f} (free 1.5)", flush=True)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/bulk_scan_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def defect_dmrg():
    """Defect spectrum by DMRG on the term lists from build_I3.py (D monomials + D-2I twisted orbitals, Ne = D).
usage: python defect_dmrg.py terms_<tag>.pkl h chi sweeps P,T,z2,n [P,T,z2,n ...]
Each state is banked as it converges, in defdmrg/<tag>_h<h>_chi<chi>_<P>_<T>_<z2>.json -> {"E": [E0, E1, ...]}.

For example defect_dmrg terms_i4A.pkl 6.0 800 14 0,0.0,1,2 0,0.5,-1,1 0,1.0,1,1 0,1.5,-1,1 0,2.0,1,1 1,0.5,-1,1 2,0.5,-1,1.
"""
    import sys, time, json, pickle, os
    s4dmrg = globals()['s4dmrg']()   # was: import s4dmrg
    DT = s4dmrg.DT

    def main():
        d = pickle.load(open(sys.argv[1], 'rb'))
        h = float(sys.argv[2]); chi = int(sys.argv[3]); sweeps = int(sys.argv[4])
        secs = [(int(a), float(b), int(c), int(n)) for a, b, c, n in (s.split(',') for s in sys.argv[5:])]
        orbs = [dict(o, lat=l) for o, l in zip(d['def_orbs'], d['def_lat'])]
        Ne = sum(1 for o in orbs if o['f'] == 0)
        tag = os.path.basename(sys.argv[1])[len('terms_'):-len('.pkl')]
        pass   # was os.makedirs(...): the store needs no output directory
        t0 = time.time()
        model, order = s4dmrg.build(orbs, d['def_terms'], h)
        print(f"model: {len(orbs)} modes, Ne={Ne}, {len(d['def_terms'])} terms, h={h} [{time.time()-t0:.0f}s]", flush=True)
        for (P, T, z2, n) in secs:
            out = f"defdmrg/{tag}_h{h:.4f}_chi{chi}_{P}_{T}_{z2}.json"
            res = store.get(out) if store.exists(out) else {"E": []}
            if len(res["E"]) >= n: continue
            occ = s4dmrg.best_product_state(orbs, d['def_terms'], h, Ne, P, T, z2)
            psis = []
            for k in range(n):
                t1 = time.time()
                E, psi = s4dmrg.run(model, order, occ, chi, sweeps, orthogonal_to=psis or None)
                psis.append(psi)
                if k >= len(res["E"]):
                    res["E"].append(float(E)); res.setdefault("chi_used", []).append(int(max(psi.chi)))
                    res.setdefault("S_mid", []).append(float(psi.entanglement_entropy()[len(orbs) // 2 - 1]))
                    store.put(out, res)
                print(f"({P},{T},{z2}) state {k}: E={E:.8f} chi={max(psi.chi)} [{time.time()-t1:.0f}s]", flush=True)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/defect_dmrg_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def fig_curves():
    """Defect levels against the bulk spacing ratio r = Delta_phi/(Delta_dphi - Delta_phi), I=2 (exact h-scan) and
I=3 (DMRG, chi=800).  The free point is r = 3/2 with psi_{3/2} = 3 and the l=1 descendant at 3.
usage: python fig_curves.py  -> fig_curves.pdf, and i3_points.json for gen_numbers.py
"""
    import os, json, glob, re
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HERE = 's4d/note'; S4D = os.path.dirname(HERE)
    cv = store.get(os.path.join(S4D, 'curve_I2_designB1c_I2.json'))

    def i3_points(tag='i3A', chi=800, chi_bulk=None):
        """Bulk ratio (from bulkscan at chi_bulk, default chi) and D-calibrated defect levels (defdmrg at chi) per field."""
        chi_bulk = chi_bulk or chi
        pts = []
        for f in sorted(store.glob(os.path.join(S4D, f'bulkscan/{tag}_chi{chi_bulk}_h*.json'))):
            b = store.get(f); h = float(re.search(r'_h([0-9.]+)\.json', f).group(1))
            if not all(k in b for k in ('E0', 'phi', 'dphi')): continue
            lev = {}
            for g in store.glob(os.path.join(S4D, f'defdmrg/{tag}_h{h:.4f}_chi{chi}_*.json')):
                P, T, z2 = re.search(r'_chi\d+_(-?\d+)_(-?[0-9.]+)_(-?\d+)\.json', g).groups()
                lev[f'{P}|{T}|{z2}'] = store.get(g)['E']
            p = {'h': h, 'ratio': (b['phi'] - b['E0']) / (b['dphi'] - b['phi']), 'alpha_bulk': 1.5 / (b['phi'] - b['E0'])}
            if '0|0.0|1' in lev and '0|1.0|1' in lev:
                e0 = lev['0|0.0|1'][0]; a = 4.0 / (lev['0|1.0|1'][0] - e0); p['alpha_D'] = a
                for key, nm in [('0|0.5|-1', 'psi12'), ('0|1.5|-1', 'psi32'), ('1|0.5|-1', 'l1'), ('2|0.5|-1', 'l2'), ('0|2.0|1', 's2')]:
                    if key in lev: p[nm] = a * (lev[key][0] - e0)
                if len(lev['0|0.0|1']) > 1: p['psipsi0'] = a * (lev['0|0.0|1'][1] - e0)
            pts.append(p)
        return pts

    def main():
        pts = i3_points()
        store.put(os.path.join(HERE, 'i3_points.json'), pts)
        ptsB = i3_points('i3B')                      # second microscopic model at I=3 (K_1 contact instead of U_1)
        store.put(os.path.join(HERE, 'i3B_points.json'), ptsB)
        pts4 = i3_points('i4A', chi=800, chi_bulk=400)   # I=4, l-mapped model A; bulk ratio from chi=400 (fixes h_c to ~0.2)
        store.put(os.path.join(HERE, 'i4_points.json'), pts4)
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
        for ax, key, lab, tgt in [(axs[0], 'psi32', r'$\hat\Delta_{\pm3/2}$', 3.0), (axs[1], 'l1', r'$\hat\Delta(\partial\psi_{1/2})$', 3.0)]:
            r2 = [c['ratio'] for c in cv]; y2 = [c[key] for c in cv]
            ax.plot(r2, y2, '-', color='C0', lw=1.6, label=r'$I=2$ (exact)')
            for c in cv:
                if abs(c['h'] - round(c['h'])) < 1e-9 and c['h'] in (3, 6, 9, 12):
                    ax.annotate(f"$h={c['h']:.0f}$", (c['ratio'], c[key]), fontsize=7, color='C0', xytext=(3, 3), textcoords='offset points')
            q = [p for p in pts if key in p]
            if q:
                ax.plot([p['ratio'] for p in q], [p[key] for p in q], 'o-', color='C3', ms=4, lw=1.2, label=r'$I=3$ (DMRG, $\chi=800$)')
                for p in q:
                    ax.annotate(f"$h={p['h']:g}$", (p['ratio'], p[key]), fontsize=7, color='C3', xytext=(3, -9), textcoords='offset points')
            qB = [p for p in ptsB if key in p]
            if qB:
                ax.plot([p['ratio'] for p in qB], [p[key] for p in qB], 's--', color='C2', ms=4, lw=1.0, mfc='none',
                        label=r'$I=3$, second model')

            q4 = [p for p in pts4 if key in p]
            if q4:
                ax.plot([p['ratio'] for p in q4], [p[key] for p in q4], 'D-', color='C4', ms=4, lw=1.2,
                        label=r'$I=4$ (DMRG, $\chi=800$)')
            ax.plot([1.5], [tgt], '*', color='k', ms=10, label='free CFT')
            ax.axvline(1.5, color='0.7', lw=0.8, ls='--'); ax.axhline(tgt, color='0.7', lw=0.8, ls='--')
            ax.set_xlabel(r'bulk ratio $\Delta_\phi/(\Delta_{\partial\phi}-\Delta_\phi)$'); ax.set_ylabel(lab + r'  ($\hat\Delta_D=4$)')
        axs[0].legend(fontsize=7, loc='lower left', frameon=False)
        fig.tight_layout(); fig.savefig(figpath('fig_curves.pdf')); print('wrote fig_curves.pdf;', len(pts), 'I=3 points')
        for p in pts: print({k: round(v, 3) for k, v in p.items()})
        for p in ptsB: print('B', {k: round(v, 3) for k, v in p.items()})
        for p in pts4: print('I4', {k: round(v, 3) for k, v in p.items()})

    def cli(argv):
        with fuzzy.logged('s4d/logs/fig_curves.log'):
            main()
    return namespace(locals(), globals())


@section
def lpar():
    """Parallel SO(3) Casimir L_par^2 on the defect Hilbert space.

The defect S^2 = {x3 = x4 = 0} is rotated by X_12, X_15, X_25 (PAIRS indices 0, 3, 6).  J_z = i X_12 has
eigenvalue P on every orbital, and L_+ = X_15 + i X_25 raises P by one (it maps x5 -> -u and kills u).
Matrix elements are torus-invariant integrands, so they are evaluated on the (chi, theta) grid at phi = 0:
A[a, b] = <a|L_+|b> for P_a = P_b + 1, same T and flavour.  With J_+ = A, the many-body Casimir is
L^2 = J_z^2 + J_z + J_- J_+, and on a state of total parallel charge P,  <L^2> = P(P+1) + <J_- J_+>.
J_- J_+ = sum (A^dag A)_cb c+_c c_b + sum conj(A_dc) A_ab c+_c c+_a c_b c_d; the one-body part is folded into
two-body terms with c+_c c_b = sum_e c+_c c+_e c_e c_b / (N_e - 1), exact at fixed particle number N_e.
usage: python lpar.py terms_<tag>.pkl   (single-particle checks)
"""
    import sys, pickle
    import numpy as np
    geom = globals()['geom'](); inter = globals()['inter']()   # was: import geom, inter

    IX12, IX15, IX25 = 0, 3, 6


    def orbitals(d, nchi=40, nth=40):
        """Wavefunctions in the order of d['def_orbs'] (untwisted LLL, then the first nband twisted states)."""
        nplus = sum(1 for o in d['def_orbs'] if o['f'] == 0)
        I = d.get('I') or next(i for i in range(1, 9) if (i + 1) * (i + 2) * (i + 3) // 6 == nplus)
        d.setdefault('nband', len(d['def_orbs']) - nplus)
        g = geom.Grid(I, nchi, nth)
        plus = inter.lll_orbitals(g, 0)
        b = np.load(f'band_I{I}.npz')
        tw = [dict(f=1, P=float(b['P'][k]), T=float(b['T'][k]), s=b['s'][k], sA=b['sA'][k]) for k in range(d['nband'])]
        orbs = plus + tw
        for o, r in zip(orbs, d['def_orbs']):
            assert o['f'] == r['f'] and abs(o['P'] - r['P']) < 1e-9 and abs(o['T'] - r['T']) < 1e-9
        assert len(orbs) == len(d['def_orbs'])
        return g, orbs


    def single_particle(g, orbs):
        n = len(orbs)
        A = np.zeros((n, n), complex); G = np.zeros((n, n), complex)
        Z = np.zeros((n, n), complex); Q = np.zeros((n, n), complex)
        W = g.W[:, None]
        ip = lambda u, v: np.sum(W * u.conj() * v)
        for a, oa in enumerate(orbs):
            for b, ob in enumerate(orbs):
                if oa['f'] != ob['f'] or abs(oa['T'] - ob['T']) > 1e-9:
                    continue
                dP = oa['P'] - ob['P']
                if abs(dP - 1) < 1e-9:
                    A[a, b] = ip(oa['s'], ob['sA'][IX15] + 1j * ob['sA'][IX25])
                elif abs(dP) < 1e-9:
                    G[a, b] = ip(oa['s'], ob['s'])
                    Z[a, b] = ip(oa['s'], 1j * ob['sA'][IX12])
                    Q[a, b] = sum(ip(oa['sA'][X], ob['sA'][X]) for X in (IX12, IX15, IX25))
        return A, G, Z, Q


    def checks(g, orbs):
        A, G, Z, Q = single_particle(g, orbs)
        P = np.array([o['P'] for o in orbs])
        blk = np.array([[o['f'] == q['f'] and abs(o['T'] - q['T']) < 1e-9 and abs(o['P'] - q['P']) < 1e-9
                         for q in orbs] for o in orbs])
        C1 = A.conj().T @ A + np.diag(P * P + P)
        out = {'gram': np.abs(G - np.eye(len(orbs)) * blk).max(),
               'Jz': np.abs(Z - np.diag(P)).max(),
               'commutator [J+,J-]-2Jz': np.abs(A @ A.conj().T - A.conj().T @ A - 2 * np.diag(P)).max(),
               'C1 vs direct Casimir form': np.abs((C1 - Q) * blk).max(),
               'imag(A)': np.abs(A.imag).max()}
        lev = {}
        for f in (0, 1):
            idx = [k for k, o in enumerate(orbs) if o['f'] == f]
            ev = np.linalg.eigvalsh(C1[np.ix_(idx, idx)])
            ell = (-1 + np.sqrt(1 + 4 * ev)) / 2
            lev[f] = np.round(ell, 6)
        return A, out, lev


    def jmjp_terms(A, orbs, tol=1e-12):
        """Two-body term list (k1, k2, k3, k4, v) = v c+_k1 c+_k2 c_k3 c_k4 for J_- J_+ at fixed N_e."""
        Ne = sum(1 for o in orbs if o['f'] == 0)
        n = len(orbs)
        nz = [(a, b, A[a, b]) for a in range(n) for b in range(n) if abs(A[a, b]) > tol]
        T = {}
        def add(key, v):
            T[key] = T.get(key, 0.0) + v
        for (d_, c, Adc) in nz:                 # (A^dag)_cd = conj(A_dc)
            for (a, b, Aab) in nz:
                if c != a and b != d_:
                    add((c, a, b, d_), np.conj(Adc) * Aab)
        B1 = A.conj().T @ A
        for c in range(n):
            for b in range(n):
                if abs(B1[c, b]) <= tol: continue
                for e in range(n):
                    if e != c and e != b:
                        add((c, e, e, b), B1[c, b] / (Ne - 1))
        bad = max((abs(v.imag) for v in T.values()), default=0.0)
        return [(k[0], k[1], k[2], k[3], float(np.real(v))) for k, v in T.items() if abs(v) > tol], bad

    def main():
        d = pickle.load(open(sys.argv[1], 'rb'))
        g, orbs = orbitals(d)
        A, out, lev = checks(g, orbs)
        for k, v in out.items():
            print(f'{k:28s} {v:.2e}')
        for f, name in ((0, 'untwisted LLL'), (1, 'twisted band')):
            vals, cnt = np.unique(lev[f], return_counts=True)
            print(f'{name}: single-particle ell (multiplicity) =', ', '.join(f'{v:g} ({c})' for v, c in zip(vals, cnt)))
        t, bad = jmjp_terms(A, orbs)
        print(f'J_- J_+ : {len(t)} two-body terms, max imag {bad:.1e}')

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/lpar_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def lpar_measure():
    """<L_par^2> on defect DMRG states (the sectors of defect_dmrg.py, recomputed, since MPS were not saved).
usage: python lpar_measure.py terms_<tag>.pkl h chi sweeps P,T,z2,n [P,T,z2,n ...]
-> lpar/<tag>_h<h>_chi<chi>.json: per sector, energies, <L^2> = P(P+1) + <J_- J_+>, and the saved energies.

For example lpar_measure terms_i3A.pkl 4.5 800 14 0,0.0,1,2 0,0.5,-1,2 0,1.0,1,2 0,1.5,-1,2 0,2.0,1,1 1,0.5,-1,2 2,0.5,-1,1.
"""
    import sys, os, json, pickle, time
    s4dmrg = globals()['s4dmrg'](); lpar = globals()['lpar']()   # was: import s4dmrg, lpar

    def main():
        d = pickle.load(open(sys.argv[1], 'rb'))
        h = float(sys.argv[2]); chi = int(sys.argv[3]); sweeps = int(sys.argv[4])
        secs = [(int(a), float(b), int(c), int(n)) for a, b, c, n in (s.split(',') for s in sys.argv[5:])]
        tag = os.path.basename(sys.argv[1])[len('terms_'):-len('.pkl')]
        g, orbs_w = lpar.orbitals(d)
        A, chk, _ = lpar.checks(g, orbs_w)
        Oterms, _ = lpar.jmjp_terms(A, orbs_w)
        orbs = [dict(o, lat=l) for o, l in zip(d['def_orbs'], d['def_lat'])]
        Ne = sum(1 for o in orbs if o['f'] == 0)
        t0 = time.time()
        model, order = s4dmrg.build(orbs, d['def_terms'], h)
        Omodel, Oorder = s4dmrg.build(orbs, Oterms, 0.0)
        assert list(order) == list(Oorder)
        print(f"{tag} h={h}: {len(orbs)} modes, {len(Oterms)} L2 terms, models built [{time.time()-t0:.0f}s]", flush=True)
        pass   # was os.makedirs(...): the store needs no output directory
        out = f"lpar/{tag}_h{h:.4f}_chi{chi}{os.environ.get('LPAR_SUFFIX', '')}.json"
        res = store.get(out) if store.exists(out) else {}
        for (P, T, z2, n) in secs:
            key = f"{P}_{T}_{z2}"
            if key in res and len(res[key]['E']) >= n: continue
            saved = f"defdmrg/{tag}_h{h:.4f}_chi{chi}_{P}_{T}_{z2}.json"
            rec = {'E': [], 'L2': [], 'E_saved': store.get(saved)['E'] if store.exists(saved) else None}
            occ = s4dmrg.best_product_state(orbs, d['def_terms'], h, Ne, P, T, z2)
            psis = []
            for k in range(n):
                t1 = time.time()
                E, psi = s4dmrg.run(model, order, occ, chi, sweeps, orthogonal_to=psis or None)
                psis.append(psi)
                L2 = P * (P + 1) + float(Omodel.H_MPO.expectation_value(psi).real)
                rec['E'].append(float(E)); rec['L2'].append(L2)
                res[key] = rec; store.put(out, res)
                print(f"({P},{T},{z2}) state {k}: E={E:.8f}  <L2>={L2:.5f}  [{time.time()-t1:.0f}s]", flush=True)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/lpar_measure_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def par_spin2():
    """Task 6: the genuine parallel-spin-two level of the 5d defect, and a descendant-consistency table.

At I = 4 (i4A, h = 6) the lowest state of the l_z = 2 sector (2, 0.5, -1) has <L_par^2> = 11.98, so it is l = 3 (or a mix),
and the tabulated "l = 2 descendant 4.68" is not an l = 2 level.  L_par^2 commutes with H, and in the sector with
l_z = P the operator J_- J_+ = L_par^2 - P(P+1) vanishes exactly on l = P states and equals l(l+1) - P(P+1) > 0 on
l > P.  So the lowest state of
    H' = H + lam J_- J_+         (sector P = 2)
is the lowest genuine l = 2 level for any lam large enough to lift l = 3 by more than the window (6 lam above it).
Its true energy is E = <H> = E' - lam <J_- J_+>, and <L^2> = 6 + <J_- J_+> must come out at 6.
mode 'ed'   (I = 2): exact, sector (2, 0.5, -1) with and without the penalty, every state labelled by ell, plus
            vacuum (0, 0, +), psi_1/2 (0, 0.5, -), D (0, 1, +) and the l = 1 level (1, 0.5, -) for the D calibration.
mode 'dmrg' (I = 3, 4): the penalized ground state(s) of (2, 0.5, -1); the plain-H MPO gives E, the L^2 MPO <L^2>.
mode 'table': D-calibrated levels 4 (E - E0)/(E_D - E0) for psi_1/2, the l = 1 level and the genuine l = 2 level,
            with the descendant spacings (l = 1) - psi and (l = 2) - psi, which a parallel descendant puts at 1 and 2.
Gates: 'ed' with lam = 0 and lam > 0 agree on every l = 2 state; 'dmrg' at I = 3 (i3A, h = 4.5, chi = 800) reproduces the
lowest (2, 0.5, -1) level of lpar/i3A_h4.5000_chi800.json, whose <L^2> is already 6.0, to 1e-3 of E_D - E0.
Stage 2, NOT prepared: matrix elements of the parallel conformal generators (int x^A H over the defect directions).
x^A breaks the torus symmetry the term lists use, so it needs a 4d real-space grid for H(x); see TASKS.md.
usage: python par_spin2.py ed terms_<tag>.pkl [h] [lam=20]
       python par_spin2.py dmrg terms_<tag>.pkl h chi sweeps [n=1] [lam=20]   -> lpar/<tag>_h<h>_chi<chi>_pen<lam>.json
       python par_spin2.py table <tag> h chi

For example par_spin2 dmrg terms_i3A.pkl 4.5 800 14 and par_spin2 dmrg terms_i4A.pkl 6.0 800 30 3.
"""
    import os
    import sys
    import json
    import time
    import pickle
    import numpy as np
    HERE = 's4d'
    SECS = {'vac': (0, 0.0, 1), 'psi12': (0, 0.5, -1), 'D': (0, 1.0, 1), 'l1': (1, 0.5, -1), 'l2': (2, 0.5, -1)}


    def setup(pkl):
        lpar = globals()['lpar']()   # was: import lpar
        d = pickle.load(open(pkl, 'rb'))
        g, orbs_w = lpar.orbitals(d)
        A, _, _ = lpar.checks(g, orbs_w)
        Oterms, _ = lpar.jmjp_terms(A, orbs_w)
        return d, Oterms


    def mode_ed(pkl, h=None, lam=20.0, k=10):
        import scipy.sparse.linalg as sla
        sys.path.insert(0, os.path.join(HERE, '..', 's3'))
        mfree2 = globals()['mfree2']()   # was: import mfree2
        M3 = globals()['s3model']()   # was: import s3model as M3
        d, Oterms = setup(pkl)
        h = d['h'] if h is None else h
        orbs = d['def_orbs']; Ne = sum(1 for o in orbs if o['f'] == 0)
        mu = np.array([-h if o['f'] == 0 else h for o in orbs])
        out = dict(pkl=os.path.basename(pkl), h=h, lam=lam, sectors={})
        for name, (P, T, z2) in SECS.items():
            B = M3.Basis(orbs, Ne, P, T, z2)
            H = mfree2.Operator(B, orbs, d['def_terms'], mu)
            O = mfree2.Operator(B, orbs, Oterms, 0 * mu)
            kk = min(k, B.dim - 2)
            E, V = sla.eigsh(H, k=kk, which='SA', tol=1e-11)
            o = np.argsort(E); E, V = E[o], V[:, o]
            L2 = np.array([P * (P + 1) + V[:, i] @ O.matvec(V[:, i]) for i in range(kk)])
            rec = dict(dim=B.dim, E=E.tolist(), L2=L2.tolist())
            if name == 'l2':
                Hp = sla.LinearOperator(H.shape, matvec=lambda x: H.matvec(x) + lam * O.matvec(x), dtype=float)
                Ep, Vp = sla.eigsh(Hp, k=min(4, B.dim - 2), which='SA', tol=1e-11)
                o = np.argsort(Ep); Vp = Vp[:, o]
                Et = [float(Vp[:, i] @ H.matvec(Vp[:, i])) for i in range(Vp.shape[1])]
                L2p = [float(P * (P + 1) + Vp[:, i] @ O.matvec(Vp[:, i])) for i in range(Vp.shape[1])]
                rec.update(E_pen=Et, L2_pen=L2p)
                l2_plain = [e for e, l in zip(E, L2) if abs(l - 6) < 1e-6]
                rec['gate_pen_vs_plain'] = float(abs(Et[0] - l2_plain[0])) if l2_plain else None
            out['sectors'][name] = rec
            print(f'{name} {(P, T, z2)} dim {B.dim}: E ' + ' '.join(f'{e:.4f}' for e in E[:6]) +
                  ' | <L2> ' + ' '.join(f'{l:.3f}' for l in L2[:6]), flush=True)
        E0 = out['sectors']['vac']['E'][0]; ED = out['sectors']['D']['E'][0]
        lev = lambda e: 4 * (e - E0) / (ED - E0)
        r = out['sectors']['l2']
        print(f"I=2: lowest l_z=2 state at {lev(r['E'][0]):.3f} with <L2> {r['L2'][0]:.3f}; penalized (lam={lam}) state at "
              f"{lev(r['E_pen'][0]):.3f} with <L2> {r['L2_pen'][0]:.4f}; gate |E_pen - E_plain(l=2)| {r['gate_pen_vs_plain']}")
        tag = os.path.basename(pkl)[len('terms_'):-len('.pkl')]
        pass   # was os.makedirs(...): the store needs no output directory
        store.put(os.path.join(HERE, 'lpar', f'{tag}_ed_par_spin2.json'), out)


    def mode_dmrg(pkl, h, chi, sweeps, n=1, lam=20.0):
        s4dmrg = globals()['s4dmrg']()   # was: import s4dmrg
        d, Oterms = setup(pkl)
        tag = os.path.basename(pkl)[len('terms_'):-len('.pkl')]
        orbs = [dict(o, lat=l) for o, l in zip(d['def_orbs'], d['def_lat'])]
        Ne = sum(1 for o in orbs if o['f'] == 0)
        P, T, z2 = SECS['l2']
        t0 = time.time()
        pen_terms = list(d['def_terms']) + [(a, b, c, e, lam * v) for (a, b, c, e, v) in Oterms]
        model, order = s4dmrg.build(orbs, pen_terms, h)
        Hmodel, Horder = s4dmrg.build(orbs, d['def_terms'], h)
        Omodel, Oorder = s4dmrg.build(orbs, Oterms, 0.0)
        assert list(order) == list(Horder) == list(Oorder)
        print(f'{tag} h={h}: penalized, plain and L2 models built [{time.time() - t0:.0f}s]', flush=True)
        out = os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}_chi{chi}_pen{lam:g}.json')
        res = store.get(out) if store.exists(out) else dict(lam=lam, P=P, T=T, z2=z2, E=[], E_pen=[], L2=[])
        occ = s4dmrg.best_product_state(orbs, pen_terms, h, Ne, P, T, z2)
        psis = []
        for k in range(n):
            t1 = time.time()
            Ep, psi = s4dmrg.run(model, order, occ, chi, sweeps, orthogonal_to=psis or None)
            psis.append(psi)
            jj = float(Omodel.H_MPO.expectation_value(psi).real)
            E = float(Hmodel.H_MPO.expectation_value(psi).real)
            if k < len(res['E']):
                res['E'][k], res['E_pen'][k], res['L2'][k] = E, float(Ep), P * (P + 1) + jj
            else:
                res['E'].append(E); res['E_pen'].append(float(Ep)); res['L2'].append(P * (P + 1) + jj)
            store.put(out, res)
            print(f'  state {k}: E = {E:.8f} (E_pen {Ep:.8f}, E_pen - lam<J-J+> = {Ep - lam * jj:.8f})  <L2> = '
                  f'{P * (P + 1) + jj:.5f}  [{time.time() - t1:.0f}s]', flush=True)


    def mode_table(tag, h, chi):
        f = lambda key: os.path.join(HERE, 'defdmrg', f'{tag}_h{h:.4f}_chi{chi}_{key}.json')
        E = {}
        for name, (P, T, z2) in SECS.items():
            fn = f(f'{P}_{T}_{z2}')
            if store.exists(fn):
                E[name] = store.get(fn)['E'][0]
        lp = os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}_chi{chi}.json')
        L = store.get(lp) if store.exists(lp) else {}
        pens = sorted(p for p in [os.path.basename(k) for k in store.glob(os.path.join(HERE, 'lpar', '*'))] if p.startswith(f'{tag}_h{h:.4f}_chi{chi}_pen'))
        if not ({'vac', 'D', 'psi12'} <= set(E)):
            print('missing defdmrg energies for', tag, h, chi, '(need vac, D, psi_1/2)'); return
        lev = lambda e: 4 * (e - E['vac']) / (E['D'] - E['vac'])
        psi = lev(E['psi12'])
        print(f'{tag} h={h} chi={chi} (D-calibrated, D at 4):  psi_1/2 {psi:.3f}')
        if 'l1' in E:
            l2l1 = L.get('1_0.5_-1', {}).get('L2', [None])[0]
            print(f'  l=1 level {lev(E["l1"]):.3f}   spacing to psi {lev(E["l1"]) - psi:+.3f} (descendant: 1)   <L2> {l2l1}')
        if 'l2' in E:
            l2l2 = L.get('2_0.5_-1', {}).get('L2', [None])[0]
            print(f'  lowest l_z=2 (plain) {lev(E["l2"]):.3f}   <L2> {l2l2}')
            plain_is_l2 = l2l2 is not None and abs(l2l2 - 6) < 0.3
        for p in pens:
            R = store.get(os.path.join(HERE, 'lpar', p))
            for k, (e, l2) in enumerate(zip(R['E'], R['L2'])):
                print(f'  genuine l=2 (lam={R["lam"]}) state {k}: {lev(e):.3f}   spacing to psi {lev(e) - psi:+.3f} '
                      f'(descendant: 2)   <L2> {l2:.4f} (must be 6)')
                if k == 0 and 'l2' in E and plain_is_l2:
                    print(f'    gate (plain lowest is already l=2): |level difference| {abs(lev(e) - lev(E["l2"])):.2e} (< 1e-3)')

    def main():
        m = sys.argv[1]
        if m == 'ed':
            mode_ed(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None,
                    float(sys.argv[4]) if len(sys.argv) > 4 else 20.0)
        elif m == 'dmrg':
            mode_dmrg(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
                      int(sys.argv[6]) if len(sys.argv) > 6 else 1, float(sys.argv[7]) if len(sys.argv) > 7 else 20.0)
        else:
            mode_table(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/par_spin2_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def corr5d():
    """5d twin of twist/defect_corr.py: the Z2 twist-defect two-point function of the order parameter on the fuzzy S^4,
against the exact free half-twist correlator, plus the bulk-to-defect couplings b_s.

Physics. At the 5d Gaussian point the Z2 twist defect (worldvolume = the great S^2 {x3 = x4 = 0} times time) is the free
half-twist, so with Delta_phi = 3/2 and the defect of dimension p = 3,
    <phi phi>_D / <phi phi>_0 = int_0^beta cos^2 / int_0^{pi/2} cos^2 = (2/pi)(beta + sin beta cos beta),
    tan beta = cos(theta/2) / sqrt(xi + sin^2(theta/2)),   xi = (|x_par,1 - x_par,2|^2 + (r_1 - r_2)^2) / (4 r_1 r_2),
r = |x3 + i x4|, theta = relative transverse angle; and the defect-channel sum
    (r_1 r_2)^{-3/2} sum_{s in Z + 1/2} b_s^2 e^{i s theta} g_{3/2+|s|}(chi),   chi = 2 + 4 xi,
    b_s^2 = Gamma(3/2 + |s|) / (Gamma(3/2) Gamma(1 + |s|))      (4/pi at s = 1/2, 16/(3 pi) at s = 3/2),
    g_D(chi) = chi^-D 2F1(D/2, (D+1)/2; D + 1 - p/2; 4/chi^2).
Amplitudes. The lowest state psi_s of (z2 = -, P = 0, T = s) has |<psi_s|Phi(x)|0_D>| = c_phi b_s (sin chi)^{|s|} at every
point of the unit S^4 (r = sin chi), with c_phi = |<phi|Phi(x)|0>| from the untwisted bulk: flat in (chi, theta).
Lattice. Phi(x) = sum_{a in +, b in -} rho_ab(x) c+_a c_b + h.c., rho_ab = s_a(x)^dag s_b(x) (fibre components contracted,
gauge invariant). The stored s are at torus angles 0; at (phi_1, phi_2) rho_ab picks up e^{i((P_b - P_a) phi_1 +
(T_b - T_a) phi_2)}, so Phi(x)|0> in sector (dP, dT) carries e^{-i(dP phi_1 + dT phi_2)}. The twisted band (6 states at
I = 2) is a truncation, like the 3d top orbital. Cylinder time tau is in units of the bulk unit u = E(d phi) - E(phi).
Gates. G1: the trivial-defect channel sum equals |x_12|^-3; G2: the s in Z + 1/2 sum equals the closed form (1e-8).
G3: the bulk ratio Delta_phi/(Delta_dphi - Delta_phi) (3/2 at the criterion field). G4: the bulk G_0 depends only on
x_1.x_2 (a polynomial fit in the cosine leaves ~1e-12). G5: c_phi is flat over the sample points.
usage: python corr5d.py exact
       python corr5d.py run terms_<tag>.pkl terms_<tag>_bulk.pkl h   -> lpar/<tag>_h<h>_corr5d.json   (ED, I = 2)
       python corr5d.py dmrg terms_<tag>.pkl terms_<bulk>.pkl h chi chi_bulk sweeps [spins=0.5,1.5]   (I >= 3, equal time; also core5d)
DMRG gate: at I = 2 the dmrg mode must reproduce the ED tau = 0 rows.

Run as corr5d exact, corr5d run terms_designB1c_I2.pkl terms_designB1c_I2_bulk.pkl 8.3 and corr5d dmrg terms_i3A.pkl terms_i3A.pkl h 600 800 14 for h = 1, 2.5, 4.5, 7, 14. The stored I = 2 file is the equal-time run (the research mass-flow script set TAUS = [0]); the default TAUS add rows at tau = 0.25, 0.5, 1, and the equal-time rows agree with the stored ones to 1e-14.
"""
    import os
    import sys
    import json
    import time
    import pickle
    import numpy as np
    from scipy.special import hyp2f1, gammaln
    HERE = 's4d'
    sys.path.insert(0, os.path.join(HERE, '..', 's3'))
    sys.path.insert(0, os.path.join(HERE, '..', 'twist'))
    lpar = globals()['lpar']()   # was: import lpar
    inter = globals()['inter']()   # was: import inter
    mfree2 = globals()['mfree2']()   # was: import mfree2
    M3 = globals()['s3model']()   # was: import s3model as M3
    _defect_corr = fuzzy; apply_cdc = _defect_corr.apply_cdc   # was: from defect_corr import apply_cdc

    P_DEF = 3
    CHIS = np.array([0.30, 0.55, 0.80, 1.05, 1.30])                        # transverse distance sin(chi) on the unit S^4
    THS = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    DPHI1 = np.array([0.0, np.pi / 2, np.pi])
    DPHI2 = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, 0.999 * np.pi])
    TAUS = np.array([0.0, 0.25, 0.5, 1.0])


    # ------------------------------------------------------------------------------------------------ exact side
    def bs2(s, dphi=1.5):
        a = np.abs(s)
        return np.exp(gammaln(dphi + a) - gammaln(dphi) - gammaln(1 + a))


    def ghat(D, chi):
        return chi ** (-D) * hyp2f1(D / 2, (D + 1) / 2, D + 1 - P_DEF / 2, 4 / chi ** 2)


    def geom5(x1, x2):
        """x = (x1, x2, x3, x4, x5); defect at x3 = x4 = 0."""
        r1, r2 = np.hypot(x1[2], x1[3]), np.hypot(x2[2], x2[3])
        dpar = np.array([x1[0] - x2[0], x1[1] - x2[1], x1[4] - x2[4]])
        xi = (dpar @ dpar + (r1 - r2) ** 2) / (4 * r1 * r2)
        th = np.arctan2(x1[3], x1[2]) - np.arctan2(x2[3], x2[2])
        return r1, r2, xi, th


    def channel_sum(alpha, x1, x2, smax=300, b2=None, dims=None):
        r1, r2, xi, th = geom5(x1, x2)
        chi = 2 + 4 * xi
        tot = 0.0 + 0.0j
        for k in range(-smax, smax + 1):
            s = k + alpha
            b = b2.get(round(s, 6), bs2(s)) if b2 else bs2(s)
            D = dims.get(round(s, 6), 1.5 + abs(s)) if dims else 1.5 + abs(s)
            tot += b * np.exp(1j * s * th) * ghat(D, chi)
        return tot / (r1 * r2) ** 1.5


    def closed_ratio(x1, x2):
        _, _, xi, th = geom5(x1, x2)
        beta = np.arctan(np.cos(th / 2) / np.sqrt(xi + np.sin(th / 2) ** 2))
        return 2 / np.pi * (beta + np.sin(beta) * np.cos(beta))


    def point(chi, th, p1, p2, scale=1.0):
        c, s = np.sqrt(1 - np.sin(chi) ** 2), np.sin(chi)
        return scale * np.array([c * np.sin(th) * np.cos(p1), c * np.sin(th) * np.sin(p1), s * np.cos(p2), s * np.sin(p2),
                                 c * np.cos(th)])


    def exact_gates(seed=1):
        rng = np.random.default_rng(seed)
        g1 = g2 = 0.0
        for _ in range(8):
            x1 = point(*rng.uniform(0.2, 1.4, 1), *rng.uniform(0.3, 2.8, 1), *rng.uniform(-3, 3, 2), scale=1.3)
            x2 = point(*rng.uniform(0.2, 1.4, 1), *rng.uniform(0.3, 2.8, 1), *rng.uniform(-3, 3, 2))
            triv = np.linalg.norm(x1 - x2) ** -3
            g1 = max(g1, abs(channel_sum(0.0, x1, x2) - triv) / triv)
            g2 = max(g2, abs(channel_sum(0.5, x1, x2) - closed_ratio(x1, x2) * triv) / triv)
        print(f'G1 (trivial-defect sum vs |x12|^-3): {g1:.1e}   G2 (half-twist sum vs closed form): {g2:.1e}')
        return g1, g2


    # ------------------------------------------------------------------------------------------------ lattice side
    def sample_points(g):
        nth = int(round(np.sqrt(g.npt)))
        chi = np.arcsin(np.sqrt(g.y.reshape(-1, nth)[:, 0]))
        th = np.arccos(g.x5.reshape(-1, nth)[0] / np.sqrt(1 - g.y.reshape(-1, nth)[0, 0]))
        ic = [int(np.argmin(np.abs(chi - c))) for c in CHIS]
        it = [int(np.argmin(np.abs(th - t))) for t in THS]
        idx = [i * nth + j for i in ic for j in it]
        return np.array(idx), np.array([chi[i] for i in ic for _ in it]), np.array([th[j] for _ in ic for j in it])


    class Side:
        """Phi(x_p)|0> per (dP, dT) sector at the sample points, and each sector's spectral decomposition."""

        def __init__(self, orbs, wf, terms, h, pts):
            t0 = time.time()
            self.Ne = sum(1 for o in orbs if o['f'] == 0)
            self.mu = np.array([-h if o['f'] == 0 else h for o in orbs])
            self.orbs, self.terms = orbs, terms
            B0 = M3.Basis(orbs, self.Ne, 0, 0.0, 1)
            e, v = self.dense(B0)
            self.E0, psi = e[0], v[:, 0]
            self.gap0 = e[1] - e[0]
            rho = {}
            for a, oa in enumerate(wf):
                for b, ob in enumerate(wf):
                    if oa['f'] == 0 and ob['f'] == 1:
                        r = np.sum(oa['s'][pts].conj() * ob['s'][pts], axis=1)          # rho_ab at the sample points
                        rho[(a, b)] = r; rho[(b, a)] = r.conj()
            acc = {}
            for (a, b), r in rho.items():
                key = (round(orbs[a]['P'] - orbs[b]['P'], 6), round(orbs[a]['T'] - orbs[b]['T'], 6))
                acc.setdefault(key, []).append((a, b, r))
            self.sec = {}
            for key, lst in sorted(acc.items()):
                Bk = M3.Basis(orbs, self.Ne, key[0], key[1], -1)
                if Bk.dim == 0:
                    continue
                U = np.zeros((Bk.dim, len(pts)), complex)
                for a, b, r in lst:
                    idx, val = apply_cdc(B0.states, psi, a, b, Bk.states)
                    np.add.at(U, idx, val[:, None] * r[None, :])
                if np.linalg.norm(U) < 1e-12:
                    continue
                e, v = self.dense(Bk)
                self.sec[key] = dict(E=e, A=v.T @ U)
            print(f'  side: {len(orbs)} modes, {len(self.sec)} sectors, E0 {self.E0:.6f} [{time.time() - t0:.0f}s]', flush=True)

        def dense(self, B):
            op = mfree2.Operator(B, self.orbs, self.terms, self.mu)
            H = op.matmat(np.eye(B.dim))
            H = (H + H.T) / 2
            return np.linalg.eigh(H)

        def lowest(self, key, k=0):
            return self.sec[key]['E'][k]

        def corr(self, tau_lat, dphi1, dphi2):
            """G[p1, p2] = <0|Phi(x_p1; dphi1, dphi2) e^{-(H - E0) tau} Phi(x_p2; 0, 0)|0>."""
            tot = 0.0
            for (dP, dT), S in self.sec.items():
                w = np.exp(-(S['E'] - self.E0) * tau_lat)
                tot = tot + np.exp(1j * (dP * dphi1 + dT * dphi2)) * (S['A'].conj().T * w[None, :]) @ S['A']
            return tot


    def compare(D, B, pts, chis, ths, taus, u, b2lat, dlat, t0):
        rows = []
        for tau in taus:
            for p1 in DPHI1:
                for p2 in DPHI2:
                    GD, GB = D.corr(tau / u, p1, p2), B.corr(tau / u, p1, p2)
                    for i in range(len(pts)):
                        for j in range(len(pts)):
                            x1 = point(chis[i], ths[i], p1, p2, np.exp(tau)); x2 = point(chis[j], ths[j], 0, 0)
                            if tau == 0 and np.linalg.norm(x1 - x2) < 1e-9:
                                continue
                            _, _, xi, _ = geom5(x1, x2)
                            Rl = complex(GD[i, j] / GB[i, j])
                            Rr = None
                            if xi > 0.02:
                                triv = np.linalg.norm(x1 - x2) ** -3
                                Rr = float((channel_sum(0.5, x1, x2, smax=100, b2=b2lat, dims=dlat) / triv).real)
                            rows.append(dict(tau=float(tau), dphi1=float(p1), dphi2=float(p2), chi1=float(chis[i]),
                                             th1=float(ths[i]), chi2=float(chis[j]), th2=float(ths[j]), xi=float(xi),
                                             R_lat=Rl.real, R_lat_im=Rl.imag, R_exact=float(closed_ratio(x1, x2)), R_rec=Rr))
            print(f'  tau {tau}: done [{time.time() - t0:.0f}s]', flush=True)
        return rows


    def run(pkl, pkl_bulk, h):
        t0 = time.time()
        g1, g2 = exact_gates()
        d = pickle.load(open(pkl, 'rb')); db = pickle.load(open(pkl_bulk, 'rb'))
        g, wf_d = lpar.orbitals(d)
        wf_b = inter.lll_orbitals(g, 0) + inter.lll_orbitals(g, 1)
        for o, r in zip(wf_b, db['bulk_orbs']):
            assert o['f'] == r['f'] and abs(o['P'] - r['P']) < 1e-9 and abs(o['T'] - r['T']) < 1e-9
        pts, chis, ths = sample_points(g)
        print(f'{os.path.basename(pkl)} h={h}: {len(pts)} sample points, chi {np.round(np.unique(chis), 3)}, '
              f'theta {np.round(np.unique(ths), 3)}', flush=True)
        B = Side(db['bulk_orbs'], wf_b, db['bulk_terms'], h, pts)
        D = Side(d['def_orbs'], wf_d, d['def_terms'], h, pts)
        Ephi = B.lowest((0.0, 0.0)); Edphi = min(B.lowest((1.0, 0.0)), B.lowest((0.0, 1.0)))
        u = Edphi - Ephi
        gates = dict(G1=g1, G2=g2, bulk_ratio=(Ephi - B.E0) / u, dphi_bulk=(Ephi - B.E0) / u,
                     dphi_components=[(B.lowest((1.0, 0.0)) - B.E0) / u, (B.lowest((0.0, 1.0)) - B.E0) / u])
        # G4: bulk G_0 at each tau is a function of x1.x2 only (polynomial in the cosine)
        g4 = 0.0
        for tau in TAUS:
            cs, vals = [], []
            for p1 in DPHI1:
                for p2 in DPHI2:
                    G = B.corr(tau / u, p1, p2)
                    for i in range(len(pts)):
                        for j in range(len(pts)):
                            cs.append(point(chis[i], ths[i], p1, p2) @ point(chis[j], ths[j], 0, 0)); vals.append(G[i, j])
            cs, vals = np.array(cs), np.array(vals)
            fit = np.polynomial.polynomial.polyfit(cs, vals.real, 2 * d['I'] + 2)
            res = vals - np.polynomial.polynomial.polyval(cs, fit)
            g4 = max(g4, float(np.max(np.abs(res)) / np.max(np.abs(vals))))
        gates['G4_bulk_rotation_invariance'] = g4
        # amplitudes
        phi_vec = B.sec[(0.0, 0.0)]['A'][0]
        cphi = np.abs(phi_vec)
        gates['G5_cphi_flatness'] = float(np.ptp(cphi) / cphi.mean())
        amps = {}
        for (dP, dT), S in D.sec.items():
            if abs(dP) > 1e-9 or abs(dT) > 2.6:
                continue
            a = np.abs(S['A'][0])
            b2 = (a / (cphi.mean() * np.sin(chis) ** abs(dT))) ** 2
            amps[dT] = dict(b2=b2.tolist(), b2_mean=float(b2.mean()), b2_spread=float(np.ptp(b2) / b2.mean()),
                            b2_exact=float(bs2(dT)), Delta_lat=float((S['E'][0] - D.E0) / u), Delta_exact=1.5 + abs(dT))
        b2lat = {round(s, 6): v['b2_mean'] for s, v in amps.items() if abs(s) <= 1.5}
        dlat = {round(s, 6): v['Delta_lat'] for s, v in amps.items() if abs(s) <= 1.5}
        rows = compare(D, B, pts, chis, ths, TAUS, u, b2lat, dlat, t0)
        dev = np.array([r['R_lat'] - r['R_exact'] for r in rows])
        devr = np.array([r['R_rec'] - r['R_exact'] for r in rows if r['R_rec'] is not None])
        tag = os.path.basename(pkl)[len('terms_'):-len('.pkl')]
        out = dict(pkl=os.path.basename(pkl), h=h, u=u, gates=gates, amplitudes={f'{s:+.1f}': v for s, v in amps.items()},
                   levels_D=dict(D=float((D.lowest((0.0, 1.0)) - D.E0) / u) if (0.0, 1.0) in D.sec else None),
                   max_dev=dict(lat=float(np.max(np.abs(dev))), rec=float(np.max(np.abs(devr)))),
                   median_dev=dict(lat=float(np.median(np.abs(dev))), rec=float(np.median(np.abs(devr)))), rows=rows,
                   time=time.time() - t0)
        store.put(os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}_corr5d.json'), out)
        print('gates:', gates)
        for s, v in sorted(amps.items()):
            print(f"  s={s:+.1f}: Delta {v['Delta_lat']:.3f} (exact {v['Delta_exact']})  b_s^2 {v['b2_mean']:.4f} "
                  f"(exact {v['b2_exact']:.4f}), spread over points {v['b2_spread']:.3f}")
        print(f"|R - exact|: lattice max {out['max_dev']['lat']:.4f} median {out['median_dev']['lat']:.4f}; "
              f"reconstructed max {out['max_dev']['rec']:.4f} median {out['median_dev']['rec']:.4f}")
        for tau in TAUS:
            sel = [r for r in rows if r['tau'] == tau]
            print(f"  tau {tau}: median |dev| {np.median([abs(r['R_lat'] - r['R_exact']) for r in sel]):.4f}, "
                  f"max |Im R| {max(abs(r['R_lat_im']) for r in sel):.1e}")


    class DmrgSide:
        """Side at equal time by DMRG: per sector the Gram matrix M_ij = <v_i|v_j> of the hopped vacua v = c+_a c_b|0>,
        with u_sector(x) = sum_i c_i(x) v_i, so <u(x1)|u(x2)> = (C^dag M C)[p1, p2]. extra: further (P, T, z2) sectors whose
        lowest state is kept, with its overlaps <psi|v_i> (amplitudes)."""

        def __init__(self, orbs, wf, terms, h, pts, chi, sweeps, extra):
            core5d = globals()['core5d']()   # was: import core5d
            _dmrg_ward = fuzzy.question('01_known_results/free_control').dmrg_ward(); apply_hop = _dmrg_ward.apply_hop   # was: from dmrg_ward import apply_hop
            t0 = time.time()
            self.pos, self.st = core5d.dmrg_states(orbs, terms, h, chi, sweeps, [(0, 0.0, 1)] + list(extra))
            self.E0, S0 = self.st[(0, 0.0, 1)]
            pairs = {}
            for a, oa in enumerate(wf):
                for b, ob in enumerate(wf):
                    if oa['f'] != ob['f']:
                        key = (round(orbs[a]['P'] - orbs[b]['P'], 6), round(orbs[a]['T'] - orbs[b]['T'], 6))
                        pairs.setdefault(key, []).append((a, b, np.sum(oa['s'][pts].conj() * ob['s'][pts], axis=1)))
            self.sec, self.amp = {}, {}
            nov = 0
            for key, lst in sorted(pairs.items()):
                vs, cs = [], []
                for a, b, r in lst:
                    v = apply_hop(S0, self.pos[a], self.pos[b])
                    if v.norm > 1e-14:
                        vs.append(v); cs.append(r)
                if not vs:
                    continue
                n = len(vs)
                M = np.zeros((n, n), complex)
                for i in range(n):
                    for j in range(i, n):
                        M[i, j] = vs[i].overlap(vs[j]); M[j, i] = np.conj(M[i, j])
                nov += n * (n + 1) // 2
                C = np.array(cs)
                self.sec[key] = dict(C=C, M=M)
                for sec in extra:
                    if (abs(sec[0] - key[0]) < 1e-9 and abs(sec[1] - key[1]) < 1e-9 and sec[2] == -1):
                        psi = self.st[sec][1]
                        self.amp[key] = np.array([psi.overlap(v) for v in vs]) @ C
            print(f'  DMRG side: {len(orbs)} modes, {len(self.sec)} sectors, {nov} overlaps, E0 {self.E0:.6f} '
                  f'[{time.time() - t0:.0f}s]', flush=True)

        def corr(self, tau_lat, dphi1, dphi2):
            assert tau_lat == 0, 'DMRG side is equal-time only'
            return sum(np.exp(1j * (dP * dphi1 + dT * dphi2)) * (S['C'].conj().T @ S['M'] @ S['C'])
                       for (dP, dT), S in self.sec.items())


    def run_dmrg(pkl, pkl_bulk, h, chi, chi_bulk, sweeps, spins=(0.5, 1.5)):
        """Equal-time correlator and amplitudes at I >= 3; also writes the core5d analysis of the same defect states."""
        core5d = globals()['core5d']()   # was: import core5d
        t0 = time.time()
        d = pickle.load(open(pkl, 'rb')); db = pickle.load(open(pkl_bulk, 'rb'))
        g, wf_d = lpar.orbitals(d)
        d.setdefault('I', g.I)
        wf_b = inter.lll_orbitals(g, 0) + inter.lll_orbitals(g, 1)
        for o, r in zip(wf_b, db['bulk_orbs']):
            assert o['f'] == r['f'] and abs(o['P'] - r['P']) < 1e-9 and abs(o['T'] - r['T']) < 1e-9
        pts, chis, ths = sample_points(g)
        tag = os.path.basename(pkl)[len('terms_'):-len('.pkl')]
        print(f'{tag} h={h} DMRG chi={chi} (bulk {chi_bulk}): {len(pts)} sample points', flush=True)
        dorbs = [dict(o, lat=l) for o, l in zip(d['def_orbs'], d['def_lat'])]
        D = DmrgSide(dorbs, wf_d, d['def_terms'], h, pts, chi, sweeps, [(0, s, -1) for s in spins] + [(0, 1.0, 1)])
        cdata = core5d.dmrg_measure(dorbs, D.pos, D.st)
        cres = core5d.analyse(d, cdata, g, wf_d, f'{tag} h={h} chi={chi}')
        core5d.report(cres)
        store.put(os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}_chi{chi}_core5d.json'), cres)
        B = DmrgSide(db['bulk_orbs'], wf_b, db['bulk_terms'], h, pts, chi_bulk, sweeps, [(0, 0.0, -1)])
        gates = {}
        cs, vals = [], []
        for p1 in DPHI1:
            for p2 in DPHI2:
                G = B.corr(0.0, p1, p2)
                for i in range(len(pts)):
                    for j in range(len(pts)):
                        cs.append(point(chis[i], ths[i], p1, p2) @ point(chis[j], ths[j], 0, 0)); vals.append(G[i, j])
        cs, vals = np.array(cs), np.array(vals)
        fit = np.polynomial.polynomial.polyfit(cs, vals.real, 2 * d['I'] + 2)
        gates['G4_bulk_rotation_invariance'] = float(np.max(np.abs(vals - np.polynomial.polynomial.polyval(cs, fit)))
                                                     / np.max(np.abs(vals)))
        cphi = np.abs(B.amp[(0.0, 0.0)])
        gates['G5_cphi_flatness'] = float(np.ptp(cphi) / cphi.mean())
        gates['E_bulk'] = dict(vac=B.E0, phi=B.st[(0, 0.0, -1)][0])
        amps = {}
        for key, a in D.amp.items():
            s = key[1]
            b2 = (np.abs(a) / (cphi.mean() * np.sin(chis) ** abs(s))) ** 2
            amps[s] = dict(b2=b2.tolist(), b2_mean=float(b2.mean()), b2_spread=float(np.ptp(b2) / b2.mean()),
                           b2_exact=float(bs2(s)))
        rows = compare(D, B, pts, chis, ths, [0.0], 1.0, None, None, t0)
        for r in rows:
            r['R_rec'] = None
        dev = np.array([r['R_lat'] - r['R_exact'] for r in rows])
        out = dict(pkl=os.path.basename(pkl), h=h, chi=chi, chi_bulk=chi_bulk, gates=gates,
                   amplitudes={f'{s:+.1f}': v for s, v in amps.items()}, max_dev=dict(lat=float(np.max(np.abs(dev)))),
                   median_dev=dict(lat=float(np.median(np.abs(dev)))), rows=rows, time=time.time() - t0)
        store.put(os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}_chi{chi}_corr5d.json'), out)
        print('gates:', gates)
        for s, v in sorted(amps.items()):
            print(f"  s={s:+.1f}: b_s^2 {v['b2_mean']:.4f} (exact {v['b2_exact']:.4f}), spread over points {v['b2_spread']:.3f}")
        print(f"equal-time |R - exact|: max {out['max_dev']['lat']:.4f}, median {out['median_dev']['lat']:.4f}  "
              f"[{time.time() - t0:.0f}s]")

    def main():
        if sys.argv[1] == 'exact':
            exact_gates()
        elif sys.argv[1] == 'dmrg':
            run_dmrg(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]),
                     [float(x) for x in sys.argv[8].split(',')] if len(sys.argv) > 8 else (0.5, 1.5))
        else:
            run(sys.argv[2], sys.argv[3], float(sys.argv[4]))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/corr5d_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def core5d():
    """5d core spin ladder: a core coupling near the Z2 twist defect on the fuzzy S^4 shifts the transverse-spin-s mode by an
amount fixed by |s|, as in 3d (twist/core_scaling.py).

Physics. A potential localized within a few magnetic lengths of the defect is a set of irrelevant defect couplings
g_s psi_s psi_-s, of dimension 3 + 2|s| on the p = 3 defect, so the shift of psi_s goes as (a/R)^{2|s|} = I^{-|s|}. At the
Gaussian point the shift is fixed by Wick's theorem: for dH = lambda int kappa n^x (n^x = n_+ - n_- ~ c :phi^2:),
    d(E_s - E_0)/d lambda = int kappa Dn_s,   Dn_s(x) = <psi_s|n^x(x)|psi_s> - <0_D|n^x(x)|0_D> = 2c |f_s(x)|^2,
    |f_s(x)| = |<psi_s|Phi(x)|0_D>| = c_phi b_s (sin chi)^{|s|},   b_s^2 = Gamma(3/2+|s|)/(Gamma(3/2) Gamma(1+|s|)),
so the ratio of the s = 3/2 and s = 1/2 slopes carries no free parameter:
    r(kappa) = (4/3) int kappa y^{3/2} / int kappa y^{1/2}  (y = sin^2 chi),  ~ 2 eps / I for kappa = exp(-I y / eps).
Tests, all from one-body density matrices and Phi amplitudes of |0_D>, psi_1/2, psi_3/2 (lowest states of
(P, T, z2) = (0, 0, +), (0, 1/2, -), (0, 3/2, -)):
  (a) mode functions: |f_s|^2 / y^{|s|} flat over the sphere, and b_{3/2}^2/b_{1/2}^2 = 4/3;
  (b) Wick: Dn_s / |f_s|^2 flat and the same for both s;
  (c) the core ladder: slopes for kappa_eps = exp(-I y/eps), eps = 0.25 ... 2, and uniform kappa; ratio against r(kappa).
usage: python core5d.py ed terms_<tag>.pkl h                      (I = 2, exact)
       python core5d.py dmrg terms_<tag>.pkl h chi sweeps [spins=0.5,1.5]   (I >= 3)   -> lpar/<tag>_h<h>[_chi<chi>]_core5d.json
With more spins the ratios are slope_s/slope_1/2 against (b_s^2/b_1/2^2) int kappa y^s / int kappa y^{1/2}.

For example core5d dmrg terms_i4A.pkl 5.66 800 16 0.5,1.5,2.5,3.5.
"""
    import os
    import sys
    import json
    import time
    import pickle
    import numpy as np
    from scipy.special import gammaln
    HERE = 's4d'
    sys.path.insert(0, os.path.join(HERE, '..', 's3'))
    sys.path.insert(0, os.path.join(HERE, '..', 'twist'))
    lpar = globals()['lpar']()   # was: import lpar

    EPS = [0.25, 0.5, 1.0, 2.0, None]                                       # None = uniform kappa
    SPINS = [0.5, 1.5]


    def pairs_same(orbs):
        return [(a, b) for a, oa in enumerate(orbs) for b, ob in enumerate(orbs)
                if oa['f'] == ob['f'] and abs(oa['P'] - ob['P']) < 1e-9 and abs(oa['T'] - ob['T']) < 1e-9]


    def pairs_flip(orbs, s):
        return [(a, b) for a, oa in enumerate(orbs) for b, ob in enumerate(orbs)
                if oa['f'] != ob['f'] and abs(oa['P'] - ob['P']) < 1e-9 and abs(oa['T'] - ob['T'] - s) < 1e-9]


    # ------------------------------------------------------------------------------------------------ states
    def ed_data(d, h):
        mfree2 = globals()['mfree2']()   # was: import mfree2
        M3 = globals()['s3model']()   # was: import s3model as M3
        _defect_corr = fuzzy; apply_cdc = _defect_corr.apply_cdc   # was: from defect_corr import apply_cdc
        orbs, terms = d['def_orbs'], d['def_terms']
        Ne = sum(1 for o in orbs if o['f'] == 0)
        mu = np.array([-h if o['f'] == 0 else h for o in orbs])

        def low(P, T, z2):
            B = M3.Basis(orbs, Ne, P, T, z2)
            H = mfree2.Operator(B, orbs, terms, mu).matmat(np.eye(B.dim))
            e, v = np.linalg.eigh((H + H.T) / 2)
            return B.states, e[0], v[:, 0], e[1]

        def gamma(st, psi):
            G = {}
            for a, b in pairs_same(orbs):
                if a == b:
                    G[(a, b)] = float(np.sum(psi ** 2 * ((st >> np.int64(a)) & 1)))
                else:
                    idx, val = apply_cdc(st, psi, a, b, st)
                    G[(a, b)] = float(psi[idx] @ val)
            return G
        st0, E0, v0, E0b = low(0, 0.0, 1)
        out = dict(E={'vac': float(E0)}, G={'vac': gamma(st0, v0)}, A={})
        for s in SPINS + [-0.5]:
            st, E, v, Eb = low(0, s, -1)
            out['E'][s] = float(E); out['E'][f'{s}_next'] = float(Eb)
            out['G'][s] = gamma(st, v)
            amp = {}
            for a, b in pairs_flip(orbs, s):
                idx, val = apply_cdc(st0, v0, a, b, st)
                amp[(a, b)] = float(v[idx] @ val)
            out['A'][s] = amp
        stD, ED_, _, _ = low(0, 1.0, 1)
        out['E']['D'] = float(ED_)
        return out


    def dmrg_states(orbs, terms, h, chi, sweeps, sectors):
        """DMRG lowest state per (P, T, z2) in `sectors`: returns pos (mode -> chain site), {sector: (E, psi)}."""
        s4dmrg = globals()['s4dmrg']()   # was: import s4dmrg
        Ne = sum(1 for o in orbs if o['f'] == 0)
        model, order = s4dmrg.build(orbs, terms, h)
        out = {}
        for sec in sectors:
            occ = s4dmrg.best_product_state(orbs, terms, h, Ne, *sec)
            E, psi = s4dmrg.run(model, order, occ, chi, sweeps)
            out[sec] = (float(E), psi)
            print(f'  DMRG {sec}: E {float(E):.6f}', flush=True)
        return {k: i for i, k in enumerate(order)}, out


    def dmrg_measure(orbs, pos, st):
        """core5d data (energies, one-body density matrices, Phi amplitudes) from the states of dmrg_states."""
        _dmrg_ward = fuzzy.question('01_known_results/free_control').dmrg_ward(); apply_hop = _dmrg_ward.apply_hop   # was: from dmrg_ward import apply_hop
        gamma = lambda psi: {(a, b): float(np.real(psi.overlap(apply_hop(psi, pos[a], pos[b])))) for a, b in pairs_same(orbs)}
        E0, S0 = st[(0, 0.0, 1)]
        out = dict(E={'vac': E0, 'D': st[(0, 1.0, 1)][0]}, G={'vac': gamma(S0)}, A={})
        for s in sorted(k[1] for k in st if k[0] == 0 and k[2] == -1 and k[1] > 0):
            E, S = st[(0, s, -1)]
            out['E'][s] = E; out['G'][s] = gamma(S)
            amp = {}
            for a, b in pairs_flip(orbs, s):
                v = apply_hop(S0, pos[a], pos[b])
                amp[(a, b)] = complex(S.overlap(v)) if v.norm > 0 else 0.0
            out['A'][s] = amp
        return out


    DEFECT_SECTORS = [(0, 0.0, 1), (0, 0.5, -1), (0, 1.5, -1), (0, 1.0, 1)]


    def dmrg_data(d, h, chi, sweeps, spins=SPINS):
        orbs = [dict(o, lat=l) for o, l in zip(d['def_orbs'], d['def_lat'])]
        pos, st = dmrg_states(orbs, d['def_terms'], h, chi, sweeps, [(0, 0.0, 1), (0, 1.0, 1)] + [(0, s, -1) for s in spins])
        return dmrg_measure(orbs, pos, st)


    # ------------------------------------------------------------------------------------------------ analysis
    def analyse(d, data, g, wf, label):
        I = d['I']
        W, y = g.W, g.y
        rho = lambda a, b: np.sum(wf[a]['s'].conj() * wf[b]['s'], axis=1)            # conj(s_a) . s_b on the grid
        sign = np.array([1.0 if o['f'] == 0 else -1.0 for o in wf])                    # n^x = n_+ - n_-
        def density(G, nx=True):
            return sum((sign[a] if nx else 1.0) * np.real(rho(a, b) * G[(a, b)]) for (a, b) in G)
        n0 = density(data['G']['vac']); m0 = density(data['G']['vac'], False)
        unit_D = (data['E']['D'] - data['E']['vac']) / 4                           # D-calibrated unit (D at 4)
        spins = sorted(s for s in data['A'] if s > 0)
        res = dict(label=label, I=I, spins=spins, levels_Dcal={str(s): (data['E'][s] - data['E']['vac']) / unit_D for s in spins})
        prof, fs2 = {}, {}
        for s in spins:
            prof[s] = density(data['G'][s]) - n0
            tot = density(data['G'][s], False) - m0
            f = sum(rho(a, b) * A for (a, b), A in data['A'][s].items())
            fs2[s] = np.abs(f) ** 2
            flat = fs2[s] / y ** abs(s)
            wick = prof[s] / fs2[s]
            mid = (y > 0.1) & (y < 0.9)                                             # away from the core and the far pole
            res[f's={s}'] = dict(
                mode_flatness=float(np.ptp(flat[mid]) / np.mean(flat[mid])),
                mode_power=float(np.polyfit(np.log(y[mid]), np.log(fs2[s][mid]), 1)[0]),
                wick_mean=float(np.sum(W * wick * fs2[s]) / np.sum(W * fs2[s])),
                wick_spread=float(np.sqrt(np.sum(W * fs2[s] * (wick - np.sum(W * wick * fs2[s]) / np.sum(W * fs2[s])) ** 2)
                                          / np.sum(W * fs2[s])) / abs(np.sum(W * wick * fs2[s]) / np.sum(W * fs2[s]))),
                total_density_change=float(np.sum(W * np.abs(tot)) / np.sum(W * np.abs(prof[s]))),     # local |n_+ + n_-| change
                nx_integral=float(np.sum(W * prof[s])))
        b2 = lambda s: np.exp(gammaln(1.5 + s) - gammaln(1.5) - gammaln(1 + s))
        up = [s for s in spins if s > 0.5]
        res['b2_ratio'] = {str(s): float(np.sum(W * fs2[s] / y ** s) / np.sum(W * fs2[0.5] / y ** 0.5)) for s in up}
        res['b2_ratio_exact'] = {str(s): float(b2(s) / b2(0.5)) for s in up}
        res['wick_ratio'] = {str(s): res[f's={s}']['wick_mean'] / res['s=0.5']['wick_mean'] for s in up}
        ladder = []
        for eps in EPS:
            kap = np.ones_like(y) if eps is None else np.exp(-I * y / eps)
            sl = {s: float(np.sum(W * kap * prof[s]) / unit_D) for s in spins}            # d Delta_s / d lambda, D units
            L = dict(eps=eps, slope={str(s): sl[s] for s in spins}, ratio={}, ratio_free={}, ratio_lattice_modes={})
            for s in up:
                L['ratio'][str(s)] = sl[s] / sl[0.5]
                L['ratio_free'][str(s)] = float(b2(s) / b2(0.5) * np.sum(W * kap * y ** s) / np.sum(W * kap * y ** 0.5))
                L['ratio_lattice_modes'][str(s)] = float(np.sum(W * kap * fs2[s]) / np.sum(W * kap * fs2[0.5]))
            ladder.append(L)
        res['ladder'] = ladder
        return res


    def report(res):
        spins = res['spins']
        print(f"{res['label']}: D-calibrated " + ', '.join(f"psi_{s:g} {res['levels_Dcal'][str(s)]:.3f}" for s in spins))
        for s in spins:
            r = res[f's={s}']
            print(f"  s={s}: |f_s|^2 ~ y^{r['mode_power']:.3f} (exact {s}), |f_s|^2/y^s spread {r['mode_flatness']:.3f}; "
                  f"Wick Dn/|f|^2 = {r['wick_mean']:.4f} (spread {r['wick_spread']:.3f}); total-density change "
                  f"{r['total_density_change']:.3f}")
        for s in res['b2_ratio']:
            print(f"  b_{s}^2/b_1/2^2 = {res['b2_ratio'][s]:.4f} (exact {res['b2_ratio_exact'][s]:.4f}); "
                  f"Wick constant ratio {s} : 1/2 = {res['wick_ratio'][s]:.4f} (1)")
        print('  core ladder (slope_s / slope_1/2: lattice | free | lattice modes):')
        for L in res['ladder']:
            e = 'unif' if L['eps'] is None else f"{L['eps']:.2f}"
            print(f"    eps {e:>5}: slope_1/2 {L['slope']['0.5']:+.4f}  " + '  '.join(
                f"s={s}: {L['ratio'][s]:.4f} | {L['ratio_free'][s]:.4f} | {L['ratio_lattice_modes'][s]:.4f}" for s in L['ratio']))


    def main(mode, pkl, h, chi=None, sweeps=None, spins=SPINS):
        t0 = time.time()
        d = pickle.load(open(pkl, 'rb'))
        g, wf = lpar.orbitals(d)
        d.setdefault('I', g.I)
        data = ed_data(d, h) if mode == 'ed' else dmrg_data(d, h, chi, sweeps, spins)
        tag = os.path.basename(pkl)[len('terms_'):-len('.pkl')]
        label = f'{tag} h={h}' + (f' chi={chi}' if chi else ' ED')
        res = analyse(d, data, g, wf, label)
        if mode == 'ed':
            res['check_s_minus'] = float(data['E'][-0.5] - data['E'][0.5])
        res['time'] = time.time() - t0
        report(res)
        store.put(os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}' + (f'_chi{chi}' if chi else '') + '_core5d.json'), res)

    def _run():
        a = sys.argv[1:]
        if a[0] == 'ed':
            main('ed', a[1], float(a[2]))
        else:
            sp = [float(x) for x in a[5].split(',')] if len(a) > 5 else SPINS
            main('dmrg', a[1], float(a[2]), int(a[3]), int(a[4]), sp)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/core5d_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            _run()
    return namespace(locals(), globals())


@section
def ward5d():
    """5d tilt Ward identity: the displacement of the Z2 twist defect on the fuzzy S^4, without calibration.

Physics. The defect worldvolume meets each time slice in the S^2 {x3 = x4 = 0} of S^4. The rotations X_{a3}, X_{a4}
(a = 1, 2, 5 parallel) are broken only at the defect, so for the tilt L = X_35 + i X_45 (T -> T + 1, P unchanged)
    <n|L|0_D> = <n|[H, L]|0_D>/(E_n - E_0),   [H, L] = int_{S^2} x_5 D_+ ,
and in the CFT L|0_D> lies in the displacement family: no weight on D itself (Delta = 4), and on d_5 D (Delta = 5)
    |a_1|^2 = (4 pi/3)^2 * 2 Delta_D * 2 C_D / (Delta_D + 1)^2 = (256 pi^2/225) C_D.
At the 5d Gaussian point the defect is the free half-twist, with C_D = 1/(2 pi^4) (twist/free_cd_d.py), so
|a_1|^2 = 0.0576 with no free parameter.
Lattice. Single-particle matrix elements of X_35 + i X_45 between the untwisted LLL and the regular twisted band are
grid overlaps of the stored generator actions (lpar.py's method). The twisted band is a truncation, like the 3d top
orbital. The many-body tilt is applied with fermion signs, and the result is decomposed into the lowest states of the
target sector (ED, I = 2).
Gates.
  (1) The one-body machinery reproduces <J_- J_+> = <L_par^2> - P(P+1) of lpar_ed_check, through J_+ applied with the
      same code.
  (2) The untwisted LLL is closed under the tilt (no leakage).
usage: python ward5d.py terms_<tag>.pkl [h] [nlev=16]

Run as ward5d terms_designB1c_I2.pkl 8.3.
"""
    import os
    import sys
    import json
    import pickle
    import numpy as np
    import scipy.sparse.linalg as sla
    HERE = 's4d'
    sys.path.insert(0, os.path.join(HERE, '..', 's3'))
    sys.path.insert(0, os.path.join(HERE, '..', 'twist'))
    lpar = globals()['lpar']()   # was: import lpar
    mfree2 = globals()['mfree2']()   # was: import mfree2
    M3 = globals()['s3model']()   # was: import s3model as M3
    _defect_corr = fuzzy; popc = _defect_corr.popc   # was: from defect_corr import popc

    IX35, IX45 = 8, 9


    def tilt_matrix(g, orbs):
        """B[a, b] = <a| X_35 + i X_45 |b>, same flavour, P_a = P_b; the T change is returned too."""
        n = len(orbs)
        W = g.W[:, None]
        ip = lambda u, v: np.sum(W * u.conj() * v)
        B = np.zeros((n, n), complex)
        dTs = set()
        for a, oa in enumerate(orbs):
            for b, ob in enumerate(orbs):
                if oa['f'] != ob['f'] or abs(oa['P'] - ob['P']) > 1e-9 or abs(abs(oa['T'] - ob['T']) - 1) > 1e-9:
                    continue
                v = ip(oa['s'], ob['sA'][IX35] + 1j * ob['sA'][IX45])
                if abs(v) > 1e-10:
                    B[a, b] = v; dTs.add(round(oa['T'] - ob['T'], 6))
        return B, dTs


    def apply_onebody(Bmat, st_in, vec, st_out):
        out = np.zeros(len(st_out), complex)
        for a, b in zip(*np.nonzero(np.abs(Bmat) > 1e-12)):
            if a == b:
                continue
            ok = (((st_in >> np.int64(b)) & 1) == 1) & (((st_in >> np.int64(a)) & 1) == 0)
            lo, hi = min(a, b), max(a, b)
            between = np.int64(((1 << int(hi)) - 1) ^ ((1 << int(lo + 1)) - 1))
            sign = 1 - 2 * (popc(st_in & between) & 1)
            new = st_in ^ (np.int64(1) << np.int64(a)) ^ (np.int64(1) << np.int64(b))
            pos = np.searchsorted(st_out, new)
            pc = np.clip(pos, 0, max(len(st_out) - 1, 0))
            hit = ok & (pos < len(st_out)) & (st_out[pc] == new)
            np.add.at(out, pc[hit], Bmat[a, b] * sign[hit] * vec[hit])
        return out


    def eig(B, orbs, terms, mu, k):
        H = mfree2.Operator(B, orbs, terms, mu)
        k = min(k, B.dim - 2)
        w, v = sla.eigsh(H, k=k, which='SA', tol=1e-12)
        o = np.argsort(w)
        return w[o], v[:, o]


    def main(pkl, h=None, nlev=16):
        d = pickle.load(open(pkl, 'rb'))
        h = d['h'] if h is None else h
        g, orbs_w = lpar.orbitals(d)
        orbs = d['def_orbs']
        Ne = sum(1 for o in orbs if o['f'] == 0)
        mu = np.array([-h if o['f'] == 0 else h for o in orbs])
        out = dict(pkl=os.path.basename(pkl), h=h, gates={})
        # gate 1: J_+ through the same one-body code reproduces <J_- J_+> = <L^2> - P(P+1)
        A, _, _ = lpar.checks(g, orbs_w)
        Bp = M3.Basis(orbs, Ne, 0, 0.5, -1)
        w, v = eig(Bp, orbs, d['def_terms'], mu, 6)
        Bq = M3.Basis(orbs, Ne, 1, 0.5, -1)
        jj = [float(np.vdot(x, x).real) for x in (apply_onebody(A, Bp.states, v[:, i], Bq.states) for i in range(len(w)))]
        out['gates']['J+ norms (P=0, T=1/2, z2=-): should be <L^2> = 0,2,0,2,6,2'] = [round(x, 6) for x in jj]
        # gate 2: tilt leakage out of the untwisted LLL
        Bt, dTs = tilt_matrix(g, orbs_w)
        W = g.W[:, None]
        leak = []
        for b, ob in enumerate(orbs_w):
            if ob['f'] != 0:
                continue
            x = ob['sA'][IX35] + 1j * ob['sA'][IX45]
            tot = float(np.sum(W * np.abs(x) ** 2).real)
            inband = float(np.sum(np.abs(Bt[:, b]) ** 2))
            leak.append((tot - inband) / max(tot, 1e-30))
        out['gates']['untwisted LLL tilt leakage (max)'] = float(max(leak))
        out['tilt_dT'] = sorted(dTs)
        dT = max(dTs) if dTs else 1.0
        # the tilt on the defect vacuum, decomposed in the (P = 0, T = +dT, z2 = +) sector
        B0 = M3.Basis(orbs, Ne, 0, 0.0, 1)
        E0, V0 = eig(B0, orbs, d['def_terms'], mu, 1)
        Bt_sec = M3.Basis(orbs, Ne, 0, dT, 1)
        E1, V1 = eig(Bt_sec, orbs, d['def_terms'], mu, nlev)
        Lv = apply_onebody(Bt, B0.states, V0[:, 0], Bt_sec.states)
        p = np.abs(V1.conj().T @ Lv) ** 2
        lev = 4 * (E1 - E0[0]) / (E1[0] - E0[0])                         # D-calibrated: lowest (0, 1, +) state = D at 4
        out.update(levels_Dcal=lev.tolist(), p=p.tolist(), norm2=float(np.vdot(Lv, Lv).real))
        i5 = [i for i in range(len(lev)) if 4.4 < lev[i] < 5.8]
        if i5:
            k = max(i5, key=lambda i: p[i])
            out['dD'] = dict(level=float(lev[k]), p=float(p[k]), C_D=float(p[k] * 225 / (256 * np.pi ** 2)))
        out['D'] = dict(level=float(lev[0]), p=float(p[0]))
        out['prediction'] = dict(p_dD_free=float(256 * np.pi ** 2 / 225 / (2 * np.pi ** 4)), C_D_free=1 / (2 * np.pi ** 4))
        tag = os.path.basename(pkl)[len('terms_'):-len('.pkl')]
        pass   # was os.makedirs(...): the store needs no output directory
        store.put(os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}_ward5d.json'), out)
        print(f'{tag} h={h}: gates {out["gates"]}  tilt changes T by {out["tilt_dT"]}')
        print('  target-sector levels (D at 4) and tilt weights:')
        for i in range(min(nlev, 14)):
            print(f'    {lev[i]:7.3f}  p {p[i]:.3e}')
        if 'dD' in out:
            print(f"  d_5 D at {out['dD']['level']:.3f}: p = {out['dD']['p']:.4f} -> C_D = {out['dD']['C_D']:.5f} "
                  f"(free 5d: p = {out['prediction']['p_dD_free']:.4f}, C_D = {out['prediction']['C_D_free']:.5f}); "
                  f"p(D)/p(dD) = {out['D']['p'] / out['dD']['p']:.3f} (CFT 0)")

    def _run():
        main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None, int(sys.argv[3]) if len(sys.argv) > 3 else 16)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/ward5d_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            _run()
    return namespace(locals(), globals())


@section
def ward5d_dmrg():
    """5d tilt Ward identity at I >= 3 by DMRG (the ED version is ward5d.py; physics and prediction are there).

Defect vacuum in (P = 0, T = 0, z2 = +), the n lowest states of (P = 0, T = 1, z2 = +), and the overlaps
<n| X_35 + i X_45 |0_D>. The tilt is a one-body operator applied hop by hop with explicit Jordan-Wigner strings
(twist/dmrg_ward.apply_hop). Levels are D-calibrated (lowest T = 1 state = D at 4).
Prediction at the 5d Gaussian point (free half-twist): weight 0 on D (exact, by the parallel SO(3)), and
|a_1|^2 = (256 pi^2/225) C_D = 0.0576 on d_5 D at 5. At I = 2 ED gives 7e-4 (the band is too small).
Gate: at I = 2 the weights must match ward5d.py (lpar/<tag>_h<h>_ward5d.json).
usage: python ward5d_dmrg.py terms_<tag>.pkl h chi sweeps [n=10] [--check]   -> lpar/<tag>_h<h>_chi<chi>_ward5d.json

Run as ward5d_dmrg terms_i3A.pkl 4.5 600 14 10 and ward5d_dmrg terms_i4A.pkl 5.66 800 16 12.
"""
    import os
    import sys
    import json
    import time
    import pickle
    import numpy as np
    HERE = 's4d'
    sys.path.insert(0, os.path.join(HERE, '..', 'twist'))
    s4dmrg = globals()['s4dmrg']()   # was: import s4dmrg
    lpar = globals()['lpar']()   # was: import lpar
    _ward5d = globals()['ward5d'](); tilt_matrix = _ward5d.tilt_matrix   # was: from ward5d import tilt_matrix
    _dmrg_ward = fuzzy.question('01_known_results/free_control').dmrg_ward(); apply_hop = _dmrg_ward.apply_hop   # was: from dmrg_ward import apply_hop


    def states(model, order, occ, n, chi, sweeps):
        psis, Es = [], []
        for _ in range(n):
            E, psi = s4dmrg.run(model, order, occ, chi, sweeps, orthogonal_to=psis or None)
            psis.append(psi); Es.append(float(E))
        o = np.argsort(Es)
        return [Es[i] for i in o], [psis[i] for i in o]


    def main(pkl, h, chi, sweeps, n=10, check=False):
        t0 = time.time()
        d = pickle.load(open(pkl, 'rb'))
        tag = os.path.basename(pkl)[len('terms_'):-len('.pkl')]
        g, orbs_w = lpar.orbitals(d)
        B, dTs = tilt_matrix(g, orbs_w)
        orbs = [dict(o, lat=l) for o, l in zip(d['def_orbs'], d['def_lat'])]
        Ne = sum(1 for o in orbs if o['f'] == 0)
        model, order = s4dmrg.build(orbs, d['def_terms'], h)
        print(f'{tag} h={h}: {len(orbs)} modes, model built [{time.time() - t0:.0f}s]', flush=True)
        E0, S0 = states(model, order, s4dmrg.best_product_state(orbs, d['def_terms'], h, Ne, 0, 0.0, 1), 1, chi, sweeps)
        dT = max(dTs)
        E1, S1 = states(model, order, s4dmrg.best_product_state(orbs, d['def_terms'], h, Ne, 0, dT, 1), n, chi, sweeps)
        pos = {k: i for i, k in enumerate(order)}
        amp = np.zeros(n, complex)
        for a, b in zip(*np.nonzero(np.abs(B) > 1e-12)):
            if a == b:
                continue
            v = apply_hop(S0[0], pos[int(a)], pos[int(b)])
            if v.norm == 0:
                continue
            for i, br in enumerate(S1):
                amp[i] += B[a, b] * br.overlap(v)
        p = np.abs(amp) ** 2
        lev = 4 * (np.array(E1) - E0[0]) / (E1[0] - E0[0])
        out = dict(pkl=os.path.basename(pkl), h=h, chi=chi, levels_Dcal=lev.tolist(), p=p.tolist(), time=time.time() - t0)
        i5 = [i for i in range(n) if 4.4 < lev[i] < 5.8]
        if i5:
            k = max(i5, key=lambda i: p[i])
            out['dD'] = dict(level=float(lev[k]), p=float(p[k]), C_D=float(p[k] * 225 / (256 * np.pi ** 2)))
        if check:
            R = store.get(os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}_ward5d.json'))
            m = min(n, len(R['p']))
            out['check'] = dict(dlev=float(np.max(np.abs(np.array(R['levels_Dcal'][:m]) - lev[:m]))),
                                dp=float(np.max(np.abs(np.array(R['p'][:m]) - p[:m]))))
            print(f"  gate vs ED: max |level diff| {out['check']['dlev']:.1e}, max |p diff| {out['check']['dp']:.1e}", flush=True)
        store.put(os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}_chi{chi}_ward5d.json'), out)
        print('  levels ' + ' '.join(f'{x:.3f}' for x in lev) + '\n  p ' + ' '.join(f'{x:.2e}' for x in p) +
              (f"\n  d_5 D {out['dD']['level']:.3f}: p {out['dD']['p']:.4f}, C_D {out['dD']['C_D']:.5f} (free 5d 0.00513)" if 'dD' in out else ''),
              flush=True)

    def _run():
        a = [x for x in sys.argv[1:] if not x.startswith('--')]
        main(a[0], float(a[1]), int(a[2]), int(a[3]), int(a[4]) if len(a) > 4 else 10, check='--check' in sys.argv)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/ward5d_dmrg_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            _run()
    return namespace(locals(), globals())


@section
def s4_summary():
    """The numbers of Sec. 6, Appendix F and Table 25, from the stored results (written for this package; it reads stored
results and recomputes nothing).
- Twisted bands at I = 2, 3, 4 (s4d/band_I*.log, from save_band): orbitals kept, band width and gap to the next
  variational level. Levels are measured from the untwisted Landau energy I^2/2 + 2I, and the band ends at the first
  gap larger than one, the rule of the research note's number generator.
- The map from the six I = 2 local couplings to the six bulk two-particle channel energies (design.OPS): its singular
  values, its rank and the combination it annihilates.
- Table 25. The field h_c where the bulk ratio Delta_phi/(Delta_dphi - Delta_phi) is 3/2, and the D-calibrated defect
  levels there, interpolated linearly in h between the DMRG fields (s4d/note/i*_points.json, from fig_curves). Then the
  levels measured directly at the table's fields (s4d/lpar/<model>_h<h>_chi800*.json, from lpar_measure), the
  parallel-Casimir spin-two search (..._pen20*.json, from par_spin2) and the I = 4 spin-three state.
- Bulk-to-defect overlaps at I = 3, h = 4.5 (core5d): b_{3/2}^2/b_{1/2}^2 against the exact 4/3, and the growth of the
  squared overlaps with the distance r from the defect sphere, r^(2 p) with p the fitted mode power.
- Equal-time defect correlator (corr5d, rows with tau = 0): median |R_lattice - R_exact| at I = 2 (h = 8.3) and at
  I = 3 for the five fields h = 1-14, over all pairs and with pairs whose angular separation is within 15 degrees of 90 removed, where
  the lattice bulk correlator passes through zero and R is ill-conditioned.
- Tilt Ward identity (ward5d, ward5d_dmrg): the tilt's weight on the computed states above the displacement and below
  5.5 (the level-five window, where d_par D mixes with psi_{3/2} psi_{-1/2}), as a fraction of the exact free weight.
"""
    import re
    import numpy as np
    design = globals()['design']()   # was: import design
    corr5d = globals()['corr5d']()   # was: import corr5d


    def lin(x, xs, ys):
        # from s4d/note/gen_numbers.py
        xs, ys = np.asarray(xs, float), np.asarray(ys, float); o = np.argsort(xs); xs, ys = xs[o], ys[o]
        if xs[0] <= x <= xs[-1]: return float(np.interp(x, xs, ys)), False
        i = 0 if x < xs[0] else len(xs) - 2
        return float(ys[i] + (x - xs[i]) * (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])), True


    def bands():
        print('twisted bands (save_band):')
        for I in (2, 3, 4):
            lev = [float(x) for x in re.findall(r'\((-?[0-9.]+), -?[0-9.]+, -?[0-9.]+\)', store.text(f's4d/band_I{I}.log'))]
            vals, counts = np.unique(np.round(lev, 4), return_counts=True)
            gi = np.flatnonzero(np.diff(vals) > 1.0)[0]
            print(f'  I={I}: {int(counts[:gi + 1].sum())} orbitals, width {vals[gi] - vals[0]:.3f}, '
                  f'gap to the next level {vals[gi + 1] - vals[gi]:.2f}')


    def coupling_rank():
        M = np.array([design.OPS[n] for n in design.NAMES]).T
        u, sv, vt = np.linalg.svd(M)
        null = vt[-1] / vt[-1][design.NAMES.index('d1')]
        print('I=2 couplings -> bulk channel energies: singular values ' + ' '.join(f'{s:.3g}' for s in sv) +
              f'; rank {int(np.sum(sv > 1e-8 * sv[0]))}')
        print('  annihilated combination: ' + ' '.join(f'{n} {c:+.4g}' for n, c in zip(design.NAMES, null)))


    KEYS = [('psi12', 'psi_1/2'), ('psi32', 'psi_3/2'), ('l1', 's_par=1'), ('l2', 's_par=2')]


    def interpolated():
        print('bulk ratio 3/2 and the levels there, interpolated between the DMRG fields (fig_curves):')
        for name, f in (('I=3 A', 'i3_points'), ('I=3 B', 'i3B_points'), ('I=4 A', 'i4_points')):
            P = store.get(f's4d/note/{f}.json')
            pb = [p for p in P if 'ratio' in p]
            hc, ex = lin(1.5, [p['ratio'] for p in pb], [p['h'] for p in pb])
            row = []
            for k, lab in KEYS:
                q = [p for p in P if k in p]
                v = lin(hc, [p['h'] for p in q], [p[k] for p in q])[0] if len(q) >= 2 else (q[0][k] if q else float('nan'))
                row.append(f'{lab} {v:.3f}')
            print(f'  {name}: h_c = {hc:.3f}{" (extrapolated)" if ex else ""};  ' + ', '.join(row))


    def direct():
        print('levels measured at fixed fields (lpar_measure, par_spin2), displacement candidate at 4:')
        for name, base, pens in (('I=3 A', 'i3A_h4.5000_chi800', ('_pen20', '_pen20_sw14')),
                                 ('I=3 B', 'i3B_h8.2500_chi800', ()),
                                 ('I=4 A', 'i4A_h6.0000_chi800', ('_pen20',))):
            R = {}
            for k in store.glob(f's4d/lpar/{base}.json') + store.glob(f's4d/lpar/{base}_b.json'):
                R.update(store.get(k))
            e0, eD = R['0_0.0_1']['E'][0], R['0_1.0_1']['E'][0]
            cal = lambda E: 4 * (E - e0) / (eD - e0)
            row = [f'{lab} {cal(R[s]["E"][0]):.3f} (L2 {R[s]["L2"][0]:.3f})'
                   for s, lab in (('0_0.5_-1', 'psi_1/2'), ('0_1.5_-1', 'psi_3/2'), ('1_0.5_-1', 's_par=1'), ('2_0.5_-1', 'lowest P=2'))]
            print(f'  {name}, h = {base.split("_h")[1][:6]}: ' + ', '.join(row))
            for p in pens:
                S = store.get(f's4d/lpar/{base}{p}.json')
                print(f'     parallel-Casimir search {p[1:]}: ' + ', '.join(f'{cal(E):.3f} (L2 {L:.3f})' for E, L in zip(S['E'], S['L2'])))


    def overlaps():
        c = store.get('s4d/lpar/i3A_h4.5000_chi600_core5d.json')
        print(f"I=3 A, h=4.5 overlaps: b_3/2^2/b_1/2^2 = {c['b2_ratio']:.3f} (exact {c['b2_ratio_exact']:.4f}); squared overlaps "
              f"grow as r^{2 * c['s=0.5']['mode_power']:.2f} and r^{2 * c['s=1.5']['mode_power']:.2f} (free r^1 and r^3)")


    def correlator():
        print('equal-time correlator, median |R_lattice - R_exact| (all pairs | pairs within 15 deg of 90 removed):')
        for k in ['s4d/lpar/designB1c_I2_h8.3000_corr5d.json'] + store.glob('s4d/lpar/i3A_h*_chi600_corr5d.json'):
            ang, dev = [], []
            for x in (r for r in store.get(k)['rows'] if r['tau'] == 0):
                a = corr5d.point(x['chi1'], x['th1'], x['dphi1'], x['dphi2'], np.exp(x['tau'])); b = corr5d.point(x['chi2'], x['th2'], 0, 0)
                ang.append(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
                dev.append(abs(x['R_lat'] - x['R_exact']))
            ang, dev = np.array(ang), np.array(dev)
            away = np.abs(ang - 90) > 15
            print(f'  {k.split("/")[-1][:-11]:24s} {np.median(dev):.3f} | {np.median(dev[away]):.3f}')


    def ward():
        pfree = store.get('s4d/lpar/designB1c_I2_h8.3000_ward5d.json')['prediction']['p_dD_free']
        print(f'tilt weight on the level-five window 4 < level < 5.5, fraction of the free {pfree:.4f}:')
        for name, k in (('I=2 ED, h=8.3', 'designB1c_I2_h8.3000'), ('I=3 A DMRG chi=600, h=4.5', 'i3A_h4.5000_chi600'),
                        ('I=4 A DMRG chi=800, h=5.66', 'i4A_h5.6600_chi800')):
            W = store.get(f's4d/lpar/{k}_ward5d.json')
            L, p = np.array(W['levels_Dcal']), np.array(W['p'])
            win = (L > 4 + 1e-9) & (L < 5.5)
            print(f'  {name}: {100 * p[win].sum() / pfree:.1f}% over {int(win.sum())} states; largest single state '
                  f'{100 * p.max() / pfree:.1f}% at {L[np.argmax(p)]:.3f}')

    def main():
        bands(); coupling_rank(); interpolated(); direct(); overlaps(); correlator(); ward()

    def cli(argv):
        main()
    return namespace(locals(), globals())


@section
def lambda5():
    """The parallel conformal generator Lambda_5 = P_5 + K_5 on the fuzzy S^4, and the descendant Ward test.

Physics. The defect worldvolume meets each time slice in the S^2 {x3 = x4 = 0}. The parallel conformal group SO(4,1)
contains P_a + K_a for the parallel directions a = 1, 2, 5. On the regulated sphere (Fardelli-Fitzpatrick-Katz)
    Lambda_5 = int dOmega x_5 H(x) = c (P_5 + K_5) + (irrelevant corrections),
with H(x) the Hamiltonian density. x_5 is invariant under both Cartan rotations X_12 (P) and X_34 (T), so Lambda_5
keeps every sector label (P, T, z2) and its matrix elements are torus-invariant integrands (exact on the 2d grid).
For a defect primary psi of dimension D, K_5|psi> = 0 and Lambda_5|psi> = c P_5|psi>: ALL of its weight off psi lies
on ONE state, d_5 psi at D + 1 (parallel spin 1, l_z = 0), with |c P_5 psi|^2 = 2 D c^2. Normalization-free tests:
  (T1) concentration: the largest single-state share of ||Lambda psi||^2 - |<psi|Lambda|psi>|^2 tends to 1;
  (T2) that state's D-calibrated level is D_psi + 1 (the level that quantum numbers alone labelled d psi);
  (T3) ratios of the absolute weights of two primaries equal the ratios of their dimensions (same c), e.g.
       W(psi)/W(D) = D_psi/D_D (2/4 for the free half-twist) and W_bulk(phi)/W(psi) = D_phi/D_psi (3/4).
  (T4) the vacua are invariant: Lambda_5|0> has no weight off |0>.
Construction. H(x) is the normal-ordered local interaction of build_I.terms_for (contact[L], cas[L], density
Laplacian in its gradient form) plus the field -h(n_+ - n_-); Lambda_5 inserts x_5 into the grid weights. The gradient
form fixes the total-derivative ambiguity of the Laplacian density (a choice, as in twist/lambda_z.py). The weighted
field is one-body but not diagonal; it is folded into two-body terms with c+_i c_j = sum_k c+_i c+_k c_k c_j/(N_e - 1),
exact at fixed N_e (lpar.py's device), so the ED sector builder and the MPO builder take it unchanged (with h = 0).
Gates. (G1) weight 1 reproduces the stored H terms and gives the identity Gram matrix; (G2) Lambda is real symmetric in
every sector built; (G3) at I = 2, Lambda with weight 1 equals H in a sector (ED).
usage: python lambda5.py gate terms_<tag>.pkl config.json
       python lambda5.py ed terms_<tag>.pkl terms_<tag>_bulk.pkl config.json h     -> lpar/<tag>_h<h>_lambda5.json

Paper, App. F.5 (Eq. s4generator, Table 26). Run as lambda5 gate terms_designB1c_I2.pkl designB1c_I2.json, lambda5 ed terms_designB1c_I2.pkl terms_designB1c_I2_bulk.pkl designB1c_I2.json 8.3, lambda5 dmrg terms_designB1c_I2.pkl designB1c_I2.json 8.3 200 10 7 (the DMRG gate against ed) and lambda5 dmrg terms_i3A.pkl i3A.json 4.5 400 10 10 1 (the I = 3 row; about 20 minutes and 6.4 GB).
"""
    import os
    import sys
    import json
    import time
    import pickle
    import numpy as np
    HERE = 's4d'
    sys.path.insert(0, HERE)
    geom = globals()['geom']()   # was: import geom
    inter = globals()['inter']()   # was: import inter
    lpar = globals()['lpar']()   # was: import lpar


    def couplings(cfg):
        c = cfg['couplings']
        U = {int(n[1:]): v for n, v in c.items() if n.startswith('c') and abs(v) > 0}
        K = {int(n[1:]): v for n, v in c.items() if n.startswith('k') and abs(v) > 0}
        return U, K, c.get('d1', 0.0)


    def two_body(g, orbs, I, cfg, wt):
        """Normal-ordered interaction terms with grid weight W * wt (wt = 1 reproduces build_I.terms_for)."""
        U, K, d1 = couplings(cfg)
        W = g.W * wt
        t = inter.pair_terms(orbs, I, W, U, K)
        if abs(d1) > 0:
            t = inter.merge(t, inter.density_terms(orbs, W, 0.0, d1))
        tl, bad = inter.to_list(t)
        return [x for x in tl if x[0] != x[1] and x[2] != x[3]], bad


    def gram(g, orbs, wt):
        """B[i, j] = int wt s_i^dag s_j for the same flavour and Cartan charges (fibre-traced)."""
        n = len(orbs)
        B = np.zeros((n, n), complex)
        W = (g.W * wt)[:, None]
        for i, a in enumerate(orbs):
            for j, b in enumerate(orbs):
                if a['f'] == b['f'] and abs(a['P'] - b['P']) < 1e-9 and abs(a['T'] - b['T']) < 1e-9:
                    B[i, j] = np.sum(W * a['s'].conj() * b['s'])
        return B


    def fold_field(orbs, B, h, Ne, tol=1e-12):
        """-h sum (B^+_ij c+_i c_j) + h sum (B^-_ij c+_i c_j) as two-body terms at fixed N_e."""
        n = len(orbs)
        out = []
        for i in range(n):
            for j in range(n):
                b = B[i, j]
                if abs(b) < tol:
                    continue
                v = (-h if orbs[i]['f'] == 0 else h) * b / (Ne - 1)
                for k in range(n):
                    if k != i and k != j:
                        out.append((i, k, k, j, v))
        return out


    def lambda_terms(g, orbs, I, cfg, h, Ne, wt):
        tb, bad = two_body(g, orbs, I, cfg, wt)
        B = gram(g, orbs, wt)
        assert np.max(np.abs(B.imag)) < 1e-10
        return tb + [(a, b, c, d, float(np.real(v))) for (a, b, c, d, v) in fold_field(orbs, B.real, h, Ne)], bad


    def bulk_orbitals(I):
        g = geom.Grid(I, 40, 40)
        return g, inter.lll_orbitals(g, 0) + inter.lll_orbitals(g, 1)


    def as_dict(tl):
        out = {}
        for (a, b, c, d, v) in tl:
            out[(a, b, c, d)] = out.get((a, b, c, d), 0.0) + v
        return out


    def gate(pkl, cfgf):
        d = pickle.load(open(pkl, 'rb'))
        cfg = store.get(cfgf)
        g, orbs = lpar.orbitals(d)
        I = g.I
        t0 = time.time()
        tb, bad = two_body(g, orbs, I, cfg, np.ones_like(g.W))
        ref, new = as_dict(d['def_terms']), as_dict(tb)
        keys = set(ref) | set(new)
        diff = max(abs(ref.get(k, 0.0) - new.get(k, 0.0)) for k in keys)
        scale = max(abs(v) for v in ref.values())
        B = gram(g, orbs, np.ones_like(g.W))
        print(f'G1 I={I}: {len(tb)} terms vs stored {len(ref)}; max |diff| {diff:.2e} (scale {scale:.2e}); imag {bad:.1e}; '
              f'Gram - 1: {np.max(np.abs(B - np.eye(len(orbs)))):.2e}  [{time.time() - t0:.0f}s]', flush=True)
        tx, badx = two_body(g, orbs, I, cfg, g.x5)
        print(f'x5-weighted: {len(tx)} terms, imag {badx:.1e}; x5 Gram max offdiag '
              f'{np.max(np.abs(gram(g, orbs, g.x5) - np.diag(np.diag(gram(g, orbs, g.x5))))):.3f}', flush=True)


    def ed(pkl, bulkpkl, cfgf, h, nfull=4000):
        mb = globals()['mb']()   # was: import mb
        d = pickle.load(open(pkl, 'rb')); db = pickle.load(open(bulkpkl, 'rb'))
        cfg = store.get(cfgf)
        tag = os.path.basename(pkl)[len('terms_'):-len('.pkl')]
        g, orbs = lpar.orbitals(d)
        I = g.I
        Ne = sum(1 for o in orbs if o['f'] == 0)
        res = dict(tag=tag, I=I, h=h, Ne=Ne, sectors={})

        def analyse(label, orbsx, Hterms, Lterms, Lterms1, sec, ref):
            H, Bs = mb.sector(orbsx, Hterms, h, Ne, *sec)
            L, _ = mb.sector(orbsx, Lterms, 0.0, Ne, *sec)
            L1, _ = mb.sector(orbsx, Lterms1, 0.0, Ne, *sec)
            Hd = H.toarray(); Ld = L.toarray()
            g3 = np.max(np.abs(L1.toarray() - Hd))
            g2 = np.max(np.abs(Ld - Ld.conj().T))
            e, v = np.linalg.eigh(Hd)
            psi = v[:, 0]
            lv = Ld @ psi
            tot = float(np.vdot(lv, lv).real - abs(np.vdot(psi, lv)) ** 2)
            amp = v.conj().T @ lv
            w = np.abs(amp) ** 2
            w[0] = 0.0
            top = np.argsort(-w)[:6]
            rows = [dict(n=int(n), E=float(e[n]), w=float(w[n]), share=float(w[n] / tot) if tot > 0 else 0.0) for n in top]
            rec = dict(sector=list(sec), dim=int(H.shape[0]), E=e[:12].tolist(), total=tot, diag=float(np.vdot(psi, lv).real),
                       top=rows, G2_herm=float(g2), G3_weight1_minus_H=float(g3))
            res['sectors'][label] = rec
            return rec

        t0 = time.time()
        one = np.ones_like(g.W)
        Hd_terms = d['def_terms']
        Ld_terms, bad = lambda_terms(g, orbs, I, cfg, h, Ne, g.x5)
        L1_terms, _ = lambda_terms(g, orbs, I, cfg, h, Ne, one)
        print(f'defect Lambda_5: {len(Ld_terms)} terms (imag {bad:.1e}) [{time.time() - t0:.0f}s]', flush=True)
        for label, sec in (('defect_vac', (0, 0.0, 1)), ('psi', (0, 0.5, -1)), ('D', (0, 1.0, 1))):
            analyse(label, orbs, Hd_terms, Ld_terms, L1_terms, sec, None)
        gb, ob = bulk_orbitals(I)
        Lb_terms, badb = lambda_terms(gb, ob, I, cfg, h, Ne, gb.x5)
        L1b_terms, _ = lambda_terms(gb, ob, I, cfg, h, Ne, np.ones_like(gb.W))
        for label, sec in (('bulk_vac', (0, 0.0, 1)), ('phi', (0, 0.0, -1))):
            analyse(label, ob, db['bulk_terms'], Lb_terms, L1b_terms, sec, None)
        S = res['sectors']
        E0d, ED = S['defect_vac']['E'][0], S['D']['E'][0]
        cal = lambda E: 4 * (E - E0d) / (ED - E0d)
        E0b, Ephi = S['bulk_vac']['E'][0], S['phi']['E'][0]
        calb = lambda E: 1.5 * (E - E0b) / (Ephi - E0b)
        for k in ('defect_vac', 'psi', 'D'):
            for r in S[k]['top']:
                r['Dcal'] = cal(r['E'])
            S[k]['Dcal0'] = cal(S[k]['E'][0])
        for k in ('bulk_vac', 'phi'):
            for r in S[k]['top']:
                r['phical'] = calb(r['E'])
        for k, rec in S.items():
            lev = 'Dcal' if k in ('defect_vac', 'psi', 'D') else 'phical'
            print(f"{k:10s} dim {rec['dim']:5d}  G2 {rec['G2_herm']:.1e}  G3 {rec['G3_weight1_minus_H']:.1e}  "
                  f"total off-diag weight {rec['total']:.4e}", flush=True)
            print('    ' + '  '.join(f"[{r[lev]:.3f}: {r['share']:.3f}]" for r in rec['top'][:5]), flush=True)
        Dpsi, DD = S['psi']['Dcal0'], 4.0
        Wpsi, WD, Wphi = S['psi']['top'][0]['w'], S['D']['top'][0]['w'], S['phi']['top'][0]['w']
        res['ratios'] = dict(Wpsi_over_WD=Wpsi / WD, Dpsi_over_DD=Dpsi / DD, Wphi_over_Wpsi=Wphi / Wpsi, Dphi_over_Dpsi=1.5 / Dpsi)
        print('ratios: ' + json.dumps({k: round(v, 4) for k, v in res['ratios'].items()}), flush=True)
        pass   # was os.makedirs(...): the store needs no output directory
        fn = os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}_lambda5.json')
        store.put(fn, res)
        print(f'-> {fn} [{time.time() - t0:.0f}s]')


    def states(model, order, occ, n, chi, sweeps):
        s4dmrg = globals()['s4dmrg']()   # was: import s4dmrg
        psis, Es = [], []
        for _ in range(n):
            E, psi = s4dmrg.run(model, order, occ, chi, sweeps, orthogonal_to=psis or None)
            psis.append(psi); Es.append(float(E))
        o = np.argsort(Es)
        return [Es[i] for i in o], [psis[i] for i in o]


    def dmrg(pkl, cfgf, h, chi, sweeps, n, nD=1):
        """The psi test (and, with nD > 1, the D test) by DMRG. Total off-psi weight = Lambda.variance(psi), exact."""
        s4dmrg = globals()['s4dmrg']()   # was: import s4dmrg
        from tenpy.networks.mpo import MPOEnvironment
        t0 = time.time()
        d = pickle.load(open(pkl, 'rb')); cfg = store.get(cfgf)
        tag = os.path.basename(pkl)[len('terms_'):-len('.pkl')]
        g, orbs_w = lpar.orbitals(d)
        I = g.I
        orbs = [dict(o, lat=l) for o, l in zip(d['def_orbs'], d['def_lat'])]
        Ne = sum(1 for o in orbs if o['f'] == 0)
        Lt, bad = lambda_terms(g, orbs_w, I, cfg, h, Ne, g.x5)
        model, order = s4dmrg.build(orbs, d['def_terms'], h)
        lmodel, lorder = s4dmrg.build(orbs, Lt, 0.0)
        assert list(order) == list(lorder)
        print(f'{tag} h={h} chi={chi}: {len(orbs)} modes, Lambda_5 {len(Lt)} terms (imag {bad:.1e}), models [{time.time() - t0:.0f}s]',
              flush=True)
        occ = lambda P, T, z: s4dmrg.best_product_state(orbs, d['def_terms'], h, Ne, P, T, z)
        E0 = states(model, order, occ(0, 0.0, 1), 1, chi, sweeps)[0][0]
        out = dict(pkl=os.path.basename(pkl), h=h, chi=chi, sweeps=sweeps, E0=E0, tests={})

        def test(label, sec, k):
            Es, S = states(model, order, occ(*sec), k, chi, sweeps)
            amp = [complex(MPOEnvironment(S[i], lmodel.H_MPO, S[0]).full_contraction(0)) for i in range(k)]
            tot = float(np.real(lmodel.H_MPO.variance(S[0])))
            w = np.abs(np.array(amp)) ** 2; w[0] = 0.0
            out['tests'][label] = dict(sector=list(sec), E=Es, amp=[[a.real, a.imag] for a in amp], w=w.tolist(), total=tot)
            print(f'  {label}: {k} states, total off-diagonal weight {tot:.4e}, captured {w.sum() / tot:.3f} [{time.time() - t0:.0f}s]',
                  flush=True)
            return Es

        ED = test('D', (0, 1.0, 1), nD)[0]
        test('psi', (0, 0.5, -1), n)
        cal = lambda E: 4 * (E - E0) / (ED - E0)
        for label, r in out['tests'].items():
            r['Dcal'] = [cal(E) for E in r['E']]
            r['share'] = [x / r['total'] for x in r['w']] if r['total'] > 0 else []
            print(f'  {label}: ' + '  '.join(f'[{a:.3f}: {s:.3f}]' for a, s in zip(r['Dcal'], r['share'])), flush=True)
        out['time'] = time.time() - t0
        fn = os.path.join(HERE, 'lpar', f'{tag}_h{h:.4f}_chi{chi}_lambda5.json')
        store.put(fn, out)
        print(f'-> {fn} [{out["time"]:.0f}s]', flush=True)

    def main():
        a = sys.argv
        if a[1] == 'gate':
            gate(a[2], a[3])
        elif a[1] == 'ed':
            ed(a[2], a[3], a[4], float(a[5]))
        elif a[1] == 'dmrg':
            dmrg(a[2], a[3], float(a[4]), int(a[5]), int(a[6]), int(a[7]), int(a[8]) if len(a) > 8 else 1)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('s4d/logs/lambda5_{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def audit_s4d():
    """Independent small-cutoff checks; leave the production data and Hamiltonians untouched.

Run with ~/anaconda3/envs/quantum/bin/python. Results are written beside this script.
The optional density improvement is fixed on the I=2 defect vacuum only;
the psi sector is an independent test, not part of that fit.

Paper, App. F.2-F.5. harmonic_checks proves the density weights f_n of Eq. (s4densityweight) exactly at I = 2, 3, 4, 8 (App. F.3); saved_checks reads the bands and the lambda5 and defect_dmrg results (band multiplets, ED against DMRG at I = 2, the chi = 400 against chi = 800 levels at I = 3); improvement_test fits the vacuum-improved moment of Table 26 on the I = 2 defect vacuum and holds the psi and D sectors out; rotation_checks tests the parallel rotation algebra of the projected generators (App. F.2). Run as audit_s4d; saved_checks needs the results of save_band, lambda5 and defect_dmrg, which are not stored here.
"""
    from pathlib import Path
    import json
    import os
    import pickle
    import sys

    import numpy as np
    import sympy as sp
    ROOT = Path('.')   # was the research folder s4d/: bands and terms are read from s4d_files/, results from the store


    def harmonic_checks():
        t = sp.symbols('t')
        records = []
        for I in (2, 3, 4, 8):
            for n in range(min(I, 5) + 1):
                poly = sp.Poly(sp.gegenbauer(n, sp.Rational(3, 2), 2*t-1), t)
                terms = poly.terms()
                norm = sp.integrate(6*t*(1-t)*poly.as_expr()**2, (t, 0, 1))
                trace = 0
                for k in range(I+1):
                    mean = sum(c*sp.rf(k+2, p[0])/sp.rf(I+4, p[0]) for p, c in terms)
                    trace += (k+1)*(I-k+1)*mean**2/norm
                measured = sp.factor(trace/sp.binomial(I+3, 3))
                exact = sp.factorial(I)*sp.factorial(I+3)/(sp.factorial(I-n)*sp.factorial(I+n+3))
                assert measured == exact
                records.append(dict(I=I, n=n, value=str(measured)))
        return records


    def saved_checks():
        bands = []
        for I, n in ((2, 6), (3, 14), (4, 26)):
            b = np.load(ROOT / f'band_I{I}.npz')
            energies = b['E']
            assert np.all(np.diff(energies) > -1e-6)
            multiplets = []
            for t in np.unique(b['T'][:n]):
                ps, counts = np.unique(np.round(b['P'][:n][abs(b['T'][:n]-t)<1e-7], 6), return_counts=True)
                weights = dict(zip(ps, counts))
                assert all(weights[p] == weights[-p] for p in weights)
                for p in sorted(x for x in weights if x >= 0):
                    multiplicity = int(weights[p]-weights.get(p+1, 0))
                    assert multiplicity >= 0
                    if multiplicity:
                        multiplets.append(dict(T=float(t), ell=float(p), multiplicity=multiplicity))
            assert round(sum((2*x['ell']+1)*x['multiplicity'] for x in multiplets)) == n
            bands.append(dict(I=I, count=n, width=float(energies[n-1]-energies[0]),
                              gap=float(energies[n]-energies[n-1]), multiplets=multiplets))
        ed = store.get('lpar/designB1c_I2_h8.3000_lambda5.json')
        dmrg2 = store.get('lpar/designB1c_I2_h8.3000_chi200_lambda5.json')
        edpsi = ed['sectors']['psi']
        check2 = dict(energy_max_error=float(np.max(np.abs(np.array(edpsi['E'][:7])-dmrg2['tests']['psi']['E']))),
                      variance_relative_error=abs(dmrg2['tests']['psi']['total']/edpsi['total']-1),
                      dominant_share_errors=[abs(dmrg2['tests']['psi']['share'][r['n']]-r['share']) for r in edpsi['top'][:2]])
        dmrg3 = store.get('lpar/i3A_h4.5000_chi400_lambda5.json')
        E800 = {}
        for name, sector in [('vac','0_0.0_1'), ('psi','0_0.5_-1'), ('D','0_1.0_1'), ('first','1_0.5_-1')]:
            E800[name] = store.get(f'defdmrg/i3A_h4.5000_chi800_{sector}.json')['E'][0]
        scale = (E800['D']-E800['vac'])/4
        check3 = dict(primary_400=dmrg3['tests']['psi']['Dcal'][0],
                      first_400=dmrg3['tests']['psi']['Dcal'][1],
                      primary_800=(E800['psi']-E800['vac'])/scale,
                      first_800=(E800['first']-E800['vac'])/scale,
                      largest_resolved_share=max(dmrg3['tests']['psi']['share']),
                      captured_share=sum(dmrg3['tests']['psi']['share']),
                      amplitude_convergence_available=False,
                      displacement_descendants_computed=False)
        return dict(bands=bands, ED_DMRG_I2=check2, I3=check3)


    def improvement_test():
        inter = globals()['inter']()   # was: import inter
        lambda5 = globals()['lambda5']()   # was: import lambda5
        lpar = globals()['lpar']()   # was: import lpar
        mb = globals()['mb']()   # was: import mb
        d = pickle.load(open(ROOT/'terms_designB1c_I2.pkl', 'rb'))
        cfg = store.get('designB1c_I2.json')
        h = 8.3
        g, orbs = lpar.orbitals(d)
        Ne = sum(o['f'] == 0 for o in orbs)
        base, _ = lambda5.lambda_terms(g, orbs, 2, cfg, h, Ne, g.x5)
        # These are x5 moments O5. Adding a*O5 to Lambda is realized by
        # adding -(a/4)*Laplacian(O) to the energy density on the unit S4.
        contact, _ = inter.to_list(inter.density_terms(orbs, g.W*g.x5, 1., 0.))
        contact = [x for x in contact if x[0] != x[1] and x[2] != x[3]]
        B = lambda5.gram(g, orbs, g.x5)
        field = lambda5.fold_field(orbs, B, 1., Ne)
        field = [(a,b,c,e,float(np.real(v))) for a,b,c,e,v in field]
        records = {}
        matrices = {}
        for name, sec in [('vac',(0,0.,1)), ('psi',(0,.5,-1)), ('D',(0,1.,1))]:
            H, basis = mb.sector(orbs, d['def_terms'], h, Ne, *sec)
            e, v = np.linalg.eigh(H.toarray())
            operators = [mb.sector(orbs, ts, 0., Ne, *sec)[0].toarray() for ts in (base, field, contact)]
            for op in operators:
                assert np.max(np.abs(op-op.T.conj())) < 1e-9
            amplitudes = np.array([v.T.conj() @ op @ v[:,0] for op in operators])
            amplitudes[:,0] = 0
            matrices[name] = (e, amplitudes)
        e, a = matrices['vac']
        coeffs, _, rank, sing = np.linalg.lstsq(a[1:].T, -a[0], rcond=1e-12)
        E0 = matrices['vac'][0][0]
        gamma = (matrices['D'][0][0]-E0)/4
        for name,(e,a) in matrices.items():
            rec = {}
            for label, amp in [('original',a[0]),('vacuum_fitted',a[0]+coeffs@a[1:])]:
                w = abs(amp)**2
                total = float(w.sum())
                top = np.argsort(-w)[:4]
                rec[label] = dict(total=total, levels=[dict(n=int(k), Dcal=float((e[k]-E0)/gamma),
                                                          share=float(w[k]/total)) for k in top])
            records[name] = rec
        reference = store.get('lpar/designB1c_I2_h8.3000_lambda5.json')
        for name in ('psi','D'):
            assert abs(records[name]['original']['total']/reference['sectors'][name]['total']-1) < 1e-8
        return dict(moment_coefficients=dict(field=float(np.real(coeffs[0])), contact=float(np.real(coeffs[1]))),
                    fit_rank=int(rank), singular_values=sing.tolist(), sectors=records,
                    scope='I=2 defect vacuum fit; psi and D held out; no Hamiltonian or saved production data changed')


    def rotation_checks():
        lpar = globals()['lpar']()   # was: import lpar
        out = []
        for I, tag in ((2, 'designB1c_I2'), (3, 'i3A'), (4, 'i4A')):
            d = pickle.load(open(ROOT/f'terms_{tag}.pkl', 'rb'))
            g, orbs = lpar.orbitals(d)
            _, checks, _ = lpar.checks(g, orbs)
            out.append(dict(I=I, checks={k:float(v) for k,v in checks.items()}))
        return out

    def main():
        result = dict(harmonic_exact_checks=harmonic_checks(), saved_data=saved_checks())
        print('Exact harmonic weights and saved-data checks passed.', flush=True)
        result['density_improvement'] = improvement_test()
        result['rotation_closure'] = rotation_checks()
        store.put('checks_2026_11_07/checks.json', result)
        print(json.dumps({k:v for k,v in result.items() if k != 'harmonic_exact_checks'}, indent=2))

    def cli(argv):
        main()
    return namespace(locals(), globals())


STEPS = {'twisted_band': twisted_band, 'save_band': save_band, 'build_I': build_I, 'build_I3': build_I3, 'design': design, 'bulk_scan': bulk_scan, 'defect_dmrg': defect_dmrg, 'fig_curves': fig_curves, 'lpar': lpar, 'lpar_measure': lpar_measure, 'par_spin2': par_spin2, 'corr5d': corr5d, 'core5d': core5d, 'ward5d': ward5d, 'ward5d_dmrg': ward5d_dmrg, 's4_summary': s4_summary, 'lambda5': lambda5, 'audit_s4d': audit_s4d}
REPORT = [('corr5d', 'exact'), ('s4_summary',), ('audit_s4d',)]

if __name__ == '__main__':
    _file = os.path.abspath(__file__)
    os.makedirs(FILES, exist_ok=True)
    os.chdir(FILES)
    fuzzy.cli(_file, STEPS, REPORT)
