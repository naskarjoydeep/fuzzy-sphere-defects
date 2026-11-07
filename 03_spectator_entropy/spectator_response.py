"""How do the untwisted spectators respond to a partial twist, and is the twisted line stable?

At large N the gap equation forces the spectators' lowest line mode onto its singular branch (eq:sqrta), which lowers
its dimension; the neutral line operators are then the zeros of the varsigma kernel K_0 (eq:K0singlet), not the
spectator double trace. The steps compute the lowest neutral line operator (Table 8) and check it independently from
the paper's equations (paper, Sec. 4.2 and App. D, "The neutral line spectrum of a partial twist").

The lattice column of Table 2 (spectator line mode 0.41, lowest neutral operator 1.23, local potential subtracted) is the
O(3) half-twist of 01_known_results/free_control.py, step core_scaling (stored as twist/core_scaling_o3wf.json).

usage: python spectator_response.py <step>      (python spectator_response.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def singlet_channel():
    """The neutral (O(2)_tw x O(N-2)_sp singlet, transverse spin 0) line spectrum of a partial twist at large N (2026-10-25).

At large N the singlet sector is governed by the varsigma propagator D = Sigma^{-1}, Sigma = (1/2) sum_i G_i G_i.  In the S^1 x H^2
frame its circle mode kappa = 0 is a sum of AdS_2 bubbles (eq:bubbles of the paper),
    K_0(nu) = (1/2)(N-2) sum_{s in Z} B[D^sp_s, D^sp_{-s}](nu) + sum_{s in Z+alpha} B[D^tw_s, D^tw_s](nu),
    B[D1,D2](nu) = sum_j e_j / (nu^2 + (D1+D2+2j-1/2)^2),
with spectator dimensions D^sp_0 = 1/2 - sqrt(varsigma) (singular branch), D^sp_s = 1/2 + sqrt(s^2+varsigma), twisted
D^tw_s = 1/2 + sqrt(s^2+varsigma), and sqrt(varsigma) = 4 A_alpha/(pi^2 N).
Physical singlet line operators are poles of D, i.e. zeros of K_0 at nu^2 = -(Dh - 1/2)^2.  Poles of K_0 (double traces,
including the spectator 'composite' phi_sp^2 at 2 D^sp_0 = 1 - 8A/(pi^2 N)) are removed from the spectrum, exactly as phi^2 is
replaced by varsigma in the bulk.  Each term is increasing in x = (Dh-1/2)^2, so exactly one zero lies between consecutive poles.
Check: alpha -> 0 (no twist, varsigma = 0) gives zeros at Dh = 2, 4, ... (the bulk varsigma restricted to a trivial line).
usage: python singlet_channel.py
"""
    import numpy as np
    from scipy.special import gammaln
    from scipy.optimize import brentq
    pi = np.pi


    def A(alpha):
        return pi / 2 * (1 - 2 * alpha) * np.tan(pi * alpha) if alpha != 0.5 else 1.0


    def poch_half(x):
        return np.exp(gammaln(x + 0.5) - gammaln(x))


    def ej(D1, D2, jmax):
        j = np.arange(jmax + 1, dtype=float); D12 = D1 + D2
        lp = lambda a, n: gammaln(a + n) - gammaln(a)            # log Pochhammer (a)_n
        loge = (lp(0.5, j) + np.log(poch_half(D12 + 2 * j)) + lp(D12 + j, j) - np.log(2 * np.sqrt(pi)) - gammaln(j + 1)
                - np.log(poch_half(D1 + j)) - np.log(poch_half(D2 + j)) - lp(D12 + j - 0.5, j))
        return np.exp(loge), D12 + 2 * j - 0.5


    def poles_and_residues(N, alpha, smax=300, jmax=300):
        sv = 4 * A(alpha) / (pi ** 2 * N) if alpha > 0 else 0.0
        var = sv ** 2
        res, pos = [], []
        for s in range(-smax, smax + 1):                         # spectators, pairs (s, -s)
            D = 0.5 - sv if s == 0 else 0.5 + np.sqrt(s * s + var)
            e, m = ej(D, D, jmax); res.append(0.5 * (N - 2) * e); pos.append(m)
        if alpha > 0:                                            # twisted pair, pairs (s, s), s in Z + alpha
            for n in range(-smax, smax + 1):
                s = n + alpha
                D = 0.5 + np.sqrt(s * s + var)
                e, m = ej(D, D, jmax); res.append(e); pos.append(m)
        else:                                                    # untwisted: the two "twisted" reals are ordinary spectators
            for s in range(-smax, smax + 1):
                D = 0.5 + abs(s)
                e, m = ej(D, D, jmax); res.append(e); pos.append(m)
        r = np.concatenate(res); m = np.concatenate(pos)
        return r, m ** 2                                          # residues and pole positions in x = (Dh - 1/2)^2


    def K0(x, r, xp):
        return np.sum(r / (xp - x))


    def lowest_singlets(N, alpha, nzero=2, **kw):
        r, xp = poles_and_residues(N, alpha, **kw)
        order = np.argsort(xp); r, xp = r[order], xp[order]
        # merge (numerically) coincident poles
        ux, inv = np.unique(np.round(xp, 12), return_inverse=True)
        ur = np.bincount(inv, weights=r)
        zeros = []
        for k in range(len(ux) - 1):
            a, b = ux[k], ux[k + 1]
            if b - a < 1e-12:
                continue
            f = lambda x: K0(x, ur, ux)
            lo, hi = a + 1e-12 * (1 + a), b - 1e-12 * (1 + b)
            if f(lo) < 0 < f(hi):
                zeros.append(0.5 + np.sqrt(brentq(f, lo, hi, xtol=1e-14)))
            if len(zeros) >= nzero:
                break
        poles = 0.5 + np.sqrt(ux[:4])
        return zeros, poles

    def main():
        z, p = lowest_singlets(100, 0.0)
        print(f'check alpha = 0: zeros {np.round(z, 6)} (expect 2, 4), poles {np.round(p, 4)} (1, 3, ...)')
        for alpha in (0.5, 0.25, 0.1):
            for N in (3, 4, 6, 10, 30, 100, 1000):
                z, p = lowest_singlets(N, alpha)
                sv = 4 * A(alpha) / (pi ** 2 * N)
                print(f'alpha={alpha:4.2f} N={N:5d}: naive composite 2D_sp = {1 - 2 * sv:.4f} (removed pole)  lowest singlets {np.round(z, 4)}  '
                      f'first poles {np.round(p[:3], 4)}')

    def cli(argv):
        with fuzzy.logged('largeN/singlet_channel.log'):
            main()
    return namespace(locals(), globals())


@section
def singlet_check():
    """Independent check of CCS's App. D.8 (lowest neutral line operator of a partial twist = lowest zero of the kappa = 0
singlet kernel K_0).  Written from the paper's Eqs. (bubbles), (K0singlet), (sqrta), not from largeN/singlet_channel.py.
usage: python singlet_check.py
"""
    import numpy as np
    from scipy.special import gammaln
    from scipy.optimize import brentq
    def e_coeffs(D1, D2, J):
        j = np.arange(J + 1)[:, None]
        D12 = D1 + D2
        half = lambda x: gammaln(x + 0.5) - gammaln(x)                     # log of (x)_{1/2}
        poch = lambda x, n: gammaln(x + n) - gammaln(x)                    # log of (x)_n
        loge = (poch(0.5, j) + half(D12 + 2 * j) + poch(D12 + j, j) - np.log(2 * np.sqrt(np.pi)) - gammaln(j + 1)
                - half(D1 + j) - half(D2 + j) - poch(D12 + j - 0.5, j))
        return np.exp(loge), D12 + 2 * j                                    # weights and pole dimensions


    def kernel(N, alpha, S=600, J=400, sigma=None):
        A = 1.0 if abs(alpha - 0.5) < 1e-12 else (0.5 * np.pi * (1 - 2 * alpha) * np.tan(np.pi * alpha) if alpha > 0 else 0.0)  # A_{1/2} = 1
        rs = 4 * A / (np.pi ** 2 * N) if sigma is None else sigma
        s_int = np.arange(-S, S + 1)
        sp = np.where(s_int == 0, 0.5 - rs, 0.5 + np.sqrt(s_int ** 2 + rs ** 2))       # spectators, singular s = 0 branch
        if alpha == 0:
            sp = 0.5 + np.sqrt(s_int ** 2 + rs ** 2)
        s_tw = np.arange(-S, S + 1) + alpha
        tw = 0.5 + np.sqrt(s_tw ** 2 + rs ** 2)
        W, P = [], []
        w, p = e_coeffs(sp[None, :], sp[::-1][None, :], J); W.append(0.5 * (N - 2) * w.ravel()); P.append(p.ravel())
        w, p = e_coeffs(tw[None, :], tw[None, :], J); W.append(w.ravel()); P.append(p.ravel())
        W, P = np.concatenate(W), np.concatenate(P)
        K = lambda D: np.sum(W / ((P - 0.5) ** 2 - (D - 0.5) ** 2))
        return K, np.unique(np.round(P[W > 1e-12], 10)), rs


    def lowest_zero(N, alpha, **kw):
        K, poles, rs = kernel(N, alpha, **kw)
        poles = np.sort(poles[poles > 0.5])
        for a, b in zip(poles[:-1], poles[1:]):
            if b - a > 1e-9:
                lo, hi = a + 1e-9 * (b - a) + 1e-12, b - 1e-9 * (b - a) - 1e-12
                if np.sign(K(lo)) != np.sign(K(hi)):
                    return brentq(K, lo, hi, xtol=1e-12), (a, b), rs
        return None

    def main():
        K, poles, _ = kernel(3, 0.0, sigma=0.0)
        ps = np.sort(poles); zs = []
        for a, b in zip(ps[:-1], ps[1:]):
            lo, hi = a + 1e-9, b - 1e-9
            if np.sign(K(lo)) != np.sign(K(hi)):
                zs.append(brentq(K, lo, hi))
            if len(zs) == 2:
                break
        print('no twist: zeros', [round(z, 4) for z in zs], '(target 2, 4)')
        for N in (3, 4, 6, 10):
            for alpha in (0.5, 0.25, 0.1):
                z, (a, b), rs = lowest_zero(N, alpha)
                print(f'N={N:2d} alpha={alpha:4.2f}: lowest neutral zero {z:.4f} between poles {a:.4f} and {b:.4f}  (sqrt sigma* {rs:.4f})')

    def cli(argv):
        with fuzzy.logged('twist/singlet_check.log'):
            main()
    return namespace(locals(), globals())


@section
def spectator_shift_ratio():
    """Spectator shifts of the O(3) defects with one twisted real component and with a twisted plane (Table 2, last row).

shift = Delta_phi - Dh_sp, both in the bulk unit of the tuned model (the tuning sets Delta_phi = 0.518942 exactly), with the
local potential subtracted. Plane: the (1,2) pair twisted, spectator phi_3, the s = 0 odd level of core_scaling_o3wf.json
(the lattice column of Table 2 extrapolates it to 0.41). Component: the reflection diag(1, 1, -1), only phi_3 twisted,
spectators phi_1, phi_2, the lowest charge-one s = 0 level (sector 1|0|+1) of o3_twists/reflection_N*.json at lam = 0.
Both are extrapolated linearly in 1/N_m over N_m = 6-9 (and 7-9 as a check). Large N predicts the ratio 1/2.
usage: python spectator_shift_ratio.py      -> spectator_shift_ratio.json
"""
    import os
    import json
    import numpy as np

    HERE = 'twist'
    DPHI = 0.518942
    NS = (6, 7, 8, 9)


    def main():
        cs = store.get(os.path.join(HERE, 'core_scaling_o3wf.json'))
        plane = np.array([cs[str(n)]['odd_s0#0']['level'] for n in NS])
        comp = np.array([store.get(os.path.join(HERE, '..', 'fun', 'combined_2026_10_22', 'analysis', 'o3_twists',
                                                   f'reflection_N{n}.json'))['lam']['0']['1|0|+1'][0] for n in NS])
        N = np.array(NS, float)
        out = dict(N=list(NS), plane=plane.tolist(), component=comp.tolist(),
                   ratio_finite=[(DPHI - c) / (DPHI - p) for c, p in zip(comp, plane)], fits={})
        for lab, sl in (('6-9', slice(None)), ('7-9', slice(1, None))):
            pe = np.polyfit(1 / N[sl], plane[sl], 1)[1]
            ce = np.polyfit(1 / N[sl], comp[sl], 1)[1]
            out['fits'][lab] = dict(plane=pe, component=ce, shift_plane=DPHI - pe, shift_component=DPHI - ce,
                                    ratio=(DPHI - ce) / (DPHI - pe))
            print(f'N_m = {lab}: plane {pe:.4f}, component {ce:.4f}; shifts {DPHI - ce:.3f} / {DPHI - pe:.3f}, '
                  f'ratio {(DPHI - ce) / (DPHI - pe):.3f}')
        print('finite-size ratios', ' '.join(f'{x:.3f}' for x in out['ratio_finite']))
        store.put(os.path.join(HERE, 'spectator_shift_ratio.json'), out)

    def cli(argv):
        with fuzzy.logged('twist/spectator_shift_ratio.log'):
            main()
    return namespace(locals(), globals())


STEPS = {'singlet_channel': singlet_channel, 'singlet_check': singlet_check, 'spectator_shift_ratio': spectator_shift_ratio}
REPORT = [('singlet_channel',), ('singlet_check',), ('spectator_shift_ratio',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
