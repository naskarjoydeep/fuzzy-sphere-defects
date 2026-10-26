"""Can cap entanglement on the fuzzy sphere measure the defect entropy?

The entanglement entropy that a monodromy line adds to a polar cap combines log g with the conformal weight,
S_D = log g - 2 pi h (eq:Sdefect). The steps cut each lowest-Landau-level orbital at the cap boundary, build the full
Schmidt matrix of the many-body ground state, and fit the twist dependence of the cap entropy of the free complex scalar
and of the O(2) model to the free shapes log g_free(alpha) and h_free(alpha) (paper, App. D, "A lattice entanglement
test"). The shape is right but the normalization is not (B_cap = 1.59, 1.55, 1.59, 1.65 at N_m = 6-9).

usage: python cap_entanglement.py <step> [args]      (python cap_entanglement.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def cap_ee():
    """Real-space entanglement entropy of a polar cap on the fuzzy sphere, and the defect contribution S_defect.

Physics. For a line defect in a 3d CFT, the entanglement entropy of a disk centred on the point where the line crosses
the time slice exceeds the defect-free value by a universal constant,
    S_defect = log g - 4 pi^2 h      (modular Hamiltonian K = 2 pi int (R^2 - r^2)/(2R) T_00, <T_00>_D = h/r^3),
with g the defect g-function and h its stress-tensor coefficient. For the regular half-twist of a free complex scalar
(on S^1 x H^2): log g = 0.0667172, h = 1/(48 pi^2), so S_defect = 0.0667172 - 1/12 = -0.016616.
Lattice. On the fuzzy sphere the cap theta < theta_c around the north pole (where the line pierces the sphere) is
L_z-symmetric, so each orbital splits into normalized inside and outside modes, c+_k = sqrt(p_k) a+_k + sqrt(1 - p_k) b+_k,
p_k = I_{sin^2(theta_c/2)}(s - m + 1, s + m + 1) (regularized incomplete beta, also for twisted half-integer m).
Expanding each ED basis state over which occupied orbitals sit inside gives the Schmidt matrix exactly. Reordering
a+ before b+ costs (-1)^(number of (b, a) pairs in the original ascending order), which depends only on the pattern of
occupied positions. The matrix is block-diagonal in the inside particle number, L_z and pair charge.
Model: o2gauss (free complex scalar), ground state of (Q = 0, K = 0), untwisted (alpha = 0) and half-twisted (alpha = 1/2)
with the core potential stripped (lam = 0) or kept (lam = 1).
Gates: (G1) a single filled Landau level gives sum_k H(p_k); (G2) Schmidt weights sum to 1; (G3) the half-twist is
north-south symmetric, so S(theta_c) = S(pi - theta_c).
usage: python cap_ee.py gate
       python cap_ee.py run N1,N2,... [lam=0]   -> cap_ee_o2gauss.json
"""
    import os
    import sys
    import json
    import time
    import itertools
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from scipy.special import betainc
    M = fuzzy.onmodel()   # was: import onmodel as M
    CS = fuzzy.core_strip()   # was: import core_strip as CS
    HERE = 'twist'
    THETAS = np.array([0.4, 0.45, 0.5, 0.55, 0.6]) * np.pi
    S_DEFECT_FREE = 0.0667171954 - 1.0 / 12.0


    def ground_state(cfg, N, alpha, lam):
        fl, orbs, koff, two, mu, _, _ = CS.build(cfg, N, alpha, lam)
        b = M.Basis(fl, orbs, koff, N, [0], 0, ())
        H = M.build_H(b, two, mu)
        if b.dim <= 2500:
            e, v = np.linalg.eigh(H.toarray()); e0, v0 = e[0], v[:, 0]
        else:
            e, v = sla.eigsh(H, k=1, which='SA', tol=1e-12); e0, v0 = e[0], v[:, 0]
        modes = []                                                     # (m, flavour pair charge) per mode, mode order
        for f, ob in enumerate(orbs):
            for m in ob:
                modes.append((m, int(fl.qpair[f, 0]) if fl.qpair.shape[1] else 0))
        return b.states, v0 / np.linalg.norm(v0), np.array([m for m, _ in modes]), np.array([q for _, q in modes]), e0


    def weights(ms, N, theta_c):
        s = (N - 1) / 2
        u = np.sin(theta_c / 2) ** 2
        return np.array([betainc(s - m + 1, s + m + 1, u) for m in ms])


    def entropy(states, vec, ms, qs, p, return_weights=False):
        """S of the cap from the exact Schmidt decomposition (blocks by N_A, 2 L_z^A, Q_A)."""
        dim = len(states)
        nm = len(ms)
        n = bin(int(states[0])).count('1')
        occ = np.zeros((dim, n), np.int64)                             # occupied modes, ascending
        cnt = np.zeros(dim, np.int64)
        for k in range(nm):
            hit = ((states >> np.int64(k)) & 1) == 1
            occ[hit, cnt[hit]] = k
            cnt[hit] += 1
        lp, lq = 0.5 * np.log(np.clip(p, 1e-300, None)), 0.5 * np.log(np.clip(1 - p, 1e-300, None))
        m2 = np.round(2 * ms).astype(np.int64)
        S, wsum, blocks = 0.0, 0.0, 0
        for NA in range(n + 1):
            keysA, keysB, vals, gk = [], [], [], []
            for pat in itertools.combinations(range(n), NA):
                inA = np.zeros(n, bool); inA[list(pat)] = True
                inv = sum(1 for i in range(n) for j in range(i + 1, n) if (not inA[i]) and inA[j])
                sgn = -1.0 if inv % 2 else 1.0
                A = occ[:, inA]; B = occ[:, ~inA]
                maskA = np.bitwise_or.reduce(np.int64(1) << A, axis=1) if NA else np.zeros(dim, np.int64)
                maskB = states ^ maskA
                lw = (lp[A].sum(1) if NA else 0.0) + (lq[B].sum(1) if NA < n else 0.0)
                val = sgn * vec * np.exp(lw)
                g = (m2[A].sum(1) if NA else np.zeros(dim, np.int64)) * 1000 + (qs[A].sum(1) if NA else 0)
                keysA.append(maskA); keysB.append(maskB); vals.append(val); gk.append(g)
            keysA, keysB, vals, gk = (np.concatenate(x) for x in (keysA, keysB, vals, gk))
            keep = np.abs(vals) > 1e-14
            keysA, keysB, vals, gk = keysA[keep], keysB[keep], vals[keep], gk[keep]
            o = np.argsort(gk, kind='stable')
            keysA, keysB, vals, gk = keysA[o], keysB[o], vals[o], gk[o]
            cuts = np.flatnonzero(np.diff(gk)) + 1
            for a, bb, v in zip(np.split(keysA, cuts), np.split(keysB, cuts), np.split(vals, cuts)):
                ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(bb, return_inverse=True)
                Mx = sp.csr_matrix((v, (ia, ib)), shape=(len(ua), len(ub)))
                G = (Mx @ Mx.T) if len(ua) <= len(ub) else (Mx.T @ Mx)
                ev = np.linalg.eigvalsh(G.toarray()) if G.shape[0] > 1 else np.array([G.toarray()[0, 0]])
                ev = ev[ev > 1e-16]
                S -= np.sum(ev * np.log(ev)); wsum += ev.sum(); blocks += 1
        return (S, wsum, blocks) if return_weights else S


    def gate():
        # G1: one filled LLL of N orbitals (a single Fock state) against sum_k H(p_k)
        N = 7; s = (N - 1) / 2
        ms = np.array([-s + j for j in range(N)] + [-s + 0.5 + j for j in range(N - 1)])
        qs = np.zeros(len(ms), np.int64)
        st = np.array([(1 << N) - 1], np.int64)
        for th in (0.5 * np.pi, 0.3 * np.pi):
            p = weights(ms, N, th)
            S, w, _ = entropy(st, np.array([1.0]), ms, qs, p, True)
            ref = -np.sum(p[:N] * np.log(p[:N]) + (1 - p[:N]) * np.log(1 - p[:N]))
            print(f'G1 filled LLL, theta_c = {th / np.pi:.2f} pi: S = {S:.10f}, sum H(p) = {ref:.10f}, weights {w:.12f}')
        # G2/G3 on the half-twist ground state at N = 6
        st, v, ms, qs, _ = ground_state('o2gauss', 6, 0.5, 0.0)
        for th in (0.4 * np.pi, 0.6 * np.pi):
            S, w, nb = entropy(st, v, ms, qs, weights(ms, 6, th), True)
            print(f'G2/G3 half-twist N=6, theta_c = {th / np.pi:.2f} pi: S = {S:.10f}, weights {w:.12f}, blocks {nb}')


    def run(Ns, lam=0.0):
        path = os.path.join(HERE, 'cap_ee_o2gauss.json')
        R = store.get(path) if store.exists(path) else {}
        for N in Ns:
            t0 = time.time()
            rec = {}
            for lab, alpha, lm in (('untwisted', 0.0, 1.0), (f'half_lam{lam:g}', 0.5, lam)):
                st, v, ms, qs, e0 = ground_state('o2gauss', N, alpha, lm)
                rec[lab] = {f'{th / np.pi:.2f}': float(entropy(st, v, ms, qs, weights(ms, N, th))) for th in THETAS}
                rec[lab + '_dim'] = int(len(st))
            rec[f'S_defect_lam{lam:g}'] = {k: rec[f'half_lam{lam:g}'][k] - rec['untwisted'][k] for k in rec['untwisted']}
            R[f'{N}_lam{lam:g}'] = rec
            store.put(path, R)
            print(f"N={N} lam={lam:g}: S_defect(theta_c/pi) = " + '  '.join(f'{k}: {x:+.5f}' for k, x in rec[f'S_defect_lam{lam:g}'].items())
                  + f"   (free CFT {S_DEFECT_FREE:+.5f})  |  S_bulk(0.5) {rec['untwisted']['0.50']:.4f}  [{time.time() - t0:.0f}s]", flush=True)

    def main():
        if sys.argv[1] == 'gate':
            gate()
        else:
            run([int(x) for x in sys.argv[2].split(',')], float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)

    def cli(argv):
        sys.argv = list(argv)
        sys.argv = list(argv)
        with fuzzy.logged('twist/cap_ee_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def cap_alpha_scan():
    """B_cap of the paper (App. D, "A lattice entanglement test"): the alpha scan of the cap entropy of the free complex scalar.

On 2026-10-24 the cap entropy minus its alpha -> 0+ value was computed at alpha = 0.1-0.5 for N = 6-9 with the functions
of cap_ee.py, fitted to A + B S_D^free(alpha), and stored as twist/cap_ee_alpha_scan.json. That run was done inline and
its code was not kept. This step refits the stored entropies against the stored free shape, which reproduces the stored
A and B exactly, and prints them; the paper quotes B at theta_c = pi/2.
"""
    import numpy as np


    def main():
        R = store.get('twist/cap_ee_alpha_scan.json')
        S = np.array([R['S_cft'][a] for a in sorted(R['S_cft'], key=float)])
        for N in sorted((k for k in R if k != 'S_cft'), key=int):
            for t, rec in R[N].items():
                (A, B), res = np.linalg.lstsq(np.vstack([np.ones(len(S)), S]).T, np.array(rec['y']), rcond=None)[:2]
                assert abs(A - rec['A']) < 1e-12 and abs(B - rec['B']) < 1e-12
                print(f'N={N} theta_c={t}pi: A = {A:+.4f}  B_cap = {B:.3f}  rms {np.sqrt(res[0] / len(S)):.1e}')

    def cli(argv):
        with fuzzy.logged('twist/cap_alpha_scan.log'):
            main()
    return namespace(locals(), globals())


@section
def cap_two_shapes():
    """Cap entanglement vs twist angle: separate log g from h (2026-10-25).

For a line defect the cap entropy exceeds the defect-free value by S_D = log g - 2 pi h (paper normalization, h = conformal
weight).  log g_free(alpha) and h_free(alpha) have different alpha dependence, so with enough angles a fit
    S_cap(alpha) - S_cap(0+) = A + B log g_free(alpha) + C h_free(alpha)
determines B and C separately; the continuum structure requires C/B = -2 pi (and B = 1 for the free target).
Reference alpha -> 0+ (alpha = 0.001), since the regular band removes one orbital per twisted flavour for any alpha != 0.
Model: o2gauss (free complex scalar target), core potential stripped (lam = 0); optional o2wf (interacting O(2) half twist).
usage: python cap_two_shapes.py <cfg> N1,N2,... [theta_c/pi = 0.4,0.5]   -> cap_two_shapes_<cfg>.json
"""
    import os
    import sys
    import json
    import numpy as np
    from scipy.integrate import quad
    _cap_ee = cap_ee(); ground_state, entropy, weights = _cap_ee.ground_state, _cap_ee.entropy, _cap_ee.weights
    HERE = 'twist'
    ALPHAS = [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


    def logg_free(a):
        f = lambda nu: nu * np.tanh(np.pi * nu) * np.log((1 - 2 * np.exp(-2 * np.pi * nu) * np.cos(2 * np.pi * a) + np.exp(-4 * np.pi * nu))
                                                           / (1 - np.exp(-2 * np.pi * nu)) ** 2)
        return quad(f, 0, np.inf, limit=200)[0]


    def h_free(a):
        return a * (1 - a) * (1 - 2 * a) * np.tan(np.pi * a) / 12 if a != 0.5 else 1 / (24 * np.pi)


    def run(cfg, Ns, thetas=(0.4, 0.5)):
        fn = os.path.join(HERE, f'cap_two_shapes_{cfg}.json')
        R = store.get(fn) if store.exists(fn) else {}
        for N in Ns:
            rec = R.setdefault(str(N), {})
            for a in ALPHAS:
                key = f'{a:g}'
                if key in rec:
                    continue
                st, v, ms, qs, e0 = ground_state(cfg, N, a, 0.0)
                rec[key] = {f'{t:.2f}': float(entropy(st, v, ms, qs, weights(ms, N, t * np.pi))) for t in thetas}
                store.put(fn, R)
            print(f'{cfg} N={N} done', flush=True)
        analyse(R, thetas)


    def analyse(R, thetas=(0.4, 0.5)):
        al = np.array(ALPHAS[1:])
        LG = np.array([logg_free(a) for a in al]); HF = np.array([h_free(a) for a in al])
        for N in sorted(R, key=int):
            for t in thetas:
                k = f'{t:.2f}'
                if not all(f'{a:g}' in R[N] for a in ALPHAS):
                    continue
                y = np.array([R[N][f'{a:g}'][k] for a in al]) - R[N]['0.001'][k]
                M3 = np.vstack([np.ones_like(al), LG, HF]).T
                c3, res3 = np.linalg.lstsq(M3, y, rcond=None)[:2]
                M2 = np.vstack([np.ones_like(al), LG - 2 * np.pi * HF]).T
                c2, res2 = np.linalg.lstsq(M2, y, rcond=None)[:2]
                r3 = np.sqrt(res3[0] / len(al)) if len(res3) else 0; r2 = np.sqrt(res2[0] / len(al)) if len(res2) else 0
                print(f'N={N} theta_c={t}pi: two shapes B={c3[1]:.3f} C={c3[2]:.3f} C/B={c3[2] / c3[1]:+.3f} (-2pi = -6.283)  rms {r3:.1e} | '
                      f'one shape S_D: B={c2[1]:.3f} rms {r2:.1e}')

    def main():
        run(sys.argv[1], [int(x) for x in sys.argv[2].split(',')],
            tuple(float(x) for x in sys.argv[3].split(',')) if len(sys.argv) > 3 else (0.4, 0.5))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/cap_two_shapes_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


STEPS = {'cap_ee': cap_ee, 'cap_alpha_scan': cap_alpha_scan, 'cap_two_shapes': cap_two_shapes}
REPORT = [('cap_alpha_scan',), ('cap_two_shapes', 'o2gauss', '6,7,8,9'), ('cap_two_shapes', 'o2wf', '6,7,8')]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
