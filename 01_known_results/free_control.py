"""Does the method reproduce the exact free monodromy defect?

The Gaussian vector model (u = 0.148) targets the free complex scalar, whose half-twist is known exactly. Results:
the exact free spectrum, the extrapolation of the lattice levels, the squared bulk-to-defect coefficients and the
equal-time correlator against the exact Green function, and the response of each mode to the regulator's local
potential, which decays with the transverse spin of the mode.

usage: python free_control.py <step> [args]      (python free_control.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def free_defect():
    """Exact defect spectrum of a free complex scalar with monodromy e^{2 pi i alpha} (d=3), and comparison
with the fuzzy-sphere O(2) Gaussian model (run_on_defect.py o2gauss).

Single-particle defect states: charge q = +-1, transverse spin s in q*alpha + Z, parallel level n >= 0,
energy 1/2 + |s| + n.  The defect Hilbert space is their bosonic Fock space, so every level in sector
(Q, L_z) is a sum of single-particle energies with total charge Q and total spin L_z.
"""
    HERE = 'twist'
    import sys
    import json
    import os
    import numpy as np


    def exact_levels(alpha, Q, K, emax=4.0):
        """Sorted energies <= emax in sector (Q, L_z = K + alpha*Q)."""
        sp = []
        for q in (+1, -1):
            for k in range(-6, 7):
                s = q * alpha + k
                for n in range(0, 6):
                    e = 0.5 + abs(s) + n
                    if e <= emax + 1e-9:
                        sp.append((e, q, s))
        sp.sort()
        out = []

        def rec(i, e, qq, ss):                      # multisets: non-decreasing index i (bosons)
            if abs(qq - Q) == 0 and abs(ss - (K + alpha * Q)) < 1e-9:
                out.append(e)
            for j in range(i, len(sp)):
                e2 = e + sp[j][0]
                if e2 > emax + 1e-9:
                    break
                rec(j, e2, qq + sp[j][1], ss + sp[j][2])
        rec(0, 0.0, 0, 0.0)
        return sorted(out)


    def fuzzy_levels(res, alpha, Q, K):
        lev = res['alpha'][f"{alpha:.4f}"]
        E0 = lev["(0,)|0|()"][0]
        ED = lev["(0,)|1|()"][0]
        key = f"({Q},)|{K}|()"
        if key not in lev:
            return None
        return [2 * (e - E0) / (ED - E0) for e in lev[key]]


    def cli(argv):
        Ns = [int(x) for x in argv[1].split(',')]
        R = {N: store.get(os.path.join(HERE, 'on_results', f'o2gauss_N{N}.json')) for N in Ns}
        for al in (0.1, 0.2, 0.3, 0.4, 0.5):
            print(f"alpha = {al}")
            for (Q, K, lab) in [(0, 0, 'neutral s=0'), (0, 1, 'neutral s=1 (D)'), (1, 0, 'Psi_a'), (1, -1, 'Psi_a-1'),
                                (1, 1, 'Psi_a+1'), (2, 0, 'Q=2 s=2a'), (2, -1, 'Q=2 s=2a-1')]:
                ex = exact_levels(al, Q, K)
                row = f"  ({Q},{K:+d}) {lab:16s} exact " + ' '.join(f"{x:.3f}" for x in ex[:4])
                for N in Ns:
                    fz = fuzzy_levels(R[N], al, Q, K)
                    if fz:
                        row += f" | N={N}: " + ' '.join(f"{x:.3f}" for x in fz[:4])
                print(row)
    return namespace(locals(), globals())


@section
def free_extrap():
    """Extrapolate the fuzzy-sphere free-scalar defect spectrum (N = 6, 8, 10) and compare with exact values.
D calibration: corrections ~ 1/N (fit a + b/N + c/N^2 through N=6,8,10).  Bulk calibration (single-particle
spacing d.phi - phi = 1): corrections ~ N^{-1/2}, the R^{-1} of the leading irrelevant Goldstone coupling
(fit a + b N^{-1/2} + c/N)."""
    import json
    import os
    import numpy as np


    def cli(argv):
        exact_levels = free_defect().exact_levels
        HERE = 'twist'
        Ns = np.array([6, 8, 10])
        R = {N: store.get(os.path.join(HERE, 'on_results', f'o2gauss_N{N}.json')) for N in Ns}

        def cal(res, al, Q, K, mode):
            lev = res['alpha'][f'{al:.4f}']; E0 = lev['(0,)|0|()'][0]
            if mode == 'D':
                g = (lev['(0,)|1|()'][0] - E0) / 2
            else:
                b = res['alpha']['0.0000']; g = b['(1,)|1|()'][0] - b['(1,)|0|()'][0]
            return [(e - E0) / g for e in lev[f'({Q},)|{K}|()']]

        def ext(y, mode):
            x1 = 1.0 / Ns if mode == 'D' else Ns ** -0.5
            x2 = 1.0 / Ns ** 2 if mode == 'D' else 1.0 / Ns
            return np.linalg.solve(np.vstack([np.ones(3), x1, x2]).T, y)[0]

        if True:
            devD, devB = [], []
            print(f"{'alpha':>5} {'level':16s} {'exact':>6} | {'D-cal N=10':>10} {'extrap':>7} | {'bulk N=10':>9} {'extrap':>7}")
            for al in (0.1, 0.2, 0.3, 0.4, 0.5):
                for Q, K, lab, idx in [(1, 0, 'Psi_a', [0, 1]), (1, -1, 'Psi_a-1', [0]), (1, 1, 'Psi_a+1', [0]),
                                       (2, 0, 'Psi_a^2', [0]), (2, -1, 'Psi_a Psi_a-1', [0]), (0, 0, 'neutral s=0', [1, 2]),
                                       (0, 1, 'D (s=1)', [0, 1])]:
                    ex = exact_levels(al, Q, K)
                    for i in idx:
                        yD = np.array([cal(R[N], al, Q, K, 'D')[i] for N in Ns])
                        yB = np.array([cal(R[N], al, Q, K, 'b')[i] for N in Ns])
                        eD, eB = ext(yD, 'D'), ext(yB, 'b')
                        if not (Q == 0 and K == 1 and i == 0):
                            devD.append(eD - ex[i]); devB.append(eB - ex[i])
                        print(f"{al:5.1f} {lab + ('' if i == idx[0] else ' #%d' % (i + 1)):16s} {ex[i]:6.3f} | {yD[-1]:10.3f} {eD:7.3f} | {yB[-1]:9.3f} {eB:7.3f}")
            devD, devB = np.array(devD), np.array(devB)
            print(f"extrapolated - exact over {len(devD)} levels: D-cal mean {devD.mean():+.3f}, rms {np.sqrt((devD**2).mean()):.3f}, max {np.abs(devD).max():.3f};"
                  f"  bulk-cal mean {devB.mean():+.3f}, rms {np.sqrt((devB**2).mean()):.3f}, max {np.abs(devB).max():.3f}")
    return namespace(locals(), globals())


@section
def defect_corr():
    """Task 9: a normalized defect two-point function on the fuzzy sphere, reconstructed and compared with the exact free
half-twist correlator; bulk-to-defect amplitudes extracted independently.

Model: o2gauss (free complex scalar, onmodel 'gauss', U = 0.148, h_c from on_results), monodromy alpha = 1/2.
Lattice field: Phi(Omega) = psi_0^+ psi_-(Omega) + psi_+^+ psi_0(Omega) (= (V_1 + i V_2)/sqrt2, charge +1), with the
untwisted / twisted orbital wavefunctions N_m e^{i m phi} cos^{s+m}(theta/2) sin^{s-m}(theta/2) (twist._lognorm), so
Phi(Omega)|0> splits into sectors (Q = 1, K) with phase e^{-i(K + alpha) phi}.
Cylinder correlator G(Omega_1, Omega_2; tau) = <0|Phi^+(Omega_1) e^{-(H - E_0) tau/u} Phi(Omega_2)|0>, u = d phi - phi of
the untwisted bulk (radius 1).  The normalized correlator is R = G_D / G_0 (defect vacuum over the untwisted vacuum, same
lattice operator), which cancels c_phi^2 and the Weyl factors; flat points x_1 = e^tau Omega_1, x_2 = Omega_2.
Exact (d = 3, Delta_phi = 1/2):
    <phi-bar phi>_D / <phi-bar phi>_0 = (2/pi) arctan( cos(theta/2) / sqrt(xi + sin^2(theta/2)) ),
    xi = (x_par^2 + (r_1 - r_2)^2) / (4 r_1 r_2),  theta = phi_1 - phi_2;
and the defect-channel sum (r_1 r_2)^{-1/2} sum_{s in Z + alpha} b_s^2 e^{i s theta} g_{1/2+|s|}(chi), chi = 2 + 4 xi,
    b_s^2 = Gamma(1/2 + |s|) / (Gamma(1/2) Gamma(1 + |s|)),
    g_D(chi) = chi^-D 2F1(D/2, (D+1)/2; D + 1/2; 4/chi^2)        (line defect, p = 1).
Amplitudes: |<psi_s|Phi(Omega)|0_D>| = c_phi b_s (sin Theta)^{|s|} with c_phi = |<phi|Phi|0>| from the untwisted bulk, for
the lowest state psi_s of (Q = 1, K), s = K + alpha.  The Theta dependence of the extracted b_s is the lattice error.
Reconstruction: the defect-channel sum with lattice b_s^2 and lattice Delta_s for |s| <= 3/2, exact beyond, against R.
Gates: G1 the alpha = 0 channel sum equals 1/|x_12| and G2 the alpha = 1/2 sum equals the closed form (1e-8, at xi > 0);
G3 defect and bulk vacua and the (1, K) levels agree with on_results to 1e-8; G4 at alpha = 0, G_0 depends only on the
angle between Omega_1 and Omega_2 (1e-10).
usage: python defect_corr.py exact               -> G1, G2 only (seconds)
       python defect_corr.py run N               -> defect_corr_N{N}.json
       python defect_corr.py report              -> table over N
"""
    import os
    import sys
    import json
    import glob
    import time
    import numpy as np
    import scipy.sparse.linalg as sla
    from scipy.special import hyp2f1, gammaln

    from fuzzy import popc, apply_cdc, wave_abs
    HERE = 'twist'
    THETAS = np.array([np.pi / 6, np.pi / 4, np.pi / 3, 5 * np.pi / 12, np.pi / 2, 7 * np.pi / 12, 2 * np.pi / 3])
    DPHIS = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi * 0.999])
    TAUS = np.array([0.0, 0.25, 0.5, 1.0])


    # ------------------------------------------------------------------------------------------------ exact side
    def bs2(s):
        a = np.abs(s)
        return np.exp(gammaln(0.5 + a) - gammaln(0.5) - gammaln(1 + a))


    def ghat(D, chi):
        return chi ** (-D) * hyp2f1(D / 2, (D + 1) / 2, D + 0.5, 4 / chi ** 2)


    def geom(x1, x2):
        r1, r2 = np.hypot(x1[0], x1[1]), np.hypot(x2[0], x2[1])
        xi = ((x1[2] - x2[2]) ** 2 + (r1 - r2) ** 2) / (4 * r1 * r2)
        th = np.arctan2(x1[1], x1[0]) - np.arctan2(x2[1], x2[0])
        return r1, r2, xi, th


    def channel_sum(alpha, x1, x2, smax=400, b2=None, dims=None):
        """(r1 r2)^-1/2 sum_s b_s^2 e^{i s theta} g(chi); b2, dims: optional {s: value} overrides (lattice data)."""
        r1, r2, xi, th = geom(x1, x2)
        chi = 2 + 4 * xi
        tot = 0.0 + 0.0j
        for k in range(-smax, smax + 1):
            s = k + alpha
            b = b2.get(round(s, 6), bs2(s)) if b2 else bs2(s)
            D = dims.get(round(s, 6), 0.5 + abs(s)) if dims else 0.5 + abs(s)
            tot += b * np.exp(1j * s * th) * ghat(D, chi)
        return tot / np.sqrt(r1 * r2)


    def closed_ratio(x1, x2):
        _, _, xi, th = geom(x1, x2)
        return 2 / np.pi * np.arctan(np.cos(th / 2) / np.sqrt(xi + np.sin(th / 2) ** 2))


    def exact_gates(seed=1):
        rng = np.random.default_rng(seed)
        g1 = g2 = 0.0
        for _ in range(8):
            a, b = rng.uniform(0.3, 2.8, 2), rng.uniform(-2.5, 2.5, 2)
            x1 = 1.3 * np.array([np.sin(a[0]) * np.cos(b[0]), np.sin(a[0]) * np.sin(b[0]), np.cos(a[0])])
            x2 = np.array([np.sin(a[1]) * np.cos(b[1]), np.sin(a[1]) * np.sin(b[1]), np.cos(a[1])])
            triv = 1 / np.linalg.norm(x1 - x2)
            g1 = max(g1, abs(channel_sum(0.0, x1, x2) - triv) / triv)
            g2 = max(g2, abs(channel_sum(0.5, x1, x2) - closed_ratio(x1, x2) * triv) / triv)
        print(f'G1 (alpha = 0 sum vs 1/|x12|): {g1:.1e}   G2 (alpha = 1/2 sum vs closed form): {g2:.1e}   (both < 1e-8)')
        return g1, g2


    # ------------------------------------------------------------------------------------------------ lattice side
    class Side:
        """One vacuum (alpha) with Phi(Theta)|0> per K sector: U[K] = (dim_K x nTheta), H[K], plus lowest levels."""

        def __init__(self, N, alpha, Ks=None, lam=1.0):
            M = fuzzy.onmodel()
            CONFIGS = fuzzy.run_on_defect().CONFIGS
            cfg = CONFIGS['o2gauss']
            R = store.get(os.path.join(HERE, 'on_results', f'o2gauss_N{N}.json'))
            self.N, self.alpha, self.hc, self.stored = N, alpha, R['h_c'], R['alpha'].get(f'{alpha:.4f}')
            core_strip = fuzzy.core_strip()
            fl, orbs, koff, two, mu, _, _ = core_strip.build('o2gauss', N, alpha, lam, self.hc)   # lam = 0: core potential stripped
            self.M, self.fl, self.orbs, self.koff, self.two, self.mu = M, fl, orbs, koff, two, mu
            self.off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
            s = (N - 1) / 2
            f0, fp, fm = fl.names.index('0'), fl.names.index('12+'), fl.names.index('12-')
            b0 = M.Basis(fl, orbs, koff, N, [0], 0, ())
            H0 = M.build_H(b0, two, mu)
            E, V = sla.eigsh(H0, k=2, which='SA', tol=1e-12)
            o = np.argsort(E)
            self.E0, psi = float(E[o[0]]), V[:, o[0]]
            self.st0 = b0.states
            acc, phases = {}, {}
            for (fa, fb) in ((f0, fm), (fp, f0)):                         # psi_0^+ psi_-  and  psi_+^+ psi_0
                for ja, ma in enumerate(orbs[fa]):
                    for jb, mb in enumerate(orbs[fb]):
                        K = (ja + koff[fa]) - (jb + koff[fb])
                        if Ks is not None and K not in Ks:
                            continue
                        phases.setdefault(K, set()).add(round(mb - ma, 9))
                        acc.setdefault(K, []).append((int(self.off[fa] + ja), int(self.off[fb] + jb),
                                                      wave_abs(s, ma, THETAS) * wave_abs(s, mb, THETAS)))
            self.check_phase = {K: sorted(p) for K, p in phases.items()}  # each must be {-(K + alpha)}
            self.U, self.H, self.lev, self.vec0 = {}, {}, {}, {}
            for K, lst in sorted(acc.items()):
                bK = M.Basis(fl, orbs, koff, N, [1], K, ())
                if bK.dim == 0:
                    continue
                U = np.zeros((bK.dim, len(THETAS)), dtype=complex)
                for a, b, f in lst:
                    idx, val = apply_cdc(self.st0, psi, a, b, bK.states)
                    np.add.at(U, idx, val[:, None] * f[None, :])
                self.U[K], self.H[K] = U, M.build_H(bK, two, mu)
                k = min(3, bK.dim - 2) if bK.dim > 3 else bK.dim
                if bK.dim > 2500:
                    e, v = sla.eigsh(self.H[K], k=k, which='SA', tol=1e-12)
                else:
                    e, v = np.linalg.eigh(self.H[K].toarray()); e, v = e[:k], v[:, :k]
                oo = np.argsort(e)
                self.lev[K], self.vec0[K] = e[oo].tolist(), v[:, oo[0]]

        def corr(self, tau_lat):
            """G[K] = (nTheta x nTheta) matrix <u_K(Th1)| e^{-(H - E0) tau} |u_K(Th2)> for each K."""
            out = {}
            for K, U in self.U.items():
                W = U if tau_lat == 0 else sla.expm_multiply(-(self.H[K] - self.E0 * _eye(self.H[K])) * tau_lat, U)
                out[K] = U.conj().T @ W
            return out


    def _eye(H):
        import scipy.sparse as sp
        return sp.identity(H.shape[0], format='csr', dtype=H.dtype)


    def total(G, alpha, dphi):
        """sum_K e^{i(K+alpha) dphi} G[K]  -> (nTheta x nTheta)."""
        return sum(np.exp(1j * (K + alpha) * dphi) * g for K, g in G.items())


    def run(N, lam=1.0):
        t0 = time.time()
        g1, g2 = exact_gates()
        B = Side(N, 0.0)
        D = Side(N, 0.5, lam=lam)
        print(f'N={N}: sides built, K sectors {sorted(D.U)}  [{time.time() - t0:.0f}s]', flush=True)
        u = B.lev[1][0] - B.lev[0][0]                                      # d phi - phi (sectors (1,1), (1,0))
        gates = dict(G1=g1, G2=g2, dphi=(B.lev[0][0] - B.E0) / u,
                     phase_D={K: v for K, v in D.check_phase.items() if v != [round(-(K + 0.5), 9)]},
                     phase_B={K: v for K, v in B.check_phase.items() if v != [round(-float(K), 9)]})
        if D.stored and B.stored and lam == 1.0:
            gates['E0_D'] = D.E0 - D.stored['(0,)|0|()'][0]
            gates['E0_B'] = B.E0 - B.stored['(0,)|0|()'][0]
            gates['levels_D'] = {K: D.lev[K][0] - D.stored[f'(1,)|{K}|()'][0] for K in (-1, 0, 1) if f'(1,)|{K}|()' in D.stored}
        # G4: rotation invariance of the untwisted correlator (pairs with equal geodesic angle)
        G0s = B.corr(0.0)
        Gb = total(G0s, 0.0, np.pi / 3)                                    # (Theta_1, Theta_4) = (pi/4, pi/2) at dphi = pi/3
        cosg = np.cos(THETAS[1]) * np.cos(THETAS[4]) + np.sin(THETAS[1]) * np.sin(THETAS[4]) * np.cos(np.pi / 3)
        g4 = total(G0s, 0.0, np.arccos(np.clip(cosg, -1, 1)))[4, 4]          # same angle, both points on the equator
        gates['G4'] = float(abs(g4 - Gb[1, 4]) / abs(Gb[1, 4]))
        # amplitudes
        cphi = np.abs(B.vec0[0].conj() @ B.U[0])                           # |<phi|Phi(Theta)|0>|, flat in Theta
        amps = {}
        for K in sorted(D.U):
            sK = K + 0.5
            if abs(sK) > 2.6:
                continue
            a = np.abs(D.vec0[K].conj() @ D.U[K])
            b = a / (cphi.mean() * np.sin(THETAS) ** abs(sK))
            amps[sK] = dict(K=K, b2_theta=(b ** 2).tolist(), b2_exact=float(bs2(sK)),
                            Delta_lat=(D.lev[K][0] - D.E0) / u, Delta_exact=0.5 + abs(sK))
        gates['cphi_flatness'] = float(np.ptp(cphi) / cphi.mean())
        mid = list(THETAS).index(np.pi / 2)
        b2lat = {round(s, 6): v['b2_theta'][mid] for s, v in amps.items() if abs(s) <= 1.5}
        dlat = {round(s, 6): v['Delta_lat'] for s, v in amps.items() if abs(s) <= 1.5}
        rows = []
        for tau in TAUS:
            GDs, GBs = D.corr(tau / u), B.corr(tau / u)
            for dp in DPHIS:
                GD, GB = total(GDs, 0.5, dp), total(GBs, 0.0, dp)
                for i1, T1 in enumerate(THETAS):
                    for i2, T2 in enumerate(THETAS):
                        x1 = np.exp(tau) * np.array([np.sin(T1) * np.cos(dp), np.sin(T1) * np.sin(dp), np.cos(T1)])
                        x2 = np.array([np.sin(T2), 0.0, np.cos(T2)])
                        _, _, xi, _ = geom(x1, x2)
                        if xi < 1e-9 and dp == 0.0:
                            continue
                        Rl = complex(GD[i1, i2] / GB[i1, i2])
                        Re = float(closed_ratio(x1, x2))
                        Rr = None
                        if xi > 0.02:
                            triv = 1 / np.linalg.norm(x1 - x2)
                            Rr = float((channel_sum(0.5, x1, x2, smax=120, b2=b2lat, dims=dlat) / triv).real)
                        rows.append(dict(tau=float(tau), dphi=float(dp), T1=float(T1), T2=float(T2), xi=float(xi),
                                         R_lat=Rl.real, R_lat_im=Rl.imag, R_exact=Re, R_rec=Rr))
            print(f'  tau {tau}: done [{time.time() - t0:.0f}s]', flush=True)
        dev = lambda key: float(np.max([abs(r[key] - r['R_exact']) for r in rows if r[key] is not None]))
        out = dict(N=N, lam=lam, h_c=D.hc, u=u, gates=gates, amplitudes={f'{s:+.1f}': v for s, v in amps.items()}, rows=rows,
                   max_dev=dict(lat=dev('R_lat'), rec=dev('R_rec')))
        store.put(os.path.join(HERE, f'defect_corr_N{N}' + ('' if lam == 1.0 else f'_lam{lam:g}') + '.json'), out)
        print('gates:', {k: v for k, v in gates.items()})
        for s, v in amps.items():
            print(f"  s={s:+.1f}: Delta {v['Delta_lat']:.4f} (exact {v['Delta_exact']})  b_s^2 at Theta=pi/2 "
                  f"{v['b2_theta'][mid]:.4f} (exact {v['b2_exact']:.4f}); spread over Theta {np.ptp(v['b2_theta']):.4f}")
        print(f"max |R - exact|: lattice {out['max_dev']['lat']:.4f}, reconstructed {out['max_dev']['rec']:.4f}")


    def report(suffix=''):
        files = [f for f in store.glob(os.path.join(HERE, f'defect_corr_N*{suffix}.json')) if suffix or '_lam' not in f]
        for fn in sorted(files, key=lambda f: int(f.split('_N')[-1].split('_')[0].split('.')[0])):
            R = store.get(fn)
            eq = [r for r in R['rows'] if r['tau'] == 0 and abs(r['T1'] - np.pi / 2) < 1e-9 and abs(r['T2'] - np.pi / 2) < 1e-9]
            line = '  '.join(f"dphi {r['dphi']:.2f}: {r['R_lat']:.3f}/{r['R_exact']:.3f}" for r in eq)
            b = R['amplitudes'].get('+0.5', {})
            print(f"N={R['N']}: max dev lat {R['max_dev']['lat']:.4f} rec {R['max_dev']['rec']:.4f} | equator (lattice/exact, "
                  f"exact = 1 - dphi/pi) {line} | b_1/2^2 {b.get('b2_theta', [np.nan] * 5)[4]:.4f} (exact {b.get('b2_exact', np.nan):.4f})")


    def cli(argv):
        if argv[1] == 'exact':
            exact_gates()
        elif argv[1] == 'run':
            run(int(argv[2]), float(argv[3]) if len(argv) > 3 else 1.0)
        else:
            report(argv[2] if len(argv) > 2 else '')
    return namespace(locals(), globals())


@section
def core_scaling():
    """Experiment A: the microscopic core of a monodromy defect as a sum of irrelevant defect couplings.

Physics question. Does the fuzzy-sphere core flow to the regular monodromy defect, and which defect operators carry
the approach? A core-localized one-body potential is, in the continuum, a line perturbation
    delta S = sum_O g_O int dtau O_hat(tau),
summed over neutral, transverse-spin-zero defect operators. Each coupling is dimensionless only after a factor
R^{1 - Delta_O}, so at first order it shifts a level n by
    delta Delta_n = sum_O g_O C_{n O n} N^{-(Delta_O - 1)/2}     (R^2 = N).
Free complex scalar, alpha = 1/2. A one-body potential is quadratic in the field, so it couples to the bilinears
psi_s psi-bar_s of the defect modes, with Delta_O = 1 + 2|s|. A single-particle state psi_s feels it at first order only
through its own bilinear. The prediction is a ladder: the core shift of psi_s scales as N^{-|s|}, i.e. N^{-1/2} for
s = 1/2 and N^{-3/2} for s = 3/2. Two-particle states inherit the slower constituent.
Interacting models (o2wf, o3wf). The exponents measure the dimension of the leading neutral defect operator that each
level couples to. They are compared with the lowest neutral s = 0 levels of the stripped spectrum.
Method. dDelta_n/dlam at lam = 0 by a symmetric difference, from levels at lam = -d, 0, +d
(twist/core_strip.build, bulk units d phi - phi = 1, D-calibrated as a cross-check). The local exponent between
consecutive N gives Delta_eff = 1 - 2 dln|slope|/dlnN.
usage: python core_scaling.py run <config> N1,N2,... [d=0.25]   -> core_scaling_<config>.json
       python core_scaling.py report <config>
"""
    import os
    import sys
    import json
    import numpy as np
    M = fuzzy.onmodel()
    CS = fuzzy.core_strip()
    CONFIGS = fuzzy.run_on_defect().CONFIGS

    HERE = 'twist'


    def levels(cfgname, N, lam, only=None):
        cfg = CONFIGS[cfgname]
        npair, npar = len(cfg['pairs']), cfg['Nv'] - 2 * len(cfg['pairs'])
        z, p, q1, q2 = [0] * npair, (+1,) * npar, [1] + [0] * (npair - 1), [2] + [0] * (npair - 1)
        fl, orbs, koff, two, mu, _, _ = CS.build(cfgname, N, 0.5, lam, only=only)
        secs = {'vac': (z, 0, p, 3), 'D': (z, 1, p, 2), 'psi1/2': (q1, 0, p, 2), 'psi3/2': (q1, 1, p, 2), 'Q2': (q2, 0, p, 1)}
        if npar:
            secs['odd_s0'] = (z, 0, (-1,) + (+1,) * (npar - 1), 2)              # untwisted component's sector (O(3))
        out = {}
        for k, (q, K, pp, n) in secs.items():
            b = M.Basis(fl, orbs, koff, N, q, K, pp)
            out[k] = sorted(M.lowest(M.build_H(b, two, mu), n))
        return out


    def run(cfgname, Ns, d=0.25, only=None):
        R = {}
        tag = '' if only is None else '_only_' + '-'.join(only)
        path = os.path.join(HERE, f'core_scaling_{cfgname}{tag}.json')
        if store.exists(path):
            R = store.get(path)
        for N in Ns:
            bulk = store.get(os.path.join(HERE, 'on_results', f'{cfgname}_N{N}.json'))['alpha']['0.0000']
            cfg = CONFIGS[cfgname]
            npair, npar = len(cfg['pairs']), cfg['Nv'] - 2 * len(cfg['pairs'])
            q1, p = (1,) + (0,) * (npair - 1), (+1,) * npar
            u = bulk[f'{q1}|1|{p}'][0] - bulk[f'{q1}|0|{p}'][0]
            L = {lam: levels(cfgname, N, lam, only) for lam in (-d, 0.0, d)}
            rec = {}
            for k in L[0.0]:
                for i in range(len(L[0.0][k])):
                    if k == 'vac' and i == 0:
                        continue
                    g = {lam: (L[lam][k][i] - L[lam]['vac'][0]) / u for lam in L}
                    gD = {lam: 2 * (L[lam][k][i] - L[lam]['vac'][0]) / (L[lam]['D'][0] - L[lam]['vac'][0]) for lam in L}
                    rec[f'{k}#{i}'] = dict(level=g[0.0], slope=(g[d] - g[-d]) / (2 * d), level_D=gD[0.0],
                                           slope_D=(gD[d] - gD[-d]) / (2 * d), curv=(g[d] + g[-d] - 2 * g[0.0]) / d ** 2)
            R[str(N)] = rec
            print(f'{cfgname} N={N}: ' + '  '.join(f"{k} {v['level']:.3f} (d/dlam {v['slope']:+.4f})" for k, v in rec.items()), flush=True)
            store.put(path, R)


    def report(cfgname, tag=''):
        R = store.get(os.path.join(HERE, f'core_scaling_{cfgname}{tag}.json'))
        Ns = sorted(int(n) for n in R)
        print(f'{cfgname}: core-coupling slopes d(Delta)/d(lam) at lam = 0 (bulk units), and Delta_eff = 1 - 2 dln|slope|/dlnN')
        for k in R[str(Ns[0])]:
            s = np.array([R[str(N)][k]['slope'] for N in Ns])
            lev = np.array([R[str(N)][k]['level'] for N in Ns])
            ex = np.diff(np.log(np.abs(s))) / np.diff(np.log(Ns))
            print(f"  {k:10s} level(lam=0) {lev[-1]:.3f}  slopes " + ' '.join(f'{x:+.4f}' for x in s) +
                  '  | local exponents ' + ' '.join(f'{e:+.2f}' for e in ex) + f'  -> Delta_eff {1 - 2 * ex[-1]:.2f}')


    def cli(argv):
        if argv[1] == 'run':
            only = argv[5].split(',') if len(argv) > 5 else None
            run(argv[2], [int(x) for x in argv[3].split(',')], float(argv[4]) if len(argv) > 4 else 0.25, only)
        else:
            report(argv[2], argv[3] if len(argv) > 3 else '')
    return namespace(locals(), globals())


@section
def free_cd():
    """Experiment B: the displacement normalization C_D of the free complex-scalar monodromy defect, exactly.

Physics. Moving a conformal defect costs a universal amount, fixed by the two-point function of the displacement,
<D_i(tau) D_j(0)> = C_D delta_ij / tau^4. In the free theory with monodromy e^{2 pi i alpha} (regular core), the
displacement is the unique neutral, spin-one, dimension-two defect operator, D = D_1 + i D_2 = C psi_alpha psibar_{alpha-1},
built from the two lowest defect modes (unit-normalized; transverse spins alpha and alpha - 1). C follows from the
translation Ward identity: moving the defect by dz0 is moving every insertion by -dz0, so
    -(d_zbar_x + d_zbar_y) <phi(x) phibar(y)>_D = (1/2) int dtau <D(tau) phi(x) phibar(y)>
                                                = (C/2) <phi(x) psibar_{alpha-1}> * <phibar(y) psi_alpha>, integrated over tau.
The bulk-to-defect functions are b_s e^{+-i s theta} r^{|s|} / (r^2 + (x_par - tau)^2)^{1/2 + |s|}, with
b_s^2 = Gamma(1/2+|s|)/(Gamma(1/2) Gamma(1+|s|)). C must come out the same at every pair of points (the check).
C_D = |C|^2 / 2, since <D Dbar> = 2 C_D / tau^4 and <psi psibar> = 1/tau^{2 Delta}.
The left side uses the exact correlator: the closed form at alpha = 1/2, the defect-channel sum otherwise.
usage: python free_cd.py [alpha1,alpha2,...]   -> free_cd.json

Paper, App. C.2 (eq:cdfree): the exact C_D = 1/(2 pi^2) of the free half-twist from the translation identity.
"""
    import os
    import sys
    import json
    import numpy as np
    from scipy.integrate import quad
    _defect_corr = globals()['defect_corr'](); bs2, channel_sum, closed_ratio = _defect_corr.bs2, _defect_corr.channel_sum, _defect_corr.closed_ratio   # was: from defect_corr import bs2, channel_sum, closed_ratio

    HERE = 'twist'


    def G(alpha, x, y):
        if abs(alpha - 0.5) < 1e-12:
            return closed_ratio(x, y) / np.linalg.norm(x - y)
        return channel_sum(alpha, x, y, smax=300)


    def lhs(alpha, x, y, h=1e-5):
        """-(d_zbar_x + d_zbar_y) G, with d_zbar = (d_1 + i d_2)/2, by central differences."""
        tot = 0.0 + 0.0j
        for i, w in ((0, 1.0), (1, 1j)):
            e = np.zeros(3); e[i] = h
            d = (G(alpha, x + e, y + e) - G(alpha, x - e, y - e)) / (2 * h)
            tot += 0.5 * w * d
        return -tot


    def rhs_over_C(alpha, x, y):
        rx, ry = np.hypot(x[0], x[1]), np.hypot(y[0], y[1])
        tx, ty = np.arctan2(x[1], x[0]), np.arctan2(y[1], y[0])
        s1, s2 = alpha, alpha - 1.0
        f = lambda t: (ry ** abs(s1) / (ry ** 2 + (y[2] - t) ** 2) ** (0.5 + abs(s1)) *
                       rx ** abs(s2) / (rx ** 2 + (x[2] - t) ** 2) ** (0.5 + abs(s2)))
        I = quad(f, -np.inf, np.inf, epsabs=1e-13, epsrel=1e-11, limit=400)[0]
        phase = np.exp(1j * s2 * tx) * np.exp(-1j * s1 * ty)
        return 0.5 * np.sqrt(bs2(s1) * bs2(s2)) * phase * I


    def run(alphas, npts=12, seed=3):
        rng = np.random.default_rng(seed)
        out = {}
        for al in alphas:
            Cs = []
            for _ in range(npts):
                a, b = rng.uniform(0.4, 2.7, 2), rng.uniform(-2.6, 2.6, 2)
                x = 1.4 * np.array([np.sin(a[0]) * np.cos(b[0]), np.sin(a[0]) * np.sin(b[0]), np.cos(a[0])])
                y = np.array([np.sin(a[1]) * np.cos(b[1]), np.sin(a[1]) * np.sin(b[1]), np.cos(a[1])])
                Cs.append(lhs(al, x, y) / rhs_over_C(al, x, y))
            Cs = np.array(Cs)
            CD = np.abs(Cs) ** 2 / 2
            out[f'{al:g}'] = dict(C_abs=np.abs(Cs).tolist(), C_phase=np.angle(Cs).tolist(), C_D=float(np.median(CD)),
                                  spread=float(np.ptp(CD) / np.median(CD)))
            print(f'alpha = {al:g}: |C| = {np.median(np.abs(Cs)):.8f}, phase spread {np.ptp(np.angle(Cs)):.2e}, '
                  f'C_D = |C|^2/2 = {np.median(CD):.8f}, relative spread over {npts} point pairs {np.ptp(CD) / np.median(CD):.1e}')
        store.put(os.path.join(HERE, 'free_cd.json'), out)

    def main():
        run([float(a) for a in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0.5, 0.25, 0.1])

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/free_cd_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def disp_tilt():
    """Task 5a: the displacement from the tilt Ward identity instead of a calibration.

The flux line through the poles breaks the two rotations that tilt it.  With L_+ the rotation generator (exact on the
untwisted orbitals; on twisted orbitals sqrt((s-m)(s+m+1)) c+_{m+1} c_m truncated at the top orbital, where the
regular core drops the singular mode),
    <n|L_+|0_D> = <n|[H, L_+]|0_D> / (E_n - E_0),     [H, L_+] = i (D_+(north) - D_+(south)),
so in the CFT L_+|0_D> lies entirely in the displacement family.  With D_+ = D_1 + i D_2,
<D_i(x) D_j(0)> = C_D delta_ij / x^4 and |k> = P^k|D_+> normalized,
    |<k|L_+|0_D>|^2 = 4 (2 Delta_D)_k 2 C_D / (k! (Delta_D + k)^2)   for k odd,   0 for k even and for other families.
At Delta_D = 2: the primary D itself has zero weight, d_par D (Delta = 3) carries |a_1|^2 = 32 C_D / 9, and
p_3/p_1 = 1.8.  The weights grow with k (the norm diverges in the continuum), so only the low window is used.
Identification: d_par D is the K = 1 state near Delta = 3 that carries the weight; D is the K = 1 state one unit
below it.  C_D = 9 |a_1|^2 / 32 needs no calibration (L_+ is the exact rotation generator in the bulk).

Backends: 'ising' (twist.py, standard point V0, V1, h = 4.75, 1, 3.16; D sector twoLz = 2, Z2 even);
'o2gauss', 'o2wf', 'o3wf' (onmodel, alpha = 0.5, h_c from on_results; D sector neutral K = 1).
Levels are bulk-calibrated (Ising: T at 3; onmodel: d phi - phi = 1), not D-calibrated.
Gates: at alpha = 0 (untwisted vacuum, an L = 0 singlet) |L_+|0>| < 1e-10; the defect vacuum and K = 1 levels agree
with qian_point / on_results to 1e-8.  For o2gauss (free complex scalar) the exact K = 1 levels at Delta = 3 are
d D and two primaries; only one may carry weight.
usage: python disp_tilt.py run <backend> N [nlev=30]     -> disp_results/tilt_<backend>_N{N}.json
       python disp_tilt.py report <backend>              -> table, C_D(N) and its extrapolation

Paper, App. C.2: run o2gauss_lam0 N and run o2gauss N for N = 6-10. The ising backend also needs the research module qian_point for one gate, which is not part of this package.
"""
    import os
    import sys
    import json
    import glob
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    HERE = 'twist'
    OUT = os.path.join(HERE, 'disp_results')
    ISING = (4.75, 1.0, 3.16)


    def lowest(H, k, vecs=False):
        if H.shape[0] <= 2500:
            w, v = np.linalg.eigh(H.toarray())
            return (w[:k], v[:, :k]) if vecs else w[:k]
        w, v = sla.eigsh(H, k=min(k, H.shape[0] - 2), which='SA', tol=1e-12)
        o = np.argsort(w)
        return (w[o], v[:, o]) if vecs else w[o]


    def raise_op(states_in, states_out, ms, offset, s, truncate_top=True):
        """Sparse matrix of sum_j sqrt((s-m_j)(s+m_j+1)) c+_{j+1} c_j on one flavour (modes offset .. offset+len(ms)-1).
        Adjacent modes, so no fermion sign.  The hop out of the top orbital has no target and is dropped."""
        rows, cols, vals = [], [], []
        for j in range(len(ms) - 1):
            a, b = offset + j + 1, offset + j
            c = np.sqrt(max((s - ms[j]) * (s + ms[j] + 1), 0.0))
            ok = (((states_in >> np.int64(b)) & 1) == 1) & (((states_in >> np.int64(a)) & 1) == 0)
            new = states_in ^ (np.int64(1) << np.int64(a)) ^ (np.int64(1) << np.int64(b))
            pos = np.searchsorted(states_out, new)
            pc = np.clip(pos, 0, max(len(states_out) - 1, 0))
            hit = ok & (pos < len(states_out)) & (states_out[pc] == new)
            idx = np.nonzero(hit)[0]
            rows.append(pc[idx]); cols.append(idx); vals.append(np.full(len(idx), c))
        r = np.concatenate(rows) if rows else np.zeros(0, int)
        return sp.csr_matrix((np.concatenate(vals) if vals else np.zeros(0), (r, np.concatenate(cols) if cols else r)),
                             shape=(len(states_out), len(states_in)))


    # ------------------------------------------------------------------------------------------------ backends
    def ising(N, alpha_twisted=True):
        _twist = fuzzy; couplings_for, TwistBasis, interaction_terms, build_H_fast = _twist.couplings_for, _twist.TwistBasis, _twist.interaction_terms, _twist.build_H_fast   # was: from twist import couplings_for, TwistBasis, interaction_terms, build_H_fast
        V0, V1, h = ISING
        G0, G1, _ = couplings_for(N, V0, V1)
        s = (N - 1) / 2
        tb = interaction_terms(N, False, G0, G1)
        bt = {k: TwistBasis(N, False, k, +1) for k in (0, 4)}
        E0b = lowest(build_H_fast(bt[0], tb, h), 1)[0]
        gam = (lowest(build_H_fast(bt[4], tb, h), 1)[0] - E0b) / 3
        td = interaction_terms(N, alpha_twisted, G0, G1)
        b0, b1 = TwistBasis(N, alpha_twisted, 0, +1), TwistBasis(N, alpha_twisted, 2, +1)
        H0, H1 = build_H_fast(b0, td, h), build_H_fast(b1, td, h)
        Lp = (raise_op(b0.states, b1.states, b0.mp, 0, s) + raise_op(b0.states, b1.states, b0.mm, b0.Np, s))
        return dict(H0=H0, H1=H1, Lp=Lp, unit=gam, unit_name='T at 3', dims=(b0.dim, b1.dim))


    def onmodel(cfgname, N, alpha=0.5, lam=1.0):
        M = fuzzy.onmodel()   # was: import onmodel as M
        _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
        cfg = CONFIGS[cfgname]
        R = store.get(os.path.join(HERE, 'on_results', f'{cfgname}_N{N}.json'))
        hc = R['h_c']
        kw = dict(pairs=cfg['pairs'], twist=cfg['twist'])
        npair, npar = len(cfg['pairs']), cfg['Nv'] - 2 * len(cfg['pairs'])
        z, p, q1 = [0] * npair, (+1,) * npar, [1] + [0] * (npair - 1)
        s = (N - 1) / 2
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], hc, **kw)
        Eb = {k: M.lowest(M.build_H(M.Basis(fl, orbs, koff, N, q, K, p), two, mu), 1)[0]
              for k, (q, K) in dict(vac=(z, 0), phi=(q1, 0), dphi=(q1, 1)).items()}
        u = Eb['dphi'] - Eb['phi']
        core_strip = fuzzy.core_strip()   # was: import core_strip
        fl, orbs, koff, two, mu, _, _ = core_strip.build(cfgname, N, alpha, lam, hc)      # lam = 0: core potential stripped
        b0, b1 = M.Basis(fl, orbs, koff, N, z, 0, p), M.Basis(fl, orbs, koff, N, z, 1, p)
        H0, H1 = M.build_H(b0, two, mu), M.build_H(b1, two, mu)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        Lp = sum(raise_op(b0.states, b1.states, list(orbs[f]), int(off[f]), s) for f in range(fl.F))
        return dict(H0=H0, H1=H1, Lp=Lp, unit=u, unit_name='d phi - phi = 1', dims=(b0.dim, b1.dim), h_c=hc,
                    dphi_check=(Eb['phi'] - Eb['vac']) / u, dphi_target=cfg['dphi'], stored=R['alpha'].get(f'{alpha:.4f}'),
                    keys=(f"{tuple(z)}|0|{p}", f"{tuple(z)}|1|{p}"))


    def build(backend, N, untwisted=False):
        if backend == 'ising':
            return ising(N, alpha_twisted=not untwisted)
        name, lam = (backend.split('_lam')[0], float(backend.split('_lam')[1])) if '_lam' in backend else (backend, 1.0)
        return onmodel(name, N, alpha=0.0 if untwisted else 0.5, lam=lam)


    # ------------------------------------------------------------------------------------------------ modes
    def wd(D, k):
        """CFT weight (up to 2 C_D) of the level-k descendant: 4 (2D)_k / (k! (D+k)^2), k odd."""
        from math import gamma, factorial
        return 0.0 if k % 2 == 0 else 4 * gamma(2 * D + k) / gamma(2 * D) / factorial(k) / (D + k) ** 2


    def identify(lev, p):
        """d_par D = the largest weight within 0.3-2 above the lowest K = 1 level (the lattice spectrum is compressed, so a
        fixed window around 3 fails for the free scalar); D = the K = 1 level nearest one unit below it."""
        cand = [i for i in range(len(lev)) if lev[0] + 0.3 < lev[i] < lev[0] + 2.0]
        if not cand:
            return {}
        i3 = max(cand, key=lambda i: p[i])
        win = [i for i in range(len(lev)) if abs(lev[i] - lev[i3]) < 0.6]
        iD = int(np.argmin(np.abs(lev - (lev[i3] - 1.0))))
        return dict(dD=dict(index=int(i3), level=float(lev[i3]), p=float(p[i3]), share=float(p[i3] / p[win].sum())),
                    D=dict(index=iD, level=float(lev[iD]), p=float(p[iD]), p_over_dD=float(p[iD] / p[i3]),
                           lowest_K1=float(lev[0])),
                    C_D=float(9 * p[i3] / 32))


    def run(backend, N, nlev=30):
        t0 = time.time()
        pass   # was os.makedirs(...): the store needs no output directory
        gate = {}
        U = build(backend, N, untwisted=True)                           # gate: L_+ annihilates the untwisted vacuum
        E, V = lowest(U['H0'], 1, vecs=True)
        gate['untwisted_norm'] = float(np.linalg.norm(U['Lp'] @ V[:, 0]))
        B = build(backend, N)
        E0, V0 = lowest(B['H0'], 1, vecs=True)
        E1, V1 = lowest(B['H1'], nlev, vecs=True)
        v = B['Lp'] @ V0[:, 0]                                         # lands in the K = 1 sector
        a2 = np.abs(V1.conj().T @ v) ** 2
        Cv = B['H1'] @ v - E0[0] * v                                    # [H, L_+]|0_D>
        n2, c2 = float(np.vdot(v, v).real), float(np.vdot(Cv, Cv).real)
        out = dict(backend=backend, N=N, dims=B['dims'], unit=B['unit'], unit_name=B['unit_name'], E0=float(E0[0]),
                   norm2=n2, comm_norm2=c2, levels=((E1 - E0[0]) / B['unit']).tolist(),
                   p=a2.tolist(), q=((E1 - E0[0]) ** 2 * a2).tolist(), gate=gate)
        out['p_window'] = float(np.sum(a2) / n2)
        out['q_window'] = float(np.sum(out['q']) / c2)
        if backend == 'ising':
            from qian_point import defect, bulk
            b = bulk(N, *ISING); d = defect(N, *ISING, b)
            gate['D_bulkcal_qian_point'] = d['D_bulkcal']
            gate['D_bulkcal_here'] = out['levels'][0]
        else:
            st = B['stored']
            if st:
                gate['E0_vs_stored'] = float(E0[0] - st[B['keys'][0]][0])
                gate['K1_vs_stored'] = float(E1[0] - st[B['keys'][1]][0])
            gate['dphi_tuned'] = [B['dphi_check'], B['dphi_target']]
        lev, p = np.array(out['levels']), np.array(out['p'])
        out.update(identify(lev, p))
        store.put(os.path.join(OUT, f'tilt_{backend}_N{N}.json'), out)
        print(f'{backend} N={N}: dims {B["dims"]}, |L+|0>|^2 {out["norm2"]:.4f}, window holds {out["p_window"]:.3f} of it '
              f'[{time.time() - t0:.0f}s]')
        print(f'  gates: ' + ', '.join(f'{k} {v}' for k, v in gate.items()))
        for i in range(min(nlev, 16)):
            print(f'   {lev[i]:7.3f}  p {p[i]:.3e}')
        if 'dD' in out:
            print(f"  d_par D: level {out['dD']['level']:.3f} (share of its window: {out['dD']['share']:.3f}); "
                  f"D: level {out['D']['level']:.3f}, p(D)/p(dD) {out['D']['p_over_dD']:.2e} (CFT 0); C_D {out['C_D']:.4f}")


    def report(backend):
        rows = []
        for fn in sorted(store.glob(os.path.join(OUT, f'tilt_{backend}_N*.json')), key=lambda f: int(f.split('_N')[-1][:-5])):
            R = store.get(fn)
            lev, p = np.array(R['levels']), np.array(R['p'])
            R.update(identify(lev, p))                                     # re-identify from the stored weights
            if 'dD' not in R:
                continue
            odd = [i for i in range(len(lev)) if abs(lev[i] - (R['D']['level'] + 3)) < 0.6]
            r31 = float(p[odd].max() / R['dD']['p']) if odd else float('nan')
            rows.append((R['N'], R['D']['level'], R['D']['lowest_K1'], R['dD']['level'], R['dD']['share'],
                         R['D']['p_over_dD'], r31, R['C_D']))
        print(f'{backend}: N  D(Ward)  lowest K=1  dD  share  p(D)/p(dD)  p3/p1 (CFT {wd(2, 3) / wd(2, 1):.2f})  C_D')
        for r in rows:
            print(f'  {r[0]:2d}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}  {r[4]:.3f}  {r[5]:.2e}  {r[6]:.2f}  {r[7]:.4f}')
        if len(rows) >= 3:
            Ns = np.array([r[0] for r in rows], float); C = np.array([r[7] for r in rows])
            for w in (0.5, 1.0):
                A = np.vstack([np.ones_like(Ns), Ns ** -w]).T
                print(f'  C_D(inf) with N^-{w}: {np.linalg.lstsq(A, C, rcond=None)[0][0]:.4f}')

    def main():
        if sys.argv[1] == 'run':
            run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 30)
        else:
            report(sys.argv[2])

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/disp_tilt_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def dmrg_ward():
    """Large-N tilt Ward identity by DMRG: the displacement normalization C_D of a monodromy defect.

Physics. The rotation that tilts the line is broken only at the defect, so
    <n|L_+|0_D> = <n|[H, L_+]|0_D>/(E_n - E_0)
and, in the CFT, L_+|0_D> lies entirely in the displacement family. The weight on d_par D gives
C_D = 9 |<dD|L_+|0_D>|^2 / 32, with no energy calibration (twist/disp_tilt.py for the ED version and the derivation).
Free complex scalar, alpha = 1/2: C_D = 1/(2 pi^2) exactly (twist/free_cd.py). That is the anchor that makes
large-N runs of o2gauss_lam0 a convergence test before Ising, O(2) and O(3) values are quoted.
Method.
1. DMRG ground state |0_D> in (q = 0, K = 0) and the n lowest states of (q = 0, K = 1).
2. For each hop c+_{f,j+1} c_{f,j} of L_+ (coefficient sqrt((s-m)(s+m+1)), truncated at each flavour's top orbital),
   c+_a c_b |0_D> is built with explicit Jordan-Wigner strings, and the overlaps with the K = 1 states are summed.
3. Levels in bulk units need u = E(d phi) - E(phi) of the untwisted bulk at the same h. This is passed in (--u), or
   computed by DMRG (--bulk), or skipped, in which case D-calibrated units are used.
4. h_c comes from --hc, or is tuned by DMRG secant iterations on the bulk ratio (--tune), as run_on_defect does by ED.
Gate. At N <= 9, run with --check: the K = 1 levels and the weights must equal the ED result
disp_results/tilt_<cfg>[_lam0]_N{N}.json to the DMRG accuracy (1e-6 at the default chi).
usage: python dmrg_ward.py <cfg> N chi [--lam 0] [--hc X] [--u U | --bulk | --tune] [--nK1 6] [--sweeps 16] [--check]
       cfg in o2gauss, o2wf, o3wf (run_on_defect.CONFIGS)   -> dmrg_ward_results/<cfg>_lam{lam}_N{N}_chi{chi}.json

Paper, App. C.2: dmrg_ward o2gauss 12 400 --lam 0 and dmrg_ward o2gauss 14 600 --lam 0 (h_c tuned by DMRG; the stored results carry it), with dmrg_ward o2gauss 6 200 --lam 0 --check as the gate against disp_tilt.
"""
    import os
    import sys
    import json
    import time
    import argparse
    import itertools
    import numpy as np
    HERE = 'twist'
    OUT = os.path.join(HERE, 'dmrg_ward_results')


    def problem(cfgname, N, alpha, lam, hc):
        """modes, charges and terms for TeNPy, with the core potential scaled by lam (core_strip)."""
        CS = fuzzy.core_strip()   # was: import core_strip as CS
        fl, orbs, koff, two, mu, _, _ = CS.build(cfgname, N, alpha, lam, hc)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        npair, par_fl = len(fl.pairs), list(fl.unpaired)
        mode_m, charges = [], []
        for f in range(fl.F):
            for jj, m in enumerate(orbs[f]):
                mode_m.append(m)
                charges.append(tuple([1, jj + koff[f]] + [int(fl.qpair[f, p]) for p in range(npair)]
                                     + [1 if f == g else 0 for g in par_fl]))
        names = ['N', 'K'] + [f'Q{p}' for p in range(npair)] + [f'P{i}' for i in range(len(par_fl))]
        mods = [1, 1] + [1] * npair + [2] * len(par_fl)
        return dict(fl=fl, orbs=orbs, koff=koff, off=off, two=two, mu=list(mu), mode_m=mode_m, charges=charges,
                    names=names, mods=mods, npair=npair, npar=len(par_fl), N=N)


    def product_state(P, q_t, K_t, par_t):
        """an occupation in the target sector, found by up to 3 particle moves out of the filled psi_0 flavour."""
        N, npair = P['N'], P['npair']
        s = (N - 1) / 2
        ch = P['charges']
        par_t = [0 if x > 0 else 1 for x in par_t]
        tag = lambda occ: (list(np.array([ch[k] for k in occ]).sum(0)[2:2 + npair]),
                           int(np.array([ch[k] for k in occ]).sum(0)[1] - round(s * N)),
                           [int(x) % 2 for x in np.array([ch[k] for k in occ]).sum(0)[2 + npair:]])
        base = set(range(N))
        others = [k for k in range(len(ch)) if k >= N]
        for n in range(4):
            for rem in itertools.combinations(sorted(base), n):
                for add in itertools.combinations(others, n):
                    occ = (base - set(rem)) | set(add)
                    if tag(occ) == (list(q_t), K_t, par_t):
                        return occ
        raise RuntimeError('no product state in the sector')


    def build(P, shift=0.0):
        from tenpy.models.model import MPOModel
        _dmrg_twist = __import__('dmrg'); build_model, compress_mpo = _dmrg_twist.build_model, _dmrg_twist.compress_mpo   # was: from dmrg_twist import build_model, compress_mpo
        model, order = build_model(P['mode_m'], P['charges'], P['names'], P['two'], [v - shift for v in P['mu']], mods=P['mods'])
        return MPOModel(model.lat, compress_mpo(model.H_MPO)), order


    def lowest(model, order, occ, n, chi, sweeps, shift):
        """n lowest states of a sector; excited states by orthogonalization, energies corrected for the shift."""
        _dmrg_twist = __import__('dmrg'); run_dmrg = _dmrg_twist.run_dmrg   # was: from dmrg_twist import run_dmrg
        states, Es = [], []
        for _ in range(n):
            E, psi, eng = run_dmrg(model, order, occ, chi_max=chi, sweeps=sweeps, orthogonal_to=list(states) or None)
            states.append(psi); Es.append(float(E + shift * len(occ)))
        o = np.argsort(Es)
        return [Es[i] for i in o], [states[i] for i in o]


    def apply_hop(psi, pa, pb):
        """c+_a c_b |psi> with full Jordan-Wigner strings (MPS positions pa != pb)."""
        v = psi.copy()
        for p in range(pb):
            v.apply_local_op(p, 'JW', unitary=True)
        v.apply_local_op(pb, 'C', unitary=False, renormalize=False)
        for p in range(pa):
            v.apply_local_op(p, 'JW', unitary=True)
        v.apply_local_op(pa, 'Cd', unitary=False, renormalize=False)
        return v


    def lplus_amplitudes(P, order, psi0, bras):
        s = (P['N'] - 1) / 2
        pos = {k: i for i, k in enumerate(order)}
        amp = np.zeros(len(bras), complex)
        for f in range(P['fl'].F):
            ms = P['orbs'][f]
            for j in range(len(ms) - 1):
                c = np.sqrt(max((s - ms[j]) * (s + ms[j] + 1), 0.0))
                a, b = int(P['off'][f] + j + 1), int(P['off'][f] + j)
                v = apply_hop(psi0, pos[a], pos[b])
                if v.norm == 0:
                    continue
                for i, br in enumerate(bras):
                    amp[i] += c * br.overlap(v)
        return amp


    def bulk_unit(cfgname, N, hc, chi, sweeps):
        """u = E(d phi) - E(phi) and Delta_phi of the untwisted bulk at h = hc, by DMRG."""
        P = problem(cfgname, N, 0.0, 1.0, hc)
        z, q1 = [0] * P['npair'], [1] + [0] * (P['npair'] - 1)
        par = [1] * P['npar']
        model, order = build(P)
        E = {}
        for nm, (q, K) in dict(vac=(z, 0), phi=(q1, 0), dphi=(q1, 1)).items():
            E[nm] = lowest(model, order, product_state(P, q, K, par), 1, chi, sweeps, 0.0)[0][0]
        u = E['dphi'] - E['phi']
        return u, (E['phi'] - E['vac']) / u


    def tune_hc(cfgname, N, chi, sweeps, h0, h1, tol=1e-6):
        _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
        D = CONFIGS[cfgname]['dphi']
        f = lambda h: bulk_unit(cfgname, N, h, chi, sweeps)[1] - D
        f0, f1 = f(h0), f(h1)
        for _ in range(12):
            h2 = h1 - f1 * (h1 - h0) / (f1 - f0)
            h0, f0, h1, f1 = h1, f1, h2, f(h2)
            print(f'  tune: h = {h1:.7f}, Delta_phi - target = {f1:+.2e}', flush=True)
            if abs(f1) < tol:
                break
        return h1


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument('cfg'); ap.add_argument('N', type=int); ap.add_argument('chi', type=int)
        ap.add_argument('--lam', type=float, default=0.0); ap.add_argument('--hc', type=float)
        ap.add_argument('--u', type=float); ap.add_argument('--bulk', action='store_true')
        ap.add_argument('--tune', nargs=2, type=float, metavar=('H0', 'H1'))
        ap.add_argument('--nK1', type=int, default=6); ap.add_argument('--sweeps', type=int, default=16)
        ap.add_argument('--check', action='store_true')
        a = ap.parse_args()
        t0 = time.time()
        if a.tune:
            a.hc = tune_hc(a.cfg, a.N, a.chi, a.sweeps, *a.tune)
        if a.hc is None:
            a.hc = store.get(os.path.join(HERE, 'on_results', f'{a.cfg}_N{a.N}.json'))['h_c']
        if a.bulk or a.tune:
            a.u = bulk_unit(a.cfg, a.N, a.hc, a.chi, a.sweeps)[0]
        P = problem(a.cfg, a.N, 0.5, a.lam, a.hc)
        z, par = [0] * P['npair'], [1] * P['npar']
        shift = 2.0
        model, order = build(P, shift)
        E0, S0 = lowest(model, order, product_state(P, z, 0, par), 1, a.chi, a.sweeps, shift)
        E1, S1 = lowest(model, order, product_state(P, z, 1, par), a.nK1, a.chi, a.sweeps, shift)
        print(f'{a.cfg} N={a.N} chi={a.chi} lam={a.lam}: E0 {E0[0]:.10f}; K=1 levels built [{time.time() - t0:.0f}s]', flush=True)
        amp = lplus_amplitudes(P, order, S0[0], S1)
        p = np.abs(amp) ** 2
        unit = a.u if a.u else None
        lev = np.array(E1) - E0[0]
        levels = (lev / unit) if unit else 2 * lev / lev[0]                  # bulk units, or D-calibrated (D = 2)
        _disp_tilt = globals()['disp_tilt'](); identify = _disp_tilt.identify   # was: from disp_tilt import identify
        ident = identify(levels, p)
        out = dict(cfg=a.cfg, N=a.N, chi=a.chi, lam=a.lam, h_c=a.hc, u=a.u, units='bulk' if unit else 'D=2',
                   E0=E0[0], E_K1=E1, levels=levels.tolist(), p=p.tolist(), **ident, chi_reached=int(max(S0[0].chi)),
                   time=time.time() - t0)
        if a.check:
            tag = a.cfg + ('_lam0' if a.lam == 0 else '')
            R = store.get(os.path.join(HERE, 'disp_results', f'tilt_{tag}_N{a.N}.json'))
            n = min(len(p), 6)
            out['check'] = dict(dlev=float(np.max(np.abs(np.array(R['levels'][:n]) - (lev[:n] / R['unit'])))),
                                dp=float(np.max(np.abs(np.array(R['p'][:n]) - p[:n]))))
            print(f"  gate vs ED: max |level diff| {out['check']['dlev']:.1e}, max |p diff| {out['check']['dp']:.1e}", flush=True)
        pass   # was os.makedirs(...): the store needs no output directory
        json.dump(out, open(os.path.join(OUT, f"{a.cfg}_lam{a.lam:g}_N{a.N}_chi{a.chi}.json"), 'w'), indent=1, default=float)
        print('  levels ' + ' '.join(f'{x:.4f}' for x in levels) + ' | p ' + ' '.join(f'{x:.3e}' for x in p) +
              (f"\n  dD {ident['dD']['level']:.4f}, D {ident['D']['level']:.4f}, C_D {ident['C_D']:.5f} "
               f"(free exact 1/(2 pi^2) = {1 / (2 * np.pi ** 2):.5f})" if ident else ''), flush=True)

    def _run():
        main()

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/dmrg_ward_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            _run()
    return namespace(locals(), globals())


@section
def cd_series():
    """The free displacement normalization from the projected rotation, App. C.2 (written for this package; it reads stored
results and recomputes nothing). C_D at N_m = 6-10 by exact diagonalization (disp_tilt, potential subtracted and
original) and at N_m = 12, 14 by DMRG (dmrg_ward), as a fraction of the exact 1/(2 pi^2), and the extrapolations of the
last three sizes of the subtracted series in N_m^-p.
"""
    import numpy as np
    EX = 1 / (2 * np.pi ** 2)
    ns = list(range(6, 11)) + [12, 14]
    r0 = [store.get(f'twist/disp_results/tilt_o2gauss_lam0_N{n}.json') for n in range(6, 11)]
    r0 += [store.get('twist/dmrg_ward_results/o2gauss_lam0_N12_chi400.json'), store.get('twist/dmrg_ward_results/o2gauss_lam0_N14_chi600.json')]
    r1 = [store.get(f'twist/disp_results/tilt_o2gauss_N{n}.json') for n in range(6, 11)]

    def main():
        print(f'exact C_D = 1/(2 pi^2) = {EX:.6f}')
        print('potential subtracted: ' + ', '.join(f'N_m={n}: {v["C_D"]:.4f} ({v["C_D"] / EX:.3f})' for n, v in zip(ns, r0)))
        print('original potential:   ' + ', '.join(f'N_m={n}: {v["C_D"]:.4f} ({v["C_D"] / EX:.3f})' for n, v in zip(ns, r1)))
        x, y = np.array(ns[-3:], float), np.array([v['C_D'] / EX for v in r0[-3:]])
        print('last three sizes, intercept of C_D/C_D^free in N_m^-p: ' +
              ', '.join(f'p={p:g}: {np.linalg.lstsq(np.vstack([np.ones(3), x ** -p]).T, y, rcond=None)[0][0]:.3f}' for p in (0.5, 1, 1.5, 2)))

    def cli(argv):
        main()
    return namespace(locals(), globals())


STEPS = {'run_on_defect': fuzzy.run_on_defect, 'core_strip': fuzzy.core_strip, 'free_defect': free_defect, 'free_extrap': free_extrap, 'defect_corr': defect_corr, 'core_scaling': core_scaling,
         'free_cd': free_cd, 'disp_tilt': disp_tilt, 'dmrg_ward': dmrg_ward, 'cd_series': cd_series}
REPORT = [('free_extrap',), ('defect_corr', 'report'), ('cd_series',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
