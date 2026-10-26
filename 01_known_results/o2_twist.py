"""What is the O(2) monodromy line at a general angle alpha, and does it depend on the regulator?

Two independent O(2) regulators: the vector model of Guo et al. (u = 0.270) and the spin-one model of Dey et al.
((V0, V1) = (4, 1), Lambda tuned). Results: the lowest charged mode at half twist, the angle alpha_c where the
charged s = alpha - 1 mode becomes relevant, the two-mode interaction energies against one loop at any alpha, and
the pole estimates of the defect three-point coefficients (with the free complex scalar, u = 0.148, as control).

usage: python o2_twist.py <step> [args]      (python o2_twist.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401
import free_control                                        # noqa: E402
import ising_twist                                        # noqa: E402


@section
def run_o2defect():
    """O(2) monodromy defect vs alpha on the fuzzy sphere (spin-1 XY model of 2604.18705 at D_c(N)).

Calibration: gamma = E_{d sigma} - E_sigma (Delta difference exactly 1) from the bulk at the same N,D.
Defect levels: Dhat = (E - E0^D)/gamma, E0^D the lowest (Q=0, L_z=0) defect state.
Transverse spin s = K + alpha Q.
"""
    HERE = 'twist'
    import sys
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from scipy.optimize import brentq
    O2Basis = fuzzy.o2twist().O2Basis
    build_terms = fuzzy.o2twist().build_terms
    build_H = fuzzy.o2twist().build_H
    from fuzzy import couplings_for

    DS = 0.519088

    def low(H, dim, k):
        if dim <= 2000:
            return np.linalg.eigvalsh(H.toarray())[:k]
        return np.sort(sla.eigsh(H, k=min(k, dim - 2), which='SA', tol=1e-11)[0])

    def nD_of(b):
        return sum(((b.states >> np.int64(b.off[f] + j)) & 1) for f in (0, 2) for j in range(b.nf[f])).astype(float)

    def Dc_of(N, G0, G1):
        two, U = build_terms(N, 0, G0, G1)
        bs = [O2Basis(N, 0, 0, 0), O2Basis(N, 0, 1, 0), O2Basis(N, 0, 1, 1), O2Basis(N, 0, 0, 1), O2Basis(N, 0, 2, 0)]
        H0 = [build_H(b, two, U, 0.0) for b in bs]; nD = [nD_of(b) for b in bs]
        def E(i, D, k=1): return low(H0[i] + sp.diags(D * nD[i]), bs[i].dim, k)
        tgt = DS / (1 + DS)
        Dc = brentq(lambda D: (E(1, D)[0] - E(0, D)[0]) / (E(2, D)[0] - E(0, D)[0]) - tgt, 2.0, 4.0, xtol=1e-6)
        e = [E(i, Dc)[0] for i in range(5)]
        gam = e[2] - e[1]
        return Dc, gam, dict(sigma=(e[1] - e[0]) / gam, dsigma=(e[2] - e[0]) / gam,
                             j=(e[3] - e[0]) / gam, t=(e[4] - e[0]) / gam)

    def cli(argv):
        N = int(argv[1]); alphas = [float(a) for a in argv[2].split(',')]
        t0 = time.time()
        G0, G1, _ = couplings_for(N, 4.0, 1.0)
        Dc, gam, bulk = Dc_of(N, G0, G1)
        print(f"N={N}: D_c={Dc:.4f} gamma={gam:.4f}  bulk check: sigma={bulk['sigma']:.4f} (0.5191)  "
              f"j={bulk['j']:.4f} (2)  t={bulk['t']:.4f} (1.2363)   [{time.time()-t0:.0f}s]", flush=True)
        sectors = [(0, 0), (0, 1), (0, 2), (1, 0), (1, -1), (1, 1), (1, -2), (2, 0), (2, -1), (2, -2)]
        for al in alphas:
            t1 = time.time()
            two, U = build_terms(N, al, G0, G1)
            res = {}
            for (Q, Kk) in sectors:
                b = O2Basis(N, al, Q, Kk)
                H = build_H(b, two, U, Dc)
                res[(Q, Kk)] = low(H, b.dim, 4)
            E0 = res[(0, 0)][0]
            line = []
            for (Q, Kk) in sectors:
                sv = Kk + al * Q
                vals = (res[(Q, Kk)] - E0) / gam
                line.append(f"Q={Q} s={sv:+.2f}: " + ' '.join(f"{v:.3f}" for v in vals[:3]))
            print(f"  alpha={al:.3f} ({time.time()-t1:.0f}s)\n     " + '\n     '.join(line), flush=True)
    return namespace(locals(), globals())


@section
def o2_ed_precise():
    """Full-precision O(2) ED energies for the note's fits (the *.log files print gaps to 3 decimals)."""
    HERE = 'twist/note'
    import sys
    import json


    def cli(argv):
        O2Basis = fuzzy.o2twist().O2Basis
        build_terms = fuzzy.o2twist().build_terms
        build_H = fuzzy.o2twist().build_H
        from fuzzy import couplings_for
        low = run_o2defect().low
        Dc = {6: 2.9563, 7: 2.9075, 8: 2.8747, 9: 2.8516, 10: 2.8347}
        jobs = [(N, 0.5, [(0, 0), (0, 1), (1, 0)]) for N in (6, 7, 8, 9)]
        jobs += [(N, a, [(0, 0), (0, 1), (1, 0), (1, -1)]) for N in (6, 7, 8, 9) for a in (0.35, 0.4, 0.45)]
        jobs += [(N, 0.0, [(0, 0), (1, 0), (1, 1), (0, 1)]) for N in (6, 7, 8, 9, 10)]      # bulk: vac, sigma, d sigma, j
        out = {}
        for N, a, secs in jobs:
            G0, G1, _ = couplings_for(N, 4.0, 1.0)
            two, U = build_terms(N, a, G0, G1)
            for sec in secs:
                b = O2Basis(N, a, *sec)
                out[f"{N}_{a}_{sec[0]}_{sec[1]}"] = float(low(build_H(b, two, U, Dc[N]), b.dim, 2)[0])
            print(N, a, 'done', flush=True)
        store.put('twist/note/o2_ed_precise.json', out)
    return namespace(locals(), globals())


@section
def analyze_on():
    """Defect dimensions for the O(N) fuzzy-sphere models (on_results/*.json from run_on_defect.py, plus
dmrg_results/<config>_*.json from dmrg_jobs.py), displacement-calibrated.

Sector keys: "(pair charges)|K|(parities)" with L_z = K + alpha * Q_twisted.
  D        : neutral, K = 1, all parities even.  For o4wf_diag the SU(2) triplet composite lies just below
             it in the same sector; it also appears at pair charges (1,-1), so D is the lowest neutral K=1
             level with no partner there.
  Psi_a    : twisted charge 1, K = 0;  Psi_{a-1}: K = -1;  Psi_{a+1}: K = +1.
  phi3     : o3wf only, the untwisted component's leading defect operator (neutral, K=0, P3 odd).
One loop (eps = 1): Delta_s = 1/2 + |s| - c alpha(1-alpha)/|s|,  c = (n_t + 1)/(M + 8).
"""
    import json
    import os
    import sys
    import glob
    import numpy as np

    HERE = 'twist'
    ONE_LOOP_C = {'o2wf': 2 / 10, 'o2gauss': 0.0, 'o3wf': 2 / 11, 'o4wf_diag': 3 / 12, 'o4wf_pair': 2 / 12}


    def keys(name):
        if name in ('o2wf', 'o2gauss'):
            z, q1, p, pm = '(0,)', '(1,)', '()', None
        elif name == 'o3wf':
            z, q1, p, pm = '(0,)', '(1,)', '(1,)', '(-1,)'
        else:
            z, q1, p, pm = '(0, 0)', '(1, 0)', '()', None
        return z, q1, p, pm


    def load(name, N):
        """ED results for (name, N); if absent, assemble the same structure from DMRG results."""
        f = os.path.join(HERE, 'on_results', f'{name}_N{N}.json')
        if store.exists(f):
            return store.get(f)
        res = {'alpha': {}, 'dmrg': True}
        for g in store.glob(os.path.join(HERE, 'dmrg_results', f'{name}_N{N}_a*.json')):
            r = store.get(g)
            q, K, par = r['sector']
            key = f"{tuple(q)}|{K}|{tuple(par)}"
            lev = res['alpha'].setdefault(f"{r['alpha']:.4f}", {})
            lev[key] = r.get('E_list', [r['E']])
            res['h_c'] = r['h']
            res.setdefault('trunc', []).append(max(r.get('trunc_list', [r['trunc']])))
        return res if res['alpha'] else None


    def displacement(name, lev, tol=2e-6):
        """Displacement energy.  For o4wf_diag an SU(2)-triplet composite lies just below D in the same sector;
    its partner at pair charges (1,-1) identifies it.  The neutral state closest to that partner is taken as
    the triplet (an exact match in ED; in DMRG the neutral-sector ground state may sit slightly above the
    partner if it has not fully converged), and D is the lowest remaining state."""
        z, q1, p, _ = keys(name)
        neutral = list(lev[f'{z}|1|{p}'])
        if name != 'o4wf_diag':
            return neutral[0]
        trip = lev.get(f'(1, -1)|1|{p}', [])
        for t in trip[:1]:
            if len(neutral) > 1:
                neutral.pop(int(np.argmin([abs(e - t) for e in neutral])))
        return neutral[0] if neutral else np.nan


    def dims(name, lev):
        """D-calibrated dimensions of the named defect operators at one alpha."""
        z, q1, p, pm = keys(name)
        E0 = lev[f'{z}|0|{p}'][0]
        ED = displacement(name, lev)
        c = lambda e: 2 * (e - E0) / (ED - E0)
        g = lambda k, i=0: c(lev[k][i]) if k in lev and len(lev[k]) > i else np.nan
        out = dict(D_raw=ED - E0, Psi_a=g(f'{q1}|0|{p}'), Psi_am1=g(f'{q1}|-1|{p}'),
                   Psi_ap1=g(f'{q1}|1|{p}'), neutral0=g(f'{z}|0|{p}', 1))
        q2 = '(2,)' if name in ('o2wf', 'o2gauss', 'o3wf') else '(2, 0)'
        out['Q2'] = g(f'{q2}|0|{p}')
        if pm:
            out['phi3'] = g(f'{z}|0|{pm}')
            out['dphi3'] = g(f'{z}|1|{pm}')
        return out


    def bulk(name, lev):
        z, q1, p, pm = keys(name)
        E0 = lev[f'{z}|0|{p}'][0]
        g = lev[f'{q1}|1|{p}'][0] - lev[f'{q1}|0|{p}'][0]
        c = lambda e: (e - E0) / g
        q2 = '(2,)' if name in ('o2wf', 'o2gauss', 'o3wf') else '(2, 0)'
        return dict(phi=c(lev[f'{q1}|0|{p}'][0]), t=c(lev[f'{q2}|0|{p}'][0]), S=c(lev[f'{z}|0|{p}'][2] if name == 'o3wf'
                    else lev[f'{z}|0|{p}'][2] if name.startswith('o4') else lev[f'{z}|0|{p}'][1]),
                    j=c(lev[f'{z}|1|{p}'][0]), T=c(lev[f'{z}|2|{p}'][0]))


    def alpha_c(name, res):
        al = [0.35, 0.4, 0.45]
        if not all(f'{a:.4f}' in res['alpha'] for a in al):
            return np.nan
        try:
            y = np.array([dims(name, res['alpha'][f'{a:.4f}'])['Psi_am1'] for a in al])
        except KeyError:
            return np.nan
        r = [x.real for x in np.roots(np.polyfit(al, y - 1, 2)) if abs(x.imag) < 1e-12 and 0.3 < x.real < 0.5]
        return r[0] if r else np.nan


    def one_loop(name, al, s):
        return 0.5 + abs(s) - ONE_LOOP_C[name] * al * (1 - al) / abs(s)


    def cli(argv):
        name = argv[1]
        Ns = sorted(set(int(f.split('_N')[1].split('.')[0]) for f in store.glob(os.path.join(HERE, 'on_results', f'{name}_N*.json')))
                    | set(int(f.split(f'{name}_N')[1].split('_')[0]) for f in store.glob(os.path.join(HERE, 'dmrg_results', f'{name}_N*.json'))))
        print(f"== {name}: bulk at alpha=0 (calibrated by dphi - phi = 1)")
        for N in Ns:
            r = load(name, N)
            if '0.0000' in r.get('alpha', {}) and not r.get('dmrg'):
                b = bulk(name, r['alpha']['0.0000'])
                print(f"  N={N:2d} h_c={r['h_c']:.5f}: " + '  '.join(f"{k} {v:.4f}" for k, v in b.items()))
        print(f"== {name}: defect, displacement calibration")
        for N in Ns:
            r = load(name, N)
            for a in sorted(r['alpha']):
                if float(a) == 0:
                    continue
                try:
                    d = dims(name, r['alpha'][a])
                except KeyError:
                    continue
                al = float(a)
                print(f"  N={N:2d} a={al:.2f}: " + '  '.join(f"{k} {v:.4f}" for k, v in d.items() if k != 'D_raw')
                      + f"   | 1-loop Psi_a {one_loop(name, al, al):.3f} Psi_a-1 {one_loop(name, al, al - 1):.3f}")
            print(f"     alpha_c(N={N}) = {alpha_c(name, r):.4f}")
    return namespace(locals(), globals())


@section
def on_tables():
    """Numbers for the free-scalar, two-model O(2), O(3) and O(4) sections of the monodromy note.

Reads ../on_results/*.json (ED, run_on_defect.py), ../dmrg_results/*.json (DMRG, dmrg_jobs.py),
spin1_neutral.json (spin1_neutral.py) and, for the spin-1 O(2) model, dmrg_tables.py.
"""
    import os
    import sys
    import json
    import numpy as np

    HERE = 'twist/note'
    exact_levels = free_control.free_defect().exact_levels
    A = analyze_on()
    T = ising_twist.dmrg_tables()

    FREE_LEVELS = [(1, 0, r'\hat\Psi_\alpha', 0), (1, -1, r'\hat\Psi_{\alpha-1}', 0), (2, 0, r'\hat\Psi_\alpha^2', 0),
                   (2, -1, r'\hat\Psi_\alpha\hat\Psi_{\alpha-1}', 0), (0, 0, r'\hat\Psi_\alpha\hat{\bar\Psi}_{-\alpha}', 1)]


    def free_scalar(alphas=(0.1, 0.3, 0.5), Ns=(6, 7, 8, 9, 10)):
        R = {N: A.load('o2gauss', N) for N in Ns}
        x = 1 / np.array(Ns, float)
        rows, devs = [], []
        for al in (0.1, 0.2, 0.3, 0.4, 0.5):
            for Q, K, lab, i in FREE_LEVELS:
                y, yb = [], []
                for N in Ns:
                    lev = R[N]['alpha'][f'{al:.4f}']; E0 = lev['(0,)|0|()'][0]
                    gD = (lev['(0,)|1|()'][0] - E0) / 2
                    b = R[N]['alpha']['0.0000']; gB = b['(1,)|1|()'][0] - b['(1,)|0|()'][0]
                    e = lev[f'({Q},)|{K}|()'][i]
                    y.append((e - E0) / gD); yb.append((e - E0) / gB)
                ex = exact_levels(al, Q, K)[i]
                quad = np.polyfit(x, y, 2)[2]
                devs.append(quad - ex)
                if al in alphas:
                    rows.append((al, lab, ex, y[-1], yb[-1], quad))
        devs = np.array(devs)
        return rows, dict(n=len(devs), mean=devs.mean(), rms=np.sqrt((devs ** 2).mean()), max=np.abs(devs).max())


    def free_exponents(Ns=(6, 8, 10)):
        """Fitted exponents of |Delta - exact| for Psi_{1/2} under both calibrations."""
        out = {}
        for mode in ('D', 'b'):
            d = []
            for N in Ns:
                r = A.load('o2gauss', N); lev = r['alpha']['0.5000']; E0 = lev['(0,)|0|()'][0]
                g = ((lev['(0,)|1|()'][0] - E0) / 2 if mode == 'D' else
                     r['alpha']['0.0000']['(1,)|1|()'][0] - r['alpha']['0.0000']['(1,)|0|()'][0])
                d.append(abs((lev['(1,)|0|()'][0] - E0) / g - 1))
            out[mode] = -np.polyfit(np.log(Ns), np.log(d), 1)[0]
        return out


    def spin1_pam1(N, al):
        """Spin-1 model Psi_{alpha-1}, D-calibrated (ED at N<=9 from o2_ed_precise.json, DMRG beyond)."""
        v = T.o2_dcal(N, al, 1, -1)
        if not np.isnan(v):
            return v
        import glob
        g = lambda sec: [store.get(f)['E'] for f in
                         store.glob(os.path.join(HERE, '..', 'dmrg_results', f'o2_N{N}_a{al}_s{sec[0]}_{sec[1]}_chi*.json'))]
        e0, ed, ep = g((0, 0)), g((0, 1)), g((1, -1))
        return 2 * (ep[0] - e0[0]) / (ed[0] - e0[0]) if (e0 and ed and ep) else np.nan


    def o2_two_models(Ns=(6, 7, 8, 9, 10, 12, 14, 16, 18)):
        st = T.o2_stab()
        acs = T.o2_alpha_c()
        neut = store.get(os.path.join(HERE, 'spin1_neutral.json')) if store.exists(os.path.join(HERE, 'spin1_neutral.json')) else {}
        rows = []
        for N in Ns:
            r = A.load('o2wf', N)
            g = {}
            if r:
                if '0.5000' in r['alpha']:
                    try:
                        d = A.dims('o2wf', r['alpha']['0.5000']); g['psi'], g['neu'] = d['Psi_a'], d['neutral0']
                    except KeyError:
                        pass
                g['ac'] = A.alpha_c('o2wf', r)
            if r and '0.4000' in r['alpha']:
                try:
                    g['pam1'] = A.dims('o2wf', r['alpha']['0.4000'])['Psi_am1']
                except KeyError:
                    pass
            rows.append(dict(N=N, s1_psi=st.get(N, {}).get('P_D', np.nan) if N in st else np.nan,
                             s1_ac=acs.get(N, np.nan), s1_neu=neut.get(str(N), [np.nan])[0],
                             s1_pam1=spin1_pam1(N, 0.4),
                             g_psi=g.get('psi', np.nan), g_ac=g.get('ac', np.nan), g_neu=g.get('neu', np.nan),
                             g_pam1=g.get('pam1', np.nan)))
        return rows


    def joint_fit(key, p_fixed=None, two_terms=True):
        """Common-limit fit of the two O(2) models: y_m(N) = L + a_m N^-p (+ b_m N^-2p).
    key: 'ac' (alpha_c) or 'psi' (Psi_1/2).  Returns (L, p, max residual, number of points)."""
        from scipy.optimize import least_squares
        rows = o2_two_models()
        D = [(r['N'], r['s1_' + key]) for r in rows if not np.isnan(r['s1_' + key])]
        G = [(r['N'], r['g_' + key]) for r in rows if not np.isnan(r['g_' + key])]
        def model(th, n, m):
            L, p = th[0], (p_fixed if p_fixed is not None else th[1])
            a, b = (th[2], th[4]) if m == 0 else (th[3], th[5])
            return L + a * n ** -p + (b * n ** (-2 * p) if two_terms else 0)
        def res(th):
            return np.array([model(th, n, 0) - y for n, y in D] + [model(th, n, 1) - y for n, y in G])
        best = None
        for p0 in (0.3, 0.5, 0.8, 1.0):
            th0 = [np.mean([y for _, y in D + G]), p0, -0.1, 0.1, 0.0, 0.0]
            f = least_squares(res, th0)
            if best is None or f.cost < best.cost:
                best = f
        p = p_fixed if p_fixed is not None else best.x[1]
        return best.x[0], p, np.abs(best.fun).max(), len(D) + len(G)


    def on_family(Ns=(6, 7, 8, 9, 10, 12)):
        rows = []
        for N in Ns:
            row = dict(N=N)
            for name in ('o2wf', 'o3wf', 'o4wf_diag'):
                r = A.load(name, N)
                if not r:
                    continue
                if '0.5000' in r['alpha']:
                    try:
                        d = A.dims(name, r['alpha']['0.5000'])
                        row[name + '_psi'] = d['Psi_a']
                        if name == 'o3wf':
                            row['phi3'] = d.get('phi3', np.nan)
                    except KeyError:
                        pass
                row[name + '_ac'] = A.alpha_c(name, r)
            rows.append(row)
        return rows


    def cli(argv):
        rows, s = free_scalar()
        print("% free scalar: alpha, level, exact, D-cal N=10, bulk-cal N=10, quadratic extrapolation (D-cal, N=6..10)")
        for al, lab, ex, yd, yb, q in rows:
            print(f"{al:.1f} & ${lab}$ & {ex:.3f} & {yd:.3f} & {yb:.3f} & {q:.3f} \\\\")
        print(f"% all 5 alphas x {len(FREE_LEVELS)} levels: extrapolated - exact: mean {s['mean']:+.3f}, rms {s['rms']:.3f}, max {s['max']:.3f}")
        print("% |Psi_1/2 - 1| exponents (N=6,8,10):", {k: round(v, 2) for k, v in free_exponents().items()})
        print("% O(2), two models: N | Psi_1/2 s1 G | Psi_a-1(0.4) s1 G | alpha_c s1 G | neutral s1 G")
        f = lambda v, n: f"{v:.{n}f}" if not np.isnan(v) else '--'
        for r in o2_two_models():
            print(f"{r['N']} & {f(r['s1_psi'],4)} & {f(r['g_psi'],4)} & {f(r['s1_pam1'],4)} & {f(r['g_pam1'],4)} & "
                  f"{f(r['s1_ac'],4)} & {f(r['g_ac'],4)} & {f(r['s1_neu'],3)} & {f(r['g_neu'],3)} \\\\")
        for key in ('ac', 'psi'):
            for pf in (None, 0.39):
                L, p, res, n = joint_fit(key, p_fixed=pf)
                print(f"% joint fit {key} (p {'free' if pf is None else 'fixed'}): L = {L:.4f}, p = {p:.2f}, max resid {res:.1e}, {n} points")
        print("% O(N) family (Guo models): N, Psi_1/2 O(2) O(3) O(4)d, alpha_c O(2) O(3) O(4)d, phi3")
        for r in on_family():
            f = lambda k, n=4: f"{r[k]:.{n}f}" if k in r and not np.isnan(r[k]) else '--'
            print(f"{r['N']} & {f('o2wf_psi')} & {f('o3wf_psi')} & {f('o4wf_diag_psi')} & {f('o2wf_ac')} & {f('o3wf_ac')} & {f('o4wf_diag_ac')} & {f('phi3')} \\\\")
    return namespace(locals(), globals())


@section
def contact_alpha():
    """Two-mode interaction energies of the O(2) monodromy defect against the twist angle (lesson from arXiv:2610.31269).

For a complex scalar with monodromy e^{2 pi i alpha} the single modes are Psi_alpha (|s| = alpha) and Psi_{alpha-1}
(|s| = 1 - alpha). Two-mode defect operators and their lattice sectors (pair charge q, K), with s = q alpha + K:
    charge 2, s = 2 alpha      [Psi_a Psi_a]          (2,0)
    charge 2, s = 2 alpha - 1  [Psi_a Psi_{a-1}]      (2,-1)
    neutral,  s = 1            D = [Psi_a Psibar_{1-a}] (0,1)
    neutral,  s = 0            two states, [Psi_a Psibar_{-a}] and [Psi_{a-1} Psibar_{1-a}], levels 1 and 2 of (0,0)
The interaction energy is the level minus the sum of its lattice single-mode levels. For the neutral s = 0 pair the
sum of both levels is used, which is independent of their mixing. One loop (eps = 1, M = 2, lambda*/(16 pi^2) = 3/10),
from the S^3 overlaps of arXiv:2610.31269 App. B, which hold for any real |s|, with flavour weight 2/3 for every
two-mode state below and the neutral-pair mixing element lambda* (2/3)/(16 pi^2):
    charge 2, s = 2a:      0.2/(1 + 2a)       charge 2, s = 2a - 1:  0.2
    D:                     0.2                neutral s = 0 sum:     0.4/(1 + 2a) + 0.4/(3 - 2a),  mixing 0.2
The free scalar (o2gauss) is the control: every interaction energy vanishes in the continuum.
Extrapolations follow arXiv:2610.31269: X_inf + c N^-1/2 and X_inf + c N^-1 on the four largest sizes.
usage: python contact_alpha.py   (reads twist/on_results/{o2gauss,o2wf}_N*.json, writes analysis/contact_alpha.json)
"""
    import os
    import json
    import glob
    import numpy as np


    def cli(argv):

        HERE = 'fun/combined_2026_10_22/analysis'
        TW = os.path.join(HERE, '..', '..', '..', 'twist', 'on_results')
        CHANNELS = ('q2_same', 'q2_diff', 'D', 'neutral_sum')


        def oneloop(a):
            return dict(q2_same=0.2 / (1 + 2 * a), q2_diff=0.2, D=0.2, neutral_sum=0.4 / (1 + 2 * a) + 0.4 / (3 - 2 * a))


        def levels(cfg):
            out = {}
            for f in sorted(store.glob(os.path.join(TW, f'{cfg}_N*.json'))):
                R = store.get(f); N = R['N']
                b = R['alpha']['0.0000']
                u = b['(1,)|1|()'][0] - b['(1,)|0|()'][0]
                for a, lev in R['alpha'].items():
                    E0 = lev['(0,)|0|()'][0]
                    g = lambda k, i=0: (lev[k][i] - E0) / u
                    pa, pb = g('(1,)|0|()'), g('(1,)|-1|()')
                    out.setdefault(a, {})[N] = dict(
                        Psi_a=pa, Psi_a1=pb,
                        q2_same=g('(2,)|0|()') - 2 * pa, q2_diff=g('(2,)|-1|()') - pa - pb, D=g('(0,)|1|()') - pa - pb,
                        neutral_sum=g('(0,)|0|()', 1) + g('(0,)|0|()', 2) - 2 * pa - 2 * pb,
                        neutral_split=g('(0,)|0|()', 2) - g('(0,)|0|()', 1))
            return out


        def extrap(Ns, v):
            Ns, v = np.array(Ns[-4:], float), np.array(v[-4:])
            r = [np.polyfit(Ns ** -w, v, 1)[1] for w in (0.5, 1.0)]
            return float(np.mean(r)), float(abs(r[0] - r[1]) / 2)


        res = {}
        for cfg in ('o2gauss', 'o2wf'):
            L = levels(cfg); res[cfg] = {}
            print(f'=== {cfg}: interaction energies in bulk units, N = {sorted(next(iter(L.values())))}; last column one loop (o2wf only)')
            for a in sorted(L, key=float):
                if float(a) == 0:
                    continue
                Ns = sorted(L[a])
                row = {}
                for ch in CHANNELS + ('neutral_split',):
                    v = [L[a][N][ch] for N in Ns]
                    row[ch] = dict(N=Ns, vals=v, extrap=extrap(Ns, v))
                row['oneloop'] = oneloop(float(a))
                res[cfg][a] = row
                print(f"  alpha {a}: " + '  '.join(f"{ch} {row[ch]['vals'][-1]:+.3f} -> {row[ch]['extrap'][0]:+.3f}({row[ch]['extrap'][1]:.3f})"
                                                 + (f" [1L {row['oneloop'][ch]:.3f}]" if cfg == 'o2wf' else '') for ch in CHANNELS))
        store.put(os.path.join(HERE, 'contact_alpha.json'), res)
    return namespace(locals(), globals())


@section
def pole_ope():
    """Defect three-point coefficients from the order-parameter density near a pole (method of arXiv:2610.31269, Eq. (C3)).

    C_{1 2 3} = <1| n(theta0) |3> / <2| n(theta0) |0_D>  * (cos theta0)^{-(D1 - D2 - D3)}
with n = Phi = psi_0^+ psi_- + psi_+^+ psi_0 (charge +1) or Phi^+ (charge -1), evaluated at azimuth 0 on the window
theta0 in {pi/12, pi/6, pi/4}; |2> is the lowest state with the transferred (charge, K).  Dimensions D are displacement
calibrated, D_D = 2.  Channels (s = q alpha + K):
    same : <(2,0)| Phi |(1,0)>  / <(1,0)| Phi |0>        [Psi_a Psi_a],        GFF sqrt 2
    diff : <(2,-1)| Phi |(1,0)> / <(1,-1)| Phi |0>       [Psi_a Psi_{a-1}],    GFF 1
    D    : <(0,1)| Phi^+ |(1,0)> / <(-1,1)| Phi^+ |0>    [Psi_a Psibar_{1-a}], GFF 1
Models: o2gauss (free complex scalar, the benchmark) and o2wf, built with twist/core_strip.build (lam = 1: model as
used, lam = 0: core potential removed).  Read-only on the twist code.
usage: python pole_ope.py <config> <N> <alpha> [lam]   -> analysis/pole_ope/<config>_N<N>_a<alpha>_lam<lam>.json
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse.linalg as sla

    HERE = 'fun/combined_2026_10_22/analysis'
    TW = os.path.join(HERE, '..', '..', '..', 'twist')
    M = fuzzy.onmodel()
    core_strip = fuzzy.core_strip()
    from fuzzy import apply_cdc, wave_abs

    TH0 = np.array([np.pi / 12, np.pi / 6, np.pi / 4])


    def eig(H, k):
        if H.shape[0] <= 3000:
            e, v = np.linalg.eigh(H.toarray()); return e[:k], v[:, :k]
        e, v = sla.eigsh(H, k=k, which='SA', tol=1e-11)
        o = np.argsort(e); return e[o], v[:, o]


    def main(cfg, N, alpha, lam=1.0):
        t0 = time.time()
        fl, orbs, koff, two, mu, _, hc = core_strip.build(cfg, N, alpha, lam)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        s = (N - 1) / 2
        f0, fp, fm = fl.names.index('0'), fl.names.index('12+'), fl.names.index('12-')
        sec = {}
        for (q, K, k) in ((0, 0, 1), (1, 0, 1), (1, -1, 1), (-1, 1, 1), (2, 0, 1), (2, -1, 1), (0, 1, 1)):
            b = M.Basis(fl, orbs, koff, N, [q], K, ())
            e, v = eig(M.build_H(b, two, mu), k)
            sec[(q, K)] = (b.states, e, v)
        E0 = sec[(0, 0)][1][0]; ED = sec[(0, 1)][1][0]
        dim = {k: float(2 * (v[1][0] - E0) / (ED - E0)) for k, v in sec.items()}

        def melt(src, dst, conj):
            """<dst| n(theta0, phi=0) |src> for n = Phi (conj False) or Phi^+ (True), lowest states."""
            st_i, _, vi = sec[src]; st_o, _, vo = sec[dst]
            out = np.zeros(len(TH0), dtype=complex)
            pairs = ((f0, fm), (fp, f0)) if not conj else ((fm, f0), (f0, fp))
            for (fa, fb) in pairs:                       # c^+_a c_b
                for ja, ma in enumerate(orbs[fa]):
                    for jb, mb in enumerate(orbs[fb]):
                        if (ja + koff[fa]) - (jb + koff[fb]) != dst[1] - src[1]:
                            continue
                        idx, val = apply_cdc(st_i, vi[:, 0], int(off[fa] + ja), int(off[fb] + jb), st_o)
                        if len(idx):
                            out += np.vdot(vo[idx, 0], val) * wave_abs(s, ma, TH0) * wave_abs(s, mb, TH0)
            return np.abs(out)

        res = dict(config=cfg, N=N, alpha=alpha, lam=lam, theta0=TH0.tolist(), dims={str(k): v for k, v in dim.items()})
        chans = {'same': ((1, 0), (2, 0), (1, 0), False, np.sqrt(2)),
                 'diff': ((1, 0), (2, -1), (1, -1), False, 1.0),
                 'D': ((1, 0), (0, 1), (-1, 1), True, 1.0)}
        for name, (i3, i1, i2, conj, gff) in chans.items():
            num = melt(i3, i1, conj); den = melt((0, 0), i2, conj)
            expo = dim[i1] - dim[i2] - dim[i3]
            r = num / den * np.cos(TH0) ** (-expo)
            res[name] = dict(ratio=r.tolist(), C=float(r.mean()), err=float((r.max() - r.min()) / 2), gff=gff)
        None
        store.put(os.path.join(HERE, 'pole_ope', f'{cfg}_N{N}_a{alpha:.2f}_lam{lam:g}.json'), res)
        print(f"{cfg} N={N} a={alpha} lam={lam}: " + '  '.join(f"{n} {res[n]['C']:.3f}({res[n]['err']:.3f}) [GFF {res[n]['gff']:.3f}]"
              for n in chans) + f"  dims Psi {dim[(1,0)]:.3f} [{time.time()-t0:.0f}s]", flush=True)


    def cli(argv):
        main(argv[1], int(argv[2]), float(argv[3]), float(argv[4]) if len(argv) > 4 else 1.0)
    return namespace(locals(), globals())


STEPS = {'run_on_defect': fuzzy.run_on_defect, 'core_strip': fuzzy.core_strip, 'run_o2defect': run_o2defect, 'o2_ed_precise': o2_ed_precise, 'analyze_on': analyze_on, 'on_tables': on_tables, 'contact_alpha': contact_alpha, 'pole_ope': pole_ope}
REPORT = [('on_tables',), ('contact_alpha',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
