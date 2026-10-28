"""Easy plane: is the monodromy transported onto the O(2) half-twist, couplings included?

The O(3) half-twist on the (1,2) plane, followed along the retuned trajectory with g < 0. Results: the defect levels
along the flow, the endpoint against an independently tuned O(2) half-twist, the bulk-to-defect coefficient beta
along the flow and its extrapolation, the universal crossover in Y = |g| N^{y_t/2}, and O(4) -> O(3).

usage: python easy_plane.py <step> [args]      (python easy_plane.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def dflow():
    """The O(3) -> XY / Ising flow in the presence of a monodromy defect on the (1,2) pair (ED, GZWH O(3)).

The defect is the Aharonov-Bohm flux of twist/onmodel.py: the charged flavours psi_+- see a flux alpha through the
poles, so a charge-Q state picks up exp(2 pi i Q alpha) around the line, and sectors are labelled by
(Q, K = L_z - alpha Q, parity P of flavour 3).  W = sum_orbitals (n1 + n2 - 2 n3) commutes with the twist.
The defect does not move the bulk critical surface, so each point (N, g) uses the bulk Delta_phi-tuned h_c(N, g) and
the bulk unit u (descendant gap of the light component) of partB_const_N*.json (gf_o3_B_N4.json for N = 4).
Defect dimensions are (E - E_0^D)/u with E_0^D the lowest state of the neutral K = 0, P = + sector.
UV: O(3) monodromy defect with one twisted pair. g < 0: the light charged components are the XY order parameter, so
the defect flows to the O(2) monodromy defect at the same alpha. g > 0: the twisted components become heavy and the
Ising order parameter phi_3 is untwisted, so the defect should flow to a trivial line.
usage: python dflow.py N alpha [lite]       -> dflow_a{alpha}_N{N}.json (resumable)
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    HERE = 'flow'
    M = fuzzy.onmodel()

    CFG = dict(model='wf', Nv=3, U=0.264, pairs=[(1, 2)], twist=[1])
    SECTORS = {                                  # name: (Q, K, P), as twist/run_on_defect.sectors_for('o3wf')
        'N0': (0, 0, 1), 'N1': (0, 1, 1), 'N2': (0, 2, 1),
        'C0': (1, 0, 1), 'Cm': (1, -1, 1), 'C1': (1, 1, 1),
        'B0': (2, 0, 1), 'Bm': (2, -1, 1),
        'Z0': (0, 0, -1), 'Z1': (0, 1, -1), 'A0': (1, 0, -1)}
    LITE = ('N0', 'N1', 'C0', 'Cm', 'Z0', 'Z1')
    W = {'0': 0.0, '12+': 1.0, '12-': 1.0, '3': -2.0}


    class DFlow:
        def __init__(self, N, alpha, cache=True):
            self.N, self.alpha = N, alpha
            kw = dict(pairs=CFG['pairs'], twist=CFG['twist'])
            fl, orbs, koff, two, mu0 = M.build_terms(CFG['model'], CFG['Nv'], N, alpha, CFG['U'], 0.0, **kw)
            _, _, _, _, mu1 = M.build_terms(CFG['model'], CFG['Nv'], N, alpha, CFG['U'], 1.0, **kw)
            self.fl, self.orbs, self.koff, self.two, self.mu0, self.dmu = fl, orbs, koff, two, mu0, mu1 - mu0
            nf = [len(o) for o in orbs]
            off = np.concatenate([[0], np.cumsum(nf)]).astype(int)
            self.wmode = np.zeros(off[-1])
            for f, name in enumerate(fl.names):
                self.wmode[off[f]:off[f + 1]] = W[name]
            self.cache, self.use_cache = {}, cache

        def sector(self, name):
            if name in self.cache:
                return self.cache[name]
            Q, K, P = SECTORS[name]
            b = M.Basis(self.fl, self.orbs, self.koff, self.N, [Q], K, (P,))
            if b.dim == 0:
                return None
            H0 = M.build_H(b, self.two, self.mu0)
            st = b.states
            nh, nw = np.zeros(b.dim), np.zeros(b.dim)
            for k in range(len(self.dmu)):
                if self.dmu[k] == 0 and self.wmode[k] == 0:
                    continue
                occ = ((st >> np.int64(k)) & 1).astype(float)
                nh += self.dmu[k] * occ
                nw += self.wmode[k] * occ
            out = dict(dim=b.dim, H0=H0, nh=nh, nw=nw)
            if self.use_cache:
                self.cache[name] = out
            return out

        def levels(self, name, h, g, k):
            s = self.sector(name)
            if s is None:
                return []
            Hm = s['H0'] + sp.diags(h * s['nh'] + g * s['nw'])
            if Hm.shape[0] <= 800:
                return np.linalg.eigvalsh(Hm.toarray())[:k].tolist()
            return np.sort(sla.eigsh(Hm, k=min(k, Hm.shape[0] - 2), which='SA', tol=1e-10)[0]).tolist()


    def bulk_points(N):
        if N >= 5:
            P = store.get(os.path.join(HERE, f'partB_const_N{N}.json'))['points']
            out = [(p['g'], p['h_c'], p['u']) for p in P.values() if p.get('h_c') is not None]
        else:
            P = store.get(os.path.join(HERE, f'gf_o3_B_N{N}.json'))['points']
            out = [(p['g'], p['h_c'], p['u']) for p in P.values() if p.get('h_c') is not None]
        seen, res = set(), []
        for g, h, u in sorted(out, key=lambda z: (abs(z[0]), z[0])):
            key = round(g, 7)
            if key in seen:
                continue
            seen.add(key); res.append((g, h, u))
        return res


    def main(N, alpha, lite=False):
        path = os.path.join(HERE, f'dflow_a{alpha:.3f}_N{N}.json')
        res = store.get(path) if store.exists(path) else dict(N=N, alpha=alpha, points={})
        names = LITE if lite else tuple(SECTORS)
        df = DFlow(N, alpha, cache=not lite)
        for g, h, u in bulk_points(N):
            key = f"{g:+.5f}"
            if key in res['points']:
                continue
            t0 = time.time()
            E = {nm: df.levels(nm, h, g, 4) for nm in names}
            e0 = E['N0'][0]
            D = {nm: [(x - e0) / u for x in v] for nm, v in E.items()}
            res['points'][key] = dict(g=g, h_c=h, u=u, E=E, D=D, time=time.time() - t0)
            store.put(path, res)
            print(f"N={N} a={alpha} g={g:+.5f}: Psi_a {D['C0'][0]:.4f}  Psi_a-1 {D['Cm'][0]:.4f}  phi3 {D['Z0'][0]:.4f}  "
                  f"D {D['N1'][0]:.4f}  [{res['points'][key]['time']:.0f}s]", flush=True)


    def cli(argv):
        main(int(argv[1]), float(argv[2]), lite=len(argv) > 3 and argv[3] == 'lite')
    return namespace(locals(), globals())


@section
def dflow_analyze():
    """Analysis of the monodromy-defect flow (dflow.py, alpha = 1/2).
XY side (g < 0): defect levels against the O(2) model's own alpha = 1/2 defect at the same N (twist/on_results/o2wf).
Ising side (g > 0): shifts of the Ising levels on the defect relative to the bulk at the same (N, g) (partB_const),
and the frozen limit g = 3 (dflow_frozen_limit.json) against first-order perturbation theory for a trivial line
perturbed by c int eps: d_eps/d_sigma = C_eee/C_sse, d_dsigma(m=+-1)/d_sigma = 1 - D_eps/(2 D_sigma),
d_dsigma(m=0)/d_sigma = (2 D_sigma - D_eps + D_eps^2)/(2 D_sigma), from <D|e^{y.K} eps(pole) e^{w.P}|D>.
usage: python dflow_analyze.py      -> dflow_summary.json, dflow_report.txt, figs/dflow.png
"""
    import os
    import json
    import glob
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HERE = 'flow'
    SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
    SEQ = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                         'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9.5, 'lines.linewidth': 1.6})
    fN = lambda N: N ** 0.895 * (1 + 0.89 * N ** -0.85)
    SIG = 8.737
    DS, DE = 0.518149, 1.412625
    PRED = dict(eps=1.532435 / 1.0518537, dsig_m1=1 - DE / (2 * DS), dsig_m0=(2 * DS - DE + DE ** 2) / (2 * DS))


    def load():
        D = {}
        for f in sorted(store.glob(os.path.join(HERE, 'dflow_a0.500_N*.json'))):
            r = store.get(f)
            D[r['N']] = sorted(r['points'].values(), key=lambda p: p['g'])
        return D


    def bulk(N):
        if N < 5:
            return {}
        P = store.get(os.path.join(HERE, f'partB_const_N{N}.json'))['points']
        return {round(p['g'], 6): p for p in P.values() if p.get('h_c') is not None}


    def o2ref(N):
        p = os.path.join(HERE, '..', 'twist', 'on_results', f'o2wf_N{N}.json')
        if not store.exists(p):
            return None
        r = store.get(p)
        a0, a5 = r['alpha']['0.0000'], r['alpha'].get('0.5000')
        if a5 is None:
            return None
        u = a0['(1,)|1|()'][0] - a0['(1,)|0|()'][0]
        e0 = a5['(0,)|0|()'][0]
        return dict(Psi=(a5['(1,)|0|()'][0] - e0) / u, D=(a5['(0,)|1|()'][0] - e0) / u, Q2=(a5['(2,)|0|()'][0] - e0) / u,
                    neutral=(a5['(0,)|0|()'][1] - e0) / u)


    def main():
        D = load()
        lines, out = [], dict(xy={}, ising={}, uv={}, pred=PRED)
        for N, pts in D.items():
            p0 = [p for p in pts if p['g'] == 0][0]
            out['uv'][N] = dict(Psi=p0['D']['C0'][0], D=p0['D']['N1'][0], phi3=p0['D']['Z0'][0],
                                neutral=p0['D']['N0'][1] if 'N0' in p0['D'] and len(p0['D']['N0']) > 1 else None)
            neg = [p for p in pts if p['g'] < 0]
            if neg:
                pm = min(neg, key=lambda p: p['g'])
                o2 = o2ref(N)
                out['xy'][N] = dict(g=pm['g'], X=SIG * abs(pm['g']) * fN(N), Psi=pm['D']['C0'][0], D=pm['D']['N1'][0],
                                    Q2=pm['D']['B0'][0] if 'B0' in pm['D'] else None,
                                    neutral=pm['D']['N0'][1] if len(pm['D']['N0']) > 1 else None, o2=o2)
            B = bulk(N)
            pos = [p for p in pts if p['g'] > 0 and round(p['g'], 6) in B]
            if pos:
                pp = max(pos, key=lambda p: p['g'])
                b = B[round(pp['g'], 6)]
                out['ising'][N] = dict(g=pp['g'], X=SIG * pp['g'] * fN(N), d_sigma=pp['D']['Z0'][0] - b['P_l0'][0],
                                       d_dsigma_m1=pp['D']['Z1'][0] - b['P_l0'][0] - 1 if 'Z1' in pp['D'] else None,
                                       d_eps=(pp['D']['N0'][1] - b['S_l0'][0]) if len(pp['D']['N0']) > 1 else None)
        lines.append('UV (g = 0), O(3) defect with one twisted pair, alpha = 1/2 (bulk units):')
        for N, v in out['uv'].items():
            lines.append(f"  N={N}: Psi_1/2 {v['Psi']:.4f}  D {v['D']:.4f}  phi_3 {v['phi3']:.4f}  neutral {v['neutral']}")
        lines.append('XY side (most negative g) against the O(2) model at the same N:')
        for N, v in out['xy'].items():
            o = v['o2'] or {}
            lines.append(f"  N={N} (g {v['g']:+.3f}, X {v['X']:.2f}): Psi_1/2 {v['Psi']:.4f} [O(2) {o.get('Psi', float('nan')):.4f}]  "
                         f"Q=2 {v['Q2'] if v['Q2'] is None else round(v['Q2'], 4)} [O(2) {o.get('Q2', float('nan')):.4f}]  D {v['D']:.4f} "
                         f"[O(2) {o.get('D', float('nan')):.4f}]  neutral {v['neutral'] if v['neutral'] is None else round(v['neutral'], 4)} "
                         f"[O(2) {o.get('neutral', float('nan')):.4f}]")
        lines.append('Ising side (largest g of the scan): defect - bulk shifts:')
        for N, v in out['ising'].items():
            lines.append(f"  N={N} (g {v['g']:+.3f}, X {v['X']:.2f}): sigma {v['d_sigma']:+.4f}  dsigma(m=1) {v['d_dsigma_m1']}  eps {v['d_eps']}")
        F = store.get(os.path.join(HERE, 'dflow_frozen_limit.json'))
        out['frozen'] = F
        lines.append(f"Frozen limit g = 3 against the trivial line + c int eps (ratios {PRED}):")
        for N, v in sorted(F.items(), key=lambda z: int(z[0])):
            lines.append(f"  N={N}: d_sigma {v['sig']:+.4f}  ratios eps {v['eps'] / v['sig']:+.3f}  dsigma(m=0) {v['dsig_m0'] / v['sig']:+.3f}  "
                         f"dsigma(m=1) {v['dsig'] / v['sig']:+.3f}")
        store.put(os.path.join(HERE, 'dflow_summary.json'), out)
        store.put(os.path.join(HERE, 'dflow_report.txt'), '\n'.join(lines) + '\n')
        print('\n'.join(lines))
        # ---- figure
        fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.1))
        Ns = [N for N in sorted(D) if N >= 5]
        ax = axs[0]
        for i, N in enumerate(Ns):
            pts = [p for p in D[N] if p['g'] < 0]
            X = [SIG * abs(p['g']) * fN(N) for p in pts]
            col = SEQ[min(i, len(SEQ) - 1)]
            ax.plot(X, [p['D']['C0'][0] for p in pts], 'o-', color=col, ms=3, label=f'N = {N}')
            ax.plot(X, [p['D']['N0'][1] for p in pts if len(p['D']['N0']) > 1], 's--', color=col, ms=3)
            o2 = o2ref(N)
            if o2:
                ax.plot([max(X) * 1.6], [o2['Psi']], '*', color=col, ms=9, mec=INK, mew=0.5)
                ax.plot([max(X) * 1.6], [o2['neutral']], '*', color=col, ms=9, mec=INK, mew=0.5)
        ax.set_xscale('log'); ax.set_xlabel('X (easy plane, g < 0)')
        ax.set_ylabel('defect dimension (bulk units)')
        ax.text(0.03, 0.12, 'Ψ₁/₂ (circles)', transform=ax.transAxes, fontsize=8, color=INK2)
        ax.text(0.03, 0.93, 'lowest neutral defect level (squares)', transform=ax.transAxes, fontsize=8, color=INK2)
        ax.set_title('(a) O(3) → XY: onto the O(2) defect (stars)', loc='left', fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc='center left')
        ax = axs[1]
        for i, N in enumerate(Ns):
            B = bulk(N)
            pts = [p for p in D[N] if p['g'] > 0 and round(p['g'], 6) in B]
            X = [SIG * p['g'] * fN(N) for p in pts]
            col = SEQ[min(i, len(SEQ) - 1)]
            ax.plot(X, [p['D']['Z0'][0] - B[round(p['g'], 6)]['P_l0'][0] for p in pts], 'o-', color=col, ms=3, label=f'N = {N}')
            ax.plot(X, [p['D']['N0'][1] - B[round(p['g'], 6)]['S_l0'][0] for p in pts], 's--', color=col, ms=3)
        ax.axhline(0, color=INK2, lw=1, ls=':')
        ax.set_xscale('log'); ax.set_xlabel('X (easy axis, g > 0)')
        ax.set_ylabel('defect − bulk, Ising levels')
        ax.text(0.03, 0.9, 'ε (squares)', transform=ax.transAxes, fontsize=8, color=INK2)
        ax.text(0.03, 0.55, 'σ (circles)', transform=ax.transAxes, fontsize=8, color=INK2)
        ax.set_title('(b) O(3) → Ising: a residual line', loc='left', fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc='upper right')
        ax = axs[2]
        Nf = np.array(sorted(int(n) for n in F))
        for key, lab, col, pk in (('eps', 'ε / σ', CAT[0], 'eps'), ('dsig_m0', '∂σ (m = 0) / σ', CAT[1], 'dsig_m0'),
                                  ('dsig', '∂σ (m = ±1) / σ', CAT[2], 'dsig_m1')):
            ax.plot(1 / Nf, [F[str(N)][key] / F[str(N)]['sig'] for N in Nf], 'o-', color=col, label=lab)
            ax.plot([0], [PRED[pk]], 's', color=col, ms=8, mfc=SURF, mew=1.6)
        ax.axhline(0, color=INK2, lw=0.8, ls=':')
        ax.set_xlim(-0.01, 0.27); ax.set_xlabel('1 / N  (frozen twisted flavours, g = 3)')
        ax.set_ylabel('ratio of level shifts')
        ax.set_title('(c) approach to a trivial line perturbed by ε', loc='left', fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc='center right')
        fig.text(0.68, 0.005, 'squares at 1/N = 0: first-order prediction for c∫ε on a trivial line', fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=1.2)
        fig.savefig(figpath('dflow.png'), dpi=160)
        print('wrote figs/dflow.png')


    def cli(argv):
        main()
    return namespace(locals(), globals())


@section
def dflow_c():
    """Experiment C: do bulk RG flows map monodromy defects onto each other, couplings included?

Physics. A monodromy defect is a boundary condition for the order parameter, so when the bulk flows from a UV to an
IR fixed point the defect should flow to the IR monodromy defect of the same symmetry element. Spectra agreeing
at the endpoint is necessary, not sufficient. The defect OPE data must flow too.
Part 'ope'. Along the O(3) -> O(2) flow (twisted pair (1,2), easy plane, g < 0), the bulk-to-defect coupling of the
    light charged order parameter phi_+ = (V_1 + i V_2)/sqrt2 to the lowest twisted mode Psi_{1/2},
        b = |<Psi|n_phi(theta = pi/2)|0_D>| / |<phi|n_phi|0>|        (equator, Tang-Zhu definition),
    at each bulk-tuned (h_c(g), g), against the same quantity in the independently tuned O(2) model (o2wf).
Part 'o4'. A second flow: O(4) -> O(3) (W = n_1 + n_2 + n_3 - 3 n_4, g < 0) with the pair (1,2) twisted, so the IR
    defect should be the O(3) defect with one twisted pair (Tang-Zhu's R_z(pi)). Levels in bulk units at the endpoint
    are compared with the O(3) model's defect at the same N.
usage: python dflow_c.py ope N        -> dflow_c_ope_N{N}.json
       python dflow_c.py o4 N         -> dflow_c_o4_N{N}.json
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    HERE = 'flow'
    M = fuzzy.onmodel()
    from fuzzy import apply_cdc, wave_abs

    O3 = dict(model='wf', Nv=3, U=0.264, pairs=[(1, 2)], twist=[1]), {'0': 0.0, '12+': 1.0, '12-': 1.0, '3': -2.0}
    O4 = dict(model='wf', Nv=4, U=0.255, pairs=[(1, 2)], twist=[1]), {'0': 0.0, '12+': 1.0, '12-': 1.0, '3': 1.0, '4': -3.0}
    O2 = dict(model='wf', Nv=2, U=0.270, pairs=[(1, 2)], twist=[1]), {'0': 0.0, '12+': 0.0, '12-': 0.0}


    class Prob:
        """One (config, N, alpha) problem with H(h, g) = H0 + h n_h + g W on demand, sectors (q, K, parities)."""

        def __init__(self, spec, N, alpha):
            cfg, w = spec
            kw = dict(pairs=cfg['pairs'], twist=cfg['twist'])
            fl, orbs, koff, two, mu0 = M.build_terms(cfg['model'], cfg['Nv'], N, alpha, cfg['U'], 0.0, **kw)
            mu1 = M.build_terms(cfg['model'], cfg['Nv'], N, alpha, cfg['U'], 1.0, **kw)[4]
            self.N, self.fl, self.orbs, self.koff, self.two, self.mu0, self.dmu = N, fl, orbs, koff, two, mu0, mu1 - mu0
            self.off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
            self.wmode = np.zeros(self.off[-1])
            for f, nm in enumerate(fl.names):
                self.wmode[self.off[f]:self.off[f + 1]] = w[nm]
            self.cache = {}

        def sector(self, q, K, par):
            key = (tuple(q), K, tuple(par))
            if key not in self.cache:
                b = M.Basis(self.fl, self.orbs, self.koff, self.N, list(q), K, tuple(par))
                st = b.states
                occ = lambda vec: sum(vec[k] * ((st >> np.int64(k)) & 1) for k in np.flatnonzero(vec))
                self.cache[key] = dict(b=b, H0=M.build_H(b, self.two, self.mu0), nh=occ(self.dmu), nw=occ(self.wmode))
            return self.cache[key]

        def eig(self, q, K, par, h, g, k=1, vecs=False):
            s = self.sector(q, K, par)
            if s['b'].dim == 0:
                return (np.zeros(0), None) if vecs else np.zeros(0)
            H = s['H0'] + sp.diags(h * s['nh'] + g * s['nw'])
            if H.shape[0] <= 1500:
                w, v = np.linalg.eigh(H.toarray()); w, v = w[:k], v[:, :k]
            else:
                w, v = sla.eigsh(H, k=min(k, H.shape[0] - 2), which='SA', tol=1e-11)
                o = np.argsort(w); w, v = w[o], v[:, o]
            return (w, v) if vecs else w

        def vplus_K0(self, vec, src, dst, theta=np.pi / 2):
            """(V_1 + i V_2)/sqrt2 restricted to Delta K = 0 at (theta, phi = 0) acting on vec (sector src) -> sector dst."""
            s = (self.N - 1) / 2
            names = self.fl.names
            f0, fp, fm = names.index('0'), names.index('12+'), names.index('12-')
            out = np.zeros(dst['b'].dim, dtype=complex)
            for (fa, fb) in ((f0, fm), (fp, f0)):
                for ja, ma in enumerate(self.orbs[fa]):
                    for jb, mb in enumerate(self.orbs[fb]):
                        if (ja + self.koff[fa]) - (jb + self.koff[fb]) != 0:
                            continue
                        c = wave_abs(s, ma, theta) * wave_abs(s, mb, theta)
                        idx, val = apply_cdc(src['b'].states, vec, int(self.off[fa] + ja), int(self.off[fb] + jb), dst['b'].states)
                        np.add.at(out, idx, c * val)
            return out


    def bulk_points(name, N):
        fn = {'o3': f'partB_const_N{N}.json', 'o4': f'gf_o4a_B_N{N}.json'}[name]
        P = store.get(os.path.join(HERE, fn))['points']
        pts = sorted({round(p['g'], 6): (p['g'], p['h_c'], p['u']) for p in P.values() if p.get('h_c') is not None}.values())
        return [p for p in pts if p[0] <= 0]


    def b_coupling(prob_bulk, prob_def, h, g, par):
        """b = |<Psi|n|0_D>| / |<phi|n|0>| at the equator; also the levels used."""
        out = {}
        for lab, P in (('bulk', prob_bulk), ('def', prob_def)):
            E0, V0 = P.eig([0], 0, par, h, g, 1, vecs=True)
            E1, V1 = P.eig([1], 0, par, h, g, 1, vecs=True)
            w = P.vplus_K0(V0[:, 0], P.sector([0], 0, par), P.sector([1], 0, par))
            out[lab] = dict(amp=float(abs(np.vdot(V1[:, 0], w))), gap=float(E1[0] - E0[0]))
        return out['def']['amp'] / out['bulk']['amp'], out


    def mode_ope(N):
        t0 = time.time()
        res = dict(N=N, flow={}, o2={})
        pb, pd = Prob(O3, N, 0.0), Prob(O3, N, 0.5)
        for g, h, u in bulk_points('o3', N):
            b, det = b_coupling(pb, pd, h, g, (1,))
            res['flow'][f'{g:+.5f}'] = dict(g=g, h=h, u=u, b=b, Psi=det['def']['gap'] / u)
            print(f'O(3) N={N} g={g:+.4f}: b_phi_Psi = {b:.4f}  Psi {det["def"]["gap"] / u:.4f}  [{time.time() - t0:.0f}s]', flush=True)
        hc2 = store.get(os.path.join(HERE, '..', 'twist', 'on_results', f'o2wf_N{N}.json'))['h_c']
        qb, qd = Prob(O2, N, 0.0), Prob(O2, N, 0.5)
        b2, det2 = b_coupling(qb, qd, hc2, 0.0, ())
        res['o2'] = dict(h_c=hc2, b=b2)
        print(f'O(2) model N={N}: b_phi_Psi = {b2:.4f}', flush=True)
        store.put(os.path.join(HERE, f'dflow_c_ope_N{N}.json'), res)


    def levels_table(P, h, g, u, secs):
        E0 = P.eig(*secs['vac'], h, g, 1)[0]
        return {k: [(e - E0) / u for e in P.eig(*v, h, g, 2)] for k, v in secs.items() if k != 'vac'}


    def mode_o4(N):
        t0 = time.time()
        res = dict(N=N, flow={})
        pd = Prob(O4, N, 0.5)
        s4 = {'vac': ([0], 0, (1, 1)), 'D': ([0], 1, (1, 1)), 'Psi_a': ([1], 0, (1, 1)), 'Psi_a-1': ([1], -1, (1, 1)),
              'Psi_a+1': ([1], 1, (1, 1)), 'phi3_mode': ([0], 0, (-1, 1)), 'phi4_mode': ([0], 0, (1, -1))}
        for g, h, u in bulk_points('o4', N):
            res['flow'][f'{g:+.5f}'] = dict(g=g, h=h, u=u, levels=levels_table(pd, h, g, u, s4))
            L = res['flow'][f'{g:+.5f}']['levels']
            print(f"O(4) N={N} g={g:+.4f}: " + '  '.join(f'{k} {v[0]:.3f}' for k, v in L.items()) + f'  [{time.time() - t0:.0f}s]', flush=True)
        # target: the O(3) one-pair defect at its own bulk critical point, bulk units of its own light field
        R3 = store.get(os.path.join(HERE, '..', 'twist', 'on_results', f'o3wf_N{N}.json'))
        b3 = R3['alpha']['0.0000']
        u3 = b3['(1,)|1|(1,)'][0] - b3['(1,)|0|(1,)'][0]
        p3 = Prob(O3, N, 0.5)
        s3 = {'vac': ([0], 0, (1,)), 'D': ([0], 1, (1,)), 'Psi_a': ([1], 0, (1,)), 'Psi_a-1': ([1], -1, (1,)),
              'Psi_a+1': ([1], 1, (1,)), 'phi3_mode': ([0], 0, (-1,))}
        res['o3_target'] = levels_table(p3, R3['h_c'], 0.0, u3, s3)
        print('O(3) one-pair defect target: ' + '  '.join(f'{k} {v[0]:.3f}' for k, v in res['o3_target'].items()), flush=True)
        store.put(os.path.join(HERE, f'dflow_c_o4_N{N}.json'), res)


    def cli(argv):
        (mode_ope if argv[1] == 'ope' else mode_o4)(int(argv[2]))
    return namespace(locals(), globals())


@section
def dflow_c_endpoint():
    """Calculation 3: the easy-plane endpoint as continuum defect data.

Adds N = 9 (O(3)) and N = 9, 10 (O(2) model) to the experiment-C coupling series b = |<Psi|n_phi(pi/2)|0_D>|/|<phi|n_phi|0>|
and the Psi_1/2 level (bulk units), only at the UV point g = 0 and the endpoint g = -0.09 (dflow_c.py scans every g).
Then extrapolates each series in N and compares the flowed endpoint with the O(2) defect and with Tang-Zhu's 0.859.
usage: python dflow_c_endpoint.py run      -> dflow_c_endpoint.json
       python dflow_c_endpoint.py report
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    C = dflow_c()

    HERE = 'flow'
    OUT = os.path.join(HERE, 'dflow_c_endpoint.json')


    def run():
        t0 = time.time()
        R = store.get(OUT) if store.exists(OUT) else {}
        N = 9
        pts = {round(g, 5): (g, h, u) for g, h, u in C.bulk_points('o3', N)}
        pb, pd = C.Prob(C.O3, N, 0.0), C.Prob(C.O3, N, 0.5)
        for gk in (-0.09, 0.0):
            if gk not in pts:
                print(f'N={N}: g = {gk} not stored', flush=True)
                continue
            g, h, u = pts[gk]
            b, det = C.b_coupling(pb, pd, h, g, (1,))
            R[f'o3_N{N}_g{g:+.3f}'] = dict(N=N, g=g, h=h, u=u, b=b, Psi=det['def']['gap'] / u)
            print(f'O(3) N={N} g={g:+.3f}: b {b:.4f}  Psi {det["def"]["gap"] / u:.4f}  [{time.time() - t0:.0f}s]', flush=True)
            store.put(OUT, R)
        del pb, pd
        for N in (9, 10):
            hc2 = store.get(os.path.join(HERE, '..', 'twist', 'on_results', f'o2wf_N{N}.json'))['h_c']
            qb, qd = C.Prob(C.O2, N, 0.0), C.Prob(C.O2, N, 0.5)
            b2, det2 = C.b_coupling(qb, qd, hc2, 0.0, ())
            R[f'o2_N{N}'] = dict(N=N, h_c=hc2, b=b2)
            print(f'O(2) model N={N}: b {b2:.4f}  [{time.time() - t0:.0f}s]', flush=True)
            store.put(OUT, R)


    def report():
        R = store.get(OUT)
        ser = {'O(3), g = 0': {}, 'O(3) endpoint, g = -0.09': {}, 'O(2) model': {}}
        for N in (6, 7, 8):
            F = store.get(os.path.join(HERE, f'dflow_c_ope_N{N}.json'))
            ser['O(3) endpoint, g = -0.09'][N] = F['flow']['-0.09000']['b']
            g0 = [v for v in F['flow'].values() if abs(v['g']) < 1e-9]
            if g0:
                ser['O(3), g = 0'][N] = g0[0]['b']
            ser['O(2) model'][N] = F['o2']['b']
        for k, v in R.items():
            if k.startswith('o3'):
                ser['O(3) endpoint, g = -0.09' if v['g'] < -0.05 else 'O(3), g = 0'][v['N']] = v['b']
            else:
                ser['O(2) model'][v['N']] = v['b']
        for name, s in ser.items():
            Ns = np.array(sorted(s)); b = np.array([s[n] for n in Ns])
            ext = {}
            for p in (0.5, 1.0):
                A = np.stack([np.ones_like(Ns, float), Ns ** -p], 1)
                c, *_ = np.linalg.lstsq(A[-3:], b[-3:], rcond=None)
                ext[p] = c[0]
            print(f'{name:26s} ' + '  '.join(f'N={n} {x:.4f}' for n, x in zip(Ns, b)) +
                  f'  | extrapolated (last 3, N^-1/2, N^-1): {ext[0.5]:.4f}, {ext[1.0]:.4f}')
        print('Tang-Zhu (O(2) half-twist, extrapolated): 0.859;  free value sqrt(2/pi) = 0.798')


    def cli(argv):
        run() if argv[1] == 'run' else report()
    return namespace(locals(), globals())


@section
def crossover_collapse():
    """Crossover scaling of the defect spectrum along the easy-plane branch (O(3) half-twist -> O(2) half-twist).

For each channel n and size N, the normalized crossover f_n(x; N) = [D_n(g) - D_n(0)] / [D_n(IR) - D_n(0)], with D_n in
bulk units (dflow_a0.500_N*.json), D_n(IR) the most negative g available, and x = |g| N^{y/2}.  If the defect follows a
single relevant bulk coupling, f_n collapses for y = y_t = 3 - Delta_t = 1.7905 (Delta_t = 1.20954, Chester et al.).
The exponent y is fitted by minimizing the spread of f_n across N on a common log-grid of x, which tests the transport
claim without any input from the endpoint comparison.  Bootstrap errors come from leaving out one size.
Only sizes whose most negative g reaches f > 0.97 at the largest common x are used for the IR value.
usage: python crossover_collapse.py           -> crossover_collapse.json
"""
    import os
    import json
    import numpy as np
    from scipy.optimize import minimize_scalar

    HERE = 'flow'
    CHAN = {'charge 2 (B0)': ('B0', 0), 'displacement (N1)': ('N1', 0), 'charge 1 (C0)': ('C0', 0),
            'lowest neutral (N0[1])': ('N0', 1)}
    YT = 3 - 1.20954


    def load():
        raw = {}
        for N in range(4, 10):
            d = store.get(os.path.join(HERE, f'dflow_a0.500_N{N}.json'))['points']
            for p in d.values():
                if p['g'] > 1e-12:
                    continue
                E, u = p['E'], p['u']
                for nm, (s, i) in CHAN.items():
                    if s in E and len(E[s]) > i:
                        raw.setdefault(nm, {}).setdefault(N, []).append((abs(p['g']), (E[s][i] - E['N0'][0]) / u))
        return {nm: {N: np.array(sorted(v)) for N, v in dd.items()} for nm, dd in raw.items()}


    def f_curves(dd, y, Ns):
        out = {}
        for N in Ns:
            a = dd[N]
            uv, ir = a[0, 1], a[-1, 1]
            out[N] = (a[:, 0] * N ** (y / 2), (a[:, 1] - uv) / (ir - uv))
        return out


    def spread(dd, y, Ns, xs):
        fc = f_curves(dd, y, Ns)
        vals = []
        for x in xs:
            row = [np.interp(np.log(x), np.log(fc[N][0][1:]), fc[N][1][1:]) for N in Ns
                   if fc[N][0][1] <= x <= fc[N][0][-1]]
            if len(row) >= 3:
                vals.append(np.var(row))
        return np.mean(vals)


    def main():
        data = load()
        res = {}
        for nm, dd in data.items():
            Ns = sorted(dd)
            xs = np.geomspace(0.004, 0.25, 25)
            opt = minimize_scalar(lambda y: spread(dd, y, Ns, xs), bounds=(1.0, 2.6), method='bounded')
            jack = [minimize_scalar(lambda y: spread(dd, y, [n for n in Ns if n != k], xs), bounds=(1.0, 2.6),
                                    method='bounded').x for k in Ns]
            fc = f_curves(dd, YT, Ns)
            tab = {x: [round(float(np.interp(np.log(x), np.log(fc[N][0][1:]), fc[N][1][1:])), 4) for N in Ns]
                   for x in (0.01, 0.02, 0.05, 0.1, 0.2)}
            res[nm] = dict(y_fit=opt.x, y_jack=jack, y_err=float(np.std(jack) * np.sqrt(len(jack) - 1)),
                           spread_at_yt=float(np.sqrt(spread(dd, YT, Ns, xs))), spread_at_fit=float(np.sqrt(opt.fun)),
                           amplitude={N: float(dd[N][-1, 1] - dd[N][0, 1]) for N in Ns}, f_at_yt=tab)
            print(f"{nm}: y_fit {opt.x:.3f} +- {res[nm]['y_err']:.3f} (y_t = {YT:.4f}); rms spread of f at y_t "
                  f"{res[nm]['spread_at_yt']:.4f}, at fit {res[nm]['spread_at_fit']:.4f}; UV->IR change "
                  + ' '.join(f'{N}:{v:+.4f}' for N, v in res[nm]['amplitude'].items()))
            for x, v in tab.items():
                print(f'    f(x={x}) over N={Ns[0]}-{Ns[-1]}: ' + ' '.join(f'{z:.3f}' for z in v))
        store.put(os.path.join(HERE, 'crossover_collapse.json'), res)


    def cli(argv):
        main()
    return namespace(locals(), globals())


@section
def beta_eps():
    """Bulk-to-defect coefficient of a free scalar with a 1/r^2 potential on a codimension-two defect, checked by direct
mode integration, and its epsilon expansion.

Claim: for the radial mode nu = sqrt(s^2 + v), the coefficient of (r r')^nu |y|^{-(d-2)-2nu} in the two-point function,
divided by the bulk normalization Gamma(d/2-1)/(4 pi^{d/2}), is b^2 = Gamma(nu + d/2 - 1)/(Gamma(d/2 - 1) Gamma(nu + 1)).
Check: G_s(r, r', y) = (1/2pi) int d^{d-2}k/(2pi)^{d-2} e^{iky} I_nu(k r<) K_nu(k r>) at small r, r' and large |y|.

Paper, App. A (eq:betanu, eq:betaeps): the mean-field coefficient and its O(eps) term. The 'numeric' column does
not converge (the oscillatory Fourier integral at r = 1e-3 defeats the quadrature) and checks nothing; the
check of eq:betanu that holds is beta_check3d.
"""
    from mpmath import mp, mpf, besseli, besselk, quad, gamma, pi, inf, besselj, psi, sqrt, log
    mp.dps = 30


    def b2_formula(nu, d):
        return gamma(nu + d / 2 - 1) / (gamma(d / 2 - 1) * gamma(nu + 1))


    def b2_numeric(nu, d, r=mpf('1e-3'), rp=mpf('2e-3'), y=mpf(1)):
        """Fourier integral in d-2 = 2 transverse-parallel... use d=4 (parallel dim 2) and d=3 (parallel dim 1)."""
        p = d - 2
        if p == 1:
            f = lambda k: besseli(nu, k * r) * besselk(nu, k * rp) * 2 * mp.cos(k * y) / (2 * pi)
        elif p == 2:
            f = lambda k: besseli(nu, k * r) * besselk(nu, k * rp) * k * besselj(0, k * y) / (2 * pi)
        G = quad(f, [0, 1, 10, 100, 1000, inf]) / (2 * pi)
        Nphi = gamma(d / 2 - 1) / (4 * pi ** (d / 2))
        return G / ((r * rp) ** nu * y ** (-(d - 2) - 2 * nu)) / Nphi

    def main():
        for d in (3, 4):
            for nu in (mpf('0.5'), mpf('0.37'), mpf('1.5')):
                print(f'd={d} nu={float(nu):.2f}: numeric {float(b2_numeric(nu, d)):.6f}  formula {float(b2_formula(nu, d)):.6f}')

        # epsilon expansion at |s| = 1/2: b^2 = 1 - (eps/2)(psi(3/2) - psi(1)) + O(eps^2)
        s = mpf('0.5')
        c1 = -(psi(0, s + 1) - psi(0, 1)) / 2
        print('O(eps) coefficient of b_1/2^2:', float(c1), ' -> b(eps=1) =', float(sqrt(1 + c1)), ' free d=3:', float(sqrt(b2_formula(s, 3))))
        for s in (mpf('0.5'), mpf('1.5'), mpf('2.5')):
            print(f'|s|={float(s)}: O(eps) b^2 = {float(1 - (psi(0, s + 1) - psi(0, 1)) / 2):.4f}  free d=3 b^2 = {float(b2_formula(s, 3)):.4f}')

    def cli(argv):
        mp.dps = 30
        with fuzzy.logged('analytic/beta_eps.log'):
            main()
    return namespace(locals(), globals())


@section
def beta_check3d():
    """d = 3 check of b^2(nu) = Gamma(nu+1/2)/(sqrt(pi) Gamma(nu+1)) through the closed form of the mode propagator.

Paper, App. A: the d = 3 check of eq:betanu through the Legendre-Q form of the mode propagator.
"""
    from mpmath import mp, mpf, besseli, besselk, quad, gamma, pi, inf, cos, legenq, sqrt
    mp.dps = 25

    def main():
        r, rp, y = mpf('0.3'), mpf('0.5'), mpf('1.0')
        chi = (r**2 + rp**2 + y**2) / (2 * r * rp)
        for nu in (mpf('0.5'), mpf('0.37'), mpf('1.3')):
            num = quad(lambda k: cos(k * y) * besseli(nu, k * r) * besselk(nu, k * rp), [0, 5, 20, 60, inf], maxdegree=10)
            closed = legenq(nu - mpf(1) / 2, 0, chi, type=3).real / (2 * sqrt(r * rp))
            # far-field coefficient: Q_{nu-1/2}(chi) -> sqrt(pi) Gamma(nu+1/2) / (Gamma(nu+1) (2 chi)^{nu+1/2})
            big = mpf(10) ** 6
            ratio = legenq(nu - mpf(1) / 2, 0, big, type=3).real * (2 * big) ** (nu + mpf(1) / 2) / (sqrt(pi) * gamma(nu + mpf(1) / 2) / gamma(nu + 1))
            b2 = 4 * pi * (1 / (4 * pi**2)) * sqrt(pi) * gamma(nu + mpf(1) / 2) / gamma(nu + 1)
            print(f'nu={float(nu)}: integral {float(num):.10f}  closed form {float(closed):.10f}  far-field ratio {float(ratio):.8f}  '
                  f'b^2 {float(b2):.6f} vs formula {float(gamma(nu + mpf(1)/2) / (sqrt(pi) * gamma(nu + 1))):.6f}')

    def cli(argv):
        mp.dps = 25
        with fuzzy.logged('analytic/beta_check3d.log'):
            main()
    return namespace(locals(), globals())


@section
def sunset_tensor():
    """Sunset tensor structure for V = (lam/4!)(phi^2)^2 with diagonal, component-dependent propagators G_a.
Claim: Sigma_a = (lam^2/18) G_a [sum_c G_c^2 + 2 G_a^2].

Paper, App. A: the tensor structure of the sunset self-energy Sigma_a.
"""
    import sympy as sp


    def main():
        N = 5
        G = sp.symbols('G0:%d' % N)
        lam = sp.symbols('lam')
        d = lambda i, j: 1 if i == j else 0
        V = lambda a, b, c, e: lam/3*(d(a, b)*d(c, e) + d(a, c)*d(b, e) + d(a, e)*d(b, c))
        for a in range(N):
            S = sp.Rational(1, 6)*sum(V(a, b, c, e)**2*G[b]*G[c]*G[e] for b in range(N) for c in range(N) for e in range(N))
            claim = lam**2/18*G[a]*(sum(g**2 for g in G) + 2*G[a]**2)
            assert sp.expand(S - claim) == 0
        print('sunset tensor formula verified for N =', N)

    def cli(argv):
        with fuzzy.logged('analytic/open_questions/sunset_tensor.log'):
            main()
    return namespace(locals(), globals())


@section
def beta_eps2():
    """O(eps^2) bulk-to-defect coupling of the half twist: sunset residues on S^1 x H^3 (d = 4).

Mode s of the propagator obeys G~_s(nu) = 1/(nu^2 + s^2 - Sigma~_s(nu)); its large-chi coefficient is
Z_s = 1 + d Sigma~_s/d(nu^2) at nu = i|s|.  Sigma = (lam^2/18) G_a [sum_c G_c^2 + 2 G_a^2].
With G = K/(512 pi^6) shapes: K0 = 1/(c - cos)^3, Kt = cos^3(th/2)/(cosh^3(chi/2)(c - cos)^3), Km = cos(th/2)/(cosh(chi/2)(c-cos)^3).
I_X(s) = 4 pi int dchi sinh(chi) K_{X,s}(chi) [sinh(k chi) - k chi cosh(k chi)]/(2 k^3), k = |s|,
regularized by adding 2 pi^2 e^{-chi}/chi (the common log divergence).  Below chi = A the integrand is
replaced by its exact Taylor series, so no cancellation of chi^{-5} terms is done in floating point.

Paper, App. A (eq:betaeps2): the residue integrals and the bulk calibration J0.
"""
    import sympy as sp
    from mpmath import mp, mpf, quad, inf, pi, psi, euler
    mp.dps = 30
    x = sp.symbols('chi', positive=True)
    A = sp.Rational(1, 20)
    ORDER = 24


    def Kint(m):
        f = sp.exp(-abs(m)*x)/sp.sinh(x)
        dc = lambda g: sp.diff(g, x)/sp.sinh(x)
        return sp.pi*dc(dc(f))


    def kernel(kind, s):
        h = sp.Rational(1, 2)
        if kind == '0':
            return Kint(int(s))
        if kind == 't':
            return (3*(Kint(int(s - h)) + Kint(int(s + h))) + Kint(int(s - 3*h)) + Kint(int(s + 3*h)))/(8*sp.cosh(x/2)**3)
        if kind == 'm':
            return (Kint(int(s - h)) + Kint(int(s + h)))/(2*sp.cosh(x/2))


    def integrand(kind, s):
        k = abs(sp.nsimplify(s))
        w = -x**3/6 if k == 0 else (sp.sinh(k*x) - k*x*sp.cosh(k*x))/(2*k**3)
        return 4*sp.pi*sp.sinh(x)*kernel(kind, s)*w + 2*sp.pi**2*sp.exp(-x)/x


    def I(kind, s):
        f = integrand(kind, s)
        ser = sp.series(f, x, 0, ORDER).removeO()
        assert all(t.as_coeff_exponent(x)[1] >= 0 for t in sp.Add.make_args(sp.expand(ser))), 'singular term left'
        head = sp.integrate(ser, (x, 0, A))
        fn = sp.lambdify(x, f, 'mpmath')
        tail = quad(fn, [mpf(A), mpf('0.2'), 1, 3, 8, 20, 60, inf])
        return mpf(sp.N(head, 40)) + tail

    def main():
        print('calibration: J0(s) = I_0(s) - pi^2 (dg(1+s) + gamma) must not depend on s')
        J0s = []
        for s in [0, 1, 2, 3]:
            v = I('0', s)
            J0s.append(v - pi**2*(psi(0, 1 + s) + euler))
            print(f'  s = {s}: I_0 = {mp.nstr(v, 16)}   J0 = {mp.nstr(J0s[-1], 16)}', flush=True)
        J0 = J0s[0]
        print('twisted (t) and mixed (m) kernels, minus J0:')
        for s in [sp.Rational(1, 2), sp.Rational(3, 2), sp.Rational(5, 2)]:
            it, im = I('t', s), I('m', s)
            print(f'  s = {s}: I_t - J0 = {mp.nstr(it - J0, 16)}   I_m - J0 = {mp.nstr(im - J0, 16)}', flush=True)

    def cli(argv):
        mp.dps = 30
        with fuzzy.logged('analytic/open_questions/beta_eps2.log'):
            main()
    return namespace(locals(), globals())


@section
def beta_eps2_pslq():
    """High-precision sunset constants S_t(s) = I_t - J0, S_m(s) = I_m - J0 and integer-relation search.

Runs when first used (by beta_eps2_modes), as importing the research script did: the constants S_t, S_m at 50 digits and their integer relations, S_t(1/2) = 4 pi^2 (4 - 5 log 2) and S_m(1/2) = 4 pi^2 (1 - log 2).
"""
    import sympy as sp
    from mpmath import mp, mpf, quad, inf, pi, psi, euler, log, zeta, catalan, pslq
    B = globals()['beta_eps2']()   # was: import beta_eps2 as B
    mp.dps = 50
    B.ORDER = 34
    B.A = sp.Rational(1, 40)


    def I(kind, s):
        x = B.x
        f = B.integrand(kind, s)
        ser = sp.series(f, x, 0, B.ORDER).removeO()
        head = sp.integrate(ser, (x, 0, B.A))
        fn = sp.lambdify(x, f, 'mpmath')
        return mpf(sp.N(head, 60)) + quad(fn, [mpf(B.A), mpf('0.1'), mpf('0.5'), 1, 2, 4, 8, 16, 32, 64, inf])


    J0 = I('0', 0)
    print('J0 =', mp.nstr(J0, 40))
    print('J0 check s=1:', mp.nstr(I('0', 1) - pi**2*(psi(0, 2) + euler) - J0, 5))
    L2 = log(2)
    basis_names = ['1', 'pi^2', 'log2', 'pi^2 log2', 'zeta3', 'log2^2', 'G']
    basis = [mpf(1), pi**2, L2, pi**2*L2, zeta(3), L2**2, catalan]
    for s in [sp.Rational(1, 2), sp.Rational(3, 2), sp.Rational(5, 2)]:
        for kind in ['t', 'm']:
            v = I(kind, s) - J0
            rel = pslq([v] + basis, maxcoeff=10**6, maxsteps=10**6)
            print(f'S_{kind}({s}) = {mp.nstr(v, 35)}   relation: {rel}', flush=True)
    return namespace(locals(), globals())


@section
def beta_eps2_modes():
    """Closed forms of the sunset constants for higher modes, and the full O(eps^2) coupling of psi_{1/2}.
Here nt counts twisted REAL components (= 2 n_t in the paper, whose n_t counts twisted planes).

Paper, App. A (eq:betaeps2): the full O(eps^2) coefficient of log beta_1/2^2 and beta(eps = 1) = 0.8480 (O(2)), 0.8453 (O(3)). Loading this section first runs beta_eps2_pslq, whose output is printed but not logged.
"""
    import sympy as sp
    from mpmath import mp, mpf, pi, log, pslq, psi, euler, sqrt
    P = globals()['beta_eps2_pslq']()   # was: import beta_eps2_pslq as P
    mp.dps = 50

    def main():
        J0 = P.J0
        for s in [sp.Rational(3, 2), sp.Rational(5, 2)]:
            v = P.I('t', s) - J0
            rel = pslq([v/pi**2, mpf(1), log(2)], maxcoeff=10**9, maxsteps=10**7)
            print(f'S_t({s})/pi^2 = {mp.nstr(v/pi**2, 30)}   relation [v/pi^2, 1, log2]: {rel}', flush=True)

        # full O(eps^2) result for s = 1/2, v = 1/2
        L2 = log(2)
        def logbeta2(N, nt, eps):
            a1 = -(1 - L2)
            pure = pi**2/24 - mpf(1)/2
            cross = (nt + 2)*(pi**2/2 - 4)/(8*(N + 8))
            sun = ((nt + 2)*(4 - 5*L2) + (N - nt)*(1 - L2))/(N + 8)**2
            return a1*eps + (pure + cross + sun)*eps**2, (a1, pure, cross, sun)
        for name, N, nt in [('O(2) full twist', 2, 2), ('O(3) R_z(pi)', 3, 2), ('O(4), one pair', 4, 2)]:
            L, parts = logbeta2(N, nt, mpf(1))
            a1, pure, cross, sun = parts
            b2 = pure + cross + sun
            beta_lin = 1 + a1/2 + (b2/2 + a1**2/8)   # strict expansion of sqrt(exp(L)) to eps^2 at eps=1
            print(f'{name:16s}: eps^2 coeff of log beta^2 = {mp.nstr(b2, 8)} (pure {mp.nstr(pure, 6)}, cross {mp.nstr(cross, 6)}, sunset {mp.nstr(sun, 6)})'
                  f'   beta(eps=1) to O(eps^2) = {mp.nstr(beta_lin, 6)}')
        print('O(2) - O(3) at eps = 1:', mp.nstr((logbeta2(2, 2, 1)[1][2] + logbeta2(2, 2, 1)[1][3] - logbeta2(3, 2, 1)[1][2] - logbeta2(3, 2, 1)[1][3])/2, 6))

    def cli(argv):
        mp.dps = 50
        with fuzzy.logged('analytic/open_questions/beta_eps2_modes.log'):
            main()
    return namespace(locals(), globals())


@section
def dflow_fill_B0():
    """Fill the charge-two level B0 = (Q, K, P) = (2, 0, +) at one point of an existing lite dflow run.

dflow_a0.500_N9.json was run with `dflow.py 9 0.5 lite`, which skips the sectors N2, C1, B0, Bm and A0, so Fig. transport(a)
has no N_m = 9 charge-two point. This recomputes N0 (to check against the stored run) and B0 at the stored (g, h_c, u)
with the same DFlow class, and writes dflow_fill_B0_a{alpha}_N{N}.json next to it (a name the dflow_a*_N*.json globs of dflow_analyze.py and
crossover_collapse.py do not match); the original file is not modified.
usage: python dflow_fill_B0.py N alpha g        e.g. python dflow_fill_B0.py 9 0.5 -0.09
"""
    import os
    import sys
    import json
    import time
    dflow = globals()['dflow']()   # was: import dflow

    HERE = 'flow'


    def main(N, alpha, g):
        src = store.get(os.path.join(HERE, f'dflow_a{alpha:.3f}_N{N}.json'))
        key = f'{g:+.5f}'
        p = src['points'][key]
        h, u = p['h_c'], p['u']
        df = dflow.DFlow(N, alpha, cache=False)
        out = dict(N=N, alpha=alpha, g=p['g'], h_c=h, u=u, source=f'dflow_a{alpha:.3f}_N{N}.json', E={}, dims={}, time={})
        for nm in ('N0', 'B0'):
            t0 = time.time()
            s = df.sector(nm)
            out['dims'][nm] = s['dim']
            out['E'][nm] = df.levels(nm, h, p['g'], 4)
            out['time'][nm] = time.time() - t0
            print(f'{nm}: dim {s["dim"]}, E {out["E"][nm]}  [{out["time"][nm]:.0f}s]', flush=True)
        e0 = out['E']['N0'][0]
        out['N0_check'] = abs(e0 - p['E']['N0'][0])
        out['D'] = {nm: [(x - e0) / u for x in v] for nm, v in out['E'].items()}
        print(f"N0 ground vs stored run: |diff| = {out['N0_check']:.2e};  B0 gap = {out['D']['B0'][0]:.10f} (bulk units)", flush=True)
        store.put(os.path.join(HERE, f'dflow_fill_B0_a{alpha:.3f}_N{N}.json'), out)

    def cli(argv):
        main(int(argv[1]), float(argv[2]), float(argv[3]))
    return namespace(locals(), globals())


STEPS = {'dflow': dflow, 'dflow_analyze': dflow_analyze, 'dflow_c': dflow_c, 'dflow_c_endpoint': dflow_c_endpoint, 'crossover_collapse': crossover_collapse, 'beta_eps': beta_eps,
         'beta_check3d': beta_check3d, 'sunset_tensor': sunset_tensor, 'beta_eps2': beta_eps2, 'beta_eps2_modes': beta_eps2_modes, 'dflow_fill_B0': dflow_fill_B0}
REPORT = [('dflow_analyze',), ('dflow_c_endpoint', 'report'), ('crossover_collapse',), ('sunset_tensor',), ('beta_check3d',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
