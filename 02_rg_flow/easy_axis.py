"""Easy axis: is the monodromy erased, leaving a trivial line?

The O(3) half-twist with g > 0, where the twisted flavours become massive. Results: the level shifts in the frozen
limit and their origin in the regulator's local potential, the domain wall of a pinned spectator with and without
that potential, and the overlap g-function of the residual line, with two controls (phi_3 pinned in the same
Hamiltonian, and the Ising magnetic line of Zhou, Gaiotto, He and Zou).

usage: python easy_axis.py <step> [args]      (python easy_axis.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401
import easy_plane                                        # noqa: E402


@section
def dflow_frozen():
    """Frozen limit of the easy-axis defect flow: g = 3, where the twisted flavours are frozen out (dflow.py, alpha = 1/2).
Tunes the bulk h_c(N, g = 3) on the light component (Delta_phi of O(3)), then records the shifts of the Ising levels on
the defect relative to the bulk at the same couplings: sigma, dsigma(m = +-1), dsigma(m = 0), eps (all in units of the
bulk descendant gap u), as in dflow_frozen_limit.json for N = 4-8.
The Newton start is extrapolated from the stored h_c at smaller N (converged to ~1e-4); if Newton fails, the branch is
walked out from the largest g of partB_const_N{N}.json.
usage: python dflow_frozen.py N        -> adds/updates key N of dflow_frozen_limit.json
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    G = fuzzy.gflow()
    DF = easy_plane.dflow()

    HERE = 'flow'
    OUT = os.path.join(HERE, 'dflow_frozen_limit.json')
    GF = 3.0


    def tune(fm, N, F):
        Ns = sorted(int(k) for k in F if int(k) < N)
        h = [F[str(n)]['h_c'] for n in Ns[-3:]]
        guess = h[-1] + (h[-1] - h[-2]) * (h[-1] - h[-2]) / (h[-2] - h[-3]) if len(h) == 3 else h[-1]
        print(f'N={N}: Newton at g = {GF} from {guess:.6f}', flush=True)
        hc, it = fm.tune_newton(GF, guess, 'L', G.DPHI['o3'])
        if hc is not None:
            print(f'  converged in {it} iterations: h_c = {hc:.9f}', flush=True)
            return hc
        P = store.get(os.path.join(HERE, f'partB_const_N{N}.json'))['points'].values()
        prev = sorted([(p['g'], p['h_c']) for p in P if p.get('h_c') is not None and p['g'] > 0])[-2:]
        for g in [x for x in (0.35, 0.6, 1.0, 1.6, 2.4, GF) if x > prev[-1][0]]:
            (ga, ha), (gb, hb) = prev[-2], prev[-1]
            hc, _ = fm.tune_newton(g, hb + (hb - ha) / (gb - ga) * (g - gb), 'L', G.DPHI['o3'])
            if hc is None:
                hc, _ = fm.tune_scan(g, hb, 'L', G.DPHI['o3'], span=0.05, n=11)
            if hc is None:
                raise RuntimeError(f'lost the branch at g = {g}')
            prev.append((g, hc))
            print(f'  ladder g = {g}: h_c = {hc:.9f}', flush=True)
        return hc


    def main(N):
        t0 = time.time()
        F = store.get(OUT)
        fm = G.GFlow('o3', N, verbose=False)
        hc = tune(fm, N, F)
        e0 = fm.levels('S0', hc, GF, k=3)
        l0 = fm.levels('L0', hc, GF, k=3)
        l1 = fm.levels('L1', hc, GF, k=2)
        u = l1[0] - l0[0]
        bs, beps = (l0[0] - e0[0]) / u, (e0[1] - e0[0]) / u
        df = DF.DFlow(N, 0.5, cache=False)
        E = {nm: df.levels(nm, hc, GF, 3) for nm in ('N0', 'Z0', 'Z1', 'N1')}
        d = lambda nm, i=0: (E[nm][i] - E['N0'][0]) / u
        r = dict(h_c=hc, sig=d('Z0') - bs, dsig=d('Z1') - bs - 1, eps=d('N0', 1) - beps, N1=d('N1'), beps=beps,
                 dsig_m0=d('Z0', 1) - (l0[1] - e0[0]) / u)
        F = store.get(OUT)
        F[str(N)] = r
        store.put(OUT, F)
        print(f"N={N} g={GF}: bulk sigma {bs:.6f} eps {beps:.4f} dsigma(m=0) {(l0[1] - e0[0]) / u:.4f}  |  shifts sigma "
              f"{r['sig']:+.4f}  ratios eps {r['eps'] / r['sig']:+.3f}  dsigma(m=0) {r['dsig_m0'] / r['sig']:+.3f}  "
              f"dsigma(m=1) {r['dsig'] / r['sig']:+.3f}  [{time.time() - t0:.0f}s]", flush=True)
        Ns = np.array(sorted(int(k) for k in F))
        v = np.array([F[str(n)]['sig'] for n in Ns])
        print('sigma local exponents', dict(zip([f'{a}-{b}' for a, b in zip(Ns, Ns[1:])],
                                                np.round(np.log(v[1:] / v[:-1]) / np.log(Ns[1:] / Ns[:-1]), 3))))


    def cli(argv):
        main(int(argv[1]))
    return namespace(locals(), globals())


@section
def core_residue():
    """Task 4: the defect-core residue. Rescale the pole-localized psi_0 ordering potential of the half-twist.

The products in the GZWH Hamiltonian are kept as written, so their ordering gives a one-body term mu(m) for each
orbital. In the twisted problem psi_0 feels the twisted flavours' orbital sets through the gradient term, and its
one-body term deviates from the untwisted constant near the poles,
    delta_0(m) = mu_0^{alpha=1/2}(m) - mu_0^{alpha=0}(m)    (psi_0 is untwisted, so its orbitals agree).
We add (lam - 1) sum_m delta_0(m) n_{m0}, so lam = 1 is the model as used so far, lam = 0 removes the residue and
lam = 2 doubles it. psi_3 is untwisted too, and its delta vanishes identically (checked).

mode 'frozen': the easy-axis frozen limit of dflow (g = 3, h_c from dflow_frozen_limit.json), shifts of the Ising
    levels on the defect relative to the bulk at the same couplings, as in dflow_frozen.py.
mode 'pin': Codex's spectator pinning of the half-twist at the bulk critical point (twist/spectator_pinning/scan.py),
    strong-pinning domain-wall dimension Delta_+- = (E0_+- - E0_++)/gamma_B by exact projection.
usage: python core_residue.py frozen N lam1,lam2,...   -> core_frozen_N{N}.json
       python core_residue.py pin N lam1,lam2,...      -> core_pin_N{N}.json
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp

    HERE = 'flow'
    DF = easy_plane.dflow()
    G = fuzzy.gflow()
    M = DF.M


    def delta_mu(N, h=0.0):
        """psi_0 and psi_3 deviations of the ordering one-body term, twisted minus untwisted."""
        kw = dict(pairs=DF.CFG['pairs'], twist=DF.CFG['twist'])
        fl, orbs, _, _, mu_t = M.build_terms(DF.CFG['model'], DF.CFG['Nv'], N, 0.5, DF.CFG['U'], h, **kw)
        _, orbs_u, _, _, mu_u = M.build_terms(DF.CFG['model'], DF.CFG['Nv'], N, 0.0, DF.CFG['U'], h, **kw)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        offu = np.concatenate([[0], np.cumsum([len(o) for o in orbs_u])]).astype(int)
        out = {}
        for name in ('0', '3'):
            f = fl.names.index(name)
            assert np.allclose(orbs[f], orbs_u[f]), name
            out[name] = mu_t[off[f]:off[f + 1]] - mu_u[offu[f]:offu[f + 1]]
        cvec = np.zeros(off[-1])
        f0 = fl.names.index('0')
        cvec[off[f0]:off[f0 + 1]] = out['0']
        return out, cvec


    def diag_occ(states, cvec):
        d = np.zeros(len(states))
        for k in np.flatnonzero(cvec):
            d += cvec[k] * ((states >> np.int64(k)) & 1)
        return d


    class CoreFlow(DF.DFlow):
        def __init__(self, N):
            super().__init__(N, 0.5, cache=True)
            self.delta, self.cvec = delta_mu(N)

        def levels_lam(self, name, h, g, k, lam):
            s = self.sector(name)
            if 'nc' not in s:
                b = M.Basis(self.fl, self.orbs, self.koff, self.N, [DF.SECTORS[name][0]], DF.SECTORS[name][1],
                            (DF.SECTORS[name][2],))
                s['nc'] = diag_occ(b.states, self.cvec)
            Hm = s['H0'] + sp.diags(h * s['nh'] + g * s['nw'] + (lam - 1.0) * s['nc'])
            if Hm.shape[0] <= 800:
                return np.linalg.eigvalsh(Hm.toarray())[:k].tolist()
            return np.sort(DF.sla.eigsh(Hm, k=min(k, Hm.shape[0] - 2), which='SA', tol=1e-10)[0]).tolist()


    def mode_frozen(N, lams):
        t0 = time.time()
        F = store.get(os.path.join(HERE, 'dflow_frozen_limit.json'))[str(N)]
        hc, g = F['h_c'], 3.0
        fm = G.GFlow('o3', N, verbose=False)
        e0 = fm.levels('S0', hc, g, k=3); l0 = fm.levels('L0', hc, g, k=3); l1 = fm.levels('L1', hc, g, k=2)
        u = l1[0] - l0[0]
        bs, beps, bdm0 = (l0[0] - e0[0]) / u, (e0[1] - e0[0]) / u, (l0[1] - e0[0]) / u
        cf = CoreFlow(N)
        out = dict(N=N, h_c=hc, g=g, delta0=cf.delta['0'].tolist(), delta3_max=float(np.abs(cf.delta['3']).max()),
                   bulk=dict(sigma=bs, eps=beps, dsig_m0=bdm0), lam={})
        for lam in lams:
            E = {nm: cf.levels_lam(nm, hc, g, 3, lam) for nm in ('N0', 'Z0', 'Z1')}
            d = lambda nm, i=0: (E[nm][i] - E['N0'][0]) / u
            out['lam'][f'{lam:g}'] = dict(sig=d('Z0') - bs, dsig=d('Z1') - bs - 1, eps=d('N0', 1) - beps,
                                          dsig_m0=d('Z0', 1) - bdm0)
            r = out['lam'][f'{lam:g}']
            print(f"N={N} lam={lam:g}: sigma {r['sig']:+.5f}  eps {r['eps']:+.5f}  dsig(m=0) {r['dsig_m0']:+.5f}  "
                  f"dsig(m=1) {r['dsig']:+.5f}  [{time.time() - t0:.0f}s]", flush=True)
        store.put(os.path.join(HERE, f'core_frozen_N{N}.json'), out)


    def mode_pin(N, lams):
        SC = scan()
        t0 = time.time()
        S = SC.PinnedSector(N, 0.5, 0, 0)
        delta, cvec = delta_mu(N, h=S.bulk['h_c'])
        nc = sp.diags(diag_occ(S.basis.states, cvec))
        H0 = S.H0
        out = dict(N=N, gamma_B=S.bulk['gamma_B'], delta0=delta['0'].tolist(), lam={})
        for lam in lams:
            S.H0 = (H0 + (lam - 1.0) * nc).tocsr()
            r = {eta: S.infinite(eta)['E_core_subtracted'][0] for eta in (1, -1)}
            out['lam'][f'{lam:g}'] = dict(delta_wall=(r[-1] - r[1]) / S.bulk['gamma_B'], E_equal=r[1], E_opposite=r[-1])
            print(f"N={N} lam={lam:g}: Delta_+- = {out['lam'][f'{lam:g}']['delta_wall']:.6f}  [{time.time() - t0:.0f}s]",
                  flush=True)
        S.H0 = H0
        store.put(os.path.join(HERE, f'core_pin_N{N}.json'), out)


    def cli(argv):
        mode, N, lams = argv[1], int(argv[2]), [float(x) for x in argv[3].split(',')]
        (mode_frozen if mode == 'frozen' else mode_pin)(N, lams)
    return namespace(locals(), globals())


@section
def core_residue_analyze():
    """Analysis of task 4 (core_residue.py): how much of the defect-core residue is the psi_0 ordering potential.

Checks (stop with a message if they fail):
  * lam = 1 reproduces dflow_frozen_limit.json (frozen limit) and Codex's twist/spectator_pinning/results (pinning)
    to 1e-8;
  * psi_3 has no ordering deviation (delta3_max < 1e-12).
Frozen limit (g = 3, twisted flavours frozen out):
  * each shift s(lam) for sigma, eps, dsigma(m=0), dsigma(m=+-1) is fitted by a quadratic in lam; s(0) is the residue
    that does not come from the psi_0 potential, s'(0) its linear response;
  * the ratios of the linear responses s'/s'_sigma are compared with first-order c int eps on a trivial line,
    1.457, 1.562 and -0.363 (dflow_analyze.PRED), separately from the ratios of the full shifts at lam = 1.
Spectator pinning (bulk critical, strong pinning):
  * Delta_+-(lam) at N = 5-8; the surviving twist shift Delta_+-(lam = 0) - Delta_+-(untwisted control).
usage: python core_residue_analyze.py   -> core_residue_summary.json, core_residue_report.txt, figs/core_residue.png
"""
    import os
    import sys
    import json
    import glob
    import numpy as np

    HERE = 'flow'
    PIN = os.path.join(HERE, '..', 'twist', 'spectator_pinning', 'results')
    PRED = dict(eps=1.532435 / 1.0518537, dsig_m0=(2 * 0.518149 - 1.412625 + 1.412625 ** 2) / (2 * 0.518149),
                dsig=1 - 1.412625 / (2 * 0.518149))
    KEYS = ('sig', 'eps', 'dsig_m0', 'dsig')


    def fail(msg):
        sys.exit('CHECK FAILED: ' + msg)


    def frozen():
        ref = store.get(os.path.join(HERE, 'dflow_frozen_limit.json'))
        out = {}
        for fn in sorted(store.glob(os.path.join(HERE, 'core_frozen_N*.json'))):
            R = store.get(fn); N = R['N']
            if R['delta3_max'] > 1e-12:
                fail(f'psi_3 ordering deviation {R["delta3_max"]} at N={N}')
            if '1' in R['lam'] and str(N) in ref:
                for k in KEYS:
                    if abs(R['lam']['1'][k] - ref[str(N)][k]) > 1e-8:
                        fail(f'frozen lam=1 {k} at N={N}: {R["lam"]["1"][k]} vs {ref[str(N)][k]}')
            lams = np.array(sorted(float(x) for x in R['lam']))
            row = dict(lams=lams.tolist(), delta0_sum=float(np.sum(R['delta0'])))
            for k in KEYS:
                y = np.array([R['lam'][f'{l:g}'][k] for l in lams])
                c = np.polyfit(lams, y, min(2, len(lams) - 1))
                row[k] = dict(values=y.tolist(), at0=float(np.polyval(c, 0)), slope0=float(np.polyval(np.polyder(c), 0)),
                              curvature=float(c[0]) if len(c) == 3 else 0.0)
            for k in ('eps', 'dsig_m0', 'dsig'):
                row[k]['ratio_slope'] = row[k]['slope0'] / row['sig']['slope0']
                row[k]['ratio_full'] = row[k]['values'][list(lams).index(1.0)] / row['sig']['values'][list(lams).index(1.0)] \
                    if 1.0 in lams else None
            out[N] = row
        return out


    def pinning():
        out = {}
        for fn in sorted(store.glob(os.path.join(HERE, 'core_pin_N*.json'))):
            R = store.get(fn); N = R['N']
            ctl = os.path.join(PIN, f'N{N}_a0.000_Q0_K0.json')
            tw = os.path.join(PIN, f'N{N}_a0.500_Q0_K0.json')
            control = store.get(ctl)['delta_infinite'] if store.exists(ctl) else None
            if '1' in R['lam'] and store.exists(tw):
                ref = store.get(tw)['delta_infinite']
                if abs(R['lam']['1']['delta_wall'] - ref) > 1e-8:
                    fail(f'pinning lam=1 at N={N}: {R["lam"]["1"]["delta_wall"]} vs Codex {ref}')
            row = dict(control=control, lam={k: v['delta_wall'] for k, v in R['lam'].items()})
            if control is not None:
                row['shift'] = {k: v - control for k, v in row['lam'].items()}
            out[N] = row
        return out


    def main():
        F, P = frozen(), pinning()
        L = ['TASK 4: the psi_0 ordering potential of the half-twist core, rescaled by lam (1 = model, 0 = removed)']
        L.append(f'first-order trivial-line ratios: eps {PRED["eps"]:.3f}, dsigma(m=0) {PRED["dsig_m0"]:.3f}, '
                 f'dsigma(m=+-1) {PRED["dsig"]:.3f}')
        for N, r in sorted(F.items()):
            L.append(f'N={N}  sum delta_0 = {r["delta0_sum"]:+.4f}')
            for k in KEYS:
                q = r[k]
                extra = '' if k == 'sig' else f'   slope ratio {q["ratio_slope"]:+.3f}, full ratio {q["ratio_full"]:+.3f}'
                L.append(f'   {k:8s} lam=0 {q["at0"]:+.5f}  lam=1 {q["values"][r["lams"].index(1.0)]:+.5f}  '
                         f'd/dlam {q["slope0"]:+.5f}{extra}')
        for N, r in sorted(P.items()):
            s = ' '.join(f'lam={k}: {v:.6f}' for k, v in sorted(r['lam'].items(), key=lambda kv: float(kv[0])))
            sh = ' '.join(f'{k}: {v:+.6f}' for k, v in sorted(r.get('shift', {}).items(), key=lambda kv: float(kv[0])))
            L.append(f'pinning N={N}: {s}   control {r["control"]}   twist shift {sh}')
        store.put(os.path.join(HERE, 'core_residue_summary.json'), dict(frozen=F, pinning=P, pred=PRED))
        store.put(os.path.join(HERE, 'core_residue_report.txt'), '\n'.join(L) + '\n')
        print('\n'.join(L))
        if F or P:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(14, 4))
            for N, r in sorted(F.items()):
                ax[0].plot(r['lams'], r['sig']['values'], 'o-', ms=3, label=f'N={N}')
            ax[0].axhline(0, color='0.5', lw=0.8, ls=':'); ax[0].set_xlabel('lam'); ax[0].set_ylabel('sigma shift (g = 3)')
            ax[0].legend(fontsize=7, frameon=False)
            Ns = sorted(F)
            for k, lab in (('eps', 'eps'), ('dsig_m0', 'dsigma m=0'), ('dsig', 'dsigma m=+-1')):
                ax[1].plot([1 / n for n in Ns], [F[n][k]['ratio_slope'] for n in Ns], 'o-', label=f'{lab}, linear response')
                ax[1].plot([1 / n for n in Ns], [F[n][k]['ratio_full'] for n in Ns], 's--', label=f'{lab}, full at lam=1')
                ax[1].plot([0], [PRED[k]], 'k*')
            ax[1].set_xlabel('1/N'); ax[1].set_ylabel('ratio to sigma'); ax[1].legend(fontsize=6, frameon=False)
            for N, r in sorted(P.items()):
                ks = sorted(r.get('shift', {}), key=float)
                ax[2].plot([float(k) for k in ks], [r['shift'][k] for k in ks], 'o-', label=f'N={N}')
            ax[2].axhline(0, color='0.5', lw=0.8, ls=':'); ax[2].set_xlabel('lam')
            ax[2].set_ylabel('pinning: twisted - untwisted Delta_+-'); ax[2].legend(fontsize=7, frameon=False)
            fig.tight_layout(); fig.savefig(figpath('core_residue.png'), dpi=150)


    def cli(argv):
        main()
    return namespace(locals(), globals())


@section
def core_pin_full():
    """Task 4 follow-up: spectator pinning with the WHOLE twist-induced core potential stripped (all flavours,
twist/core_strip.py), against stripping psi_0 only (core_residue.py pin). Delta_+- = (E_+- - E_++)/gamma_B by the
strong-pinning projection; shift = twisted - untwisted control (Codex results).
usage: python core_pin_full.py N1,N2,...   -> core_pin_full.json"""
    import os
    import sys
    import json
    import numpy as np
    import scipy.sparse as sp
    HERE = 'flow'


    def cli(argv):
        SC = scan()
        CS = fuzzy.core_strip()
        delta_mu = core_residue().delta_mu
        diag_occ = core_residue().diag_occ
        PIN = os.path.join(HERE, '..', 'twist', 'spectator_pinning', 'results')
        out = {}
        for N in [int(x) for x in argv[1].split(',')]:
            S = SC.PinnedSector(N, 0.5, 0, 0)
            hc = S.bulk['h_c']
            _, cvec0 = delta_mu(N, h=hc)
            dev = CS.build('o3wf', N, 0.5, 1.0, hc)[5]
            H0 = S.H0
            ctl = store.get(os.path.join(PIN, f'N{N}_a0.000_Q0_K0.json'))['delta_infinite']
            row = dict(control=ctl, dev_sum={'psi0': float(cvec0.sum()), 'all': float(dev.sum())})
            for name, vec in (('model', None), ('psi0_stripped', cvec0), ('all_stripped', dev)):
                S.H0 = H0 if vec is None else (H0 - sp.diags(diag_occ(S.basis.states, vec))).tocsr()
                r = {eta: S.infinite(eta)['E_core_subtracted'][0] for eta in (1, -1)}
                row[name] = (r[-1] - r[1]) / S.bulk['gamma_B'] - ctl
            S.H0 = H0
            out[N] = row
            print(f"N={N}: twist shift of Delta_+-: model {row['model']:+.5f}, psi_0 stripped {row['psi0_stripped']:+.5f}, "
                  f"all stripped {row['all_stripped']:+.5f}   (sum dev: psi0 {row['dev_sum']['psi0']:+.4f}, all {row['dev_sum']['all']:+.4f})", flush=True)
        store.put(os.path.join(HERE, 'core_pin_full.json'), out)
    return namespace(locals(), globals())


@section
def gfunc_core():
    """g-function of the easy-axis line: is the residual defect of the frozen O(3) half-twist a trivial line?

In the frozen limit of the easy-axis flow (g = 3, dflow_frozen_limit.json) the twisted flavours are gapped out and the
residual shifts come from the singlet core potential delta_0(m) (core_residue.py: lambda = 0 removes > 99.5%).  That
potential is a local change of the Hamiltonian in the untwisted Fock space, so the defect it creates can end at a pole,
and the overlap method of Zhou, Gaiotto, He, Zou (arXiv:2401.00039; gate: twist/gfunc_ising.py) applies.
Defects (north, south): 0 = none, a = lam * delta_0 on the psi_0 orbitals of that hemisphere (m > 0 north, m < 0 south,
an equatorial m = 0 orbital split evenly).  With A^{abcd} = <phi^{cd}|phi^{ab}> (ground states, K = 0, Q = 0, P3 = +):
  g_a = (A^{a000}/A^{a0aa})^2,   2^{-2 Delta^{a0}} = A^{aa00}/(A^{a000} A^{aaa0}).
A trivial endpoint has g = 1 and Delta^{a0} = 0.  The script also checks that the core-only line reproduces the
spectral shifts of the full frozen defect (core_frozen_N{N}.json at the same lam).
usage: python gfunc_core.py N lam1,lam2,...   -> gfunc_core_N{N}.json
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    HERE = 'flow'
    G = fuzzy.gflow()
    CR = core_residue()
    M = G.M
    GF = 3.0


    def lowest(Hm, k):
        if Hm.shape[0] <= 1500:
            w, v = np.linalg.eigh(Hm.toarray())
            return w[:k], v[:, :k]
        w, v = sla.eigsh(Hm.tocsr(), k=k, which='SA', tol=1e-13)
        o = np.argsort(w)
        return w[o], v[:, o]


    def run(N, lams):
        t0 = time.time()
        hc = store.get(os.path.join(HERE, 'dflow_frozen_limit.json'))[str(N)]['h_c']
        gf = G.GFlow('o3', N, cache=['S0', 'L0', 'L1'], verbose=False)
        delta, _ = CR.delta_mu(N)
        d0 = np.asarray(delta['0'])
        s = (N - 1) / 2
        m = np.arange(N) - s
        wN = np.where(m > 0, 1.0, np.where(m == 0, 0.5, 0.0))
        f0 = gf.fl.names.index('0')
        off0 = int(np.concatenate([[0], np.cumsum([len(o) for o in gf.orbs])])[f0])
        sec = {}
        for name in ('S0', 'L0', 'L1'):
            Qs, K, P = gf.spec['sectors'][name]
            b = M.Basis(gf.fl, gf.orbs, gf.koff, N, list(Qs), K, tuple(P))
            occ0 = np.stack([((b.states >> np.int64(off0 + j)) & 1).astype(float) for j in range(N)], 1)
            sec[name] = dict(H=gf.H(name, hc, GF), cN=occ0 @ (d0 * wN), cS=occ0 @ (d0 * (1 - wN)))
        H = lambda name, lam, a, b: sec[name]['H'] + sp.diags(lam * (a * sec[name]['cN'] + b * sec[name]['cS']))
        E00, v00 = lowest(H('S0', 0, 0, 0), 2)
        l0 = lowest(H('L0', 0, 0, 0), 1)[0][0]
        u = lowest(H('L1', 0, 0, 0), 1)[0][0] - l0
        bulk_sig, bulk_eps = (l0 - E00[0]) / u, (E00[1] - E00[0]) / u
        ref = store.get(os.path.join(HERE, f'core_frozen_N{N}.json'))['lam']
        out = dict(N=N, h_c=hc, g=GF, u=u, delta0=d0.tolist(), equator=float(d0[m == 0][0]) if N % 2 else None, lam={})
        for lam in lams:
            Ea0, va0 = lowest(H('S0', lam, 1, 0), 1)
            Eaa, vaa = lowest(H('S0', lam, 1, 1), 2)
            E0a, v0a = lowest(H('S0', lam, 0, 1), 1)
            la = lowest(H('L0', lam, 1, 1), 1)[0][0]
            A = lambda x, y: float(abs(x[:, 0] @ y[:, 0]))
            a000, a0aa, aa00, a0a0 = A(va0, v00), A(va0, vaa), A(vaa, v00), A(v0a, v00)
            g = (a000 / a0aa) ** 2
            d_a0 = -np.log2(aa00 / (a000 * a0aa)) / 2
            sig, eps = (la - Eaa[0]) / u - bulk_sig, (Eaa[1] - Eaa[0]) / u - bulk_eps
            full = ref.get(f'{lam:g}')
            r = dict(g=g, Delta_a0=d_a0, A_a000=a000, A_a0aa=a0aa, A_aa00=aa00, A_0a00=a0a0,
                     sig=sig, eps=eps, full_sig=full['sig'] if full else None, full_eps=full['eps'] if full else None,
                     creation_energy=float((Ea0[0] - 0.5 * E00[0] - 0.5 * Eaa[0]) / u))
            out['lam'][f'{lam:g}'] = r
            cmp = f"  full defect sig {full['sig']:+.5f} eps {full['eps']:+.5f}" if full else ''
            print(f"N={N} lam={lam:g}: g {g:.6f}  Delta^a0 {d_a0:+.5f}  (E-route {r['creation_energy']:+.5f})  "
                  f"core-only sig {sig:+.5f} eps {eps:+.5f}{cmp}  [{time.time() - t0:.0f}s]", flush=True)
        store.put(os.path.join(HERE, f'gfunc_core_N{N}.json'), out)


    def pin_control(N, b=8.0):
        """Positive control in the same frozen-limit Hamiltonian: pin phi_3 = psi_0^+ psi_3 + h.c. at the pole orbitals
    (hopping -b, as flow/dmrg_pin.py).  The bulk is Ising here, so g should approach the magnetic-line value 0.602."""
        import itertools
        SC = scan()
        t0 = time.time()
        hc = store.get(os.path.join(HERE, 'dflow_frozen_limit.json'))[str(N)]['h_c']
        gf = G.GFlow('o3', N, cache=[], verbose=False)
        secs = [M.Basis(gf.fl, gf.orbs, gf.koff, N, [0], 0, (p,)) for p in (1, -1)]
        st = np.sort(np.concatenate([x.states for x in secs]))
        from types import SimpleNamespace
        bas = SimpleNamespace(states=st, dim=len(st), nmodes=secs[0].nmodes)
        diag = sum(((st >> np.int64(k)) & 1) * (hc * gf.dmu[k] + GF * gf.wmode[k])
                   for k in range(len(gf.dmu)) if gf.dmu[k] or gf.wmode[k])
        H0 = (M.build_H(bas, gf.two, gf.mu0) + sp.diags(diag)).tocsr()
        off = np.concatenate([[0], np.cumsum([len(o) for o in gf.orbs])]).astype(int)
        f0, f3 = gf.fl.names.index('0'), gf.fl.names.index('3')
        PN = SC.hopping(st, int(off[f0] + N - 1), int(off[f3] + N - 1))
        PS = SC.hopping(st, int(off[f0]), int(off[f3]))
        gs = lambda a, c: lowest((H0 - b * (a * PN + c * PS)).tocsr(), 1)[1]
        v00, vp0, vpp = gs(0, 0), gs(1, 0), gs(1, 1)
        A = lambda x, y: float(abs(x[:, 0] @ y[:, 0]))
        g = (A(vp0, v00) / A(vp0, vpp)) ** 2
        d = -np.log2(A(vpp, v00) / (A(vp0, v00) * A(vp0, vpp))) / 2
        print(f"N={N} pinned phi_3 (b={b:g}): g {g:.5f}  Delta^+0 {d:.4f}  dim {len(st)}  [{time.time() - t0:.0f}s]", flush=True)
        fn = os.path.join(HERE, f'gfunc_core_N{N}.json')
        out = store.get(fn) if store.exists(fn) else {}
        out['pin_control'] = dict(b=b, g=g, Delta_p0=d)
        store.put(fn, out)


    def cli(argv):
        if argv[1] == 'pin':
            pin_control(int(argv[2]))
        else:
            run(int(argv[1]), [float(x) for x in argv[2].split(',')])
    return namespace(locals(), globals())


@section
def gfunc_ising():
    """Gate for the overlap g-function: the 3d Ising magnetic line (Zhou, Gaiotto, He, Zou, arXiv:2401.00039).

Defect labels follow that paper, north then south: H^{ab} has pinning a at the north pole (m = +s) and b at the south
pole (m = -s); 0 = no defect, + = infinite field along z, imposed by fixing the pole orbital to spin up (fs.FastBasis
`fixed`), so every ground state lives in the same Fock space as the bulk vacuum.  With A^{abcd} = <phi^{cd}|phi^{ab}>,
  g_+ = (A^{+000} / A^{+0++})^2,            (their Eq. 40)
  2^{-2 Delta^{+0}} = A^{++00} / (A^{+000} A^{+++0}),   2^{-2 Delta^{+-}} = A^{++--} / (A^{+++-})^2   (their Eq. 53),
the second by north/south symmetry A^{+++0} = A^{+0++}.  Their result: g = 0.602(2), Delta^{+0} = 0.108(5),
Delta^{+-} = 0.84(5).  Model of Zhu et al. (V0, V1) = (4.75, 1), h = 3.16.
usage: python gfunc_ising.py N1,N2,...   -> gfunc_ising_results.json (appended)
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse.linalg as spla

    HERE = 'twist'
    from fuzzy import FastBasis
    from fuzzy import build_hamiltonian_fast
    from fuzzy import haldane_V

    VL, H = (4.75, 1.0), 3.16
    OUT = os.path.join(HERE, 'gfunc_ising_results.json')


    def ground(b, V):
        Hm = build_hamiltonian_fast(b, VL, H, V=V)
        if b.dim <= 3000:
            w, v = np.linalg.eigh(Hm.toarray())
            return w[0], v[:, 0]
        w, v = spla.eigsh(Hm, k=1, which='SA', tol=1e-13)
        return w[0], v[:, 0]


    def overlap(b1, v1, b2, v2):
        """<v2|v1> with both embedded in the full Fock space (bit-pattern match)."""
        idx, found = b2.lookup(b1.states)
        return float(np.dot(v2[idx[found]], v1[found]))


    def run(N):
        s, M0, n = (N - 1) / 2, N * (N - 1) // 2, N - 1           # half filling N_e = N, L_z = 0
        V = haldane_V(s, VL)
        up, dn = 0, 1                                              # spin index within an orbital (mode 2a + t)
        pin = lambda a, t: {2 * a + t: 1, 2 * a + (1 - t): 0}
        cfg = {'00': {}, '+0': pin(n, up), '++': {**pin(n, up), **pin(0, up)}, '+-': {**pin(n, up), **pin(0, dn)}}
        t0 = time.time()
        B, E, G = {}, {}, {}
        for k, fx in cfg.items():
            B[k] = FastBasis(s, N, M0, fixed=fx)
            E[k], G[k] = ground(B[k], V)
        A = lambda ab, cd: overlap(B[ab], G[ab], B[cd], G[cd])     # A^{ab cd} = <cd|ab>
        a_p000, a_p0pp, a_pp00, a_pppm = A('+0', '00'), A('+0', '++'), A('++', '00'), A('+0', '+-')
        a_ppmm = A('++', '+-')                                     # <+-|++> vanishes by symmetry only if... kept as a check
        g = (a_p000 / a_p0pp) ** 2
        d_p0 = -np.log2(abs(a_pp00 / (a_p000 * a_p0pp))) / 2
        rec = dict(N=N, dims={k: B[k].dim for k in B}, E={k: float(E[k]) for k in E},
                   A={'+000': a_p000, '+0++': a_p0pp, '++00': a_pp00, '+0+-': a_pppm, '+++-': a_ppmm}, g=g,
                   Delta_p0_ratio=d_p0, seconds=time.time() - t0)
        print(f"N={N}: g {g:.5f}  Delta^+0 (ratio) {d_p0:.4f}  |A+000| {abs(a_p000):.5f}  |A+0++| {abs(a_p0pp):.5f}  "
              f"dims {rec['dims']}  [{rec['seconds']:.0f}s]", flush=True)
        data = store.get(OUT) if store.exists(OUT) else {}
        data[str(N)] = rec
        store.put(OUT, data)


    def cli(argv):
        for N in [int(x) for x in argv[1].split(',')]:
            run(N)
    return namespace(locals(), globals())


@section
def scan():
    """Spectator pinning of the O(3) half-twist, with finite and infinite pole fields.

H(b, eta) = H_twist - b (O_N + eta O_S), eta = +1 or -1,
O_P = c^dagger_{P,0} c_{P,3} + h.c.

The physical density V_3(P) in onmodel's R^2=N convention is O_P/(4*pi).
Thus b is an orbital hopping energy; the coefficient of -V_3(P) is 4*pi*b.
The bulk h_c and gamma_B are read from flow/partB_const_N*.json at g=0.
The junction estimate is [E_+-(b)-E_++(b)]/gamma_B. At infinite b the
two occupied, aligned pole orbitals are projected exactly, and their common
energy -2*b is removed before diagonalization.
"""
    import argparse
    import gc
    import json
    import os
    from pathlib import Path
    import sys
    import time
    from types import SimpleNamespace

    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    HERE = 'twist/spectator_pinning'
    M = fuzzy.onmodel()
    from fuzzy import popcount

    CONFIG = dict(model="wf", Nv=3, U=0.264, pairs=[(1, 2)], twist=[1])
    DEFAULT_FIELDS = [0, .002, .005, .01, .02, .05, .1, .2, .5, 1, 2, 4, 8]


    def bulk_parameters(N):
        path = f"flow/partB_const_N{N}.json"
        p = store.get(path)["points"]["+0.00000"]
        return dict(h_c=p["h_c"], gamma_B=p["u"], source=str(path),
                    tuning="bulk light-order-parameter/descendant, target 0.518942")


    def hopping(states, a, b):
        """c^dagger_a c_b + h.c. in a sorted, invariant Fock basis."""
        rows, cols, vals = [], [], []
        for create, destroy in [(a, b), (b, a)]:
            good = (((states >> np.int64(destroy)) & 1) == 1) & (
                ((states >> np.int64(create)) & 1) == 0)
            source = np.flatnonzero(good)
            target = states[good] ^ np.int64((1 << create) | (1 << destroy))
            dest = np.searchsorted(states, target)
            assert np.all(dest < len(states)) and np.array_equal(states[dest], target)
            lo, hi = sorted((create, destroy))
            between = np.int64((1 << hi) - (1 << (lo + 1)))
            signs = 1.0 - 2.0 * (popcount(states[good] & between) % 2)
            rows.append(dest); cols.append(source); vals.append(signs)
        return sp.coo_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(states), len(states))).tocsr()


    def eigenpairs(H, k=3, v0=None):
        """Smallest eigenpairs, with explicit absolute residuals."""
        n = H.shape[0]
        if n <= 450:
            import scipy.linalg
            w, v = scipy.linalg.eigh(H.toarray(), subset_by_index=(0, min(k, n) - 1))
        else:
            rng = np.random.default_rng(20261022 + n)
            initial = rng.normal(size=n)
            if v0 is not None:
                initial = v0 + .01 * initial / np.linalg.norm(initial)
            w, v = sla.eigsh(H, k=min(k, n - 2), which="SA", tol=2e-12,
                             ncv=max(32, 5*k), v0=initial, maxiter=20000)
            order = np.argsort(w); w, v = w[order], v[:, order]
        residuals = np.linalg.norm(H @ v - v * w, axis=0)
        if max(residuals) > 2e-9:
            raise RuntimeError(f"Lanczos residual {max(residuals):.3g}")
        return w, v, residuals


    class PinnedSector:
        def __init__(self, N, alpha=.5, Q=0, K=0):
            self.N, self.alpha, self.Q, self.K = N, alpha, Q, K
            self.bulk = bulk_parameters(N)
            t0 = time.time()
            fl, orbs, koff, two, mu = M.build_terms(
                nm=N, alpha=alpha, h=self.bulk["h_c"], **CONFIG)
            sectors = [M.Basis(fl, orbs, koff, N, [Q], K, (par,)) for par in (1, -1)]
            self.parity_dims = [b.dim for b in sectors]
            states = np.sort(np.concatenate([b.states for b in sectors]))
            self.basis = SimpleNamespace(states=states, dim=len(states), nmodes=sectors[0].nmodes)
            self.H0 = M.build_H(self.basis, two, mu).tocsr()
            off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
            spectator = fl.names.index("3")
            self.poles = dict(N=(N-1, int(off[spectator]+N-1)),
                              S=(0, int(off[spectator])))
            self.ON = hopping(states, *self.poles["N"])
            self.OS = hopping(states, *self.poles["S"])
            self.spectator_parity = 1 - 2*(
                popcount(states & np.int64(((1 << N)-1) << int(off[spectator]))) % 2)
            self.checks = dict(hermiticity=float(abs(self.H0-self.H0.T).max()),
                               pole_commutator=float(abs(self.ON@self.OS-self.OS@self.ON).max()))
            assert self.checks["hermiticity"] < 1e-11
            assert self.checks["pole_commutator"] < 1e-14
            self.build_seconds = time.time()-t0

        def H(self, b, eta):
            return self.H0 - b * (self.ON + eta*self.OS)

        def projector(self, eta):
            """Isometry occupying (c_N0+ c_N3)/sqrt(2), (c_S0+eta c_S3)/sqrt(2).

        Each column is c_N^dagger c_S^dagger |free>; the sequential creation
        below retains the fermion signs across all intervening modes.
        """
            states = self.basis.states
            n0, n3 = self.poles["N"]; s0, s3 = self.poles["S"]
            pole_mask = sum(1 << p for p in (n0, n3, s0, s3))
            reference = np.int64((1 << n0) | (1 << s0))
            free = states[(states & np.int64(pole_mask)) == reference] ^ reference
            rows, cols, vals = [], [], []
            for j, free_state in enumerate(free):
                for south, weight_s in [(s0, 1), (s3, eta)]:
                    for north in (n0, n3):
                        st = int(free_state)
                        sign = (-1) ** ((st & ((1 << south)-1)).bit_count())
                        st |= 1 << south
                        sign *= (-1) ** ((st & ((1 << north)-1)).bit_count())
                        st |= 1 << north
                        i = int(np.searchsorted(states, st))
                        assert i < len(states) and states[i] == st
                        rows.append(i); cols.append(j); vals.append(.5*weight_s*sign)
            P = sp.coo_matrix((vals, (rows, cols)), shape=(len(states), len(free))).tocsr()
            identity_error = float(abs(P.T@P-sp.eye(len(free))).max())
            alignment_error = float(abs((self.ON+eta*self.OS)@P-2*P).max())
            assert identity_error < 1e-14 and alignment_error < 1e-14
            return P, dict(isometry_error=identity_error, alignment_error=alignment_error)

        def finite(self, b, eta, k=3, v0=None):
            t0 = time.time()
            w, v, residuals = eigenpairs(self.H(b, eta), k=k, v0=v0)
            ground = v[:, 0]
            result = dict(b=float(b), physical_density_field=float(4*np.pi*b),
                          E=w.tolist(), residuals=residuals.tolist(),
                          pole_N=float(ground @ (self.ON @ ground)),
                          pole_S=float(ground @ (self.OS @ ground)),
                          seconds=time.time()-t0)
            return result, ground

        def infinite(self, eta, k=3):
            t0 = time.time()
            P, checks = self.projector(eta)
            H = (P.T@self.H0@P).tocsr()
            w, v, residuals = eigenpairs(H, k=k)
            return dict(E_core_subtracted=w.tolist(), dim=H.shape[0],
                        residuals=residuals.tolist(), seconds=time.time()-t0, **checks)


    def atomic_json(path, result):
        store.put(path, result)


    def run(N, fields=DEFAULT_FIELDS, alpha=.5, Q=0, K=0):
        path = os.path.join(HERE, "results", f"N{N}_a{alpha:.3f}_Q{Q}_K{K}.json")
        result = store.get(path) if store.exists(path) else dict(
            N=N, alpha=alpha, Q=Q, K=K, config=CONFIG,
            convention="H=H_twist-b*(O_N+eta*O_S); physical density V3=O/(4*pi)",
            finite={}, infinite={})
        if all(f"{b:.8g}" in result["finite"] for b in fields) and len(result["infinite"]) == 2:
            print(f"N={N} Q={Q} K={K}: already complete", flush=True)
            return
        S = PinnedSector(N, alpha, Q, K)
        result.update(bulk=S.bulk, dim=S.basis.dim, parity_dims=S.parity_dims,
                      build_seconds=S.build_seconds, checks=S.checks)
        print(f"N={N} Q={Q} K={K}: dim {S.basis.dim}, H nnz {S.H0.nnz}, "
              f"build {S.build_seconds:.1f}s", flush=True)
        # The exact infinite-field projection also controls the finite-b plateau.
        for eta, label in [(1, "equal"), (-1, "opposite")]:
            if label not in result["infinite"]:
                result["infinite"][label] = S.infinite(eta)
                atomic_json(path, result)
        result["delta_infinite"] = (result["infinite"]["opposite"]["E_core_subtracted"][0]-
                                     result["infinite"]["equal"]["E_core_subtracted"][0])/S.bulk["gamma_B"]
        atomic_json(path, result)
        print(f"  b=inf: Delta_wall={result['delta_infinite']:.9f}", flush=True)
        previous = {1: None, -1: None}
        for b in sorted(fields):
            key = f"{b:.8g}"
            if key in result["finite"]:
                continue
            point = {}
            for eta, label in [(1, "equal"), (-1, "opposite")]:
                if b == 0 and label == "opposite":
                    point[label] = dict(point["equal"])
                else:
                    point[label], previous[eta] = S.finite(b, eta, v0=previous[eta])
                projected = result["infinite"][label]["E_core_subtracted"][0]
                shifted = point[label]["E"][0] + 2*b
                if shifted > projected + 1e-8:
                    raise RuntimeError("finite-field energy exceeds the projected variational bound")
                point[label]["distance_to_projected_energy"] = projected-shifted
            point["delta_wall"] = (point["opposite"]["E"][0]-point["equal"]["E"][0])/S.bulk["gamma_B"]
            result["finite"][key] = point
            atomic_json(path, result)
            print(f"  b={b:g}: Delta_wall={point['delta_wall']:.9f}, "
                  f"M_N={point['equal']['pole_N']:.6f}/{point['opposite']['pole_N']:.6f}, "
                  f"M_S={point['equal']['pole_S']:.6f}/{point['opposite']['pole_S']:.6f}", flush=True)
        del S
        gc.collect()


    def cli(argv):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--N", type=int, nargs="+", required=True)
        parser.add_argument("--fields", type=float, nargs="+", default=DEFAULT_FIELDS)
        parser.add_argument("--alpha", type=float, default=.5)
        parser.add_argument("--Q", type=int, default=0)
        parser.add_argument("--K", type=int, default=0)
        args = parser.parse_args()
        for size in args.N:
            run(size, fields=args.fields, alpha=args.alpha, Q=args.Q, K=args.K)
    return namespace(locals(), globals())


STEPS = {'dflow_frozen': dflow_frozen, 'core_residue': core_residue, 'core_residue_analyze': core_residue_analyze, 'core_pin_full': core_pin_full, 'gfunc_core': gfunc_core, 'gfunc_ising': gfunc_ising, 'scan': scan}
REPORT = [('core_residue_analyze',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
