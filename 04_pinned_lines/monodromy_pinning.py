"""Where does pinning a charged mode of the O(2) monodromy lead, and how does the twist change its walls?

The coupling h_pin int (psi_s + psi_s^dagger) on the O(2) monodromy line (eq:monopinaction) starts a line flow. At weak
coupling the two real components of the doublet shift in the ratio 3:1 predicted for a single complex coupling (Table 17,
the time-reversal-even pin at h_p = 0.4, ED and DMRG to N_m = 12). At strong coupling (h_pin = 1000) the endpoint has a
tilt at the protected value for alpha <= 0.3, a finite stiffness and pin-independent levels, and its junctions give the
twist dependence of the stiffness, shape factor and wall (Tables 4 and 18-20; the alpha = 0 line with the same pin is the
reference). The large-N saddles with every component twisted (Table 21) and with one twisted pair complete the comparison
(paper, Sec. 5.3 and App. E.7, "Pinning a charged mode of the monodromy").

The ED runs take seconds to minutes and the DMRG runs 0.3-2 hours; their outputs are stored. The report rebuilds the table
of continuum extrapolations from them.

usage: python monodromy_pinning.py <step> [args]      (python monodromy_pinning.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def pin_flow():
    """Experiment E: monodromy pinning. Does the O(2) half-twist flow to a U(1)-breaking conformal line with a tilt at 1?

Physics. The O(2) pi-monodromy defect has a relevant charged mode Psi_a (Delta ~ 0.9 < 1, twisted charge 1,
K = L_z - alpha Q = 0). Perturbing the line by h_p int dtau (Psi_a + Psi_a^dagger) breaks U(1) on the line but keeps K,
so the defect must flow. If the infrared line is a conformal defect on which U(1) is broken, the broken-current Ward
identity (d J ~ delta_line t) puts a tilt operator t at Delta = 1 exactly. That is a protected number. The topological
surface ending on the line cannot be removed by a line perturbation, so the endpoint is a pinned monodromy line, not
the trivial line.
Prediction. In the K = 0 space (all charges Q), the doublet Psi_a, Psi_a^dagger (degenerate at h_p = 0, Delta ~ 0.9)
splits. One combination, the tilt, goes to Delta = 1 as h_p -> infinity; the other goes up.
Lattice. The pinning operator is the K-conserving part of the charge-raising bilinear (V_1 + i V_2)/sqrt2 just off
each pole (theta_0 and pi - theta_0, phi = 0), plus its conjugate; the twisted orbitals vanish exactly at the pole.
Levels are in bulk units (d phi - phi = 1 of the untwisted bulk), and lam = 0 strips the core potential.
usage: python pin_flow.py <config> N hp1,hp2,... [lam=0] [theta0=0.35]   -> pin_flow_<config>_N{N}_lam{lam}.json
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    M = fuzzy.onmodel()   # was: import onmodel as M
    CS = fuzzy.core_strip()   # was: import core_strip as CS
    _defect_corr = fuzzy; popc, wave_abs = _defect_corr.popc, _defect_corr.wave_abs   # was: from defect_corr import popc, wave_abs

    HERE = 'twist'


    def op_matrix(st_in, st_out, terms):
        """sum_t coef_t c+_a c_b as a sparse (dim_out x dim_in) matrix; terms = [(a, b, coef)], a != b."""
        rows, cols, vals = [], [], []
        for a, b, c in terms:
            ok = (((st_in >> np.int64(b)) & 1) == 1) & (((st_in >> np.int64(a)) & 1) == 0)
            lo, hi = min(a, b), max(a, b)
            between = np.int64(((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
            sign = 1 - 2 * (popc(st_in & between) & 1)
            new = st_in ^ (np.int64(1) << np.int64(a)) ^ (np.int64(1) << np.int64(b))
            pos = np.searchsorted(st_out, new)
            pc = np.clip(pos, 0, max(len(st_out) - 1, 0))
            hit = ok & (pos < len(st_out)) & (st_out[pc] == new)
            idx = np.nonzero(hit)[0]
            rows.append(pc[idx]); cols.append(idx); vals.append(c * sign[idx])
        if not rows:
            return sp.csr_matrix((len(st_out), len(st_in)))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(st_out), len(st_in)))


    def pin_terms(fl, orbs, koff, off, N, theta0):
        """K-conserving part of (V_1 + i V_2)/sqrt2 at (theta0, 0) and (pi - theta0, 0): [(a, b, coef)] raising Q by 1."""
        s = (N - 1) / 2
        f0, fp, fm = fl.names.index('0'), fl.names.index('12+'), fl.names.index('12-')
        acc = {}
        for th in (theta0, np.pi - theta0):
            for (fa, fb) in ((f0, fm), (fp, f0)):
                for ja, ma in enumerate(orbs[fa]):
                    for jb, mb in enumerate(orbs[fb]):
                        if (ja + koff[fa]) - (jb + koff[fb]) != 0:
                            continue
                        key = (int(off[fa] + ja), int(off[fb] + jb))
                        acc[key] = acc.get(key, 0.0) + wave_abs(s, ma, th) * wave_abs(s, mb, th)
        tot = np.sqrt(sum(v * v for v in acc.values()))
        return [(a, b, v / tot) for (a, b), v in acc.items()]              # unit-normalized coefficient vector


    def run(cfgname, N, hps, lam=0.0, theta0=0.35, k=8, alpha=0.5):
        t0 = time.time()
        R = store.get(os.path.join(HERE, 'on_results', f'{cfgname}_N{N}.json'))
        bulk = R['alpha']['0.0000']
        u = bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0]
        fl, orbs, koff, two, mu, _, _ = CS.build(cfgname, N, alpha, lam)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        Qs = [q for q in range(-N, N + 1) if M.Basis(fl, orbs, koff, N, [q], 0, ()).dim > 0]
        B = {q: M.Basis(fl, orbs, koff, N, [q], 0, ()) for q in Qs}
        dims = [B[q].dim for q in Qs]
        offs = np.concatenate([[0], np.cumsum(dims)]).astype(int)
        blocks = [[None] * len(Qs) for _ in Qs]
        for i, q in enumerate(Qs):
            blocks[i][i] = M.build_H(B[q], two, mu)
        terms = pin_terms(fl, orbs, koff, off, N, theta0)
        P = [[None] * len(Qs) for _ in Qs]
        for i, q in enumerate(Qs):
            if q + 1 in B:
                j = Qs.index(q + 1)
                Op = op_matrix(B[q].states, B[q + 1].states, terms)        # Q -> Q+1
                P[j][i] = Op; P[i][j] = Op.conj().T
        H0 = sp.bmat(blocks, format='csr')
        Hp = sp.bmat([[P[i][j] if P[i][j] is not None else (sp.csr_matrix((dims[i], dims[j])) if i != j else sp.csr_matrix((dims[i], dims[i])))
                       for j in range(len(Qs))] for i in range(len(Qs))], format='csr')
        Qdiag = np.concatenate([np.full(d, q) for q, d in zip(Qs, dims)])
        print(f'{cfgname} N={N} lam={lam}: K=0 space over Q={Qs[0]}..{Qs[-1]}, dim {H0.shape[0]}, u = {u:.5f}, '
              f'|[H0,Q]| {abs(H0 @ sp.diags(Qdiag) - sp.diags(Qdiag) @ H0).max():.1e}  [{time.time() - t0:.0f}s]', flush=True)
        out = dict(config=cfgname, N=N, lam=lam, alpha=alpha, theta0=theta0, u=u, points={})
        v0 = None
        for hp in hps:
            H = (H0 + hp * u * Hp).tocsr()                                  # h_p in bulk units
            w, v = sla.eigsh(H, k=k, which='SA', tol=1e-10, v0=v0)
            o = np.argsort(w); w, v = w[o], v[:, o]
            v0 = v[:, 0]
            q2 = [float(np.real(v[:, i].conj() @ (Qdiag ** 2 * v[:, i]))) for i in range(k)]
            lev = ((w - w[0]) / u).tolist()
            out['points'][f'{hp:g}'] = dict(hp=hp, levels=lev, Q2=q2)
            print(f'  h_p = {hp:7.3f}: ' + ' '.join(f'{x:.3f}' for x in lev[1:7]) + '   <Q^2> ' + ' '.join(f'{x:.2f}' for x in q2[:7]) +
                  f'  [{time.time() - t0:.0f}s]', flush=True)
        json.dump(out, open(os.path.join(HERE, f'pin_flow_{cfgname}_N{N}_lam{lam:g}' + ('' if alpha == 0.5 else f'_a{alpha:g}') + '.json'), 'w'), indent=1)


    def run_o4(N, hps, alpha=0.5, lam=0.0, k=8):
        """O(4) with pair (1,2) twisted (alpha) and the UNTWISTED pair (3,4) pinned at the poles by its charge-raising
        bilinear (V_3 + i V_4)/sqrt2. Its spin-zero defect mode is strongly relevant (Delta ~ 0.5), and the pin breaks only
        U(1)_34 on the line, so the prediction is a single tilt at Delta = 1 exactly, in (Q_12 = 0, K = 0).
        alpha = 0 is the control: the O(4) magnetic line, same pin, no monodromy."""
        t0 = time.time()
        _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
        cfg = CONFIGS['o4wf_pair']
        Rb = store.get(os.path.join(HERE, 'on_results', f'o4wf_diag_N{N}.json'))       # same bulk model
        hc, bulk = Rb['h_c'], Rb['alpha']['0.0000']
        u = bulk['(1, 0)|1|()'][0] - bulk['(1, 0)|0|()'][0]
        if alpha > 0:
            fl, orbs, koff, two, mu, _, _ = CS.build('o4wf_pair', N, alpha, lam, hc)
        else:
            fl, orbs, koff, two, mu = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], hc, pairs=cfg['pairs'], twist=cfg['twist'])
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        s = (N - 1) / 2
        f0, fp, fm = fl.names.index('0'), fl.names.index('34+'), fl.names.index('34-')
        terms = []
        for jpole, m in ((len(orbs[f0]) - 1, s), (0, -s)):                 # north (m = s) and south (m = -s) orbitals
            c = wave_abs(s, m, 0.0 if m > 0 else np.pi) ** 2
            terms += [(int(off[f0] + jpole), int(off[fm] + jpole), c), (int(off[fp] + jpole), int(off[f0] + jpole), c)]
        nrm = np.sqrt(sum(t[2] ** 2 for t in terms)); terms = [(a, b, c / nrm) for a, b, c in terms]
        Qs = [q for q in range(-N, N + 1) if M.Basis(fl, orbs, koff, N, [0, q], 0, ()).dim > 0]
        B = {q: M.Basis(fl, orbs, koff, N, [0, q], 0, ()) for q in Qs}
        dims = [B[q].dim for q in Qs]
        blocks = [[sp.csr_matrix((dims[i], dims[j])) for j in range(len(Qs))] for i in range(len(Qs))]
        for i, q in enumerate(Qs):
            blocks[i][i] = M.build_H(B[q], two, mu)
        pin = [[sp.csr_matrix((dims[i], dims[j])) for j in range(len(Qs))] for i in range(len(Qs))]
        for i, q in enumerate(Qs):
            if q + 1 in B:
                j = Qs.index(q + 1)
                Op = op_matrix(B[q].states, B[q + 1].states, terms)
                pin[j][i] = Op; pin[i][j] = Op.conj().T
        H0, Hp = sp.bmat(blocks, format='csr'), sp.bmat(pin, format='csr')
        Qd = np.concatenate([np.full(d, q) for q, d in zip(Qs, dims)])
        print(f'O(4) pin of pair (3,4), alpha={alpha}, N={N}, lam={lam}: (Q12=0, K=0) space over Q34={Qs[0]}..{Qs[-1]}, dim {H0.shape[0]}, '
              f'u = {u:.5f}  [{time.time() - t0:.0f}s]', flush=True)
        out = dict(N=N, alpha=alpha, lam=lam, u=u, points={})
        v0 = None
        for hp in hps:
            w, v = sla.eigsh((H0 + hp * u * Hp).tocsr(), k=k, which='SA', tol=1e-10, v0=v0)
            o = np.argsort(w); w, v = w[o], v[:, o]; v0 = v[:, 0]
            lev = ((w - w[0]) / u).tolist()
            q2 = [float(np.real(v[:, i].conj() @ (Qd ** 2 * v[:, i]))) for i in range(k)]
            out['points'][f'{hp:g}'] = dict(hp=hp, levels=lev, Q2=q2)
            print(f'  h_p = {hp:7.3f}: ' + ' '.join(f'{x:.3f}' for x in lev[1:7]) + f'   <Q34^2>_0 {q2[0]:.2f}  [{time.time() - t0:.0f}s]', flush=True)
        store.put(os.path.join(HERE, f'pin_flow_o4_a{alpha:g}_N{N}.json'), out)

    def main():
        if sys.argv[1] == 'o4':
            run_o4(int(sys.argv[2]), [float(x) for x in sys.argv[3].split(',')], float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
        else:
            hps = [float(x) for x in sys.argv[3].split(',')]
            run(sys.argv[1], int(sys.argv[2]), hps, float(sys.argv[4]) if len(sys.argv) > 4 else 0.0,
                float(sys.argv[5]) if len(sys.argv) > 5 else 0.35, alpha=float(sys.argv[6]) if len(sys.argv) > 6 else 0.5)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/pin_flow_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def kr_test():
    """Fuzzy-sphere test of the monodromy pinning defects of Kravchuk and Radcliffe (arXiv:2510.02281) in the O(2) model.

The O(2) monodromy defect at angle alpha has a charged mode Psi_alpha (charge 1, transverse spin alpha, K = L_z - alpha Q
= 0) that is relevant for alpha < v*.  Perturbing the line by h int (Psi_alpha + Psi_alpha^+) breaks U(1) and transverse
rotations separately but keeps K; the infrared defect then has
  * a tilt at Delta = 1 exactly (the broken generator; the pinning phase is exactly marginal),
  * the displacement at Delta = 2 (K = +-1, Q = 0 at h = 0),
  * near v*, where Delta_UV = 1 - delta, a perturbative fixed point with Delta_IR(Re Psi) = 1 + 2 delta (ratio -2).
The lattice perturbation is pin_flow.pin_terms (the K-conserving charge-raising bilinear just off each pole, theta0) with
the core potential stripped (lam = 0).  Levels in bulk units u, relative to the K = 0 ground state, in the K = 0 space
(all charges; the doublet and its split) and the K = 1 space (displacement candidate).
usage: python kr_test.py N alpha hp1,hp2,... [theta0=0.35]   -> kr_results/o2wf_N{N}_a{alpha}.json
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    M = fuzzy.onmodel()   # was: import onmodel as M
    CS = fuzzy.core_strip()   # was: import core_strip as CS
    _pin_flow = globals()['pin_flow'](); op_matrix, pin_terms = _pin_flow.op_matrix, _pin_flow.pin_terms   # was: from pin_flow import op_matrix, pin_terms

    HERE = 'twist'
    OUT = os.path.join(HERE, 'kr_results')


    def space(fl, orbs, koff, two, mu, N, K, terms):
        Qs = [q for q in range(-N, N + 1) if M.Basis(fl, orbs, koff, N, [q], K, ()).dim > 0]
        B = {q: M.Basis(fl, orbs, koff, N, [q], K, ()) for q in Qs}
        dims = [B[q].dim for q in Qs]
        blocks = [[None] * len(Qs) for _ in Qs]
        P = [[None] * len(Qs) for _ in Qs]
        for i, q in enumerate(Qs):
            blocks[i][i] = M.build_H(B[q], two, mu)
            if q + 1 in B:
                j = Qs.index(q + 1)
                Op = op_matrix(B[q].states, B[q + 1].states, terms)
                P[j][i] = Op
                P[i][j] = Op.conj().T
        Z = lambda i, j: sp.csr_matrix((dims[i], dims[j]))
        H0 = sp.bmat([[blocks[i][j] if blocks[i][j] is not None else Z(i, j) for j in range(len(Qs))] for i in range(len(Qs))], format='csr')
        Hp = sp.bmat([[P[i][j] if P[i][j] is not None else Z(i, j) for j in range(len(Qs))] for i in range(len(Qs))], format='csr')
        return H0, Hp


    def low(H, k, v0=None):
        if H.shape[0] <= 2000:
            return np.linalg.eigvalsh(H.toarray())[:k]
        return np.sort(sla.eigsh(H, k=k, which='SA', tol=1e-11, v0=v0)[0])


    def run(N, alpha, hps, theta0=0.35):
        pass   # was os.makedirs(...): the store needs no output directory
        t0 = time.time()
        R = store.get(os.path.join(HERE, 'on_results', f'o2wf_N{N}.json'))
        bulk = R['alpha']['0.0000']
        u = bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0]
        fl, orbs, koff, two, mu, _, _ = CS.build('o2wf', N, alpha, 0.0)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        terms = pin_terms(fl, orbs, koff, off, N, theta0)
        H0a, Hpa = space(fl, orbs, koff, two, mu, N, 0, terms)
        H0b, Hpb = space(fl, orbs, koff, two, mu, N, 1, terms)
        out = dict(N=N, alpha=alpha, theta0=theta0, u=u, points={})
        for hp in hps:
            e0 = low((H0a + hp * u * Hpa).tocsr(), 5)
            e1 = low((H0b + hp * u * Hpb).tocsr(), 3)
            lv0 = ((e0 - e0[0]) / u).tolist()
            lv1 = ((e1 - e0[0]) / u).tolist()
            out['points'][f'{hp:g}'] = dict(hp=hp, K0=lv0, K1=lv1)
            print(f'N={N} alpha={alpha} h_p={hp:6.3f}: K=0 ' + ' '.join(f'{x:.4f}' for x in lv0[1:4]) +
                  '  K=1 ' + ' '.join(f'{x:.4f}' for x in lv1[:2]) + f'  [{time.time() - t0:.0f}s]', flush=True)
        store.put(os.path.join(OUT, f'o2wf_N{N}_a{alpha:g}.json'), out)

    def main():
        run(int(sys.argv[1]), float(sys.argv[2]), [float(x) for x in sys.argv[3].split(',')],
            float(sys.argv[4]) if len(sys.argv) > 4 else 0.35)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/kr_test_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def kr_delta():
    """delta-scan of the Kravchuk-Radcliffe 3:1 ratio (2026-10-24): pin the charged mode of transverse spin s = alpha + shift.

kr_test.py pins the s = alpha mode (shift 0, conserved K = L_z - alpha Q); at alpha = 0.45 its doublet sits at delta ~ 0.09.
The s = alpha - 1 mode (shift -1, conserved K' = L_z - (alpha - 1) Q = K + Q) crosses marginality near alpha = 0.39-0.45 on
these lattices, so scanning alpha sweeps delta through zero.  KR: along the flow the gap shifts are Re : Im = 3 : 1 at leading
order as delta -> 0 (U(1) structure of beta = h F(|h|^2)), with O(delta) corrections.  At alpha = 0.5 the two pins are related
by the reflection s -> -s, so shift -1 must reproduce shift 0 there.
Pin: the K'-conserving charge-raising bilinear off each pole, as pin_flow.pin_terms but with (j_a + koff_a) - (j_b + koff_b) =
shift; the space is the direct sum over Q of the onmodel blocks with K = K'_target + shift * Q.
usage: python kr_delta.py N alpha shift hp1,hp2,... [theta0=0.35]   -> kr_results/delta_o2wf_N{N}_a{alpha}_s{shift}[_th{theta0}].json
       theta0 = -1 is the T-even pole pin (R1 + R2 at each pole; pin_even_junction.py).
       theta0 = 0.001 is the polar (half) pin: only the orbital pair nearest each pole survives (a defect-local operator), while a
       fixed theta0 is a cone around the line whose defect OPE keeps O(1) admixtures of higher charged defect primaries.
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    M = fuzzy.onmodel()   # was: import onmodel as M
    CS = fuzzy.core_strip()   # was: import core_strip as CS
    _pin_flow = globals()['pin_flow'](); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix
    _defect_corr = fuzzy; wave_abs = _defect_corr.wave_abs   # was: from defect_corr import wave_abs
    _kr_test = globals()['kr_test'](); low = _kr_test.low   # was: from kr_test import low

    HERE = 'twist'
    OUT = os.path.join(HERE, 'kr_results')


    def pin_terms_even(fl, orbs, koff, off, shift):
        """T-even pin (twist/pin_even_junction.py, generalized to shift): at each pole R1 + R2 with weight 1/2 each,
        R1 = f0^+ f12- on the pole-most f12- orbital, R2 = f12+^+ f0 on the pole-most f12+ orbital, the f0 partners fixed by the
        K' label (k_a - k_b = shift).  One bilinear per pole (the theta0 -> 0 limit of pin_terms_s) also pins a T-odd vector."""
        f0, fp, fm = fl.names.index('0'), fl.names.index('12+'), fl.names.index('12-')
        kk = lambda f, i: i + koff[f]
        def partner(f, k):
            for i in range(len(orbs[f])):
                if kk(f, i) == k:
                    return i
            return None
        terms = []
        for top in (True, False):
            im = len(orbs[fm]) - 1 if top else 0                     # R1: f0(a) <- f12-(b), k_a = k_b + shift
            ia = partner(f0, kk(fm, im) + shift)
            ip = len(orbs[fp]) - 1 if top else 0                     # R2: f12+(a) <- f0(b), k_b = k_a - shift
            ib = partner(f0, kk(fp, ip) - shift)
            assert ia is not None and ib is not None, (top, shift)
            terms += [(int(off[f0] + ia), int(off[fm] + im), 0.5), (int(off[fp] + ip), int(off[f0] + ib), 0.5)]
        return terms


    def pin_terms_s(fl, orbs, koff, off, N, theta0, shift):
        if theta0 < 0:                                               # theta0 = -1 selects the T-even pole pin
            return pin_terms_even(fl, orbs, koff, off, shift)
        s = (N - 1) / 2
        f0, fp, fm = fl.names.index('0'), fl.names.index('12+'), fl.names.index('12-')
        acc = {}
        for th in (theta0, np.pi - theta0):
            for (fa, fb) in ((f0, fm), (fp, f0)):
                for ja, ma in enumerate(orbs[fa]):
                    for jb, mb in enumerate(orbs[fb]):
                        if (ja + koff[fa]) - (jb + koff[fb]) != shift:
                            continue
                        key = (int(off[fa] + ja), int(off[fb] + jb))
                        acc[key] = acc.get(key, 0.0) + wave_abs(s, ma, th) * wave_abs(s, mb, th)
        tot = np.sqrt(sum(v * v for v in acc.values()))
        return [(a, b, v / tot) for (a, b), v in acc.items()]


    def space(fl, orbs, koff, two, mu, N, Kp, shift, terms):
        B = {}
        for q in range(-N, N + 1):
            b = M.Basis(fl, orbs, koff, N, [q], Kp + shift * q, ())
            if b.dim > 0:
                B[q] = b
        Qs = sorted(B)
        dims = [B[q].dim for q in Qs]
        Z = lambda i, j: sp.csr_matrix((dims[i], dims[j]))
        H = [[Z(i, j) for j in range(len(Qs))] for i in range(len(Qs))]
        P = [[Z(i, j) for j in range(len(Qs))] for i in range(len(Qs))]
        for i, q in enumerate(Qs):
            H[i][i] = M.build_H(B[q], two, mu)
            if q + 1 in B:
                j = Qs.index(q + 1)
                Op = op_matrix(B[q].states, B[q + 1].states, terms)
                P[j][i] = Op
                P[i][j] = Op.conj().T
        return sp.bmat(H, format='csr'), sp.bmat(P, format='csr')


    def run(N, alpha, shift, hps, theta0=0.35):
        pass   # was os.makedirs(...): the store needs no output directory
        t0 = time.time()
        R = store.get(os.path.join(HERE, 'on_results', f'o2wf_N{N}.json'))
        bulk = R['alpha']['0.0000']
        u = bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0]
        fl, orbs, koff, two, mu, _, _ = CS.build('o2wf', N, alpha, 0.0)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        terms = pin_terms_s(fl, orbs, koff, off, N, theta0, shift)
        H0a, Hpa = space(fl, orbs, koff, two, mu, N, 0, shift, terms)
        H0b, Hpb = space(fl, orbs, koff, two, mu, N, 1, shift, terms)
        out = dict(N=N, alpha=alpha, shift=shift, s=alpha + shift, theta0=theta0, u=u, nterms=len(terms), points={})
        for hp in hps:
            e0 = low((H0a + hp * u * Hpa).tocsr(), 5)
            e1 = low((H0b + hp * u * Hpb).tocsr(), 3)
            lv0 = ((e0 - e0[0]) / u).tolist()
            lv1 = ((e1 - e0[0]) / u).tolist()
            out['points'][f'{hp:g}'] = dict(hp=hp, K0=lv0, K1=lv1)
            print(f'N={N} alpha={alpha} s={alpha + shift:+.2f} h_p={hp:5.3f}: K\'=0 ' + ' '.join(f'{x:.5f}' for x in lv0[1:4]) +
                  "  K'=1 " + ' '.join(f'{x:.5f}' for x in lv1[:2]) + f'  [{time.time() - t0:.0f}s]', flush=True)
        tag = '' if abs(theta0 - 0.35) < 1e-12 else f'_th{theta0:g}'          # theta0 -> 0 is the polar (defect-local) pin
        store.put(os.path.join(OUT, f'delta_o2wf_N{N}_a{alpha:g}_s{shift}{tag}.json'), out)
        return out

    def main():
        run(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), [float(x) for x in sys.argv[4].split(',')],
            float(sys.argv[5]) if len(sys.argv) > 5 else 0.35)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/kr_delta_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def dmrg_kr():
    """DMRG for the Kravchuk-Radcliffe 3:1 test (kr_test.py beyond ED), 2026-10-24.

Same Hamiltonian as kr_test.py: O(2) GZWH (o2wf) monodromy at angle alpha with the core potential stripped (lam = 0), plus
h_p u (P + P^+), P = the unit-normalized K-conserving charge-raising bilinear off each pole (pin_flow.pin_terms, theta0).
The pin breaks Q, so only N_e and K = L_z - alpha Q are conserved; every h_p (including 0) uses the same blocking, so
truncation errors are alike across the scan.  K = 0: the defect ground state and the lowest two excited states (the Im and Re
branches of the charged doublet, split at second order in h_p).  K = 1 (with 'k1'): the lowest two states, the second being
the displacement candidate D.
Test: along the flow Delta_Re - Delta_UV = 3 beta_3 g^2 and Delta_Im - Delta_UV = beta_3 g^2, so at second order in h_p the gap
shifts are in the ratio Re : Im = 3 : 1 (unit-free).
h_c and the bulk unit u: stored ED values for N <= 10; beyond, h_c from dmrg_theta.h_of (fit to the stored N) and u from a
charge-basis DMRG of the untwisted bulk (dmrg_hq.bulk_unit).
usage: python dmrg_kr.py N alpha chi hp1,hp2,... [k1] [theta0=0.35]   -> kr_results/dmrg_o2wf_N{N}_a{alpha}_chi{chi}.json
       python dmrg_kr.py gate N alpha chi hp                           (against kr_results/o2wf_N{N}_a{alpha}.json)
       python dmrg_kr.py seed N alpha chi theta0                       (h, u and the h_p = 0 point for another pin)
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    HERE = 'twist'
    sys.path.insert(0, HERE)
    OUT = os.path.join(HERE, 'kr_results')
    SHIFT = 1.0      # -SHIFT per particle (exact, N_e fixed): with all energies positive, TeNPy's orthogonal_to projection would
                     # otherwise return the projected-out direction at eigenvalue 0


    def hc_u(N, chi):
        f = os.path.join(HERE, 'on_results', f'o2wf_N{N}.json')
        if store.exists(f):
            R = store.get(f)
            bulk = R['alpha']['0.0000']
            return R['h_c'], bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0], 'ED'
        _dmrg_theta = fuzzy.question('04_pinned_lines/junctions').dmrg_theta(); h_of = _dmrg_theta.h_of   # was: from dmrg_theta import h_of
        _dmrg_hq = fuzzy.question('04_pinned_lines/ward_stiffness').dmrg_hq(); bulk_unit = _dmrg_hq.bulk_unit   # was: from dmrg_hq import bulk_unit
        h = h_of('o2wf', N)[0]
        return h, bulk_unit('o2wf', N, h, chi), 'DMRG'


    def build(N, alpha, hp, h, u, theta0=0.35, K=0):
        CS = fuzzy.core_strip()   # was: import core_strip as CS
        _pin_flow = globals()['pin_flow'](); pin_terms = _pin_flow.pin_terms   # was: from pin_flow import pin_terms
        _dmrg_twist = __import__('dmrg'); fermion_site, compress_mpo = _dmrg_twist.fermion_site, _dmrg_twist.compress_mpo   # was: from dmrg_twist import fermion_site, compress_mpo
        from tenpy.models.model import CouplingModel, MPOModel
        from tenpy.models.lattice import TrivialLattice
        fl, orbs, koff, two, mu, _, _ = CS.build('o2wf', N, alpha, 0.0, hc=h)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        if theta0 < 0:                                 # T-even pole pin (R1 + R2 at each pole), kr_delta.pin_terms_even
            _kr_delta = globals()['kr_delta'](); pin_terms_even = _kr_delta.pin_terms_even   # was: from kr_delta import pin_terms_even
            terms = pin_terms_even(fl, orbs, koff, off, 0)
        else:
            terms = pin_terms(fl, orbs, koff, off, N, theta0)
        mode_m = [m for o in orbs for m in o]
        kk = [jj + koff[f] for f, o in enumerate(orbs) for jj in range(len(o))]
        charges = [(1, k) for k in kk]
        order = sorted(range(len(mode_m)), key=lambda k: (mode_m[k], k))
        pos = {k: i for i, k in enumerate(order)}
        lat = TrivialLattice([fermion_site(charges[k], ['N', 'K'], [1, 1]) for k in order], bc_MPS='finite')
        model = CouplingModel(lat)
        li = lambda p: list(lat.mps2lat_idx(p))
        for (k1, k2, k3, k4, v) in two:
            model.add_local_term(v, [('Cd', li(pos[k1])), ('Cd', li(pos[k2])), ('C', li(pos[k3])), ('C', li(pos[k4]))])
        for k, v in enumerate(mu):
            model.add_onsite_term(v - SHIFT, pos[k], 'N')
        if hp != 0:
            for (a, b, c) in terms:                    # c+_a c_b raises Q by one; add the conjugate
                model.add_local_term(hp * u * c, [('Cd', li(pos[a])), ('C', li(pos[b]))])
                model.add_local_term(hp * u * c, [('Cd', li(pos[b])), ('C', li(pos[a]))])
        base = set(range(N))                           # psi_0 filled: K = 0
        if K == 0:
            occ = base
        else:                                          # one psi_0 mode moved to another flavour with K label higher by K
            occ = next((base - {j}) | {k} for j in sorted(base, reverse=True) for k in range(N, len(kk)) if kk[k] - kk[j] == K)
        return MPOModel(lat, compress_mpo(model.calc_H_MPO())), order, occ


    def states(N, alpha, hp, h, u, chi, K=0, nstates=3, theta0=0.35, sweeps=30):
        _dmrg_twist = __import__('dmrg'); run_dmrg = _dmrg_twist.run_dmrg   # was: from dmrg_twist import run_dmrg
        model, order, occ = build(N, alpha, hp, h, u, theta0, K)
        out, psis = [], []
        for i in range(nstates):
            t = time.time()
            E, psi, eng = run_dmrg(model, order, occ, chi_max=chi, sweeps=sweeps, orthogonal_to=list(psis))
            psis.append(psi)
            out.append(dict(E=float(E) + SHIFT * N, trunc=float(eng.sweep_stats['max_trunc_err'][-1]), chi=int(max(psi.chi)),
                            sweeps=len(eng.sweep_stats['E']), t=time.time() - t))
        return out


    def run(N, alpha, chi, hps, k1=False, theta0=0.35):
        pass   # was os.makedirs(...): the store needs no output directory
        tag = '' if abs(theta0 - 0.35) < 1e-12 else f'_th{theta0:g}'          # theta0 -> 0: the polar (defect-local) pin
        fn = os.path.join(OUT, f'dmrg_o2wf_N{N}_a{alpha:g}_chi{chi}{tag}.json')
        if store.exists(fn):
            res = store.get(fn)
        else:
            h, u, src = hc_u(N, chi)
            res = dict(N=N, alpha=alpha, chi=chi, theta0=theta0, h=h, u=u, unit_source=src, points={})
            store.put(fn, res)
        for hp in hps:
            key = f'{hp:g}'
            P = res['points'].setdefault(key, dict(hp=hp))
            for lab, K, ns in (('K0', 0, 3),) + ((('K1', 1, 2),) if k1 else ()):
                if lab in P:
                    continue
                P[lab] = states(N, alpha, hp, res['h'], res['u'], chi, K, ns, theta0)
                store.put(fn, res)
            e0 = P['K0'][0]['E']
            lv = [(s['E'] - e0) / res['u'] for s in P['K0'][1:]]
            msg = f"N={N} alpha={alpha} chi={chi} h_p={hp:g}: K=0 " + ' '.join(f'{x:.5f}' for x in lv)
            if 'K1' in P:
                msg += '  K=1 ' + ' '.join(f"{(s['E'] - e0) / res['u']:.5f}" for s in P['K1'])
            msg += f"  trunc {max(s['trunc'] for s in P['K0']):.0e}  [{sum(s['t'] for s in P['K0']):.0f}s]"
            print(msg, flush=True)
        return res


    def gate(N, alpha, chi, hp):
        ed = store.get(os.path.join(OUT, f'o2wf_N{N}_a{alpha:g}.json'))['points'][f'{hp:g}']
        res = run(N, alpha, chi, [hp])
        P = res['points'][f'{hp:g}']
        e0 = P['K0'][0]['E']
        for i in (1, 2):
            print(f"gate N={N} h_p={hp}: K=0 level {i}: ED {ed['K0'][i]:.6f}  DMRG {(P['K0'][i]['E'] - e0) / res['u']:.6f}")


    def seed(N, alpha, chi, theta0):
        """Copy h, u and the pin-independent h_p = 0 point from the theta0 = 0.35 file into the file for another pin."""
        src = store.get(os.path.join(OUT, f'dmrg_o2wf_N{N}_a{alpha:g}_chi{chi}.json'))
        fn = os.path.join(OUT, f'dmrg_o2wf_N{N}_a{alpha:g}_chi{chi}_th{theta0:g}.json')
        res = store.get(fn) if store.exists(fn) else dict(N=N, alpha=alpha, chi=chi, theta0=theta0, h=src['h'], u=src['u'],
                                                                     unit_source=src['unit_source'], points={})
        res['points'].setdefault('0', src['points']['0'])
        store.put(fn, res)

    def main():
        if sys.argv[1] == 'seed':
            seed(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
        elif sys.argv[1] == 'gate':
            gate(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
        else:
            args = sys.argv[1:]
            k1 = 'k1' in args
            args = [a for a in args if a != 'k1']
            run(int(args[0]), float(args[1]), int(args[2]), [float(x) for x in args[3].split(',')], k1,
                float(args[4]) if len(args) > 4 else 0.35)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/dmrg_kr_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def kr_descendant_overlap():
    """SL(2) parity test of a lattice pin (2026-10-25): fractions of R|0> (R = charge-raising part of the pin, at h = 0) in the lowest
four Q = 1, K' = 0 states.  A defect primary inserted with equal strength at both poles (x = +-1 on the line) creates only EVEN
SL(2) descendants of the doublet (odd ones cancel between the poles), so weight on the first descendant (state n = 1, at ~Delta+1)
measures north/south asymmetry, i.e. a junction-like perturbation instead of a uniform pin.
N = 8, alpha = 0.45: even pin 0.0004, one-bilinear "polar" pin 0.28, theta0 = 0.35 cone 0.10, theta0 = 0.6 0.02.
usage: python kr_descendant_overlap.py N alpha   (pins from kr_delta.pin_terms_s; theta0 = -1 is the T-even pole pin)
"""
    import sys, os, json, numpy as np, scipy.sparse.linalg as sla
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    M = fuzzy.onmodel()   # was: import onmodel as M
    CS = fuzzy.core_strip()   # was: import core_strip as CS
    _kr_delta = globals()['kr_delta'](); pin_terms_s = _kr_delta.pin_terms_s   # was: from kr_delta import pin_terms_s
    _pin_flow = globals()['pin_flow'](); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix

    def main():
        N, alpha = int(sys.argv[1]), float(sys.argv[2])
        for shift in (0, -1):
            fl, orbs, koff, two, mu, _, _ = CS.build('o2wf', N, alpha, 0.0)
            off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
            B0 = M.Basis(fl, orbs, koff, N, [0], 0, ()); B1 = M.Basis(fl, orbs, koff, N, [1], shift, ())
            H0 = M.build_H(B0, two, mu); H1 = M.build_H(B1, two, mu)
            w0, v0 = sla.eigsh(H0, k=1, which='SA'); w1, v1 = sla.eigsh(H1, k=4, which='SA'); o = np.argsort(w1); w1, v1 = w1[o], v1[:, o]
            for th in (-1, 0.001, 0.2, 0.35, 0.6):
                t = pin_terms_s(fl, orbs, koff, off, N, th, shift)
                R = op_matrix(B0.states, B1.states, t)
                w = R @ v0[:, 0]
                f = np.abs(v1.T @ w) ** 2 / (w @ w)
                print(f'N={N} alpha={alpha} s={alpha + shift:+.2f} pin {"even" if th < 0 else th}: |<Psi_n|R|0>|^2 fractions (n=0..3) ' + ' '.join(f'{x:.4f}' for x in f) + f'   E_n-E_0: ' + ' '.join(f'{x:.3f}' for x in (w1 - w0[0])))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/kr_descendant_overlap_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def kr_pt2():
    """Second-order perturbation theory for the KR doublet on the lattice (2026-10-24): where the Re : Im ratio comes from.
Exact O(h^2) gap shifts from resolvents, and their decomposition by intermediate states:
  * vacuum only (with the pin at both poles): Re = 4c^2/D, Im = 2c^2/D, ratio exactly 2 (c = <Psi|P|0>, D = E_Psi - E_0);
  * the remainder: higher Q = 0 states (diagonal and off-diagonal) and Q = +-2 states (diagonal only), plus the vacuum's own shift
    from higher Q = +-1 states.
For a generalized free field the remainder cancels the vacuum term exactly (gaps unchanged), so the universal (interaction) part
is what is left after a large cancellation.
usage: python kr_pt2.py N alpha shift [theta0=0.35] [nlow=40]   (shift 0: s = alpha; -1: s = alpha - 1)
"""
    import sys
    import json
    import os
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    M = fuzzy.onmodel()   # was: import onmodel as M
    CS = fuzzy.core_strip()   # was: import core_strip as CS
    _kr_delta = globals()['kr_delta'](); pin_terms_s, space = _kr_delta.pin_terms_s, _kr_delta.space   # was: from kr_delta import pin_terms_s, space
    _pin_flow = globals()['pin_flow'](); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix

    HERE = 'twist'


    def blocks(N, alpha, shift, theta0):
        fl, orbs, koff, two, mu, _, _ = CS.build('o2wf', N, alpha, 0.0)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        terms = pin_terms_s(fl, orbs, koff, off, N, theta0, shift)
        B, H = {}, {}
        for q in range(-3, 4):
            b = M.Basis(fl, orbs, koff, N, [q], shift * q, ())
            if b.dim > 0:
                B[q] = b; H[q] = M.build_H(b, two, mu).tocsr()
        Rq = {q: op_matrix(B[q].states, B[q + 1].states, terms).tocsr() for q in B if q + 1 in B}   # Q -> Q+1
        return B, H, Rq


    def run(N, alpha, shift, theta0=0.35, nlow=40):
        R = store.get(os.path.join(HERE, 'on_results', f'o2wf_N{N}.json'))
        bulk = R['alpha']['0.0000']
        u = bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0]
        B, H, Rq = blocks(N, alpha, shift, theta0)
        eig = {}
        for q in (-2, -1, 0, 1, 2):
            k = min(nlow, H[q].shape[0] - 2)
            w, v = sla.eigsh(H[q], k=k, which='SA', tol=1e-12)
            o = np.argsort(w); eig[q] = (w[o], v[:, o])
        E0, v0 = eig[0][0][0], eig[0][1][:, 0]
        Ep, vp = eig[1][0][0], eig[1][1][:, 0]          # |Psi>, Q = +1
        Em, vm = eig[-1][0][0], eig[-1][1][:, 0]        # |Psibar>, Q = -1
        D = Ep - E0
        # P = R + R^+ with R raising Q.  Vectors: P|Psi> has Q = 0 part R^+_{0<-1}|Psi> ... careful with directions
        Pp_to0 = Rq[0].T.conj() @ vp                     # Q=1 -> Q=0 (lowering part of P)
        Pp_to2 = Rq[1] @ vp                              # Q=1 -> Q=2
        Pm_to0 = Rq[-1] @ vm                             # Q=-1 -> Q=0
        P0_to1 = Rq[0] @ v0; P0_tom1 = Rq[-1].T.conj() @ v0
        c = v0.conj() @ Pp_to0                           # <0|P|Psi>
        def resolvent(Hq, E, rhs, proj=None):
            """(E - Hq)^{-1} rhs on the complement of proj (list of vectors), by a sparse solve with deflation."""
            r = rhs.copy()
            for p in (proj or []):
                r = r - p * (p.conj() @ r)
            A = (E * sp.identity(Hq.shape[0]) - Hq).tocsr()
            if Hq.shape[0] <= 20000:
                x = sla.spsolve(A.tocsc(), r)
            else:                                        # symmetric indefinite: MINRES, deflated rhs keeps it well posed
                x, info = sla.minres(A, r, rtol=1e-12, maxiter=20000)
                assert info == 0, info
            for p in (proj or []):
                x = x - p * (p.conj() @ x)
            return x
        # doublet effective matrix (degenerate PT): M_ab = <a|P G P|b>, G = (E_Psi - H0)^{-1} outside the doublet
        G0p = resolvent(H[0], Ep, Pp_to0)                # Q=0 intermediate (vacuum included: E_Psi - E_0 > 0, no singularity)
        G2p = resolvent(H[2], Ep, Pp_to2)
        Mdiag = (Pp_to0.conj() @ G0p + Pp_to2.conj() @ G2p).real
        Moff = (Pm_to0.conj() @ G0p).real                # <Psibar|P G P|Psi> through Q = 0
        # vacuum shift: <0|P G P|0>, G = (E0 - H)^{-1} on Q = +-1
        dvac = (P0_to1.conj() @ resolvent(H[1], E0, P0_to1) + P0_tom1.conj() @ resolvent(H[-1], E0, P0_tom1)).real
        dRe, dIm = (Mdiag + abs(Moff)) - dvac, (Mdiag - abs(Moff)) - dvac
        # vacuum-only pieces
        vac_diag = abs(c) ** 2 / D; vac_off = abs(c) ** 2 / D; vac_dvac = -2 * abs(c) ** 2 / D
        # partial sums over the nlow lowest Q=0 states (spectral), to show how the cancellation builds
        w0, V0 = eig[0]
        a_p = V0.conj().T @ Pp_to0; a_m = V0.conj().T @ Pm_to0
        contrib_diag = (abs(a_p) ** 2) / (Ep - w0); contrib_off = (a_m.conj() * a_p).real / (Ep - w0)
        s = 1 / u ** 2 / u                                # shifts in bulk units per (h_p)^2: (h_p u)^2 * M / u
        s = u                                             # (h_p u)^2 M / u = h_p^2 u M
        print(f'N={N} alpha={alpha} s={alpha + shift:+.2f}: D/u = {D / u:.5f}, |c|^2/(D u)... ')
        print(f'  exact O(h^2) gap shifts per h_p^2 (bulk units): Re {dRe * s:.6f}  Im {dIm * s:.6f}  ratio {dRe / dIm:.4f}')
        print(f'  vacuum-only: Re {(2 * vac_diag - vac_dvac) * s:.6f}  Im {(0 - vac_dvac) * s:.6f}  ratio 2')
        print(f'  pieces (x h_p^2 u): diag {Mdiag * s:+.6f} [vac {vac_diag * s:+.6f}]  off {Moff * s:+.6f} [vac {vac_off * s:+.6f}]  dvac {dvac * s:+.6f} [vac-doublet {vac_dvac * s:+.6f}]')
        cs = np.cumsum(contrib_diag); co = np.cumsum(contrib_off)
        for n in (1, 2, 3, 5, 10, 20, len(w0) - 1):
            n = min(n, len(w0) - 1)
            print(f'  Q=0 states up to E-E0 = {(w0[n] - E0) / u:6.3f} (n={n}): diag partial {cs[n] * s:+.6f}  off partial {co[n] * s:+.6f}')
        out = dict(N=N, alpha=alpha, s=alpha + shift, theta0=theta0, D=D / u, Re=dRe * s, Im=dIm * s, ratio=dRe / dIm,
                   x=Moff / (Mdiag - dvac), vacRe=(2 * vac_diag - vac_dvac) * s, vacIm=-vac_dvac * s, diag=Mdiag * s, off=Moff * s, dvac=dvac * s)
        fn = os.path.join(HERE, 'kr_results', 'pt2.json')
        allr = store.get(fn) if store.exists(fn) else {}
        allr[f'N{N}_a{alpha:g}_s{shift}_th{theta0:g}'] = out
        store.put(fn, allr)
        return dict(N=N, alpha=alpha, s=alpha + shift, Re=dRe * s, Im=dIm * s, vacRe=(2 * vac_diag - vac_dvac) * s, vacIm=-vac_dvac * s,
                    diag=Mdiag * s, off=Moff * s, dvac=dvac * s)

    def main():
        run(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.35,
            int(sys.argv[5]) if len(sys.argv) > 5 else 40)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/kr_pt2_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def kr_delta_table():
    """Table of the KR ratio from the kr_delta.py runs: delta = 1 - (UV doublet level), O(h^2) shifts from h_p = 0.2, 0.4
(d = a h^2 + b h^4), ratio = a_Re / a_Im, for every (alpha, s, theta0) and N in kr_results/delta_o2wf_*.json.
usage: python kr_delta_table.py
"""
    import os
    import re
    import glob
    import json
    HERE = 'twist'


    def a2(d2, d4):
        return (16 * d2 - d4) / (12 * 0.04)


    def main():
        rows = []
        for f in store.glob(os.path.join(HERE, 'kr_results', 'delta_o2wf_N*_a*_s*.json')):
            R = store.get(f)
            th = re.search(r'_th([0-9.]+)\.json$', f)
            th = float(th.group(1)) if th else 0.35
            P = R['points']
            if not all(k in P for k in ('0', '0.2', '0.4')):
                continue
            i = [P[k]['K0'][1] for k in ('0', '0.2', '0.4')]
            r = [P[k]['K0'][2] for k in ('0', '0.2', '0.4')]
            aI, aR = a2(i[1] - i[0], i[2] - i[0]), a2(r[1] - r[0], r[2] - r[0])
            rows.append((R['s'], th, R['N'], 1 - i[0], aI, aR, aR / aI))
        rows.sort(key=lambda x: (x[1], x[0], x[2]))
        print(f"{'s':>6} {'theta0':>6} {'N':>3} {'delta':>8} {'a_Im':>8} {'a_Re':>8} {'ratio':>7}")
        last = None
        for s, th, N, d, aI, aR, rt in rows:
            if last and (s, th) != last:
                print()
            last = (s, th)
            print(f'{s:>+6.2f} {th:>6.2f} {N:>3} {d:>+8.4f} {aI:>8.5f} {aR:>8.5f} {rt:>7.3f}')

    def _run():
        main()

    def cli(argv):
        with fuzzy.logged('twist/kr_delta_table.log'):
            _run()
    return namespace(locals(), globals())


@section
def kr_final():
    """All KR 3:1 data in one place and their large-N extrapolation (2026-10-24/25).

ratio = d_Re / d_Im of the O(h_p^2) gap shifts = (1 + x)/(1 - x), x = off / (diag - dvac) (kr_pt2.py); KR: ratio 3, x = 1/2.
Sources: ED at h_p = 0.2, 0.4 (kr_test.py for s = alpha = 0.45, theta0 = 0.35; kr_delta.py for the rest), d = a h^2 + b h^4;
DMRG at h_p = 0.4 with the pin-independent h_p = 0 levels (dmrg_kr.py; the h^4 term moves the ratio by < 0.005 in ED).
Fits: x(N) = x_inf - A/N (last three points) and x_inf - A/N + B/N^2 (all points, if >= 4).
usage: python kr_final.py [Nmin=6]
"""
    import os
    import re
    import sys
    import glob
    import json
    import numpy as np
    HERE = 'twist'
    OUT = os.path.join(HERE, 'kr_results')


    def a2(d2, d4):
        return (16 * d2 - d4) / (12 * 0.04)


    def collect():
        S = {}                                             # (s, theta0) -> {N: (delta, ratio, source)}
        def put(s, th, N, delta, ratio, src):
            S.setdefault((round(s, 3), th), {})
            old = S[(round(s, 3), th)].get(N)
            if old is None or (src.startswith('ED') and not old[2].startswith('ED')):
                S[(round(s, 3), th)][N] = (delta, ratio, src)
        for f in store.glob(os.path.join(OUT, 'o2wf_N*_a0.45.json')):        # kr_test: s = +0.45, theta0 = 0.35
            R = store.get(f); P = R['points']
            if all(k in P for k in ('0', '0.2', '0.4')):
                i = [P[k]['K0'][1] for k in ('0', '0.2', '0.4')]; r = [P[k]['K0'][2] for k in ('0', '0.2', '0.4')]
                put(0.45, 0.35, R['N'], 1 - i[0], a2(r[1] - r[0], r[2] - r[0]) / a2(i[1] - i[0], i[2] - i[0]), 'ED')
        for f in store.glob(os.path.join(OUT, 'delta_o2wf_N*_a*_s*.json')):
            R = store.get(f); P = R['points']
            m = re.search(r'_th([0-9.e-]+)\.json$', f)
            th = float(m.group(1)) if m else 0.35
            if all(k in P for k in ('0', '0.2', '0.4')):
                i = [P[k]['K0'][1] for k in ('0', '0.2', '0.4')]; r = [P[k]['K0'][2] for k in ('0', '0.2', '0.4')]
                put(R['s'], th, R['N'], 1 - i[0], a2(r[1] - r[0], r[2] - r[0]) / a2(i[1] - i[0], i[2] - i[0]), 'ED')
        for f in store.glob(os.path.join(OUT, 'dmrg_o2wf_N*_a*_chi*.json')):
            R = store.get(f); P = R['points']
            th = R.get('theta0', 0.35)
            if '0' in P and '0.4' in P and len(P['0'].get('K0', [])) >= 3 and len(P['0.4'].get('K0', [])) >= 3:
                lv = lambda k: [(x['E'] - P[k]['K0'][0]['E']) / R['u'] for x in P[k]['K0'][1:3]]
                l0, l4 = lv('0'), lv('0.4')
                put(R['alpha'], th, R['N'], 1 - l0[0], (l4[1] - l0[1]) / (l4[0] - l0[0]), f"DMRG{R['chi']}")
        return S


    def fit(Ns, xs):
        Ns, xs = np.array(Ns, float), np.array(xs)
        out = {}
        if len(Ns) >= 3:
            c = np.polyfit(1 / Ns[-3:], xs[-3:], 1); out['1/N (last 3)'] = c[1]
            c = np.polyfit(1 / Ns[-2:], xs[-2:], 1); out['1/N (last 2)'] = c[1]
        if len(Ns) >= 4:
            c = np.polyfit(1 / Ns, xs, 2); out['1/N+1/N^2 (all)'] = c[2]
        return out


    def main(Nmin=6):
        S = collect()
        for (s, th), d in sorted(S.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            Ns = sorted(n for n in d if n >= Nmin)
            if len(Ns) < 2:
                continue
            xs = [(d[n][1] - 1) / (d[n][1] + 1) for n in Ns]
            print(f"s = {s:+.2f}, theta0 = {th:g}:  " + '  '.join(f"N{n}[{d[n][2]}] d={d[n][0]:+.3f} r={d[n][1]:.3f} x={x:.4f}" for n, x in zip(Ns, xs)))
            for k, xi in fit(Ns, xs).items():
                print(f"      {k:>18}: x_inf = {xi:.4f}  ratio_inf = {(1 + xi) / (1 - xi):.3f}")

    def _run():
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/kr_final_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            _run()
    return namespace(locals(), globals())


@section
def pin_even_junction():
    """Time-reversal-even pin of the monodromy line (corrects pin_polar*.py), junction and Ward data.

At alpha = 0 the ordinary pin along e_1 is the T-even sum R1 + R2 of the two charge-raising bilinears at each pole,
R1 = f0^dag f12-, R2 = f12+^dag f0.  Keeping only one of them (pin_polar.py) also pins a T-odd vector, and at alpha = 0 that
'half' pin gives C^HQ ~ 0.11 and Dh_j(pi) ~ 0.39 at N_m = 6, far from the ordinary pinned line (0.30, 0.78), while the T-even
pin with R2 moved one orbital in from the pole gives C^ang 0.336 and Dh_j(pi) 0.745 (pin_check_full.py 'shift').
Here, for any alpha, each pole carries R1 + R2 (coefficient 1/2 each, K = L_z - alpha Q conserving):
  R1 = f0(o0)^dag f12-(om), R2 = f12+(op)^dag f0(o0'), with op / om the pole-most orbital of the twisted flavour and o0, o0' the
  K-matched singlet orbitals.  For alpha -> 0+ this is the 'shift' pin.  South phase e^{i pi eta} gives the junction.
usage: python pin_even_junction.py N alpha hp eta1,...  -> pin_even_o2wf_N{N}_a{alpha}.json
"""
    import os, sys, json, time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    M = fuzzy.onmodel()   # was: import onmodel as M
    CS = fuzzy.core_strip()   # was: import core_strip as CS
    _pin_flow = globals()['pin_flow'](); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix

    HERE = 'twist'


    def even_terms(fl, orbs, koff, off, shift_at_zero=True):
        f0, fp, fm = fl.names.index('0'), fl.names.index('12+'), fl.names.index('12-')
        K = lambda f, i: i + koff[f]
        n0, np_, nm = len(orbs[f0]), len(orbs[fp]), len(orbs[fm])
        idx = lambda f, i: int(off[f] + i)
        def pair(fa, ia, fb):          # partner index in flavour fb with matching K label
            for ib in range(len(orbs[fb])):
                if K(fb, ib) == K(fa, ia):
                    return ib
            raise ValueError
        # north: R1 uses the top orbital of f12-, R2 the top orbital of f12+
        iN_m = nm - 1; iN_0a = pair(fm, iN_m, f0)
        iN_p = np_ - 1; iN_0b = pair(fp, iN_p, f0)
        # south: R2 uses the bottom orbital of f12+, R1 the bottom orbital of f12-
        iS_p = 0; iS_0b = pair(fp, iS_p, f0)
        iS_m = 0; iS_0a = pair(fm, iS_m, f0)
        if shift_at_zero and np_ == n0:      # alpha = 0: move R2 (north) and R1 (south) one orbital in, as alpha -> 0+
            iN_p, iN_0b = np_ - 2, pair(fp, np_ - 2, f0)
            iS_m, iS_0a = 1, pair(fm, 1, f0)
        c = 0.5
        tN = [(idx(f0, iN_0a), idx(fm, iN_m), c), (idx(fp, iN_p), idx(f0, iN_0b), c)]
        tS = [(idx(fp, iS_p), idx(f0, iS_0b), c), (idx(f0, iS_0a), idx(fm, iS_m), c)]
        desc = dict(north=[(float(orbs[f0][iN_0a]), float(orbs[fm][iN_m])), (float(orbs[fp][iN_p]), float(orbs[f0][iN_0b]))],
                    south=[(float(orbs[fp][iS_p]), float(orbs[f0][iS_0b])), (float(orbs[f0][iS_0a]), float(orbs[fm][iS_m]))])
        return tN, tS, desc


    def run(N, alpha, hp, etas, lam=0.0, k=6, cfgname='o2wf', w2=1.0):
        t0 = time.time()
        R = store.get(os.path.join(HERE, 'on_results', f'{cfgname}_N{N}.json'))
        bulk = R['alpha']['0.0000']
        u = bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0]
        fl, orbs, koff, two, mu, _, _ = CS.build(cfgname, N, alpha, lam)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        tN, tS, desc = even_terms(fl, orbs, koff, off)
        if w2 != 1.0:                         # relative weight of the second bilinear (robustness check)
            tN = [tN[0], (tN[1][0], tN[1][1], tN[1][2] * w2)]; tS = [tS[0], (tS[1][0], tS[1][1], tS[1][2] * w2)]
        Qs = [q for q in range(-N, N + 1) if M.Basis(fl, orbs, koff, N, [q], 0, ()).dim > 0]
        B = {q: M.Basis(fl, orbs, koff, N, [q], 0, ()) for q in Qs}
        dims = [B[q].dim for q in Qs]
        Z = lambda i, j: sp.csr_matrix((dims[i], dims[j]), dtype=complex)
        H0 = sp.block_diag([M.build_H(B[q], two, mu) for q in Qs], format='csr').astype(complex)
        PN = [[Z(i, j) for j in range(len(Qs))] for i in range(len(Qs))]
        PS = [[Z(i, j) for j in range(len(Qs))] for i in range(len(Qs))]
        for i, q in enumerate(Qs):
            if q + 1 in B:
                j = Qs.index(q + 1)
                PN[j][i] = op_matrix(B[q].states, B[q + 1].states, tN).astype(complex)
                PS[j][i] = op_matrix(B[q].states, B[q + 1].states, tS).astype(complex)
        PN, PS = sp.bmat(PN, format='csr'), sp.bmat(PS, format='csr')
        Qd = np.concatenate([np.full(d, q) for q, d in zip(Qs, dims)]).astype(float)
        print(f'N={N} alpha={alpha} hp={hp} w2={w2}: K=0 dim {H0.shape[0]}  pin {desc}  [{time.time() - t0:.0f}s]', flush=True)
        out = dict(config=cfgname, N=N, alpha=alpha, hp=hp, w2=w2, lam=lam, u=u, pin='even', desc=desc, points={})
        v0 = None
        for eta in etas:
            ph = np.exp(1j * np.pi * eta)
            H = (H0 + hp * u * (PN + PN.conj().T + ph * PS + np.conj(ph) * PS.conj().T)).tocsr()
            w, v = sla.eigsh(H, k=k, which='SA', tol=1e-11, v0=v0)
            o = np.argsort(w); w, v = w[o], v[:, o]; v0 = v[:, 0]
            rec = dict(eta=eta, E=w.tolist())
            if eta == 0:
                lev = (w - w[0]) / u
                rec['levels'] = lev.tolist()
                rec['ward'] = [float((lev[n] * abs(v[:, n].conj() @ (Qd * v[:, 0]))) ** 2 / 4) for n in range(k)]
                print(f'  levels ' + ' '.join(f'{x:.4f}' for x in lev[1:5]) + '  C^HQ ' + ' '.join(f'{x:.4f}' for x in rec['ward'][1:5]), flush=True)
            out['points'][f'{eta:g}'] = rec
        e0 = out['points']['0']['E'][0]
        out['Dj'] = {kk: (p['E'][0] - e0) / u for kk, p in out['points'].items()}
        print('  Dh_j: ' + ' '.join(f'{kk}:{x:.5f}' for kk, x in out['Dj'].items()) + f'  [{time.time() - t0:.0f}s]', flush=True)
        tag = '' if w2 == 1.0 else f'_w{w2:g}'
        store.put(os.path.join(HERE, f'pin_even_o2wf_N{N}_a{alpha:g}{tag}.json'), out)

    def main():
        run(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), [float(x) for x in sys.argv[4].split(',')],
            w2=float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/pin_even_junction_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def dmrg_even():
    """DMRG for the T-EVEN pinned monodromy line (pin_even_junction.py beyond ED), 2026-10-25.
Copy of dmrg_polar.py with the T-even pin (R1 + R2 at each pole, see pin_even_junction.py); the half pin of dmrg_polar.py
realizes a different lattice line at alpha = 0 and is superseded.

O(2) GZWH (o2wf) monodromy at angle alpha, core potential stripped (lam = 0), plus
  h_p u [ P_N + e^{i pi eta} P_S + h.c. ],  P_N = f0(m=j)^dag f12-(m=j-alpha)/sqrt2,  P_S = f12+(m=-j+alpha)^dag f0(m=-j)/sqrt2.
Conserved: N_e and K = L_z - alpha Q.  Per (N, eta): K = 0 ground state; at eta = 0 also two excited K = 0 states and the two
lowest K = 1 states, plus <n|Q|0> for the Ward estimator C^HQ = (level_n |<n|Q|0>|)^2 / 4.
usage: python dmrg_polar.py N alpha chi hp eta1,eta2,...  -> polar_results/dmrg_o2wf_N{N}_a{alpha}_chi{chi}.json
       python dmrg_polar.py gate N alpha chi hp             (against pin_junction_o2wf_N{N}_a{alpha}.json, ED)
"""
    import os, sys, json, time
    import numpy as np
    HERE = 'twist'
    sys.path.insert(0, HERE)
    OUT = os.path.join(HERE, 'even_results')
    SHIFT = 1.0          # -SHIFT per particle (exact at fixed N_e), see dmrg_kr.py


    def hc_u(N, chi):
        _dmrg_kr = globals()['dmrg_kr'](); f = _dmrg_kr.hc_u   # was: from dmrg_kr import hc_u as f
        return f(N, chi)


    def build(N, alpha, hp, h, u, eta=0.0, K=0, with_Q=False):
        CS = fuzzy.core_strip()   # was: import core_strip as CS
        _dmrg_twist = __import__('dmrg'); fermion_site, compress_mpo = _dmrg_twist.fermion_site, _dmrg_twist.compress_mpo   # was: from dmrg_twist import fermion_site, compress_mpo
        from tenpy.models.model import CouplingModel, MPOModel
        from tenpy.models.lattice import TrivialLattice
        fl, orbs, koff, two, mu, _, _ = CS.build('o2wf', N, alpha, 0.0, hc=h)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        f0, fp, fm = fl.names.index('0'), fl.names.index('12+'), fl.names.index('12-')
        _pin_even_junction = globals()['pin_even_junction'](); even_terms = _pin_even_junction.even_terms   # was: from pin_even_junction import even_terms
        tN, tS, _ = even_terms(fl, orbs, koff, off)
        mode_m = [m for o in orbs for m in o]
        kk = [jj + koff[f] for f, o in enumerate(orbs) for jj in range(len(o))]
        qq = [{'0': 0, '12+': 1, '12-': -1}[fl.names[f]] for f, o in enumerate(orbs) for jj in range(len(o))]
        charges = [(1, k) for k in kk]
        order = sorted(range(len(mode_m)), key=lambda k: (mode_m[k], k))
        pos = {k: i for i, k in enumerate(order)}
        lat = TrivialLattice([fermion_site(charges[k], ['N', 'K'], [1, 1]) for k in order], bc_MPS='finite')
        li = lambda p: list(lat.mps2lat_idx(p))
        model = CouplingModel(lat)
        for (k1, k2, k3, k4, v) in two:
            model.add_local_term(v, [('Cd', li(pos[k1])), ('Cd', li(pos[k2])), ('C', li(pos[k3])), ('C', li(pos[k4]))])
        for k, v in enumerate(mu):
            model.add_onsite_term(v - SHIFT, pos[k], 'N')
        if hp != 0:
            ph = np.exp(1j * np.pi * eta)
            for terms, c0 in ((tN, 1.0), (tS, ph)):
                for (a, b, c) in terms:
                    g = hp * u * c * c0
                    model.add_local_term(g, [('Cd', li(pos[a])), ('C', li(pos[b]))])
                    model.add_local_term(np.conj(g), [('Cd', li(pos[b])), ('C', li(pos[a]))])
        base = set(range(N))
        if K == 0:
            occ = base
        else:
            occ = next((base - {j}) | {k} for j in sorted(base, reverse=True) for k in range(N, len(kk)) if kk[k] - kk[j] == K)
        M = MPOModel(lat, compress_mpo(model.calc_H_MPO()))
        if not with_Q:
            return M, order, occ
        Qm = CouplingModel(lat)
        for k in range(len(kk)):
            if qq[k]:
                Qm.add_onsite_term(float(qq[k]), pos[k], 'N')
        return M, order, occ, Qm.calc_H_MPO()


    def solve(N, alpha, hp, h, u, chi, eta, K, nstates, sweeps=30, ward=False):
        _dmrg_twist = __import__('dmrg'); run_dmrg = _dmrg_twist.run_dmrg   # was: from dmrg_twist import run_dmrg
        from tenpy.networks.mpo import MPOEnvironment
        built = build(N, alpha, hp, h, u, eta, K, with_Q=ward)
        model, order, occ = built[:3]
        out, psis = [], []
        for i in range(nstates):
            t = time.time()
            E, psi, eng = run_dmrg(model, order, occ, chi_max=chi, sweeps=sweeps, orthogonal_to=list(psis))
            psis.append(psi)
            out.append(dict(E=float(np.real(E)) + SHIFT * N, trunc=float(eng.sweep_stats['max_trunc_err'][-1]),
                            chi=int(max(psi.chi)), t=time.time() - t))
        if ward:
            Qmpo = built[3]
            for i in range(nstates):
                env = MPOEnvironment(psis[i], Qmpo, psis[0])
                out[i]['Q0'] = abs(complex(env.full_contraction(0)))
        return out


    def run(N, alpha, chi, hp, etas):
        pass   # was os.makedirs(...): the store needs no output directory
        fn = os.path.join(OUT, f'dmrg_o2wf_N{N}_a{alpha:g}_chi{chi}.json')
        res = store.get(fn) if store.exists(fn) else None
        if res is None:
            h, u, src = hc_u(N, chi)
            res = dict(N=N, alpha=alpha, chi=chi, hp=hp, h=h, u=u, unit_source=src, points={})
            store.put(fn, res)
        for eta in etas:
            key = f'{eta:g}'
            P = res['points'].setdefault(key, dict(eta=eta))
            todo = (('K0', 0, 3, True), ('K1', 1, 2, False)) if eta == 0 else (('K0', 0, 1, False),)
            for lab, K, ns, ward in todo:
                if lab in P:
                    continue
                P[lab] = solve(N, alpha, hp, res['h'], res['u'], chi, eta, K, ns, ward=ward)
                store.put(fn, res)
            print(f"N={N} a={alpha} chi={chi} eta={eta:g}: E0 {P['K0'][0]['E']:.8f}  trunc {max(s['trunc'] for s in P['K0']):.0e}  "
                  f"[{sum(s['t'] for s in P['K0']):.0f}s]", flush=True)
        p0 = res['points']['0']
        e0, u = p0['K0'][0]['E'], res['u']
        lev = [(s['E'] - e0) / u for s in p0['K0']]
        msg = f"N={N}: K=0 levels " + ' '.join(f'{x:.5f}' for x in lev[1:])
        msg += '  C^HQ ' + ' '.join(f"{(lev[i] * p0['K0'][i]['Q0']) ** 2 / 4:.5f}" for i in range(1, len(lev)))
        if 'K1' in p0:
            msg += '  K=1 ' + ' '.join(f"{(s['E'] - e0) / u:.5f}" for s in p0['K1'])
        msg += '  Dj ' + ' '.join(f"{k}:{(p['K0'][0]['E'] - e0) / u:.5f}" for k, p in res['points'].items() if k != '0')
        print(msg, flush=True)
        return res


    def gate(N, alpha, chi, hp):
        ed = store.get(os.path.join(HERE, f'pin_even_o2wf_N{N}_a{alpha:g}.json'))
        res = run(N, alpha, chi, hp, [0, 0.0625, 1])
        e0d = res['points']['0']['K0'][0]['E']
        print('gate: K=0 levels ED ' + ' '.join(f'{x:.6f}' for x in ed['points']['0']['levels'][1:3]) +
              '  DMRG ' + ' '.join(f"{(s['E'] - e0d) / res['u']:.6f}" for s in res['points']['0']['K0'][1:]))
        print('gate: Dj(1/16) ED %.6f DMRG %.6f;  Dj(1) ED %.6f DMRG %.6f' % (
            ed['Dj']['0.0625'], (res['points']['0.0625']['K0'][0]['E'] - e0d) / res['u'],
            ed['Dj']['1'], (res['points']['1']['K0'][0]['E'] - e0d) / res['u']))
        print('gate: C^HQ ED %.6f' % ed['points']['0']['ward'][1])

    def main():
        if sys.argv[1] == 'gate':
            gate(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
        else:
            run(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), [float(x) for x in sys.argv[5].split(',')])

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/dmrg_even_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def even_series():
    """Combined ED (N_m <= 10) + DMRG (N_m >= 11) series for the T-even pinned monodromy line, with continuum fits.
C^ang uses eta = 1/16 for every size (the angle DMRG computes).
"""
    import json, glob, re, sys
    import numpy as np
    c16 = 1 - np.cos(np.pi / 16)


    def load(alpha):
        S = {}
        for f in store.glob(f'twist/pin_even_o2wf_N*_a{alpha:g}.json'):
            if '_w' in f:
                continue
            d = store.get(f); p0 = d['points']['0']
            S[d['N']] = dict(tilt=p0['levels'][1], CHQ=p0['ward'][1], Cang=d['Dj']['0.0625'] / c16, Dpi=d['Dj']['1'], src='ED')
        for f in store.glob(f'twist/even_results/dmrg_o2wf_N*_a{alpha:g}_chi*.json'):
            d = store.get(f); P = d['points']
            if '0' not in P or 'K0' not in P['0'] or '1' not in P:
                continue
            e0, u = P['0']['K0'][0]['E'], d['u']
            lev = [(s['E'] - e0) / u for s in P['0']['K0']]
            S[d['N']] = dict(tilt=lev[1], CHQ=(lev[1] * P['0']['K0'][1]['Q0']) ** 2 / 4,
                             Cang=(P['0.0625']['K0'][0]['E'] - e0) / u / c16, Dpi=(P['1']['K0'][0]['E'] - e0) / u,
                             D1=(P['0']['K1'][1]['E'] - e0) / u if 'K1' in P['0'] else None, src='DMRG')
        return dict(sorted(S.items()))


    def fit(N, y, form):
        N, y = np.asarray(N, float), np.asarray(y)
        cols = {'1/N': [np.ones_like(N), 1 / N], '1/N+1/N2': [np.ones_like(N), 1 / N, 1 / N ** 2], '1/N2': [np.ones_like(N), 1 / N ** 2]}[form]
        return np.linalg.lstsq(np.vstack(cols).T, y, rcond=None)[0][0]


    def geom(y):
        d1, d2 = y[-2] - y[-3], y[-1] - y[-2]
        q = d2 / d1
        return y[-1] + d2 * q / (1 - q) if 0 < q < 0.95 else float('nan')

    def main():
        for a in [float(x) for x in sys.argv[1:]]:
            S = load(a); Ns = list(S)
            print(f'alpha = {a}: N = {Ns} ({",".join(S[n]["src"] for n in Ns)})')
            for k in ['tilt', 'CHQ', 'Cang', 'Dpi']:
                y = [S[n][k] for n in Ns]
                hi = [n for n in Ns if n >= 8]; yh = [S[n][k] for n in hi]
                print(f'  {k:5s}: ' + ' '.join(f'{v:.4f}' for v in y) +
                      f'   1/N(N>=10) {fit([n for n in Ns if n >= 10], [S[n][k] for n in Ns if n >= 10], "1/N"):.4f}'
                      f'  1/N+1/N2(N>=8) {fit(hi, yh, "1/N+1/N2"):.4f}  1/N2(N>=10) {fit([n for n in Ns if n >= 10], [S[n][k] for n in Ns if n >= 10], "1/N2"):.4f}'
                      f'  geom {geom(y):.4f}')
            r = [S[n]['Dpi'] / (2 * S[n]['Cang']) for n in Ns]; rho = [S[n]['Dpi'] / (2 * S[n]['CHQ']) for n in Ns]
            print('  r    : ' + ' '.join(f'{v:.4f}' for v in r)); print('  rho  : ' + ' '.join(f'{v:.4f}' for v in rho))
            d1 = [(n, S[n].get('D1')) for n in Ns if S[n].get('D1')]
            if d1:
                print('  displacement candidate (DMRG): ' + ' '.join(f'N{n}:{v:.4f}' for n, v in d1))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/even_series_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def even_table():
    """Continuum table for the T-even pinned monodromy line (paper App. E.7 (b)), with the fit spreads used there.
C^HQ: a + b N^-2 and a + b N^-3 (last 3 sizes), and a fitted power (last 4).  C^ang, Delta_{+-}, r: linear in 1/N
(last 3), 1/N + 1/N^2 (N >= 8), and linear in the tilt excess (last 3 and last 4).  Ratios to alpha = 0 at equal N:
linear in 1/N (last 3), 1/N + 1/N^2 (all), and linear in the tilt excess of the twisted line (last 3).
Writes even_table.json.
"""
    import json
    import numpy as np
    from scipy.optimize import curve_fit
    _even_series = globals()['even_series'](); load = _even_series.load   # was: from even_series import load

    S0 = load(0.0)


    def lin(x, y):
        return np.polyfit(x, y, 1)[-1]


    def poly_inv(N, y, deg):
        return np.polyfit(1 / np.asarray(N, float), y, deg)[-1]


    def powfit(N, y):
        N = np.asarray(N, float)
        try:
            p, _ = curve_fit(lambda n, a, b, c: a + b * n ** (-c), N, y, p0=[y[-1], -1.0, 2.0], maxfev=20000)
            return p[0], p[2]
        except Exception:
            return np.nan, np.nan

    def main():
        out = {}
        for a in [0.0, 0.2, 0.3, 0.5]:
            S = load(a); Ns = np.array(list(S)); x = np.array([S[n]['tilt'] - 1 for n in Ns])
            g = lambda k: np.array([S[n][k] for n in Ns])
            CHQ, Cang, D = g('CHQ'), g('Cang'), g('Dpi'); r = D / (2 * Cang)
            hi = Ns >= 8
            def spread(y):
                return [poly_inv(Ns[-3:], y[-3:], 1), poly_inv(Ns[hi], y[hi], 2), lin(x[-3:], y[-3:]), lin(x[-4:], y[-4:])]
            chq = [np.polyfit(Ns[-3:] ** -2.0, CHQ[-3:], 1)[-1], np.polyfit(Ns[-3:] ** -3.0, CHQ[-3:], 1)[-1]]
            pw, pexp = powfit(Ns[-4:], CHQ[-4:]); chq.append(pw)
            row = dict(N=[int(n) for n in Ns], tilt_last=float(S[Ns[-1]]['tilt']), CHQ=chq, CHQ_power=pexp,
                       Cang=spread(Cang), Dpm=spread(D), r=spread(r),
                       D1=[(int(n), S[n]['D1']) for n in Ns if S[n].get('D1')])
            if a > 0:
                Nr = [n for n in Ns if n in S0]
                rat = {}
                for k, f in [('r', lambda s: s['Dpi'] / (2 * s['Cang'])), ('Cang', lambda s: s['Cang']), ('Dpm', lambda s: s['Dpi'])]:
                    y = np.array([f(S[n]) / f(S0[n]) for n in Nr]); xr = np.array([S[n]['tilt'] - 1 for n in Nr])
                    rat[k] = [poly_inv(Nr[-3:], y[-3:], 1), poly_inv(Nr, y, 2), lin(xr[-3:], y[-3:])]
                    rat[k + '_lastN'] = (int(Nr[-1]), float(y[-1]))
                row['ratio'] = rat
            out[a] = row
            mm = lambda v: f'{np.nanmin(v):.3f}-{np.nanmax(v):.3f}'
            print(f'alpha={a}: N={row["N"]} tilt(N={Ns[-1]})={row["tilt_last"]:.4f}  CHQ {mm(chq)} (p={pexp:.2f})  Cang {mm(row["Cang"])}'
                  f'  D+- {mm(row["Dpm"])}  r {mm(row["r"])}')
            print('   Cang fits', ' '.join(f'{v:.4f}' for v in row['Cang']), '| D fits', ' '.join(f'{v:.4f}' for v in row['Dpm']),
                  '| r fits', ' '.join(f'{v:.4f}' for v in row['r']), '| CHQ fits', ' '.join(f'{v:.4f}' for v in chq))
            if a > 0:
                print('   ratios: ' + '  '.join(f'{k} {mm(rat[k])} (N{rat[k+"_lastN"][0]}: {rat[k+"_lastN"][1]:.4f})' for k in ['r', 'Cang', 'Dpm']))
        store.put('twist/even_table.json', {str(k): v for k, v in out.items()})

    def cli(argv):
        with fuzzy.logged('twist/even_table.log'):
            main()
    return namespace(locals(), globals())


@section
def even_ratio():
    """Equal-size ratios of the T-even pinned line at twist alpha to the same pin at alpha = 0, with continuum fits.
Usage: python even_ratio.py alpha [alpha0 = 0].
"""
    import sys
    import numpy as np
    _even_series = globals()['even_series'](); load, fit, geom = _even_series.load, _even_series.fit, _even_series.geom   # was: from even_series import load, fit, geom

    def main():
        a = float(sys.argv[1]); a0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
        S, S0 = load(a), load(a0)
        Ns = [n for n in S if n in S0]
        print(f'ratio alpha={a} / alpha={a0}: N = {Ns}')
        for k, f in [('Cang', lambda s: s['Cang']), ('CHQ', lambda s: s['CHQ']), ('Dpi', lambda s: s['Dpi']),
                     ('r', lambda s: s['Dpi'] / (2 * s['Cang'])), ('tilt-1', lambda s: s['tilt'] - 1)]:
            y = [f(S[n]) / f(S0[n]) for n in Ns]
            out = f'  {k:6s}: ' + ' '.join(f'{v:.4f}' for v in y)
            if len(Ns) >= 3:
                out += f'   1/N(last3) {fit(Ns[-3:], y[-3:], "1/N"):.4f}  1/N+1/N2(all) {fit(Ns, y, "1/N+1/N2"):.4f}  geom {geom(y):.4f}'
            print(out)

    def cli(argv):
        sys.argv = list(argv)
        sys.argv = list(argv)
        with fuzzy.logged('twist/even_ratio_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def even_tiltfit():
    """Fits in the tilt excess x = tilt - 1 (the paper's Table tab:shape method) for the T-even pin at each alpha:
linear in x over the last k sizes, and quadratic over all sizes; plus the same for the alpha/0 ratio of r.
Usage: python even_tiltfit.py alpha [alpha ...]
"""
    import sys
    import numpy as np
    _even_series = globals()['even_series'](); load = _even_series.load   # was: from even_series import load


    def ext(x, y, deg):
        return np.polyfit(x, y, deg)[-1]

    def main():
        for a in [float(s) for s in sys.argv[1:]]:
            S = load(a); Ns = list(S); x = np.array([S[n]['tilt'] - 1 for n in Ns])
            q = {'r': [S[n]['Dpi'] / (2 * S[n]['Cang']) for n in Ns], 'CHQ': [S[n]['CHQ'] for n in Ns],
                 'Cang': [S[n]['Cang'] for n in Ns], 'Dpi': [S[n]['Dpi'] for n in Ns],
                 'D_Ward': [2 * S[n]['CHQ'] * S[n]['Dpi'] / (2 * S[n]['Cang']) for n in Ns]}
            print(f'alpha = {a}: N = {Ns}, tilt-1 = ' + ' '.join(f'{v:.3f}' for v in x))
            for k, y in q.items():
                y = np.array(y)
                print(f'  {k:6s}: lin(last3) {ext(x[-3:], y[-3:], 1):.4f}  lin(last4) {ext(x[-4:], y[-4:], 1):.4f}  quad(all) {ext(x, y, 2):.4f}')

    def cli(argv):
        sys.argv = list(argv)
        sys.argv = list(argv)
        with fuzzy.logged('twist/even_tiltfit_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def polar_analyze():
    """Tables and extrapolations for the polar-pinned monodromy line (pin_polar_junction.py output).

The superseded polar pin; kept because even_fits uses its fit_shape.
"""
    import json, glob, re, sys
    import numpy as np
    import os
    PREFIX = os.environ.get('PIN', 'pin_junction')


    def fit_shape(eta, D):
        """Dh_j = C t + v2 t^2 + v3 t^3 with t = 1 - cos(eta): least squares on all angles."""
        t = 1 - np.cos(np.pi * eta)
        A = np.vstack([t, t ** 2, t ** 3]).T
        c = np.linalg.lstsq(A, D, rcond=None)[0]
        return c


    def table(alpha):
        rows = []
        for f in sorted(store.glob(f'twist/{PREFIX}_o2wf_N*_a{alpha:g}.json'), key=lambda s: int(re.search(r'_N(\d+)_', s).group(1))):
            d = store.get(f)
            etas = np.array(sorted(float(k) for k in d['Dj']))
            D = np.array([d['Dj'][f'{e:g}'] for e in etas])
            if len(etas) < 5:
                continue
            p0 = d['points']['0']
            lev, ward = p0['levels'], p0['ward']
            it = int(np.argmax(ward[1:])) + 1                 # state carrying the Ward weight
            c = fit_shape(etas, D)
            small = etas[(etas > 0)].min()
            Cs = D[etas == small][0] / (1 - np.cos(np.pi * small))
            rows.append(dict(N=d['N'], tilt=lev[it], it=it, CHQ=ward[it], Cang=c[0], Csmall=Cs, Dpi=D[-1], r=D[-1] / (2 * c[0]),
                             rHQ=D[-1] / (2 * ward[it]), lev=lev[1:4]))
        return rows

    def main():
        for a in [float(x) for x in (sys.argv[1:] or ['0.3', '0.2', '0.5'])]:
            R = table(a)
            if not R:
                continue
            print(f'alpha = {a}')
            print('  N   tilt(level#)   C^HQ    C^ang(fit) C^ang(small)  Dj(pi)   r=Dj/2C^ang  r_HQ   low K=0 levels')
            for r in R:
                print(f"  {r['N']:2d}  {r['tilt']:.4f} ({r['it']})   {r['CHQ']:.4f}   {r['Cang']:.4f}     {r['Csmall']:.4f}     {r['Dpi']:.4f}   {r['r']:.4f}     {r['rHQ']:.4f}  "
                      + ' '.join(f'{x:.4f}' for x in r['lev']))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/polar_analyze_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def even_fits():
    """Continuum fits for the T-even pinned line: a + b/N (last 3 sizes), a + b/N + c/N^2 (all), geometric (last 3).
"""
    import json, glob, re, sys
    import numpy as np
    _polar_analyze = globals()['polar_analyze'](); fit_shape = _polar_analyze.fit_shape   # was: from polar_analyze import fit_shape


    def series(alpha):
        out = {}
        for f in store.glob(f'twist/pin_even_o2wf_N*_a{alpha:g}.json'):
            if '_w' in f:
                continue
            d = store.get(f); N = d['N']
            et = np.array(sorted(float(k) for k in d['Dj'])); D = np.array([d['Dj'][f'{e:g}'] for e in et])
            if len(et) < 5:
                continue
            p0 = d['points']['0']; it = int(np.argmax(p0['ward'][1:])) + 1
            C = fit_shape(et, D)[0]
            out[N] = dict(tilt=p0['levels'][it], CHQ=p0['ward'][it], Cang=C, Dpi=D[-1], r=D[-1] / (2 * C), rho=D[-1] / (2 * p0['ward'][it]))
        return dict(sorted(out.items()))


    def fits(Ns, y):
        Ns, y = np.array(Ns, float), np.array(y)
        res = {}
        A = np.vstack([np.ones(3), 1 / Ns[-3:]]).T; res['1/N(last3)'] = np.linalg.lstsq(A, y[-3:], rcond=None)[0][0]
        A = np.vstack([np.ones(len(Ns)), 1 / Ns, 1 / Ns ** 2]).T; res['1/N+1/N2(all)'] = np.linalg.lstsq(A, y, rcond=None)[0][0]
        d1, d2 = y[-2] - y[-3], y[-1] - y[-2]
        q = d2 / d1 if d1 != 0 else 0
        res['geom(last3)'] = y[-1] + d2 * q / (1 - q) if abs(q) < 0.95 else float('nan')
        return res

    def main():
        for a in [float(x) for x in sys.argv[1:]]:
            S = series(a)
            Ns = list(S)
            print(f'alpha = {a}: N = {Ns}')
            for key in ['tilt', 'CHQ', 'Cang', 'Dpi', 'r', 'rho']:
                ys = [S[n][key] for n in Ns]
                f = fits(Ns, ys) if len(Ns) >= 4 else {}
                print(f'  {key:5s}: ' + ' '.join(f'{v:.4f}' for v in ys) + '   ->  ' + '  '.join(f'{k} {v:.4f}' for k, v in f.items()))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/even_fits_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def disp_even():
    """Displacement of the T-even pinned monodromy line from the rotation Ward identity (issue 3: is the displacement at 2?).

Same identity as disp_tilt.py: <n|L_+|0_D> = <n|[H, L_+]|0_D> / (E_n - E_0) with [H, L_+] = i (D_+(north) - D_+(south)),
so L_+|0_D> lies in the displacement family and, the two poles entering with opposite sign, only its odd parallel
descendants carry weight: D itself has zero weight and d_par D (Delta = 3) the leading one.  The displacement is then the
L_z - alpha Q = 1 level one unit below the state that carries the weight, with no extrapolation of a level.
The pin mixes charge sectors, so both sectors K = 0 and K = 1 are built over all Q blocks, as in ed_even_k1.py.
usage: python disp_even.py N alpha [hp=1000] [nlev=30]   -> disp_results/even_o2wf_N{N}_a{alpha}.json
"""
    import os, sys, json, time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    M = fuzzy.onmodel()   # was: import onmodel as M
    CS = fuzzy.core_strip()   # was: import core_strip as CS
    _pin_flow = globals()['pin_flow'](); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix
    _pin_even_junction = globals()['pin_even_junction'](); even_terms = _pin_even_junction.even_terms   # was: from pin_even_junction import even_terms
    _disp_tilt = fuzzy.question('01_known_results/free_control').disp_tilt(); raise_op, identify = _disp_tilt.raise_op, _disp_tilt.identify   # was: from disp_tilt import raise_op, identify

    HERE = 'twist'
    OUT = os.path.join(HERE, 'disp_results')


    def sector(fl, orbs, koff, N, K, two, mu, terms, hpu):
        Qs = [q for q in range(-N, N + 1) if M.Basis(fl, orbs, koff, N, [q], K, ()).dim > 0]
        B = {q: M.Basis(fl, orbs, koff, N, [q], K, ()) for q in Qs}
        dims = [B[q].dim for q in Qs]
        H0 = sp.block_diag([M.build_H(B[q], two, mu) for q in Qs], format='csr')
        P = [[sp.csr_matrix((dims[i], dims[j])) for j in range(len(Qs))] for i in range(len(Qs))]
        for i, q in enumerate(Qs):
            if q + 1 in B:
                P[Qs.index(q + 1)][i] = op_matrix(B[q].states, B[q + 1].states, terms)
        P = sp.bmat(P, format='csr')
        return (H0 + hpu * (P + P.T)).tocsr(), Qs, B, dims


    def run(N, alpha, hp=1000.0, nlev=30):
        t0 = time.time()
        R = store.get(os.path.join(HERE, 'on_results', f'o2wf_N{N}.json'))
        bulk = R['alpha']['0.0000']; u = bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0]
        fl, orbs, koff, two, mu, _, _ = CS.build('o2wf', N, alpha, 0.0)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        tN, tS, _ = even_terms(fl, orbs, koff, off)
        H0, Q0, B0, d0 = sector(fl, orbs, koff, N, 0, two, mu, tN + tS, hp * u)
        H1, Q1, B1, d1 = sector(fl, orbs, koff, N, 1, two, mu, tN + tS, hp * u)
        s = (N - 1) / 2
        L = [[sp.csr_matrix((d1[i], d0[j])) for j in range(len(Q0))] for i in range(len(Q1))]
        for j, q in enumerate(Q0):                     # L_+ conserves the charge: block (q in K = 1) <- (q in K = 0)
            if q in B1:
                L[Q1.index(q)][j] = sum(raise_op(B0[q].states, B1[q].states, list(orbs[f]), int(off[f]), s)
                                        for f in range(fl.F))
        Lp = sp.bmat(L, format='csr')
        w0, v0 = sla.eigsh(H0, k=2, which='SA', tol=1e-12); o = np.argsort(w0); E0, g = w0[o[0]], v0[:, o[0]]
        w1, v1 = sla.eigsh(H1, k=nlev, which='SA', tol=1e-11); o = np.argsort(w1); w1, v1 = w1[o], v1[:, o]
        v = Lp @ g
        a2 = np.abs(v1.T @ v) ** 2
        lev, p = (w1 - E0) / u, a2
        out = dict(N=N, alpha=alpha, hp=hp, dims=[int(sum(d0)), int(sum(d1))], levels=lev.tolist(), p=p.tolist(),
                   norm2=float(v @ v), p_window=float(p.sum() / (v @ v)), seconds=time.time() - t0)
        out.update(identify(lev, p))
        pass   # was os.makedirs(...): the store needs no output directory
        store.put(os.path.join(OUT, f'even_o2wf_N{N}_a{alpha:g}.json'), out)
        print(f'N={N} alpha={alpha}: dims {out["dims"]}, window holds {out["p_window"]:.3f} of |L+|0>|^2 [{out["seconds"]:.0f}s]')
        for i in range(min(nlev, 14)):
            print(f'   {lev[i]:7.4f}  p {p[i]:.3e}')
        if 'dD' in out:
            print(f"  d_par D at {out['dD']['level']:.4f} (share {out['dD']['share']:.3f});  D at {out['D']['level']:.4f}, "
                  f"p(D)/p(dD) {out['D']['p_over_dD']:.2e} (CFT 0)", flush=True)

    def main():
        run(int(sys.argv[1]), float(sys.argv[2]), *(float(x) for x in sys.argv[3:4]),
            *(int(x) for x in sys.argv[4:5]))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/disp_even_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def ed_even_k1():
    """K = 1 (and K = 0 reference) levels of the saturated T-even pinned monodromy line at eta = 0, for the displacement check.
usage: python ed_even_k1.py N alpha hp -> prints levels relative to the K = 0 ground state, bulk units.
"""
    import os, sys, json
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    M = fuzzy.onmodel()   # was: import onmodel as M
    CS = fuzzy.core_strip()   # was: import core_strip as CS
    _pin_flow = globals()['pin_flow'](); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix
    _pin_even_junction = globals()['pin_even_junction'](); even_terms = _pin_even_junction.even_terms   # was: from pin_even_junction import even_terms

    HERE = 'twist'


    def levels(N, alpha, hp, K, fl, orbs, koff, two, mu, terms, u, k=5):
        Qs = [q for q in range(-N, N + 1) if M.Basis(fl, orbs, koff, N, [q], K, ()).dim > 0]
        B = {q: M.Basis(fl, orbs, koff, N, [q], K, ()) for q in Qs}
        dims = [B[q].dim for q in Qs]
        H0 = sp.block_diag([M.build_H(B[q], two, mu) for q in Qs], format='csr')
        P = [[sp.csr_matrix((dims[i], dims[j])) for j in range(len(Qs))] for i in range(len(Qs))]
        for i, q in enumerate(Qs):
            if q + 1 in B:
                j = Qs.index(q + 1)
                P[j][i] = op_matrix(B[q].states, B[q + 1].states, terms)
        P = sp.bmat(P, format='csr')
        w = sla.eigsh((H0 + hp * u * (P + P.T)).tocsr(), k=k, which='SA', tol=1e-11)[0]
        return np.sort(w)


    def run(N, alpha, hp):
        R = store.get(os.path.join(HERE, 'on_results', f'o2wf_N{N}.json'))
        bulk = R['alpha']['0.0000']; u = bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0]
        fl, orbs, koff, two, mu, _, _ = CS.build('o2wf', N, alpha, 0.0)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        tN, tS, _ = even_terms(fl, orbs, koff, off)
        w0 = levels(N, alpha, hp, 0, fl, orbs, koff, two, mu, tN + tS, u)
        w1 = levels(N, alpha, hp, 1, fl, orbs, koff, two, mu, tN + tS, u)
        e0 = w0[0]
        print(f'N={N} alpha={alpha}: K=0 ' + ' '.join(f'{(x - e0) / u:.4f}' for x in w0[1:4]) +
              '   K=1 ' + ' '.join(f'{(x - e0) / u:.4f}' for x in w1[:4]), flush=True)

    def main():
        run(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/ed_even_k1_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def pin_check_full():
    """Calibration of pin_polar_junction.py at alpha = 0: the FULL ordinary O(2) pin (both charge-raising bilinears at each
pole, i.e. f0^dag f1 + h.c. along e_1) must reproduce the paper's vector O(2) junction table (tilt 1.3782, C^ang 0.3721,
Dh_j(pi) 0.7823 at N = 6), while the 'half' pin (one bilinear per pole, the alpha -> 0 limit of the polar monodromy pin)
is compared with it.  usage: python pin_check_full.py N hp mode(full|half) eta1,...  (no files written)
"""
    import os, sys, json, time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    M = fuzzy.onmodel()   # was: import onmodel as M
    CS = fuzzy.core_strip()   # was: import core_strip as CS
    _pin_flow = globals()['pin_flow'](); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix

    HERE = 'twist'


    def run(N, hp, mode, etas, k=6):
        R = store.get(os.path.join(HERE, 'on_results', f'o2wf_N{N}.json'))
        bulk = R['alpha']['0.0000']
        u = bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0]
        fl, orbs, koff, two, mu, _, _ = CS.build('o2wf', N, 0.0, 0.0)
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
        f0, fp, fm = fl.names.index('0'), fl.names.index('12+'), fl.names.index('12-')
        top = lambda f: int(off[f] + len(orbs[f]) - 1); bot = lambda f: int(off[f])
        if mode == 'full':
            c = 0.5
            tN = [(top(f0), top(fm), c), (top(fp), top(f0), c)]
            tS = [(bot(f0), bot(fm), c), (bot(fp), bot(f0), c)]
        elif mode == 'shift':          # T-even pair realizable at alpha > 0: second bilinear one orbital in from the pole
            c = 0.5
            tN = [(top(f0), top(fm), c), (top(fp) - 1, top(f0) - 1, c)]
            tS = [(bot(fp), bot(f0), c), (bot(f0) + 1, bot(fm) + 1, c)]
        else:
            c = 1 / np.sqrt(2)
            tN = [(top(f0), top(fm), c)]; tS = [(bot(fp), bot(f0), c)]
        Qs = [q for q in range(-N, N + 1) if M.Basis(fl, orbs, koff, N, [q], 0, ()).dim > 0]
        B = {q: M.Basis(fl, orbs, koff, N, [q], 0, ()) for q in Qs}
        dims = [B[q].dim for q in Qs]
        Z = lambda i, j: sp.csr_matrix((dims[i], dims[j]), dtype=complex)
        H0 = sp.block_diag([M.build_H(B[q], two, mu) for q in Qs], format='csr').astype(complex)
        PN = [[Z(i, j) for j in range(len(Qs))] for i in range(len(Qs))]
        PS = [[Z(i, j) for j in range(len(Qs))] for i in range(len(Qs))]
        for i, q in enumerate(Qs):
            if q + 1 in B:
                j = Qs.index(q + 1)
                PN[j][i] = op_matrix(B[q].states, B[q + 1].states, tN).astype(complex)
                PS[j][i] = op_matrix(B[q].states, B[q + 1].states, tS).astype(complex)
        PN, PS = sp.bmat(PN, format='csr'), sp.bmat(PS, format='csr')
        Qd = np.concatenate([np.full(d, q) for q, d in zip(Qs, dims)]).astype(float)
        E = {}
        for eta in etas:
            ph = np.exp(1j * np.pi * eta)
            H = (H0 + hp * u * (PN + PN.conj().T + ph * PS + np.conj(ph) * PS.conj().T)).tocsr()
            w, v = sla.eigsh(H, k=k, which='SA', tol=1e-11)
            o = np.argsort(w); w, v = w[o], v[:, o]
            E[eta] = w[0]
            if eta == 0:
                lev = (w - w[0]) / u
                ward = [(lev[n] * abs(v[:, n].conj() @ (Qd * v[:, 0]))) ** 2 / 4 for n in range(k)]
                print(f'N={N} {mode} hp={hp}: levels ' + ' '.join(f'{x:.4f}' for x in lev[1:4]) + '  C^HQ ' + ' '.join(f'{x:.4f}' for x in ward[1:4]))
        s = min(e for e in etas if e > 0)
        print(f'   C^ang(eta={s}) = {(E[s] - E[0]) / u / (1 - np.cos(np.pi * s)):.4f}   Dh_j(pi) = {(E[1] - E[0]) / u:.4f}', flush=True)

    def main():
        run(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], [float(x) for x in sys.argv[4].split(',')])

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/pin_check_full_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def largeN_straight():
    """Large-N straight monodromy pinning line on S^1 x H^2 (d = 3).

Background Phi^1 = sqrt(N) phi_* e^{i v theta} (constant on H^2); it solves the conformally coupled equation only for
varsigma_* = 1/4 - v^2, so the twisted modes have Dh_s = 1/2 + sqrt(s^2 + 1/4 - v^2), s in Z + v, and the s = v mode sits at
Dh = 1 (the tilt).  The constraint (<phi^2> equal to its bulk value) fixes
  phi_*^2 = (1/4 pi^2) Reg[ sum_{Z+v} dg(1/2 + sqrt(s^2 + 1/4 - v^2)) - sum_Z dg(1/2 + |s|) ]
          = (1/4 pi^2) [ sum_{Z+v} (dg(1/2 + sqrt(s^2+1/4-v^2)) - log|s|) + log(2 sin(pi v)) ],
using Reg[sum_Z dg(1/2+|s|) - sum_{Z+v} log|s|] = -log(2 sin pi v) (the full-twist gap equation at varsigma = 0).
Stationarity in varsigma gives C_tilt = 2 N phi_*^2 exactly at large N (as for the ordinary pinned line).
v -> 0 must reproduce phi_0^2 = 0.0444132 (ordinary pinned line, Eq. (v0) of the paper).
"""
    from mpmath import mp, mpf, psi, log, sin, pi, sqrt, nsum, inf
    mp.dps = 25


    def phi2(v):
        v = mpf(v)
        c = mpf(1) / 4 - v ** 2
        f = lambda s: psi(0, mpf(1) / 2 + sqrt(s ** 2 + c)) - log(abs(s))
        S = nsum(lambda n: f(v + n), [0, inf]) + nsum(lambda n: f(v - 1 - n), [0, inf])
        return (S + log(2 * sin(pi * v))) / (4 * pi ** 2)


    def phi2_untwisted():
        c = mpf(1) / 4
        return nsum(lambda n: (psi(0, mpf(1) / 2 + sqrt(n ** 2 + c)) - psi(0, mpf(1) / 2 + abs(n))), [-inf, inf]) / (4 * pi ** 2)

    def main():
        print("ordinary pinned line phi_0^2 =", mp.nstr(phi2_untwisted(), 10), "(paper: 0.0444132)")
        for v in ['0.001', '0.01', '0.1', '0.2', '0.25', '0.3', '0.4', '0.45', '0.5']:
            p = phi2(v)
            print(f'v = {v:5s}: phi_*^2 = {mp.nstr(p, 8)}   C_tilt/N = 2 phi_*^2 = {mp.nstr(2 * p, 8)}   a_phi^2/N = 4 pi phi_*^2 = {mp.nstr(4 * pi * p, 8)}')

    def cli(argv):
        mp.dps = 25
        with fuzzy.logged('analytic/monodromy_pinning/largeN_straight.log'):
            main()
    return namespace(locals(), globals())


@section
def junction_twisted_largeN():
    """Large-N junction of the monodromy pinning line (all N complex components twisted by v, Phi^1 pinned), d = 3.

Generalizes ../../largeN/junction_largeM.py (CCS, ordinary pin, v = 0).  Straight line: Phi^1 = sqrt(N) phi_* e^{i v theta},
varsigma_* = 1/4 - v^2 (largeN_straight.py gives phi_*^2).  Junction: Phi^1/sqrt(N) = e^{i v theta}[A(w) + e^{i eta} B(w)] on
H^2 = {(t, w)}, with -A'' + dsig A/sin^2 w = 0 (dsig = varsigma - varsigma_*), A(pi) = B(0) = phi_*, A(0) = B(pi) = 0, and
    G_varsigma(x,x) - G_varsigma_*(x,x) = phi_*^2 - |A + e^{i eta} B|^2,
    G(x,x) = (1/2pi) sum_{s in Z+v} int dw/2pi g_{s,w},  [-d^2 + w^2 + (s^2 - v^2 + dsig)/sin^2] g = delta.
The s = v mode is massless in H^2 (the tilt at Dh = 1).  As for v = 0, dF/deta = -2 pi phi_* sin(eta) B'(pi), F(dsig=0) =
2 phi_*^2 (1 - cos eta), so C_t = 2 N phi_*^2 and r_inf(v) = (1/2) int_0^pi sin(eta) b(eta), b = -pi B'(pi)/phi_*.
v = 0 must reproduce r_inf = 1.551 (CCS).
"""
    import sys, json, time
    import numpy as np
    sys.path.insert(0, '../../largeN')
    _junction_largeM = fuzzy.question('04_pinned_lines/junction_theory').junction_largeM(); Grid, tridiag_diag_inv, omega_nodes, lap, solve_V = _junction_largeM.Grid, _junction_largeM.tridiag_diag_inv, _junction_largeM.omega_nodes, _junction_largeM.lap, _junction_largeM.solve_V   # was: from junction_largeM import Grid, tridiag_diag_inv, omega_nodes, lap, solve_V
    _largeN_straight = globals()['largeN_straight'](); phi2 = _largeN_straight.phi2   # was: from largeN_straight import phi2

    pi = np.pi


    def modes(v, S):
        return [v + n for n in range(-S - 1, S + 1) if abs(v + n) <= S + 0.5]


    def tail(v, S, s2):
        big = np.array([v + n for n in range(-S - 4000, S + 4001) if abs(v + n) > S + 0.5])[:, None]
        return ((1 / (pi * (big ** 2 / s2[None, :]))).sum(0)) / (2 * pi)


    def dG(grid, dsig, v, S, nw, c=2.0):
        h, s2 = grid.h, grid.s2
        om, wq = omega_nodes(nw, c)
        out = np.zeros(grid.N); off = -1 / h ** 2
        for s in modes(v, S):
            base = 2 / h ** 2 + (s ** 2 - v ** 2) / s2
            d0 = base[None, :] + om[:, None] ** 2
            d1 = d0 + dsig[None, :] / s2[None, :]
            out += (wq[:, None] * (tridiag_diag_inv(d1, off) - tridiag_diag_inv(d0, off))).sum(0) / h / pi / (2 * pi)
        out += -dsig / (4 * s2) * tail(v, S, s2)
        return out


    def jac(grid, dsig, v, S, nw, c=2.0):
        h, s2, N = grid.h, grid.s2, grid.N
        om, wq = omega_nodes(nw, c)
        J = np.zeros((N, N)); L0 = lap(grid) + np.diag(dsig / s2)
        for s in modes(v, S):
            for w, q in zip(om, wq):
                Gi = np.linalg.inv(L0 + np.diag((s ** 2 - v ** 2) / s2 + w ** 2))
                J -= q / pi / (2 * pi) * (Gi ** 2) / h / s2[None, :]
        J -= np.diag(tail(v, S, s2) / (4 * s2))
        return J


    def residual(grid, dsig, eta, v, p, S, nw):
        A, B, L = solve_V(grid, dsig, p)
        V2 = A ** 2 + B ** 2 + 2 * np.cos(eta) * A * B
        return dG(grid, dsig, v, S, nw) - p ** 2 + V2, A, B, L


    def solve(grid, eta, v, p, S, nw, dsig0=None, tol=1e-10, maxit=40):
        dsig = np.zeros(grid.N) if dsig0 is None else dsig0.copy()
        R, A, B, L = residual(grid, dsig, eta, v, p, S, nw)
        for it in range(maxit):
            nr = np.max(abs(R))
            if nr < tol:
                break
            JG = jac(grid, dsig, v, S, nw)
            Li = np.linalg.inv(L)
            dA = -Li * (A / grid.s2)[None, :]; dB = -Li * (B / grid.s2)[None, :]
            JV = 2 * (A + np.cos(eta) * B)[:, None] * dA + 2 * (B + np.cos(eta) * A)[:, None] * dB
            step = -np.linalg.solve(JG + JV, R)
            lam = 1.0
            while lam > 1e-4:
                tr = dsig + lam * step
                Rt, At, Bt, Lt = residual(grid, tr, eta, v, p, S, nw)
                if np.all(np.isfinite(Rt)) and np.max(abs(Rt)) < (1 - 0.3 * lam) * nr:
                    break
                lam /= 2
            dsig, R, A, B, L = tr, Rt, At, Bt, Lt
        Bp = (-4 * B[-1] + B[-2]) / (2 * grid.h)
        return dict(dsig=dsig, b=-pi * Bp / p, res=float(np.max(abs(R))), it=it)


    def r_inf(v, N=120, S=30, nw=40, neta=12):
        t0 = time.time()
        grid = Grid(N); p = float(np.sqrt(float(phi2(v)))) if v > 0 else float(np.sqrt(0.04441318577))
        x, w = np.polynomial.legendre.leggauss(neta)
        etas = pi / 2 * (x + 1); we = pi / 2 * w
        bs, prev = [], None
        for eta in etas:
            r = solve(grid, eta, v, p, S, nw, dsig0=prev)
            assert r['res'] < 1e-8, (v, eta, r['res'])
            prev = r['dsig']; bs.append(r['b'])
        rr = 0.5 * float(np.sum(we * np.sin(etas) * np.array(bs)))
        rp = solve(grid, pi, v, p, S, nw, dsig0=prev)
        out = dict(v=v, N=N, S=S, nw=nw, neta=neta, phi2=p ** 2, Ct_over_N=2 * p ** 2, r_inf=rr, wall_over_N=4 * p ** 2 * rr,
                   b_pi=float(rp['b']), sigma_min_pi=float(0.25 - v ** 2 + rp['dsig'].min()), seconds=time.time() - t0)
        print(f"v={v}: phi_*^2={p**2:.6f}  C_t/N={2*p**2:.6f}  r_inf={rr:.6f}  Delta+-/N={4*p**2*rr:.6f}  b(pi)={rp['b']:.4f}  "
              f"sigma_min(pi)={out['sigma_min_pi']:+.4f}  [{out['seconds']:.0f}s]", flush=True)
        return out

    def main():
        vs = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0.0, 0.2, 0.3, 0.5]
        cfg = tuple(int(x) for x in sys.argv[2].split(',')) if len(sys.argv) > 2 else (120, 30, 40, 12)
        res = [r_inf(v, *cfg) for v in vs]
        store.put(f'analytic/monodromy_pinning/junction_twisted_largeN_N{cfg[0]}_S{cfg[1]}.json', res)

    def cli(argv):
        sys.argv = list(argv)
        sys.argv = list(argv)
        with fuzzy.logged('analytic/monodromy_pinning/junction_twisted_largeN_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def junction_twotwist_largeN():
    """Large-N junction of the monodromy pinning line with ONE twisted complex pair (pinned) and N - 2 untwisted spectators, d = 3
(the partial-twist model of the paper's entropy section).  At leading order in 1/N the constraint is carried by the
spectators: their circle modes are s in Z, in the background varsigma_* = 1/4 - v^2 fixed by the pinned field e^{i v theta},
so the propagator potential is (s^2 - v^2 + dsig)/sin^2 as in the all-twisted solver, only the mode set changes, and
  phi_*^2 = (1/4 pi^2) sum_{s in Z} [dg(1/2 + sqrt(s^2 + 1/4 - v^2)) - dg(1/2 + |s|)].
Derived from junction_twisted_largeN.py (all components twisted), whose docstring follows.

Large-N junction of the monodromy pinning line (all N complex components twisted by v, Phi^1 pinned), d = 3.

Generalizes ../../largeN/junction_largeM.py (CCS, ordinary pin, v = 0).  Straight line: Phi^1 = sqrt(N) phi_* e^{i v theta},
varsigma_* = 1/4 - v^2 (largeN_straight.py gives phi_*^2).  Junction: Phi^1/sqrt(N) = e^{i v theta}[A(w) + e^{i eta} B(w)] on
H^2 = {(t, w)}, with -A'' + dsig A/sin^2 w = 0 (dsig = varsigma - varsigma_*), A(pi) = B(0) = phi_*, A(0) = B(pi) = 0, and
    G_varsigma(x,x) - G_varsigma_*(x,x) = phi_*^2 - |A + e^{i eta} B|^2,
    G(x,x) = (1/2pi) sum_{s in Z+v} int dw/2pi g_{s,w},  [-d^2 + w^2 + (s^2 - v^2 + dsig)/sin^2] g = delta.
The s = v mode is massless in H^2 (the tilt at Dh = 1).  As for v = 0, dF/deta = -2 pi phi_* sin(eta) B'(pi), F(dsig=0) =
2 phi_*^2 (1 - cos eta), so C_t = 2 N phi_*^2 and r_inf(v) = (1/2) int_0^pi sin(eta) b(eta), b = -pi B'(pi)/phi_*.
v = 0 must reproduce r_inf = 1.551 (CCS).
"""
    import sys, json, time
    import numpy as np
    sys.path.insert(0, '../../largeN')
    _junction_largeM = fuzzy.question('04_pinned_lines/junction_theory').junction_largeM(); Grid, tridiag_diag_inv, omega_nodes, lap, solve_V = _junction_largeM.Grid, _junction_largeM.tridiag_diag_inv, _junction_largeM.omega_nodes, _junction_largeM.lap, _junction_largeM.solve_V   # was: from junction_largeM import Grid, tridiag_diag_inv, omega_nodes, lap, solve_V
    from mpmath import mp, mpf, psi as dg, sqrt as msqrt, nsum, inf, pi as mpi


    def phi2(v):
        mp.dps = 20
        c = mpf(1) / 4 - mpf(v) ** 2
        f = lambda n: dg(0, mpf(1) / 2 + msqrt(n ** 2 + c)) - dg(0, mpf(1) / 2 + abs(n))
        return float((f(0) + 2 * nsum(f, [1, inf])) / (4 * mpi ** 2))

    pi = np.pi


    def modes(v, S):
        return list(range(-S, S + 1))                    # spectators: untwisted circle modes


    def tail(v, S, s2):
        big = np.array([n for n in range(-S - 4000, S + 4001) if abs(n) > S])[:, None]
        return ((1 / (pi * (big ** 2 / s2[None, :]))).sum(0)) / (2 * pi)


    def dG(grid, dsig, v, S, nw, c=2.0):
        h, s2 = grid.h, grid.s2
        om, wq = omega_nodes(nw, c)
        out = np.zeros(grid.N); off = -1 / h ** 2
        for s in modes(v, S):
            base = 2 / h ** 2 + (s ** 2 - v ** 2) / s2
            d0 = base[None, :] + om[:, None] ** 2
            d1 = d0 + dsig[None, :] / s2[None, :]
            out += (wq[:, None] * (tridiag_diag_inv(d1, off) - tridiag_diag_inv(d0, off))).sum(0) / h / pi / (2 * pi)
        out += -dsig / (4 * s2) * tail(v, S, s2)
        return out


    def jac(grid, dsig, v, S, nw, c=2.0):
        h, s2, N = grid.h, grid.s2, grid.N
        om, wq = omega_nodes(nw, c)
        J = np.zeros((N, N)); L0 = lap(grid) + np.diag(dsig / s2)
        for s in modes(v, S):
            for w, q in zip(om, wq):
                Gi = np.linalg.inv(L0 + np.diag((s ** 2 - v ** 2) / s2 + w ** 2))
                J -= q / pi / (2 * pi) * (Gi ** 2) / h / s2[None, :]
        J -= np.diag(tail(v, S, s2) / (4 * s2))
        return J


    def residual(grid, dsig, eta, v, p, S, nw):
        A, B, L = solve_V(grid, dsig, p)
        V2 = A ** 2 + B ** 2 + 2 * np.cos(eta) * A * B
        return dG(grid, dsig, v, S, nw) - p ** 2 + V2, A, B, L


    def solve(grid, eta, v, p, S, nw, dsig0=None, tol=1e-10, maxit=40):
        dsig = np.zeros(grid.N) if dsig0 is None else dsig0.copy()
        R, A, B, L = residual(grid, dsig, eta, v, p, S, nw)
        for it in range(maxit):
            nr = np.max(abs(R))
            if nr < tol:
                break
            JG = jac(grid, dsig, v, S, nw)
            Li = np.linalg.inv(L)
            dA = -Li * (A / grid.s2)[None, :]; dB = -Li * (B / grid.s2)[None, :]
            JV = 2 * (A + np.cos(eta) * B)[:, None] * dA + 2 * (B + np.cos(eta) * A)[:, None] * dB
            step = -np.linalg.solve(JG + JV, R)
            lam = 1.0
            while lam > 1e-4:
                tr = dsig + lam * step
                Rt, At, Bt, Lt = residual(grid, tr, eta, v, p, S, nw)
                if np.all(np.isfinite(Rt)) and np.max(abs(Rt)) < (1 - 0.3 * lam) * nr:
                    break
                lam /= 2
            dsig, R, A, B, L = tr, Rt, At, Bt, Lt
        Bp = (-4 * B[-1] + B[-2]) / (2 * grid.h)
        return dict(dsig=dsig, b=-pi * Bp / p, res=float(np.max(abs(R))), it=it)


    def r_inf(v, N=120, S=30, nw=40, neta=12):
        t0 = time.time()
        grid = Grid(N); p = float(np.sqrt(float(phi2(v)))) if v > 0 else float(np.sqrt(0.04441318577))
        x, w = np.polynomial.legendre.leggauss(neta)
        etas = pi / 2 * (x + 1); we = pi / 2 * w
        bs, prev = [], None
        for eta in etas:
            r = solve(grid, eta, v, p, S, nw, dsig0=prev)
            assert r['res'] < 1e-8, (v, eta, r['res'])
            prev = r['dsig']; bs.append(r['b'])
        rr = 0.5 * float(np.sum(we * np.sin(etas) * np.array(bs)))
        rp = solve(grid, pi, v, p, S, nw, dsig0=prev)
        out = dict(v=v, N=N, S=S, nw=nw, neta=neta, phi2=p ** 2, Ct_over_N=2 * p ** 2, r_inf=rr, wall_over_N=4 * p ** 2 * rr,
                   b_pi=float(rp['b']), sigma_min_pi=float(0.25 - v ** 2 + rp['dsig'].min()), seconds=time.time() - t0)
        print(f"v={v}: phi_*^2={p**2:.6f}  C_t/N={2*p**2:.6f}  r_inf={rr:.6f}  Delta+-/N={4*p**2*rr:.6f}  b(pi)={rp['b']:.4f}  "
              f"sigma_min(pi)={out['sigma_min_pi']:+.4f}  [{out['seconds']:.0f}s]", flush=True)
        return out

    def main():
        vs = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0.0, 0.2, 0.3, 0.5]
        cfg = tuple(int(x) for x in sys.argv[2].split(',')) if len(sys.argv) > 2 else (120, 30, 40, 12)
        res = [r_inf(v, *cfg) for v in vs]
        store.put(f'analytic/monodromy_pinning/junction_twotwist_largeN_N{cfg[0]}_S{cfg[1]}.json', res)

    def cli(argv):
        sys.argv = list(argv)
        sys.argv = list(argv)
        with fuzzy.logged('analytic/monodromy_pinning/junction_twotwist_largeN_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


STEPS = {'pin_flow': pin_flow, 'kr_test': kr_test, 'kr_delta': kr_delta, 'dmrg_kr': dmrg_kr, 'kr_descendant_overlap': kr_descendant_overlap, 'kr_pt2': kr_pt2, 'kr_delta_table': kr_delta_table, 'kr_final': kr_final, 'pin_even_junction': pin_even_junction, 'dmrg_even': dmrg_even, 'even_series': even_series, 'even_table': even_table, 'even_ratio': even_ratio, 'even_tiltfit': even_tiltfit, 'polar_analyze': polar_analyze, 'even_fits': even_fits, 'disp_even': disp_even, 'ed_even_k1': ed_even_k1, 'pin_check_full': pin_check_full, 'largeN_straight': largeN_straight, 'junction_twisted_largeN': junction_twisted_largeN, 'junction_twotwist_largeN': junction_twotwist_largeN}
REPORT = [('kr_delta_table',), ('even_table',), ('largeN_straight',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
