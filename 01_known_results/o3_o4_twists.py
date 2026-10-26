"""How do the O(3) and O(4) monodromy lines compare with O(2)?

The O(3) model with one twisted plane (R_z(pi)), the centre twist g = -1 and the reflection diag(1, 1, -1), with and
without the regulator's local potential, and the O(4) model with two twisted planes. The vector-model defect levels
themselves are produced by the shared step run_on_defect (configs o3wf, o4wf_diag, o4wf_pair).

usage: python o3_o4_twists.py <step> [args]      (python o3_o4_twists.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def o3_twists():
    """The two further Z2 monodromies of the O(3) vector model studied in arXiv:2610.31269, with and without the core
potential: the centre twist g = -1 (all three components twisted) and the reflection g = diag(1, 1, -1) (only phi_3).

The (1,2) pair is diagonalised into psi_+- (charge S^z = +-1); its twist is on (centre) or off (reflection).  The
unpaired psi_3 gets the half-shifted orbital grid by patching onmodel.Flavours so that its twisted charge is 1 at
alpha = 1/2 (flux 1/2), which is the sign monodromy of phi_3 = psi_0^+ psi_3 + h.c.  Sectors are labelled by S^z, the exact
L_z (Basis with Lz, mscale = 2) and the parity P_3 = (-1)^{N_3}, which for a twisted psi_3 is tied to 2 L_z and S^z.  The core potential is stripped as in twist/core_strip.py: lam = 0 removes
the deviation of every flavour's ordering one-body term from its bulk constant.  Gate: with the patch off this
reproduces twist/core_strip_results/o3wf (the one-pair half-twist) to 1e-10.
Levels are in bulk units u = E(dphi) - E(phi) of the untwisted model at the same N (stored h_c of on_results/o3wf).
usage: python o3_twists.py <centre|reflection|pair> N   -> analysis/o3_twists/<defect>_N<N>.json
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
    CONFIGS = fuzzy.run_on_defect().CONFIGS

    _Orig = M.Flavours
    TWIST3 = {'on': False}


    class Patched(_Orig):
        def __init__(self, Nv, pairs=(), twist=None):
            super().__init__(Nv, pairs, twist)
            if TWIST3['on']:
                k = self.names.index('3')
                self.qtw = self.qtw.copy(); self.qtw[k] = 1


    M.Flavours = Patched
    DEFECTS = {'centre': dict(twist=[1], t3=True), 'reflection': dict(twist=[0], t3=True), 'pair': dict(twist=[1], t3=False)}


    def build(defect, N, lam, hc):
        cfg = CONFIGS['o3wf']; d = DEFECTS[defect]
        TWIST3['on'] = d['t3']
        kw = dict(pairs=cfg['pairs'], twist=d['twist'])
        fl, ob, _, _, mub = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], hc, **kw)
        offb = np.concatenate([[0], np.cumsum([len(o) for o in ob])]).astype(int)
        fl, ot, kt, two, mut = M.build_terms(cfg['model'], cfg['Nv'], N, 0.5, cfg['U'], hc, **kw)
        offt = np.concatenate([[0], np.cumsum([len(o) for o in ot])]).astype(int)
        dev = np.zeros_like(mut)
        for f in range(fl.F):
            c = mub[offb[f]:offb[f + 1]]
            dev[offt[f]:offt[f + 1]] = mut[offt[f]:offt[f + 1]] - c[0]
        return fl, ot, kt, two, mut - (1.0 - lam) * dev


    def lowest(H, k):
        if H.shape[0] <= 3000:
            e = np.linalg.eigvalsh(H.toarray()); return e[:k]
        return np.sort(sla.eigsh(H, k=k, which='SA', tol=1e-10, return_eigenvectors=False))


    def main(defect, N):
        t0 = time.time()
        R = store.get(os.path.join(TW, 'on_results', f'o3wf_N{N}.json'))
        hc = R['h_c']; b = R['alpha']['0.0000']
        u = b['(1,)|1|(1,)'][0] - b['(1,)|0|(1,)'][0]
        out = dict(defect=defect, N=N, u=u, lam={})
        sectors = [(sz, lz, p3) for sz in (0, 1, 2) for lz in (0.0, 0.5, 1.0, 1.5) for p3 in (1, -1)]
        for lam in (1.0, 0.0):
            fl, orbs, koff, two, mu = build(defect, N, lam, hc)
            lev = {}
            for (sz, lz, p3) in sectors:
                bs = M.Basis(fl, orbs, koff, N, [sz], 0, (p3,), Lz=lz, mscale=2)
                if bs.dim < 5:
                    continue
                lev[f'{sz}|{lz:g}|{p3:+d}'] = lowest(M.build_H(bs, two, mu), 4 if sz == 0 else 3).tolist()
            E0 = min(v[0] for v in lev.values())
            out['lam'][f'{lam:g}'] = {k: [(e - E0) / u for e in v] for k, v in lev.items()}
            print(f'{defect} N={N} lam={lam:g} [{time.time() - t0:.0f}s]: ' + '  '.join(
                f"{k}:" + '/'.join(f'{x:.3f}' for x in v[:3]) for k, v in out['lam'][f'{lam:g}'].items()), flush=True)
        None
        store.put(os.path.join(HERE, 'o3_twists', f'{defect}_N{N}.json'), out)


    def cli(argv):
        main(argv[1], int(argv[2]))
    return namespace(locals(), globals())


STEPS = {'run_on_defect': fuzzy.run_on_defect, 'o3_twists': o3_twists,
         'on_tables': fuzzy.question('01_known_results/o2_twist').on_tables}   # the O(2), O(3) and O(4) tables
REPORT = [('on_tables',), ('o3_twists', 'centre', '6'), ('o3_twists', 'reflection', '6')]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
