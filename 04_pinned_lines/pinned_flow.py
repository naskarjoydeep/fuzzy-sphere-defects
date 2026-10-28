"""Does pinning commute with the easy-plane flow, and what does the untwisted spectator do to the walls?

The pinned O(3) half-twist, with the time-reversal-even pin of monodromy_pinning.py (h_pin = 1000, local potential
subtracted), is followed along the tuned easy-plane trajectory of 02_rg_flow (paper, Sec. 5.1, Fig. 4). Each quantity
is compared with the pinned O(2) monodromy line at the same N_m, stored as twist/pin_even_o2wf_N*_a0.5.json:
  gate N             runs the O(2) model through the same code and reproduces that line;
  run N g1,g2,...    the tilt, Ward and angle stiffness, shape factor and wall of the pinned O(3) line along the flow;
  robust N           the endpoint test with the local potential kept, and with the second pin bilinear reweighted by
                     0.7 and 1.4, in both models.
The paper uses N_m = 6-8, by exact diagonalization (N_m = 8 needs about 2.3 GB).

usage: python pinned_flow.py <step> [args]      (python pinned_flow.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace          # noqa: E402,F401


@section
def pinned_flow():
    """Pinned O(3) half-twist followed along the easy-plane flow: does pinning commute with the bulk flow?

Physics. The O(3) half-twist (twist on the (1,2) pair, alpha = 1/2) with the time-reversal-even pin of its lowest charged
mode (twist/pin_even_junction.py: R1 + R2 at each pole, K = L_z - alpha Q conserving) is a monodromy pinning line with an
untwisted spectator phi_3. Along the easy-plane flow g < 0 the spectator becomes massive and the line should approach the
pinned O(2) monodromy line of Sec. 5.3 (o2wf). Per (N, g): the lowest levels of the pinned (K = 0, P_3 = +) sector at eta = 0
(tilt and Ward C^HQ with the U(1) charge Q_12) and the junction energies at eta = 1/16 (C^angle) and eta = 1 (wall), in the
bulk unit u of the tuned trajectory.
Hamiltonian. dflow.py's H = H_two + h n_h + g W at the tuned (h_c(N, g), u(N, g)) of partB_const, with the core potential
stripped as in Sec. 5.3 (lam = 0: every flavour's one-body term is its untwisted, orbital-independent value), plus
h_pin u [P_N + e^{i pi eta} P_S + h.c.], h_pin = 1000.
Gate. cfg o2wf through the same code (g = 0, h_c of on_results) reproduces pin_even_o2wf_N{N}_a0.5.json.
usage: python pinned_flow.py dims N | python pinned_flow.py gate N | python pinned_flow.py run N g1,g2,...
       -> pinned_flow_N{N}.json (resumable)
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    HERE = 'flow'
    M = fuzzy.onmodel()   # was: import onmodel as M
    _mp = fuzzy.question('04_pinned_lines/monodromy_pinning'); even_terms = _mp.pin_even_junction().even_terms; op_matrix = _mp.pin_flow().op_matrix   # was: from pin_even_junction import even_terms; from pin_flow import op_matrix
    bulk_points = fuzzy.question('02_rg_flow/easy_plane').dflow().bulk_points   # was: from dflow import bulk_points

    CFGS = {'o3wf': dict(model='wf', Nv=3, U=0.264, pairs=[(1, 2)], twist=[1]),
            'o2wf': dict(model='wf', Nv=2, U=0.270, pairs=[(1, 2)], twist=[1])}
    W = {'0': 0.0, '12+': 1.0, '12-': 1.0, '3': -2.0}
    ALPHA, HP, C16 = 0.5, 1000.0, 1 - np.cos(np.pi / 16)


    class Pinned:
        def __init__(self, cfgname, N, core='strip', w2=1.0):
            cfg = CFGS[cfgname]
            kw = dict(pairs=cfg['pairs'], twist=cfg['twist'])
            fl, orbs, koff, two, mt0 = M.build_terms(cfg['model'], cfg['Nv'], N, ALPHA, cfg['U'], 0.0, **kw)
            _, _, _, _, mt1 = M.build_terms(cfg['model'], cfg['Nv'], N, ALPHA, cfg['U'], 1.0, **kw)
            _, ob, _, _, mb0 = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], 0.0, **kw)
            _, _, _, _, mb1 = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], 1.0, **kw)
            offb = np.concatenate([[0], np.cumsum([len(o) for o in ob])]).astype(int)
            off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
            c0 = np.zeros(off[-1]); c1 = np.zeros(off[-1]); w = np.zeros(off[-1])
            for f, name in enumerate(fl.names):                     # stripped core: untwisted constant per flavour
                a0, a1 = mb0[offb[f]:offb[f + 1]], mb1[offb[f]:offb[f + 1]]
                assert np.allclose(a0, a0[0]) and np.allclose(a1, a1[0])
                c0[off[f]:off[f + 1]] = a0[0]; c1[off[f]:off[f + 1]] = a1[0] - a0[0]
                w[off[f]:off[f + 1]] = W.get(name, 0.0)
            if core == 'raw':                                        # the Hamiltonian as written (twisted one-body term)
                c0, c1 = np.array(mt0, float), np.array(mt1, float) - np.array(mt0, float)
            self.N, self.cfgname = N, cfgname
            self.fl, self.orbs, self.koff, self.two = fl, orbs, koff, two
            self.c0, self.c1, self.w, self.off = c0, c1, w, off
            self.tN, self.tS, self.desc = even_terms(fl, orbs, koff, off)
            if w2 != 1.0:                                            # relative weight of the second bilinear
                self.tN = [self.tN[0], (self.tN[1][0], self.tN[1][1], self.tN[1][2] * w2)]
                self.tS = [self.tS[0], (self.tS[1][0], self.tS[1][1], self.tS[1][2] * w2)]
            par = (1,) if 'Nv' in cfg and cfg['Nv'] == 3 else ()
            self.Qs = [q for q in range(-N, N + 1) if M.Basis(fl, orbs, koff, N, [q], 0, par).dim > 0]
            self.B = {q: M.Basis(fl, orbs, koff, N, [q], 0, par) for q in self.Qs}
            self.dims = [self.B[q].dim for q in self.Qs]

        def matrices(self):
            zero = np.zeros_like(self.c0)
            H2 = sp.block_diag([M.build_H(self.B[q], self.two, zero) for q in self.Qs], format='csr').astype(complex)
            occ = np.concatenate([((self.B[q].states[:, None] >> np.arange(len(self.c0))[None, :].astype(np.int64)) & 1)
                                  for q in self.Qs]).astype(float)
            n0, nh, nw = occ @ self.c0, occ @ self.c1, occ @ self.w
            n = len(self.Qs)
            PN = [[None] * n for _ in range(n)]; PS = [[None] * n for _ in range(n)]
            for i, q in enumerate(self.Qs):
                for j in range(n):
                    PN[i][j] = sp.csr_matrix((self.dims[i], self.dims[j]), dtype=complex)
                    PS[i][j] = sp.csr_matrix((self.dims[i], self.dims[j]), dtype=complex)
            for i, q in enumerate(self.Qs):
                if q + 1 in self.B:
                    j = self.Qs.index(q + 1)
                    PN[j][i] = op_matrix(self.B[q].states, self.B[q + 1].states, self.tN).astype(complex)
                    PS[j][i] = op_matrix(self.B[q].states, self.B[q + 1].states, self.tS).astype(complex)
            Qd = np.concatenate([np.full(d, q) for q, d in zip(self.Qs, self.dims)]).astype(float)
            return H2, n0, nh, nw, sp.bmat(PN, format='csr'), sp.bmat(PS, format='csr'), Qd

        def point(self, mats, h, g, u, etas=(0.0, 0.0625, 1.0), k=6):
            H2, n0, nh, nw, PN, PS, Qd = mats
            Hb = (H2 + sp.diags(n0 + h * nh + g * nw)).tocsr()
            rec, v0 = {}, None
            for eta in etas:
                ph = np.exp(1j * np.pi * eta)
                H = (Hb + HP * u * (PN + PN.conj().T + ph * PS + np.conj(ph) * PS.conj().T)).tocsr()
                kk = k if eta == 0 else 2
                wv, v = sla.eigsh(H, k=kk, which='SA', tol=1e-11, v0=v0)
                o = np.argsort(wv); wv, v = wv[o], v[:, o]; v0 = v[:, 0]
                r = dict(E=wv.tolist())
                if eta == 0:
                    lev = (wv - wv[0]) / u
                    r['levels'] = lev.tolist()
                    r['ward'] = [float((lev[n] * abs(v[:, n].conj() @ (Qd * v[:, 0]))) ** 2 / 4) for n in range(kk)]
                rec[f'{eta:g}'] = r
            e0 = rec['0']['E'][0]
            Dj = {kk: (r['E'][0] - e0) / u for kk, r in rec.items()}
            out = dict(h=h, g=g, u=u, tilt=rec['0']['levels'][1], CHQ=rec['0']['ward'][1], levels=rec['0']['levels'],
                       Cang=Dj['0.0625'] / C16, wall=Dj['1'])
            out['r'] = out['wall'] / (2 * out['Cang'])
            return out


    def dims(N):
        p = Pinned('o3wf', N)
        print(f'o3wf N={N}: K=0, P3=+ sectors over Q: {dict(zip(p.Qs, p.dims))}, total {sum(p.dims)}')


    def gate(N):
        t0 = time.time()
        R = store.get(os.path.join(HERE, '..', 'twist', 'on_results', f'o2wf_N{N}.json'))
        bulk = R['alpha']['0.0000']; u = bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0]
        p = Pinned('o2wf', N)
        out = p.point(p.matrices(), R['h_c'], 0.0, u)
        ref = store.get(os.path.join(HERE, '..', 'twist', f'pin_even_o2wf_N{N}_a0.5.json'))
        print(f"gate o2wf N={N}: tilt {out['tilt']:.8f} vs {ref['points']['0']['levels'][1]:.8f}; C^HQ {out['CHQ']:.8f} vs "
              f"{ref['points']['0']['ward'][1]:.8f}; wall {out['wall']:.8f} vs {ref['Dj']['1']:.8f}; "
              f"Dj(1/16) {out['Cang'] * C16:.8f} vs {ref['Dj']['0.0625']:.8f}  [{time.time() - t0:.0f}s]")


    def run(N, gs):
        fn = os.path.join(HERE, f'pinned_flow_N{N}.json')
        res = store.get(fn) if store.exists(fn) else dict(N=N, alpha=ALPHA, hp=HP, points={})
        pts = {round(g, 7): (g, h, u) for g, h, u in bulk_points(N)}
        t0 = time.time()
        p = Pinned('o3wf', N)
        mats = p.matrices()
        print(f'o3wf N={N}: dim {sum(p.dims)}, matrices [{time.time() - t0:.0f}s]', flush=True)
        for g in gs:
            key = f'{g:g}'
            if key in res['points']:
                continue
            gg, h, u = pts[round(g, 7)]
            out = p.point(mats, h, gg, u)
            res['points'][key] = out
            store.put(fn, res)
            print(f"  g={g:+.4f}: tilt {out['tilt']:.4f}  C^HQ {out['CHQ']:.4f}  C^ang {out['Cang']:.4f}  wall {out['wall']:.4f}  "
                  f"r {out['r']:.4f}  [{time.time() - t0:.0f}s]", flush=True)


    def robust(N):
        """Endpoint test for microscopic variants: (core, w2) in raw/1, strip/0.7, strip/1.4; O(3) at g = 0, -0.09 and O(2)."""
        fn = os.path.join(HERE, f'pinned_flow_robust_N{N}.json')
        res = store.get(fn) if store.exists(fn) else {}
        pts = {round(g, 7): (g, h, u) for g, h, u in bulk_points(N)}
        R = store.get(os.path.join(HERE, '..', 'twist', 'on_results', f'o2wf_N{N}.json'))
        bulk = R['alpha']['0.0000']; u2 = bulk['(1,)|1|()'][0] - bulk['(1,)|0|()'][0]
        t0 = time.time()
        for core, w2 in (('raw', 1.0), ('strip', 0.7), ('strip', 1.4)):
            tag = f'{core}_w{w2:g}'
            if tag in res:
                continue
            p2 = Pinned('o2wf', N, core, w2); o2 = p2.point(p2.matrices(), R['h_c'], 0.0, u2)
            p3 = Pinned('o3wf', N, core, w2); m3 = p3.matrices()
            rec = dict(o2=o2)
            for g in (0.0, -0.09):
                gg, h, u = pts[round(g, 7)]
                rec[f'{g:g}'] = p3.point(m3, h, gg, u)
            res[tag] = rec
            store.put(fn, res)
            r0, r9 = rec['0'], rec['-0.09']
            print(f"N={N} {tag}: wall O3(g=0)/O2 {r0['wall'] / o2['wall']:.4f}  O3(g=-0.09)/O2 {r9['wall'] / o2['wall']:.4f}  "
                  f"C^HQ {r0['CHQ'] / o2['CHQ']:.4f} / {r9['CHQ'] / o2['CHQ']:.4f}  C^ang {r0['Cang'] / o2['Cang']:.4f} / "
                  f"{r9['Cang'] / o2['Cang']:.4f}  [{time.time() - t0:.0f}s]", flush=True)


    def main():
        a = sys.argv
        if a[1] == 'robust':
            robust(int(a[2]))
        if a[1] == 'dims':
            dims(int(a[2]))
        elif a[1] == 'gate':
            gate(int(a[2]))
        elif a[1] == 'run':
            run(int(a[2]), [float(x) for x in a[3].split(',')])

    def cli(argv):
        main()
    return namespace(locals(), globals())


STEPS = {'pinned_flow': pinned_flow}
REPORT = [('pinned_flow', 'gate', '6')]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
