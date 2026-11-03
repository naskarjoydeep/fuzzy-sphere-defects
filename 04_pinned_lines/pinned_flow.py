"""Does pinning commute with the bulk flows, and which pin survives which branch?

Section pinned_flow: the pinned O(3) half-twist, with the time-reversal-even pin of monodromy_pinning.py (h_pin = 1000,
local potential subtracted), is followed along the tuned easy-plane trajectory of 02_rg_flow (paper, Sec. 5.1, Fig. 4).
Each quantity is compared with the pinned O(2) monodromy line at the same N_m, stored as twist/pin_even_o2wf_N*_a0.5.json:
  gate N             runs the O(2) model through the same code and reproduces that line;
  run N g1,g2,...    the tilt, Ward and angle stiffness, shape factor and wall of the pinned O(3) line along the flow;
  robust N           the endpoint test with the local potential kept, and with the second pin bilinear reweighted by
                     0.7 and 1.4, in both models.
The paper uses N_m = 6-8, by exact diagonalization (N_m = 8 needs about 2.3 GB).

Sections pin_branches and ising_pinned_ref (paper, App. E.9, Table 21, Fig. 1): the same charged pin and a neutral pin on
the untwisted component phi_3 (eq:pin3), followed along both branches. On the easy-axis branch the phi_3 pin ends on the
pinned Ising line, which ising_pinned_ref realizes independently in the Ising model; the charged pin decouples there, and
the phi_3 pin decouples on the easy-plane branch. Steps: pin_branches gate N, neutral N g1,..., neutral_lm 9 g1,...
(low memory, N_m = 9), charged N g1,..., finite N; ising_pinned_ref N1,N2,... ('frozen' in a g list is g = 3).
The pin_branches docstring keeps its research note "not in the draft", written before App. E.9 existed.

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


@section
def pin_branches():
    """Which pins survive which bulk flow?  Exploration for Fig. 1 (not in the draft).

Physics.  The O(3) half-twist (twist on the (1,2) pair) can be pinned in two ways:
  * charged pin (Sec. 5.1, flow/pinned_flow.py): couples to the lowest charged mode of the twisted pair;
  * neutral pin (this file): couples to the untwisted component phi_3, O_P = c+_{P,0} c_{P,3} + h.c. at the pole orbitals
    (twist/spectator_pinning/scan.py), at infinite field by the exact pole projection.
Prediction: pins follow the light fields.
  easy axis (g > 0, phi_3 light):  neutral pin -> pinned Ising line (twist disappears, pin survives);
                                    charged pin -> trivial Ising line (pin disappears with the twisted pair).
  easy plane (g < 0, phi_3 heavy): neutral pin -> unpinned O(2) half-twist (pin disappears);
                                    charged pin -> pinned O(2) monodromy line (Fig. 5).
Hamiltonian: dflow's H = H_two + h n_h + g W on the tuned trajectory (h_c(N, g), u(N, g)) of partB_const, core potential
stripped as in pinned_flow.py; frozen easy-axis limit g = 3 with h_c from dflow_frozen_limit.json and u = E(dsigma) - E(sigma)
computed here from the untwisted, unpinned bulk.
Observables (bulk units): wall = [E_{+-} - E_{++}]/u (lowest Q = 0 states, K = 0), pinned levels of the ++ line in Q = 0
and Q = 1 (K = 0).  Reference for 'twist disappeared': the same pin on the untwisted model (alpha = 0) at the same couplings.
usage:
  python pin_branches.py gate N              (core = raw, g = 0: reproduces Codex's spectator-pinning energies)
  python pin_branches.py neutral N g1,g2,... (g may include 'frozen')   -> pin_branches_neutral_N{N}.json
  python pin_branches.py charged N g1,...    (charged pin on the easy-axis branch) -> pin_branches_charged_N{N}.json
"""
    import os
    import sys
    import json
    import time
    from types import SimpleNamespace
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    HERE = 'flow'
    M = fuzzy.onmodel()   # was: import onmodel as M
    sys.path.insert(0, os.path.join(fuzzy.ROOT, '02_rg_flow'))   # easy_axis.py imports its sibling easy_plane
    hopping = fuzzy.question('02_rg_flow/easy_axis').scan().hopping   # was: from scan import hopping
    bulk_points = fuzzy.question('02_rg_flow/easy_plane').dflow().bulk_points   # was: from dflow import bulk_points

    CFG = dict(model='wf', Nv=3, U=0.264, pairs=[(1, 2)], twist=[1])
    W = {'0': 0.0, '12+': 1.0, '12-': 1.0, '3': -2.0}


    def trajectory(N):
        pts = {round(g, 7): (g, h, u) for g, h, u in bulk_points(N)}
        fz = store.get(os.path.join(HERE, 'dflow_frozen_limit.json'))[str(N)]
        return pts, fz['h_c']


    class Neutral:
        """O(3) model at flux alpha in a fixed U(1)_12 charge Q and K = L_z - alpha Q sector, both phi_3 parities."""

        def __init__(self, N, alpha, Q, K=0, core='strip', pars=(1, -1)):
            kw = dict(pairs=CFG['pairs'], twist=CFG['twist'])
            fl, orbs, koff, two, mt0 = M.build_terms(CFG['model'], CFG['Nv'], N, alpha, CFG['U'], 0.0, **kw)
            _, _, _, _, mt1 = M.build_terms(CFG['model'], CFG['Nv'], N, alpha, CFG['U'], 1.0, **kw)
            _, ob, _, _, mb0 = M.build_terms(CFG['model'], CFG['Nv'], N, 0.0, CFG['U'], 0.0, **kw)
            _, _, _, _, mb1 = M.build_terms(CFG['model'], CFG['Nv'], N, 0.0, CFG['U'], 1.0, **kw)
            off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
            offb = np.concatenate([[0], np.cumsum([len(o) for o in ob])]).astype(int)
            c0 = np.zeros(off[-1]); c1 = np.zeros(off[-1]); w = np.zeros(off[-1])
            for f, name in enumerate(fl.names):
                a0, a1 = mb0[offb[f]:offb[f + 1]], mb1[offb[f]:offb[f + 1]]
                c0[off[f]:off[f + 1]] = a0[0]; c1[off[f]:off[f + 1]] = a1[0] - a0[0]
                w[off[f]:off[f + 1]] = W.get(name, 0.0)
            if core == 'raw':
                c0, c1 = np.array(mt0, float), np.array(mt1, float) - np.array(mt0, float)
            secs = [M.Basis(fl, orbs, koff, N, [Q], K, (p,)) for p in pars]
            states = np.sort(np.concatenate([b.states for b in secs if b.dim > 0]))
            self.N, self.alpha, self.Q, self.K = N, alpha, Q, K
            self.basis = SimpleNamespace(states=states, dim=len(states), nmodes=secs[0].nmodes)
            self.H2 = M.build_H(self.basis, two, np.zeros_like(c0)).tocsr()
            occ = ((states[:, None] >> np.arange(off[-1])[None, :].astype(np.int64)) & 1).astype(float)
            self.n0, self.nh, self.nw = occ @ c0, occ @ c1, occ @ w
            f0, f3 = fl.names.index('0'), fl.names.index('3')
            self.poles = dict(N=(int(off[f0] + N - 1), int(off[f3] + N - 1)), S=(int(off[f0]), int(off[f3])))
            if len(pars) == 2:                                       # the pin flips the phi_3 parity
                self.ON = hopping(states, *self.poles['N']); self.OS = hopping(states, *self.poles['S'])

        def H(self, h, g):
            return (self.H2 + sp.diags(self.n0 + h * self.nh + g * self.nw)).tocsr()

        def projector(self, eta):
            """Isometry occupying (c_N0 + c_N3)/sqrt2 and (c_S0 + eta c_S3)/sqrt2 (Codex's construction)."""
            states = self.basis.states
            n0, n3 = self.poles['N']; s0, s3 = self.poles['S']
            mask = sum(1 << p for p in (n0, n3, s0, s3))
            ref = np.int64((1 << n0) | (1 << s0))
            free = states[(states & np.int64(mask)) == ref] ^ ref
            rows, cols, vals = [], [], []
            for j, fs in enumerate(free):
                for south, ws in ((s0, 1), (s3, eta)):
                    for north in (n0, n3):
                        st = int(fs)
                        sign = (-1) ** ((st & ((1 << south) - 1)).bit_count()); st |= 1 << south
                        sign *= (-1) ** ((st & ((1 << north) - 1)).bit_count()); st |= 1 << north
                        i = int(np.searchsorted(states, st))
                        rows.append(i); cols.append(j); vals.append(.5 * ws * sign)
            P = sp.coo_matrix((vals, (rows, cols)), shape=(len(states), len(free))).tocsr()
            assert abs(P.T @ P - sp.eye(len(free))).max() < 1e-13
            assert abs((self.ON + eta * self.OS) @ P - 2 * P).max() < 1e-13
            return P

        def pinned(self, h, g, eta, k):
            P = self.projector(eta)
            Hp = (P.T @ self.H(h, g) @ P).tocsr()
            return eig(Hp, k)

        def unpinned(self, h, g, k):
            return eig(self.H(h, g), k)


    def eig(H, k):
        if H.shape[0] <= 600:
            return np.linalg.eigvalsh(H.toarray())[:k].tolist()
        w = sla.eigsh(H, k=k, which='SA', tol=1e-11, return_eigenvectors=False)
        return np.sort(w).tolist()


    def frozen_unit(N, h):
        """u = E(dsigma) - E(sigma) of the untwisted, unpinned bulk at g = 3: lowest two phi_3-odd K = 0 states."""
        b = Neutral(N, 0.0, 0, 0, pars=(-1,))
        odd = b.unpinned(h, 3.0, 2)
        ev = Neutral(N, 0.0, 0, 0, pars=(1,)).unpinned(h, 3.0, 1)
        u = odd[1] - odd[0]
        return u, (odd[0] - ev[0]) / u


    def gate(N):
        ref = store.get(os.path.join(HERE, '..', 'twist', 'spectator_pinning', 'results', f'N{N}_a0.500_Q0_K0.json'))
        h = ref['bulk']['h_c']
        t0 = time.time()
        S = Neutral(N, 0.5, 0, 0, core='raw')
        for eta, lab in ((1, 'equal'), (-1, 'opposite')):
            mine = S.pinned(h, 0.0, eta, 3)
            theirs = ref['infinite'][lab]['E_core_subtracted'][:3]
            print(f'gate N={N} {lab}: max |diff| = {max(abs(a - b) for a, b in zip(mine, theirs)):.2e}  (dim {S.basis.dim}) '
                  f'[{time.time() - t0:.0f}s]', flush=True)


    def neutral(N, gs):
        fn = os.path.join(HERE, f'pin_branches_neutral_N{N}.json')
        res = store.get(fn) if store.exists(fn) else dict(N=N, points={})
        pts, h_frozen = trajectory(N)
        t0 = time.time()
        built = {}
        for gl in gs:
            key = gl
            if key in res['points']:
                continue
            if gl == 'frozen':
                g, h = 3.0, h_frozen
                u, sig = frozen_unit(N, h)
            else:
                g, h, u = pts[round(float(gl), 7)]; sig = None
            rec = dict(g=g, h=h, u=u, sigma_check=sig)
            for alpha in (0.5, 0.0):
                for Q in (0, 1):
                    if (alpha, Q) not in built:
                        built[(alpha, Q)] = Neutral(N, alpha, Q, 0)
                S0, S1 = built[(alpha, 0)], built[(alpha, 1)]
                pp = S0.pinned(h, g, 1, 4); pm = S0.pinned(h, g, -1, 2); q1 = S1.pinned(h, g, 1, 2)
                un = S0.unpinned(h, g, 3); un1 = S1.unpinned(h, g, 2)
                e0 = pp[0]
                rec[f'a{alpha:g}'] = dict(
                    wall=(pm[0] - e0) / u, levels_Q0=[(x - e0) / u for x in pp], levels_Q1=[(x - e0) / u for x in q1],
                    unpinned_Q0=[(x - un[0]) / u for x in un], unpinned_Q1=[(x - un[0]) / u for x in un1], E=dict(pp=pp, pm=pm, q1=q1))
            res['points'][key] = rec
            store.put(fn, res)
            a, b = rec['a0.5'], rec['a0']
            print(f"N={N} g={gl}: wall twisted {a['wall']:.4f} untwisted {b['wall']:.4f} | Q0 levels tw "
                  f"{' '.join(f'{x:.4f}' for x in a['levels_Q0'][1:])} untw {' '.join(f'{x:.4f}' for x in b['levels_Q0'][1:])} | "
                  f"Q1 tw {a['levels_Q1'][0]:.4f} (unpinned tw {a['unpinned_Q1'][0]:.4f})  [{time.time() - t0:.0f}s]", flush=True)


    def neutral_lm(N, gs, unpinned_gs=('0', '-0.01', '-0.045', '-0.09')):
        """Low-memory variant for large N: one sector at a time, projected pieces built once per sector and eta.
        alpha = 1/2: pinned ++, +- (Q = 0) and ++ (Q = 1) at every g; unpinned Q = 0, 1 only at unpinned_gs.
        alpha = 0: pinned ++, +- (Q = 0) at every g (the 'twist disappeared' reference)."""
        import gc
        fn = os.path.join(HERE, f'pin_branches_neutral_N{N}.json')
        res = store.get(fn) if store.exists(fn) else dict(N=N, points={})
        pts, h_frozen = trajectory(N)
        t0 = time.time()
        cpl = {}
        for gl in gs:
            if gl == 'frozen':
                u, sig = frozen_unit(N, h_frozen); cpl[gl] = (3.0, h_frozen, u, sig)
            else:
                g, h, u = pts[round(float(gl), 7)]; cpl[gl] = (g, h, u, None)
        print(f'N={N}: couplings ready [{time.time() - t0:.0f}s]', flush=True)
        E = {gl: {} for gl in gs}
        for alpha, Q, etas in ((0.5, 0, (1, -1)), (0.5, 1, (1,)), (0.0, 0, (1, -1))):
            S = Neutral(N, alpha, Q, 0)
            for eta in etas:
                P = S.projector(eta)
                parts = [(P.T @ S.H2 @ P).tocsr()] + [(P.T @ sp.diags(x) @ P).tocsr() for x in (S.n0, S.nh, S.nw)]
                for gl in gs:
                    g, h, u, _ = cpl[gl]
                    Hp = parts[0] + parts[1] + h * parts[2] + g * parts[3]
                    E[gl][(alpha, Q, eta)] = eig(Hp, 4 if (Q == 0 and eta == 1) else 2)
                del P, parts; gc.collect()
            if alpha == 0.5:
                for gl in gs:
                    if gl in unpinned_gs:
                        g, h, u, _ = cpl[gl]
                        E[gl][(alpha, Q, 'un')] = eig(S.H(h, g), 3 if Q == 0 else 2)
            print(f'N={N}: sector alpha={alpha} Q={Q} done, dim {S.basis.dim} [{time.time() - t0:.0f}s]', flush=True)
            del S; gc.collect()
        for gl in gs:
            g, h, u, sig = cpl[gl]
            rec = dict(g=g, h=h, u=u, sigma_check=sig)
            for alpha in (0.5, 0.0):
                pp, pm = E[gl][(alpha, 0, 1)], E[gl][(alpha, 0, -1)]
                e0 = pp[0]
                r = dict(wall=(pm[0] - e0) / u, levels_Q0=[(x - e0) / u for x in pp])
                if alpha == 0.5:
                    r['levels_Q1'] = [(x - e0) / u for x in E[gl][(0.5, 1, 1)]]
                    if (0.5, 0, 'un') in E[gl]:
                        un, un1 = E[gl][(0.5, 0, 'un')], E[gl][(0.5, 1, 'un')]
                        r['unpinned_Q0'] = [(x - un[0]) / u for x in un]; r['unpinned_Q1'] = [(x - un[0]) / u for x in un1]
                rec[f'a{alpha:g}'] = r
            res['points'][gl] = rec
            a, b = rec['a0.5'], rec['a0']
            un1 = a.get('unpinned_Q1', [float('nan')])[0]
            print(f"N={N} g={gl}: wall twisted {a['wall']:.4f} untwisted {b['wall']:.4f} | Q0 levels tw "
                  f"{' '.join(f'{x:.4f}' for x in a['levels_Q0'][1:])} untw {' '.join(f'{x:.4f}' for x in b['levels_Q0'][1:])} | "
                  f"Q1 tw {a['levels_Q1'][0]:.4f} (unpinned tw {un1:.4f})", flush=True)
        store.put(fn, res)


    def charged(N, gs):
        Pinned = globals()['pinned_flow']().Pinned   # was: from pinned_flow import Pinned
        fn = os.path.join(HERE, f'pin_branches_charged_N{N}.json')
        res = store.get(fn) if store.exists(fn) else dict(N=N, points={})
        pts, h_frozen = trajectory(N)
        t0 = time.time()
        p = Pinned('o3wf', N)
        mats = p.matrices()
        for gl in gs:
            if gl in res['points']:
                continue
            if gl == 'frozen':
                g, h = 3.0, h_frozen; u, _ = frozen_unit(N, h)
            else:
                g, h, u = pts[round(float(gl), 7)]
            out = p.point(mats, h, g, u)
            res['points'][gl] = out
            store.put(fn, res)
            print(f"N={N} charged g={gl}: wall {out['wall']:.5f}  C^ang {out['Cang']:.5f}  C^HQ {out['CHQ']:.5f}  tilt {out['tilt']:.4f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)


    def finite(N):
        """Finite pin strengths (App. E.9 prose): the decoupling seen without the infinite-field pole locking.
        charged pin (pinned_flow.Pinned, strength hp in bulk units u): frozen-limit even levels and wall for hp = 0 ... 1000,
        and the wall along g = 0, 0.01, 0.03, 0.09, frozen for hp = 3, 10;
        neutral pin, H - b u (O_N + eta O_S) with b = 0, 1, 3, 10: wall and lowest Q = 1 level at g = 0 and -0.09.
        -> pin_branches_finite_N{N}.json"""
        Pinned = globals()['pinned_flow']().Pinned   # was: from pinned_flow import Pinned

        def low(H, k):
            if H.shape[0] <= 600:
                return np.linalg.eigvalsh(H.toarray())[:k]
            return np.sort(sla.eigsh(H, k=k, which='SA', tol=1e-10, return_eigenvectors=False))
        t0 = time.time()
        pts, hz = trajectory(N)
        u3, _ = frozen_unit(N, hz)
        cpl = {f'{g:g}': pts[round(g, 7)] for g in (0.0, 0.01, 0.03, 0.09, -0.09)}
        cpl['frozen'] = (3.0, hz, u3)
        out = dict(N=N, charged_frozen={}, charged_wall={}, neutral={})
        p = Pinned('o3wf', N)
        H2, n0, nh, nw, PN, PS, Qd = p.matrices()

        def charged(g, h, u, hp, k):
            Hb = (H2 + sp.diags(n0 + h * nh + g * nw)).tocsr()
            E = {}
            for eta in (0.0, 1.0):
                ph = np.exp(1j * np.pi * eta)
                E[eta] = low((Hb + hp * u * (PN + PN.conj().T + ph * PS + np.conj(ph) * PS.conj().T)).tocsr(), k)
            return [float((x - E[0.0][0]) / u) for x in E[0.0]], float((E[1.0][0] - E[0.0][0]) / u)
        g, h, u = cpl['frozen']
        for hp in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 1000.0):
            lev, wall = charged(g, h, u, hp, 4)
            out['charged_frozen'][f'{hp:g}'] = dict(even_levels=lev[1:], wall=wall)
        for hp in (3.0, 10.0):
            out['charged_wall'][f'{hp:g}'] = {lab: charged(*cpl[lab], hp, 2)[1] for lab in ('0', '0.01', '0.03', '0.09', 'frozen')}
        del p, H2, n0, nh, nw, PN, PS, Qd
        S0, S1 = Neutral(N, 0.5, 0, 0), Neutral(N, 0.5, 1, 0)
        for lab in ('0', '-0.09'):
            g, h, u = cpl[lab]
            rec = {}
            for b in (0.0, 1.0, 3.0, 10.0):
                e0 = low(S0.H(h, g) - b * u * (S0.ON + S0.OS), 1)[0]
                em = low(S0.H(h, g) - b * u * (S0.ON - S0.OS), 1)[0]
                e1 = low(S1.H(h, g) - b * u * (S1.ON + S1.OS), 1)[0]
                rec[f'{b:g}'] = dict(wall=float((em - e0) / u), Q1=float((e1 - e0) / u))
            out['neutral'][lab] = rec
        store.put(os.path.join(HERE, f'pin_branches_finite_N{N}.json'), out)
        cf, cw, nt = out['charged_frozen'], out['charged_wall'], out['neutral']
        print(f"N={N} finite [{time.time() - t0:.0f}s]: charged frozen eps(hp=0,3,1000) = {cf['0']['even_levels'][0]:.4f}, "
              f"{cf['3']['even_levels'][0]:.4f}, {cf['1000']['even_levels'][0]:.4f}; wall hp=3 g=0 {cw['3']['0']:.4f} frozen "
              f"{cw['3']['frozen']:.2e}; neutral b=1: Q1 g=0 {nt['0']['1']['Q1']:.4f} (b=0 {nt['0']['0']['Q1']:.4f}), "
              f"g=-0.09 {nt['-0.09']['1']['Q1']:.4f} (b=0 {nt['-0.09']['0']['Q1']:.4f}), wall {nt['-0.09']['1']['wall']:.4f}", flush=True)


    def main():
        a = sys.argv
        if a[1] == 'gate':
            gate(int(a[2]))
        elif a[1] == 'neutral':
            neutral(int(a[2]), a[3].split(','))
        elif a[1] == 'neutral_lm':
            neutral_lm(int(a[2]), a[3].split(','))
        elif a[1] == 'finite':
            finite(int(a[2]))
        elif a[1] == 'charged':
            charged(int(a[2]), a[3].split(','))

    def cli(argv):
        main()
    return namespace(locals(), globals())


@section
def ising_pinned_ref():
    """Reference for pin_branches.py: the pinned (magnetic) line of the genuine 3d Ising model on the fuzzy sphere.

Model of Zhu et al., (V0, V1) = (4.75, 1), h = 3.16, half filling; infinite pinning field by fixing the pole orbitals
(fs.FastBasis `fixed`, as in twist/gfunc_ising.py).  Unit u = E(dsigma) - E(sigma) of the unpinned bulk, the same
light-field convention as the O(3) runs (checks: E(sigma)/u -> 0.518, E(eps)/u -> 1.41).
Output per N: wall = [E_{+-} - E_{++}]/u (L_z = 0) and the ++ line levels in L_z = 0 and 1, relative to E_{++}.
usage: python ising_pinned_ref.py N1,N2,...   -> ising_pinned_ref.json
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse.linalg as spla

    HERE = 'flow'
    _fs = fuzzy; FastBasis, build_hamiltonian_fast, haldane_V = _fs.FastBasis, _fs.build_hamiltonian_fast, _fs.haldane_V   # was: from fs.fockfast import FastBasis; from fs.opsfast import build_hamiltonian_fast; from fs.pseudo import haldane_V

    VL, HT = (4.75, 1.0), 3.16
    OUT = os.path.join(HERE, 'ising_pinned_ref.json')


    def levels(b, V, k):
        Hm = build_hamiltonian_fast(b, VL, HT, V=V)
        if b.dim <= 1500:
            return np.linalg.eigvalsh(Hm.toarray())[:k]
        return np.sort(spla.eigsh(Hm, k=k, which='SA', tol=1e-12, return_eigenvectors=False))


    def run(N):
        s, M0, n = (N - 1) / 2, N * (N - 1) // 2, N - 1
        V = haldane_V(s, VL)
        pin = lambda a, t: {2 * a + t: 1, 2 * a + (1 - t): 0}       # t = 0 up, 1 down
        t0 = time.time()
        b0 = levels(FastBasis(s, N, M0), V, 3)                   # 1, sigma, eps
        b1 = levels(FastBasis(s, N, M0 + 1), V, 1)               # dsigma
        u = b1[0] - b0[1]
        pp0 = levels(FastBasis(s, N, M0, fixed={**pin(n, 0), **pin(0, 0)}), V, 4)
        pp1 = levels(FastBasis(s, N, M0 + 1, fixed={**pin(n, 0), **pin(0, 0)}), V, 2)
        pm0 = levels(FastBasis(s, N, M0, fixed={**pin(n, 0), **pin(0, 1)}), V, 1)
        rec = dict(N=N, u=float(u), sigma=float((b0[1] - b0[0]) / u), eps=float((b0[2] - b0[0]) / u),
                   wall=float((pm0[0] - pp0[0]) / u), levels_L0=[float((x - pp0[0]) / u) for x in pp0],
                   levels_L1=[float((x - pp0[0]) / u) for x in pp1], seconds=time.time() - t0)
        print(f"Ising N={N}: sigma {rec['sigma']:.4f} eps {rec['eps']:.4f} | wall {rec['wall']:.4f} | ++ L0 "
              f"{' '.join(f'{x:.4f}' for x in rec['levels_L0'][1:])} | ++ L1 {' '.join(f'{x:.4f}' for x in rec['levels_L1'])} "
              f"[{rec['seconds']:.0f}s]", flush=True)
        data = store.get(OUT) if store.exists(OUT) else {}
        data[str(N)] = rec
        store.put(OUT, data)


    def main():
        for N in [int(x) for x in sys.argv[1].split(',')]:
            run(N)

    def cli(argv):
        main()
    return namespace(locals(), globals())


STEPS = {'pinned_flow': pinned_flow, 'pin_branches': pin_branches, 'ising_pinned_ref': ising_pinned_ref}
REPORT = [('pinned_flow', 'gate', '6'), ('pin_branches', 'gate', '6')]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
