"""How large is the domain-wall dimension of a pinned O(N) line, measured on the lattice?

Pinning the order parameter at the two poles with directions at an internal angle eta creates a junction on a straight
line; at eta = pi it is the domain wall between opposite pinnings, whose dimension Delta_{+-} decides whether the
candidate ordered line is stable (paper, Sec. 5.2 and App. E). The steps are the exact-diagonalization and DMRG angle
scans of the vector O(2), O(3), O(4) models and the spin-one O(2) regulator, the O(4) walls at N_m = 10, 11, and the
audit pipeline that turns them into the shape factor, the measured walls and their extrapolations (Tables 3, 11-14 and
16, eq:wallpredict; Figs. 5, 10 and 11).

The ED scans take seconds to an hour and the DMRG scans 1-31 hours; their outputs are stored. The report reruns the audit.

usage: python junctions.py <step> [args]      (python junctions.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def theta_on():
    """Theta junction between pinned lines in the O(N) Wilson-Fisher models (o2wf, o3wf, o4wf_diag bulks), Cartesian flavours.

Same physics as theta_junction.py (O(3), charge basis), for any N_v: flavours psi_0, psi_1..psi_Nv are kept real
(no pairs), so V_a = psi_0^+ psi_a + h.c. are real hoppings.  North pinned along e_1, south along cos(theta) e_1 +
sin(theta) e_2:
  H = H_O(Nv)(h_c) - b [ (c+_{N0} c_{N1} + h.c.) + cos(theta) (c+_{S0} c_{S1} + h.c.) + sin(theta) (c+_{S0} c_{S2} + h.c.) ].
Basis: all K = 0 states (every Z2 parity sector).  Bulk h_c and unit u = E(d phi) - E(phi) from on_results/<cfg>_N.json.
usage: python theta_on.py <cfg> N [b=8] [ntheta=8]   -> theta_junction_results/<cfg>_N{N}_b{b}.json
"""
    import os
    import sys
    import json
    import time
    import itertools
    from types import SimpleNamespace
    import numpy as np
    HERE = 'twist'
    sys.path.insert(0, os.path.join(HERE, 'spectator_pinning'))
    M = fuzzy.onmodel()   # was: import onmodel as M
    S = fuzzy.question('02_rg_flow/easy_axis').scan()   # was: import scan as S
    _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS

    OUT = os.path.join(HERE, 'theta_junction_results')


    def bulk(cfgname, N):
        R = store.get(os.path.join(HERE, 'on_results', f'{cfgname}_N{N}.json'))
        cfg = CONFIGS[cfgname]
        npair, npar = len(cfg['pairs']), cfg['Nv'] - 2 * len(cfg['pairs'])
        q1, p = (1,) + (0,) * (npair - 1), (+1,) * npar
        B = R['alpha']['0.0000']
        return R['h_c'], B[f'{q1}|1|{p}'][0] - B[f'{q1}|0|{p}'][0]


    def run(cfgname, N, b=8.0, ntheta=8, reduced=False):
        """reduced: fix the parities of flavours 3..Nv to +, which the pin in the (1, 2) plane conserves (the junction
        ground states lie there); a ~2^(Nv-2) smaller basis."""
        pass   # was os.makedirs(...): the store needs no output directory
        fn = os.path.join(OUT, f'{cfgname}_N{N}_b{b:g}' + ('_red' if reduced else '') + '.json')
        res = store.get(fn) if store.exists(fn) else dict(cfg=cfgname, N=N, b=b, theta={})
        cfg = CONFIGS[cfgname]
        Nv = cfg['Nv']
        hc, u = bulk(cfgname, N)
        t0 = time.time()
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], Nv, N, 0.0, cfg['U'], hc, pairs=())
        assert fl.names == ['0'] + [str(a) for a in range(1, Nv + 1)]
        pars = [p[:2] + (1,) * (Nv - 2) for p in itertools.product((1, -1), repeat=2)] if reduced else \
            list(itertools.product((1, -1), repeat=Nv))
        secs = [M.Basis(fl, orbs, koff, N, [], 0, par) for par in pars]
        states = np.sort(np.concatenate([x.states for x in secs]))
        basis = SimpleNamespace(states=states, dim=len(states), nmodes=secs[0].nmodes)
        H0 = M.build_H(basis, two, mu).tocsr()
        north = S.hopping(states, N - 1, N + N - 1)                     # psi_0 - psi_1 at m = +s
        s1, s2 = S.hopping(states, 0, N), S.hopping(states, 0, 2 * N)   # psi_0 - psi_1, psi_0 - psi_2 at m = -s
        res.update(dim=basis.dim, gamma_B=u, h_c=hc, Nv=Nv)
        print(f'{cfgname} N={N}: dim {basis.dim}, build {time.time() - t0:.0f}s', flush=True)
        v0 = None
        for th in np.linspace(0, np.pi, ntheta + 1):
            key = f'{th:.6f}'
            if key in res['theta']:
                continue
            t1 = time.time()
            H = H0 - b * (north + np.cos(th) * s1 + np.sin(th) * s2)
            w, v, r = S.eigenpairs(H.tocsr(), k=4, v0=v0)
            v0 = v[:, 0]
            res['theta'][key] = dict(theta=float(th), E=w.tolist())
            E00 = res['theta'][f'{0:.6f}']['E'][0]
            print(f'  theta/pi={th / np.pi:.4f}: Delta {(w[0] - E00) / u:.5f}  levels '
                  + ' '.join(f'{(x - E00) / u:.4f}' for x in w) + f'  [{time.time() - t1:.0f}s]', flush=True)
            store.put(fn, res)
        return res

    def main():
        run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 8.0,
            int(sys.argv[4]) if len(sys.argv) > 4 else 8, reduced=len(sys.argv) > 5 and sys.argv[5] == 'reduced')

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/theta_on_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def theta_dey():
    """Second regulator for the junction shape factor R: the spin-1 XY fuzzy sphere of arXiv:2604.18705 (o2twist.py).

The O(2) order parameter of this model is the in-plane spin S^+-, odd under its antiunitary symmetry (the real
quadrupoles psi_0^+ psi_a would pin a different operator).  So the pin is the spin itself, in the S^z basis
(flavours +1, 0, -1): north pole along S^x, south pole along cos(theta) S^x + sin(theta) S^y (complex Hermitian H),
hopping strength b on the pole orbitals, bulk at D_c(N) tuned by run_o2defect.Dc_of (Delta_sigma target) with unit
gamma = E(d sigma) - E(sigma).  Basis: L_z = 0, all charges Q = N_{+1} - N_{-1} (broken by the pin).
The tilt is the lowest theta = 0 state connected to the ground state by Q ([H, Q] is exactly the pin rotation).
Outputs Delta(theta), C_t = Delta''(0) (fit in 1 - cos theta), R = Delta(pi)/(2 C_t) and the tilt, as theta_on.py.
usage: python theta_dey.py N [b=8]   -> theta_junction_results/dey_N{N}_b{b}.json
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    HERE = 'twist'
    sys.path.insert(0, HERE)
    _o2twist = fuzzy.o2twist(); O2Basis, build_terms, build_H = _o2twist.O2Basis, _o2twist.build_terms, _o2twist.build_H   # was: from o2twist import O2Basis, build_terms, build_H
    _twist = fuzzy; couplings_for = _twist.couplings_for   # was: from twist import couplings_for
    _run_o2defect = fuzzy.question('01_known_results/o2_twist').run_o2defect(); Dc_of = _run_o2defect.Dc_of   # was: from run_o2defect import Dc_of
    _pin_flow = fuzzy.question('04_pinned_lines/monodromy_pinning').pin_flow(); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix

    OUT = os.path.join(HERE, 'theta_junction_results')
    ANG = [0, .125, .25, .5, .75, 1]


    def run(N, b=8.0):
        pass   # was os.makedirs(...): the store needs no output directory
        t0 = time.time()
        G0, G1, _ = couplings_for(N, 4.0, 1.0)
        Dc, gam, bulkc = Dc_of(N, G0, G1)
        two, U = build_terms(N, 0, G0, G1)
        Qs = [q for q in range(-N, N + 1) if O2Basis(N, 0, q, 0).dim > 0]
        B = {q: O2Basis(N, 0, q, 0) for q in Qs}
        st = np.concatenate([B[q].states for q in Qs])
        order = np.argsort(st)
        st = st[order]
        H0 = sp.block_diag([build_H(B[q], two, U, Dc) for q in Qs], format='csr')
        H0 = H0[order][:, order]
        Qd = np.concatenate([np.full(B[q].dim, q) for q in Qs])[order]
        off = B[Qs[0]].off                              # flavour offsets: f = 0 (+1), 1 (0), 2 (-1); alpha = 0 -> N orbitals each
        mode = lambda f, j: int(off[f] + j)
        def spin(j, comp):
            """S^x or S^y on orbital j: sum_ab c+_a (S)_ab c_b, S = Sx (real) or Sy (imaginary), as a sparse matrix."""
            p, z, m = mode(0, j), mode(1, j), mode(2, j)
            up = op_matrix(st, st, [(p, z, 1.0), (z, m, 1.0)])                 # c+_{+1} c_0 + c+_0 c_{-1}  (S^+ / sqrt2)
            if comp == 'x':
                return (up + up.T) / np.sqrt(2)
            return (-1j * up + 1j * up.T) / np.sqrt(2)
        SxN, SxS, SyS = spin(N - 1, 'x'), spin(0, 'x'), spin(0, 'y')
        res = dict(model='spin-1 XY (2604.18705)', N=N, b=b, D_c=Dc, gamma=gam, bulk_check=bulkc, dim=len(st), theta={})
        print(f'dey N={N}: D_c {Dc:.5f}  sigma {bulkc["sigma"]:.4f}  dim {len(st)}  [{time.time() - t0:.0f}s]', flush=True)
        E = {}
        for x in ANG:
            th = np.pi * x
            H = (H0 - b * (SxN + np.cos(th) * SxS + np.sin(th) * SyS)).tocsr()
            k = 6 if x == 0 else 1
            w, v = sla.eigsh(H, k=k, which='SA', tol=1e-12)
            o = np.argsort(w); w, v = w[o], v[:, o]
            E[x] = w[0]
            if x == 0:
                g0 = v[:, 0]
                amp = [abs(np.vdot(v[:, i], Qd * g0)) for i in range(1, k)]
                it = 1 + int(np.argmax(np.array(amp) > 1e-6))
                res['tilt'] = float((w[it] - w[0]) / gam)
                res['theta0_levels'] = ((w - w[0]) / gam).tolist()
                res['theta0_Qamp'] = [0.0] + [float(a) for a in amp]
            res['theta'][f'{x:.4f}'] = float(E[x])
        th = np.pi * np.array(ANG)
        D = np.array([(E[x] - E[0]) / gam for x in ANG])
        xx = 1 - np.cos(th)
        c = np.linalg.lstsq(np.stack([xx, xx ** 2, xx ** 3], 1), D, rcond=None)[0]
        res.update(Delta_pi=float(D[-1]), C_t=float(c[0]), R=float(D[-1] / (2 * c[0])), Delta=D.tolist())
        store.put(os.path.join(OUT, f'dey_N{N}_b{b:g}.json'), res)
        print(f"dey N={N}: tilt {res['tilt']:.4f}  Delta(pi) {D[-1]:.4f}  C_t {c[0]:.4f}  R {res['R']:.4f}  "
              f"Delta(pi/2)/Delta(pi) {D[3] / D[-1]:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        return res

    def main():
        run(int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 8.0)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/theta_dey_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def dmrg_pin():
    """DMRG for the domain wall between phi_3-pinned lines of the O(3) model (ordinary line, alpha = 0).

Spontaneous Z2 breaking on a line, and with it a defect transition along a symmetric easy-axis line coupling, needs the
defect-changing operator between the + and - pinned lines to be irrelevant, Delta_+- > 1.  ED reaches N = 9; Yang
et al. (arXiv:2609.03289) N_o = 10.  This extends the sequence by DMRG.

H(eta) = H_O(3)(h_c(N)) - b [ (psi_0^+ psi_3 + h.c.)_{m=+s} + eta (psi_0^+ psi_3 + h.c.)_{m=-s} ],
with the spectator parity no longer conserved (Q of the (1,2) plane and K stay conserved).  States:
  pp  : eta = +1, Q = 0 ground state;  pm : eta = -1, Q = 0 ground state;  tilt : eta = +1, Q = 1 ground state.
Delta_+- = (E_pm - E_pp) / gamma_B with gamma_B = E(dphi3) - E(phi3) from dmrg_out (same N, same h), and in tilt units
(E_pm - E_pp) / (E_tilt - E_pp).  The tilt is protected at exactly 1 in the continuum.
usage: python dmrg_pin.py N chi [b=8] [states=pp,pm,tilt]   -> dmrg_pin_out/N{N}_chi{chi}_b{b}.json
       python dmrg_pin.py gate              (N = 8: DMRG against ED at the same b)
"""
    import os
    import sys
    import json
    import time
    import glob
    import numpy as np
    HERE = 'flow'
    TW = os.path.join(HERE, '..', 'twist')
    sys.path.insert(0, TW)
    sys.path.insert(0, HERE)
    OUT = os.path.join(HERE, 'dmrg_pin_out')


    def h_of(N):
        if N <= 9:
            return store.get(os.path.join(HERE, f'partB_const_N{N}.json'))['points']['+0.00000']['h_c']
        f = sorted(store.glob(os.path.join(HERE, 'dmrg_box_out', f'N{N}_chi*.json')))[0]
        return store.get(f)['h']


    def build(N, h, eta, b, Q, alpha=0.0):
        _dmrg_jobs = __import__('dmrg'); on_problem = _dmrg_jobs.on_problem   # was: from dmrg_jobs import on_problem
        _dmrg_twist = __import__('dmrg'); fermion_site, compress_mpo = _dmrg_twist.fermion_site, _dmrg_twist.compress_mpo   # was: from dmrg_twist import fermion_site, compress_mpo
        from tenpy.models.model import CouplingModel, MPOModel
        from tenpy.models.lattice import TrivialLattice
        mode_m, charges, names, mods, terms, onsite, occ = on_problem('o3wf', N, alpha, [[Q], 0, [1]], h)
        s = (N - 1) / 2
        if alpha > 0:                                  # axial half-twist: strip the core potential (lam = 0)
            C = fuzzy.core_strip()   # was: import core_strip as C
            fl, orbs, koff, two, mu, dev, hc = C.build('o3wf', N, alpha, 0.0, hc=h)
            assert len(mu) == len(onsite)
            onsite = list(mu)
        n_tw = N - 1 if alpha > 0 else N               # twisted flavours (12+, 12-) have N-1 orbitals
        off3 = N + 2 * n_tw
        assert abs(mode_m[N - 1] - s) < 1e-12 and abs(mode_m[off3 + N - 1] - s) < 1e-12
        assert abs(mode_m[0] + s) < 1e-12 and abs(mode_m[off3] + s) < 1e-12
        ch = [tuple(c[:3]) for c in charges]                          # drop the spectator parity
        nm, md = names[:3], mods[:3]
        order = sorted(range(len(mode_m)), key=lambda k: (mode_m[k], k))
        pos = {k: i for i, k in enumerate(order)}
        lat = TrivialLattice([fermion_site(ch[k], nm, md) for k in order], bc_MPS='finite')
        model = CouplingModel(lat)
        li = lambda p: list(lat.mps2lat_idx(p))
        for (k1, k2, k3, k4, v) in terms:
            model.add_local_term(v, [('Cd', li(pos[k1])), ('Cd', li(pos[k2])), ('C', li(pos[k3])), ('C', li(pos[k4]))])
        for k, v in enumerate(onsite):
            if v != 0:
                model.add_onsite_term(v, pos[k], 'N')
        for a, c, coef in [(N - 1, off3 + N - 1, -b), (0, off3, -b * eta)]:   # north m=+s, south m=-s; flavours 0, 3
            model.add_local_term(coef, [('Cd', li(pos[a])), ('C', li(pos[c]))])
            model.add_local_term(coef, [('Cd', li(pos[c])), ('C', li(pos[a]))])
        mpo = MPOModel(lat, compress_mpo(model.calc_H_MPO()))
        return mpo, order, occ


    def ground(N, h, eta, b, Q, chi, sweeps=30, alpha=0.0):
        _dmrg_twist = __import__('dmrg'); run_dmrg = _dmrg_twist.run_dmrg   # was: from dmrg_twist import run_dmrg
        t = time.time()
        model, order, occ = build(N, h, eta, b, Q, alpha)
        E, psi, eng = run_dmrg(model, order, occ, chi_max=chi, sweeps=sweeps)
        return dict(E=float(E), trunc=float(eng.sweep_stats['max_trunc_err'][-1]), chi=int(max(psi.chi)),
                    sweeps=len(eng.sweep_stats['E']), t=time.time() - t)


    def bulk_unit(N, h):
        if N <= 9:
            return store.get(os.path.join(HERE, f'partB_const_N{N}.json'))['points']['+0.00000']['u']
        ref = {}
        for g in store.glob(os.path.join(HERE, 'dmrg_out', f'N{N}_h{h:.9f}_*_chi*.json')):
            r = store.get(g)
            ref.setdefault(r['state'], {})[r['chi']] = r['E']
        return ref['dphi3'][max(ref['dphi3'])] - ref['phi3'][max(ref['phi3'])]


    def run(N, chi, b=8.0, states=('pp', 'pm', 'tilt'), alpha=0.0):
        pass   # was os.makedirs(...): the store needs no output directory
        fn = os.path.join(OUT, f'N{N}_chi{chi}_b{b:g}' + (f'_a{alpha:g}' if alpha else '') + '.json')
        res = store.get(fn) if store.exists(fn) else dict(N=N, chi=chi, b=b, h=h_of(N), states={})
        h = res['h']
        spec = dict(pp=(+1, 0), pm=(-1, 0), tilt=(+1, 1))
        for st in states:
            if st in res['states']:
                continue
            eta, Q = spec[st]
            res['states'][st] = ground(N, h, eta, b, Q, chi, alpha=alpha)
            store.put(fn, res)
            print(f"N={N} chi={chi} {st}: E {res['states'][st]['E']:.10f} trunc {res['states'][st]['trunc']:.1e} "
                  f"[{res['states'][st]['t']:.0f}s]", flush=True)
        S = res['states']
        if 'pp' in S and 'pm' in S:
            u = bulk_unit(N, h)
            res['Delta_pm_bulk'] = (S['pm']['E'] - S['pp']['E']) / u
            if 'tilt' in S:
                res['tilt_bulk'] = (S['tilt']['E'] - S['pp']['E']) / u
                res['Delta_pm_tilt'] = (S['pm']['E'] - S['pp']['E']) / (S['tilt']['E'] - S['pp']['E'])
            store.put(fn, res)
            print(f"N={N}: Delta_+- {res['Delta_pm_bulk']:.4f} (bulk)  tilt {res.get('tilt_bulk', np.nan):.4f}  "
                  f"Delta_+- {res.get('Delta_pm_tilt', np.nan):.4f} (tilt units)", flush=True)
        return res


    def gate():
        sys.path.insert(0, os.path.join(TW, 'spectator_pinning'))
        S = fuzzy.question('02_rg_flow/easy_axis').scan()   # was: import scan as S
        N, b = 8, 8.0
        ps = S.PinnedSector(N, alpha=0.0, Q=0, K=0)
        e_pp = ps.finite(b, +1, k=2)[0]['E'][0]
        e_pm = ps.finite(b, -1, k=2)[0]['E'][0]
        ed = (e_pm - e_pp) / ps.bulk['gamma_B']
        r = run(N, 400, b, ('pp', 'pm'))
        print(f"gate N=8 b=8: ED E_pp {e_pp:.10f} E_pm {e_pm:.10f} Delta {ed:.6f} | DMRG E_pp {r['states']['pp']['E']:.10f} "
              f"E_pm {r['states']['pm']['E']:.10f} Delta {r['Delta_pm_bulk']:.6f}")

    def main():
        if sys.argv[1] == 'gate':
            gate()
        else:
            run(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 8.0,
                tuple(sys.argv[4].split(',')) if len(sys.argv) > 4 else ('pp', 'pm', 'tilt'),
                float(sys.argv[5]) if len(sys.argv) > 5 else 0.0)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('flow/dmrg_pin_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def dmrg_theta():
    """DMRG for the theta junction between pinned lines of the O(N_v) Wilson-Fisher models (theta_on.py beyond ED).

Same Hamiltonian as theta_on.py (Cartesian flavours psi_0..psi_Nv, north pinned along e_1, south along
cos(theta) e_1 + sin(theta) e_2, pole hopping b), with only N_e and K conserved, the same for every theta so that
truncation errors are alike across the angle scan.  The shape ratio R = Delta(pi) / (2 Delta''(0)) and the profile
Delta(theta) / Delta(pi) need no bulk unit, so N beyond the stored bulk tunings only needs h_c (taken from the
largest stored N; h_c drifts by < 3e-4 per N).
usage: python dmrg_theta.py <cfg> N chi [thetas/pi = 0,0.25,0.5,0.75,1] [b=8]
       -> theta_junction_results/dmrg_<cfg>_N{N}_chi{chi}_b{b}.json
       python dmrg_theta.py gate <cfg> N chi       (DMRG against theta_on.py ED at the same N)
       python dmrg_theta.py unit <cfg> N chi       (bulk unit u = E(d phi) - E(phi), charge basis, same h as the scan)
       python dmrg_theta.py tilt <cfg> N chi       (theta = 0 lowest state odd under psi_2 -> -psi_2: the tilt, in the
                                                    same file; the parity of flavour 2 is conserved only at theta = 0, pi)
"""
    import os
    import sys
    import json
    import time
    import glob
    import numpy as np
    HERE = 'twist'
    sys.path.insert(0, HERE)
    OUT = os.path.join(HERE, 'theta_junction_results')


    def h_of(cfgname, N):
        f = os.path.join(HERE, 'on_results', f'{cfgname}_N{N}.json')
        if store.exists(f):
            return store.get(f)['h_c'], True
        if cfgname == 'o3wf':                         # DMRG-tuned h_c of the O(3) box runs, as flow/dmrg_pin.py uses
            sys.path.insert(0, os.path.join(HERE, '..', 'flow'))
            dmrg_pin = globals()['dmrg_pin']()   # was: import dmrg_pin
            return float(dmrg_pin.h_of(N)), True
        # beyond the stored tunings: h_c(N) = h_inf + a N^-q fitted to the stored N (dR/dh ~ 5 at N = 9, so freezing h at the
        # last stored N would bias R by -0.001 per extra N)
        from scipy.optimize import curve_fit
        Ns = sorted(int(os.path.basename(g)[len(cfgname) + 2:-5]) for g in store.glob(os.path.join(HERE, 'on_results', f'{cfgname}_N*.json')))
        hs = [store.get(os.path.join(HERE, 'on_results', f'{cfgname}_N{n}.json'))['h_c'] for n in Ns]
        p, _ = curve_fit(lambda n, hi, a, q: hi + a * n ** -q, np.array(Ns, float), np.array(hs), p0=[hs[-1], -0.01, 1.0], maxfev=20000)
        return float(p[0] + p[1] * N ** -p[2]), False


    def build(cfgname, N, h, theta, b, odd=False):
        M = fuzzy.onmodel()   # was: import onmodel as M
        _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
        _dmrg_twist = __import__('dmrg'); fermion_site, compress_mpo = _dmrg_twist.fermion_site, _dmrg_twist.compress_mpo   # was: from dmrg_twist import fermion_site, compress_mpo
        from tenpy.models.model import CouplingModel, MPOModel
        from tenpy.models.lattice import TrivialLattice
        cfg = CONFIGS[cfgname]
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, cfg['U'], h, pairs=())
        mode_m = [m for o in orbs for m in o]
        if odd:
            assert abs(np.sin(theta)) < 1e-12
            charges = [(1, jj, int(f == 2)) for f, o in enumerate(orbs) for jj in range(len(o))]
            names, mods = ['N', 'K', 'P2'], [1, 1, 2]
        else:
            charges = [(1, jj) for o in orbs for jj in range(len(o))]
            names, mods = ['N', 'K'], [1, 1]
        order = sorted(range(len(mode_m)), key=lambda k: (mode_m[k], k))
        pos = {k: i for i, k in enumerate(order)}
        lat = TrivialLattice([fermion_site(charges[k], names, mods) for k in order], bc_MPS='finite')
        model = CouplingModel(lat)
        li = lambda p: list(lat.mps2lat_idx(p))
        for (k1, k2, k3, k4, v) in two:
            model.add_local_term(v, [('Cd', li(pos[k1])), ('Cd', li(pos[k2])), ('C', li(pos[k3])), ('C', li(pos[k4]))])
        for k, v in enumerate(mu):
            if v != 0:
                model.add_onsite_term(v, pos[k], 'N')
        hops = [(N - 1, 2 * N - 1, -b), (0, N, -b * np.cos(theta)), (0, 2 * N, -b * np.sin(theta))]
        for a, c, coef in hops:                        # north psi_0-psi_1 at m=+s; south psi_0-psi_1, psi_0-psi_2 at m=-s
            if abs(coef) > 1e-14:
                model.add_local_term(coef, [('Cd', li(pos[a])), ('C', li(pos[c]))])
                model.add_local_term(coef, [('Cd', li(pos[c])), ('C', li(pos[a]))])
        occ = set(range(N)) if not odd else (set(range(N)) - {N // 2}) | {2 * N + N // 2}   # odd: one psi_0 -> psi_2, same m
        return MPOModel(lat, compress_mpo(model.calc_H_MPO())), order, occ   # psi_0 filled: K = 0


    def ground(cfgname, N, h, theta, b, chi, sweeps=30, odd=False):
        _dmrg_twist = __import__('dmrg'); run_dmrg = _dmrg_twist.run_dmrg   # was: from dmrg_twist import run_dmrg
        t = time.time()
        model, order, occ = build(cfgname, N, h, theta, b, odd)
        E, psi, eng = run_dmrg(model, order, occ, chi_max=chi, sweeps=sweeps)
        return dict(theta=float(theta), E=float(E), trunc=float(eng.sweep_stats['max_trunc_err'][-1]),
                    chi=int(max(psi.chi)), sweeps=len(eng.sweep_stats['E']), t=time.time() - t)


    def run(cfgname, N, chi, thetas=(0, .25, .5, .75, 1), b=8.0):
        pass   # was os.makedirs(...): the store needs no output directory
        fn = os.path.join(OUT, f'dmrg_{cfgname}_N{N}_chi{chi}_b{b:g}.json')
        h, exact = h_of(cfgname, N)
        res = store.get(fn) if store.exists(fn) else dict(cfg=cfgname, N=N, chi=chi, b=b, h=h, h_stored=exact, theta={})
        for x in thetas:
            key = f'{x:.4f}'
            if key in res['theta']:
                continue
            res['theta'][key] = ground(cfgname, N, res['h'], np.pi * x, b, chi)
            store.put(fn, res)
            r = res['theta'][key]
            print(f"{cfgname} N={N} chi={chi} theta/pi={x:.4f}: E {r['E']:.10f} trunc {r['trunc']:.1e} chi {r['chi']} [{r['t']:.0f}s]", flush=True)
        T = res['theta']
        if '0.0000' in T and '1.0000' in T:
            e0, ep = T['0.0000']['E'], T['1.0000']['E']
            print(f"  Delta(theta)/Delta(pi): " + ' '.join(f"{k}:{(v['E'] - e0) / (ep - e0):.4f}" for k, v in sorted(T.items())), flush=True)
        return res


    def tilt(cfgname, N, chi, b=8.0):
        fn = os.path.join(OUT, f'dmrg_{cfgname}_N{N}_chi{chi}_b{b:g}.json')
        res = store.get(fn)
        if 'tilt' not in res:
            res['tilt'] = ground(cfgname, N, res['h'], 0.0, b, chi, odd=True)
            store.put(fn, res)
        T = res['theta']
        r = (res['tilt']['E'] - T['0.0000']['E']) / (T['1.0000']['E'] - T['0.0000']['E'])
        print(f"{cfgname} N={N} chi={chi} tilt: E {res['tilt']['E']:.10f} trunc {res['tilt']['trunc']:.1e} [{res['tilt']['t']:.0f}s]"
              f"  tilt/Delta(pi) {r:.4f}", flush=True)


    def unit(cfgname, N, chi, b=8.0):
        _dmrg_jobs = __import__('dmrg'); on_problem = _dmrg_jobs.on_problem   # was: from dmrg_jobs import on_problem
        _dmrg_twist = __import__('dmrg'); build_model, run_dmrg = _dmrg_twist.build_model, _dmrg_twist.run_dmrg   # was: from dmrg_twist import build_model, run_dmrg
        fn = os.path.join(OUT, f'dmrg_{cfgname}_N{N}_chi{chi}_b{b:g}.json')
        res = store.get(fn)
        for lab, K in (('phi', 0), ('dphi', 1)):
            if lab in res.get('unit', {}):
                continue
            t = time.time()
            mode_m, charges, names, mods, terms, onsite, occ = on_problem(cfgname, N, 0.0, [[1], K, []], res['h'])
            model, order = build_model(mode_m, charges, names, terms, onsite, mods)
            E, psi, eng = run_dmrg(model, order, occ, chi_max=chi, sweeps=30)
            res.setdefault('unit', {})[lab] = dict(E=float(E), trunc=float(eng.sweep_stats['max_trunc_err'][-1]), t=time.time() - t)
            store.put(fn, res)
        U = res['unit']
        u = U['dphi']['E'] - U['phi']['E']
        print(f"{cfgname} N={N} chi={chi} unit u {u:.8f}" + (f"  tilt-1 {(res['tilt']['E'] - res['theta']['0.0000']['E']) / u - 1:.4f}" if 'tilt' in res else ''), flush=True)


    def gate(cfgname, N, chi):
        ed = store.get(os.path.join(OUT, f'{cfgname}_N{N}_b8.json'))
        r = run(cfgname, N, chi, (0, .5, 1))
        for x in ('0.0000', '0.5000', '1.0000'):
            e_ed = ed['theta'][f'{np.pi * float(x):.6f}']['E'][0]
            print(f'gate {cfgname} N={N} theta/pi={x}: ED {e_ed:.10f} DMRG {r["theta"][x]["E"]:.10f} diff {r["theta"][x]["E"] - e_ed:.2e}')

    def main():
        if sys.argv[1] == 'gate':
            gate(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        elif sys.argv[1] == 'unit':
            unit(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        elif sys.argv[1] == 'tilt':
            tilt(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        else:
            run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                tuple(float(x) for x in sys.argv[4].split(',')) if len(sys.argv) > 4 else (0, .25, .5, .75, 1),
                float(sys.argv[5]) if len(sys.argv) > 5 else 8.0)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/dmrg_theta_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def dmrg_dey():
    """DMRG for the pinned-line junction of the spin-1 XY regulator (theta_dey.py beyond ED), Phase B item 1.

Same Hamiltonian as theta_dey.py: bulk of o2twist.py at the tuned D_c(N), north pole orbital pinned along S^x and south
pole orbital along cos(theta) S^x + sin(theta) S^y with strength b (complex Hermitian for 0 < theta < pi), conserving only
N_e and L_z.  The tilt is the first excited theta = 0 state, as in ED for N_m = 6-10 (gap to the next level ~0.35); its Ward
amplitude <t|Q|0> with Q = N_{+1} - N_{-1} gives C^HQ = (level |<t|Q|0>|)^2 / 4 in bulk units.
Bulk unit gamma = E(d sigma) - E(sigma) and D_c: ED file for N_m <= 10, stored bulk DMRG (dmrg_results/o2_N*_a0) beyond.
usage: python dmrg_dey.py N chi [thetas/pi = 0,0.125,0.25,1] [b=1024]   -> theta_junction_results/dmrg_dey_N{N}_chi{chi}_b{b}.json
       python dmrg_dey.py tilt N chi [b=1024]                          (first excited theta = 0 state and its Ward amplitude)
       python dmrg_dey.py gate N chi                                   (against theta_dey.py ED at the same N)
"""
    import os, sys, json, time, glob
    import numpy as np
    HERE = 'twist'
    sys.path.insert(0, HERE)
    OUT = os.path.join(HERE, 'theta_junction_results')


    def bulk(N):
        ed = os.path.join(OUT, f'dey_N{N}_b1024.json')
        if store.exists(ed):
            d = store.get(ed); return d['D_c'], d['gamma'], 'ED'
        f = lambda Q, K: store.glob(os.path.join(HERE, 'dmrg_results', f'o2_N{N}_a0_s{Q}_{K}_chi*.json'))
        s10, s11 = store.get(f(1, 0)[0]), store.get(f(1, 1)[0])
        assert abs(s10['D'] - s11['D']) < 1e-12
        return s10['D'], s11['E'] - s10['E'], f'DMRG chi {s10["chi"]}'


    def build(N, D, theta, b):
        _twist = fuzzy; couplings_for = _twist.couplings_for   # was: from twist import couplings_for
        _o2twist = fuzzy.o2twist(); orbitals, build_terms = _o2twist.orbitals, _o2twist.build_terms   # was: from o2twist import orbitals, build_terms
        _dmrg_twist = __import__('dmrg'); fermion_site, compress_mpo = _dmrg_twist.fermion_site, _dmrg_twist.compress_mpo   # was: from dmrg_twist import fermion_site, compress_mpo
        from tenpy.models.model import CouplingModel, MPOModel
        from tenpy.models.lattice import TrivialLattice
        G0, G1, _ = couplings_for(N, 4.0, 1.0)
        orb = orbitals(N, 0.0)
        off = [0, len(orb[0]), len(orb[0]) + len(orb[1])]          # flavours: 0 = S^z +1, 1 = S^z 0, 2 = S^z -1
        mode_m, charges = [], []
        for f in range(3):
            for j, m in enumerate(orb[f]):
                mode_m.append(m); charges.append((1, j))
        two, U = build_terms(N, 0.0, G0, G1)
        onsite = [U[f][j] + (D if f != 1 else 0.0) for f in range(3) for j in range(len(orb[f]))]
        order = sorted(range(len(mode_m)), key=lambda k: (mode_m[k], k))
        pos = {k: i for i, k in enumerate(order)}
        lat = TrivialLattice([fermion_site(charges[k], ['N', 'K'], [1, 1]) for k in order], bc_MPS='finite')
        model = CouplingModel(lat)
        li = lambda p: list(lat.mps2lat_idx(p))
        for (k1, k2, k3, k4, v) in two:
            model.add_local_term(v, [('Cd', li(pos[k1])), ('Cd', li(pos[k2])), ('C', li(pos[k3])), ('C', li(pos[k4]))])
        for k, v in enumerate(onsite):
            if v != 0:
                model.add_onsite_term(v, pos[k], 'N')
        # pin: -b [S^x_N + cos(theta) S^x_S + sin(theta) S^y_S], cos S^x + sin S^y = (e^{-i theta} up + e^{i theta} up^+)/sqrt2,
        # up = c+_{+1} c_0 + c+_0 c_{-1} on the pole orbital (north j = N-1, south j = 0)
        for j, ph in ((N - 1, 0.0), (0, theta)):
            c = -b / np.sqrt(2) * np.exp(-1j * ph)
            for a, z in ((off[0] + j, off[1] + j), (off[1] + j, off[2] + j)):
                coef = c if abs(np.sin(ph)) > 1e-14 else c.real
                model.add_local_term(coef, [('Cd', li(pos[a])), ('C', li(pos[z]))])
                model.add_local_term(np.conj(coef), [('Cd', li(pos[z])), ('C', li(pos[a]))])
        occ = {off[1] + j for j in range(N)}                       # S^z = 0 flavour filled: K = 0
        Qsign = {k: (1 if k < off[1] else (-1 if k >= off[2] else 0)) for k in range(len(mode_m))}
        return MPOModel(lat, compress_mpo(model.calc_H_MPO())), order, occ, Qsign


    def ground(N, D, theta, b, chi, sweeps=30, excited=False):
        _dmrg_twist = __import__('dmrg'); run_dmrg = _dmrg_twist.run_dmrg   # was: from dmrg_twist import run_dmrg
        t = time.time()
        model, order, occ, Qs = build(N, D, theta, b)
        E, psi, eng = run_dmrg(model, order, occ, chi_max=chi, sweeps=sweeps)
        out = dict(theta=float(theta), E=float(E), trunc=float(eng.sweep_stats['max_trunc_err'][-1]), chi=int(max(psi.chi)), t=time.time() - t)
        if excited:
            from tenpy.networks.mps import MPSEnvironment
            E1, psi1, eng1 = run_dmrg(model, order, occ, chi_max=chi, sweeps=sweeps, orthogonal_to=[psi])
            env = MPSEnvironment(psi1, psi)
            nexp = env.expectation_value('N')
            amp = abs(sum(Qs[order[p]] * nexp[p] for p in range(len(order))))
            out.update(E1=float(E1), trunc1=float(eng1.sweep_stats['max_trunc_err'][-1]), Qamp=float(amp),
                       overlap=float(abs(psi1.overlap(psi))), t=time.time() - t)
        return out


    def run(N, chi, thetas=(0, .125, .25, 1), b=1024.0):
        pass   # was os.makedirs(...): the store needs no output directory
        fn = os.path.join(OUT, f'dmrg_dey_N{N}_chi{chi}_b{b:g}.json')
        D, gam, src = bulk(N)
        res = store.get(fn) if store.exists(fn) else dict(model='spin-1 XY', N=N, chi=chi, b=b, D_c=D, gamma=gam, bulk_src=src, theta={})
        for x in thetas:
            key = f'{x:.4f}'
            if key in res['theta']:
                continue
            res['theta'][key] = ground(N, D, np.pi * x, b, chi)
            store.put(fn, res)
            r = res['theta'][key]
            print(f"dey N={N} chi={chi} theta/pi={x:.4f}: E {r['E']:.10f} trunc {r['trunc']:.1e} chi {r['chi']} [{r['t']:.0f}s]", flush=True)
        return res


    def tilt(N, chi, b=1024.0):
        fn = os.path.join(OUT, f'dmrg_dey_N{N}_chi{chi}_b{b:g}.json')
        res = store.get(fn)
        if 'tilt' not in res:
            res['tilt'] = ground(N, res['D_c'], 0.0, b, chi, excited=True)
            store.put(fn, res)
        T, g = res['theta'], res['gamma']
        lev = (res['tilt']['E1'] - T['0.0000']['E']) / g
        print(f"dey N={N} chi={chi}: tilt {lev:.5f}  |<t|Q|0>| {res['tilt']['Qamp']:.5f}  C^HQ {(lev * res['tilt']['Qamp']) ** 2 / 4:.5f}"
              f"  |<t|0>| {res['tilt']['overlap']:.1e}  trunc {res['tilt']['trunc1']:.1e} [{res['tilt']['t']:.0f}s]", flush=True)


    def gate(N, chi):
        ed = store.get(os.path.join(OUT, f'dey_N{N}_b1024.json'))
        r = run(N, chi, (0, .125, .25, 1))
        for x in ('0.0000', '0.1250', '0.2500', '1.0000'):
            print(f"gate N={N} theta/pi={x}: ED {ed['theta'][x]:.10f} DMRG {r['theta'][x]['E']:.10f} diff {r['theta'][x]['E'] - ed['theta'][x]:.2e}")
        tilt(N, chi)
        print(f"gate ED: tilt {ed['theta0_levels'][1]:.5f}  |<t|Q|0>| {ed['theta0_Qamp'][1]:.5f}  C^HQ {(ed['theta0_levels'][1] * ed['theta0_Qamp'][1]) ** 2 / 4:.5f}")

    def main():
        if sys.argv[1] == 'gate':
            gate(int(sys.argv[2]), int(sys.argv[3]))
        elif sys.argv[1] == 'tilt':
            tilt(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 1024.0)
        else:
            run(int(sys.argv[1]), int(sys.argv[2]),
                tuple(float(x) for x in sys.argv[3].split(',')) if len(sys.argv) > 3 else (0, .125, .25, 1),
                float(sys.argv[4]) if len(sys.argv) > 4 else 1024.0)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/dmrg_dey_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def dmrg_wall():
    """Direct domain-wall dimension Delta_+- = [E(pi) - E(0)] / gamma_B of the pinned O(N_v) line by DMRG, reusing a dmrg_hq.py run.

dmrg_hq.py already holds E(0) (the theta = 0 pinned ground state of dmrg_theta.build), the tilt level and the bulk unit u at
its h, b and chi, so the wall needs one more ground state: theta = pi (south pin along -e_1), same N_e, K sector, same h,
b, chi and sweeps.  Also records Delta_+- / (tilt - 1) data for the tilt-excess extrapolation.
usage: python dmrg_wall.py <cfg> N chi        (needs ct_ward_results/hq_<cfg>_N{N}_chi{chi}.json)
       -> theta_junction_results/wall_<cfg>_N{N}_chi{chi}.json
       python dmrg_wall.py gate <cfg> N chi   (theta = 0 and pi by DMRG against theta_on.py ED at the same N, stored h_c)
"""
    import os
    import sys
    import json
    import numpy as np
    HERE = 'twist'
    sys.path.insert(0, HERE)
    _dmrg_theta = globals()['dmrg_theta'](); ground, h_of = _dmrg_theta.ground, _dmrg_theta.h_of   # was: from dmrg_theta import ground, h_of

    OUT = os.path.join(HERE, 'theta_junction_results')


    def run(cfgname, N, chi):
        hq = store.get(os.path.join(HERE, 'ct_ward_results', f'hq_{cfgname}_N{N}_chi{chi}.json'))
        r = ground(cfgname, N, hq['h'], np.pi, hq['b'], chi)
        D = (r['E'] - hq['E0']) / hq['u']
        res = dict(cfg=cfgname, N=N, chi=chi, b=hq['b'], h=hq['h'], E0=hq['E0'], Epi=r['E'], u=hq['u'], tilt=hq['tilt'],
                   Delta_pi=float(D), trunc_pi=r['trunc'], seconds=r['t'])
        store.put(os.path.join(OUT, f'wall_{cfgname}_N{N}_chi{chi}.json'), res)
        print(f"{cfgname} N={N} chi={chi}: Delta_+- {D:.5f}  tilt {hq['tilt']:.4f}  trunc {r['trunc']:.0e}  [{r['t']:.0f}s]", flush=True)
        return res


    def gate(cfgname, N, chi):
        ed = store.get(os.path.join(OUT, f'{cfgname}_N{N}_b8.json'))
        h, _ = h_of(cfgname, N)
        E = {}
        for x in (0.0, 1.0):
            E[x] = ground(cfgname, N, h, np.pi * x, 8.0, chi)['E']
            e_ed = ed['theta'][f'{np.pi * x:.6f}']['E'][0]
            print(f'gate {cfgname} N={N} theta/pi={x:.0f}: ED {e_ed:.10f} DMRG {E[x]:.10f} diff {E[x] - e_ed:.2e}', flush=True)
        print(f"gate {cfgname} N={N}: Delta_+- DMRG {(E[1.0] - E[0.0]) / ed['gamma_B']:.5f}", flush=True)

    def main():
        if sys.argv[1] == 'gate':
            gate(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        else:
            run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/dmrg_wall_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def o4_package():
    """O(4) junction package (2026-10-25): angle curvature C^ang, shape r = Delta(pi)/(2 C^ang), Ward stiffness C^HQ and
rho_N = Delta(pi)/(2 C^HQ) for N = 6-9, unrounded, as JSON (o4_package.json) plus a printed table.

N = 6-8: ED angle scans (theta_junction_results/o4wf_diag_N{N}_b8[_red].json, nine angles), C^HQ from tilt_test_results.
N = 9: DMRG scan dmrg_o4wf_diag_N9_chi800_b8.json (theta/pi = 0, 1/8, 1/4, 1/2, 3/4, 1; theta = 0 from dmrg_hq, pi from
dmrg_wall, same h, b, chi), C^HQ from ct_ward_results/hq_o4wf_diag_N9_chi800.json.
Fit: Delta = C t + u2 t^2 + u3 t^3, t = 1 - cos(theta), least squares over the angles; for comparability the ED sizes are also
fitted on the N = 9 angle subset.
usage: python o4_package.py

An earlier summary of the O(4) scans with the global cubic fit of the curvature, which the paper replaced by the
two-angle curvature of junction_analysis (App. E, "Shape of the junction"). Kept for its unrounded per-size numbers.
"""
    import os
    import json
    import numpy as np
    HERE = 'twist'
    SUB = (0.125, 0.25, 0.5, 0.75, 1.0)


    def fit(th, D):
        t = 1 - np.cos(np.asarray(th))
        A = np.vstack([t, t ** 2, t ** 3]).T
        c = np.linalg.lstsq(A, np.asarray(D), rcond=None)[0]
        return c


    def ed(N):
        f = os.path.join(HERE, 'theta_junction_results', f'o4wf_diag_N{N}_b8.json')
        if not store.exists(f):
            f = f.replace('_b8.json', '_b8_red.json')
        r = store.get(f)
        E = {v['theta']: v['E'][0] for v in r['theta'].values()}
        th = np.array(sorted(E)); g = r['gamma_B']
        D = np.array([(E[x] - E[0.0]) / g for x in th])
        tilt = (r['theta'][next(iter(r['theta']))]['E'][1] - E[0.0]) / g if False else None
        return th, D, f


    def main():
        out = {}
        for N in (6, 7, 8):
            th, D, f = ed(N)
            c_all = fit(th[1:], D[1:])
            sel = [i for i, x in enumerate(th) if any(abs(x - np.pi * s) < 1e-9 for s in SUB)]
            c_sub = fit(th[sel], D[sel])
            hq = store.get(os.path.join(HERE, 'tilt_test_results', f'o4wf_diag_N{N}.json'))['C_t_HQ']
            tilt = store.get(os.path.join(HERE, 'ct_ward_results', f'o4wf_diag_N{N}.json'))['tilt']
            Dpi = float(D[-1])
            out[N] = dict(source='ED ' + os.path.basename(f), Delta_pi=Dpi, tilt=tilt, C_ang=float(c_all[0]), C_ang_sub=float(c_sub[0]),
                          u2=float(c_all[1]), u3=float(c_all[2]), r_ang=Dpi / (2 * c_all[0]), r_ang_sub=Dpi / (2 * c_sub[0]),
                          C_HQ=hq, rho=Dpi / (2 * hq), profile={f'{x / np.pi:.4f}': float(d / Dpi) for x, d in zip(th, D)})
        s = store.get(os.path.join(HERE, 'theta_junction_results', 'dmrg_o4wf_diag_N9_chi800_b8.json'))
        hq = store.get(os.path.join(HERE, 'ct_ward_results', 'hq_o4wf_diag_N9_chi800.json'))
        T = s['theta']; u = s['unit_u']; e0 = T['0.0000']['E']
        keys = sorted(T, key=float)
        th = np.array([np.pi * float(k) for k in keys]); D = np.array([(T[k]['E'] - e0) / u for k in keys])
        complete = all(f'{x:.4f}' in T for x in SUB)
        c = fit(th[1:], D[1:]); Dpi = float(D[-1])
        out[9] = dict(source='DMRG chi800 dmrg_o4wf_diag_N9_chi800_b8.json', complete=complete, angles=[float(k) for k in keys],
                      Delta_pi=Dpi, tilt=hq['tilt'], C_ang=float(c[0]), C_ang_sub=float(c[0]), u2=float(c[1]), u3=float(c[2]),
                      r_ang=Dpi / (2 * c[0]), r_ang_sub=Dpi / (2 * c[0]), C_HQ=hq['C_t_HQ'], C_Ward_density=hq['C_t_Ward'],
                      rho=Dpi / (2 * hq['C_t_HQ']), trunc={k: T[k].get('trunc') for k in keys},
                      profile={k: float((T[k]['E'] - e0) / u / Dpi) for k in keys})
        store.put(os.path.join(HERE, 'o4_package.json'), out)
        print(f"{'N':>2} {'Delta(pi)':>10} {'tilt':>7} {'C_ang':>8} {'C_ang(sub)':>10} {'r_ang':>7} {'r(sub)':>7} {'C_HQ':>8} {'rho_N':>7}")
        for N, v in out.items():
            print(f"{N:>2} {v['Delta_pi']:>10.6f} {v['tilt']:>7.4f} {v['C_ang']:>8.5f} {v['C_ang_sub']:>10.5f} {v['r_ang']:>7.4f} "
                  f"{v['r_ang_sub']:>7.4f} {v['C_HQ']:>8.5f} {v['rho']:>7.4f}" + ('' if v.get('complete', True) else '   (angle 3/4 pending)'))
        Ns = np.array([6, 7, 8, 9.]); tl = np.array([out[n]['tilt'] - 1 for n in (6, 7, 8, 9)])
        for lab, key in (('C_ang (subset)', 'C_ang_sub'), ('r_ang (subset)', 'r_ang_sub'), ('rho_N', 'rho')):
            y = np.array([out[n][key] for n in (6, 7, 8, 9)])
            l1 = np.polyfit(1 / Ns[-3:], y[-3:], 1)[1]; t1 = np.polyfit(tl[-3:], y[-3:], 1)[1]; t2 = np.polyfit(tl, y, 2)[2]
            print(f'  {lab:>15}: 1/N linear (N=7-9) {l1:.4f}   tilt-1 linear (N=7-9) {t1:.4f}   tilt-1 quadratic (N=6-9) {t2:.4f}')

    def _run():
        main()

    def cli(argv):
        with fuzzy.logged('twist/o4_package.log'):
            _run()
    return namespace(locals(), globals())


@section
def junction_analysis():
    """Reproduce the junction audit from named, frozen outputs; no many-body runs.

Fit intervals are sensitivity to the listed ansatz and size window, not errors.
Run directly to refresh JUNCTION_ROUTE_AUDIT.json and AUDIT_INPUT_MANIFEST.json.

This is the paper's audit script (figure_sources/junction_analysis.py); its inputs are read from the store under
their research paths. Its nonlinear fits (curve_fit) reproduce to about 1e-10 relative across BLAS builds.
"""
    from pathlib import Path
    import hashlib
    import json
    import shutil
    import warnings
    import numpy as np
    from scipy.optimize import curve_fit, OptimizeWarning
    HERE = Path(__file__).resolve().parent
    DATA = HERE / 'data'
    ROOT = HERE.parents[2]
    INPUTS = {}

    def read(rel):
        INPUTS[rel] = 'stored'   # the paper's copy recorded the sha256 of each frozen input in AUDIT_INPUT_MANIFEST.json
        return store.get(rel)

    def curvature(th, d):
        # Small-angle curvature: D = C x + u2 x^2 through the two smallest nonzero angles (pi/8, pi/4), x = 1 - cos(eta).
        # It reproduces the exact second-order curvature E''(0) to < 1e-4 at N_m <= 9; the global cubic fit in x over all
        # angles, kept as a diagnostic, overstates it by an amount that grows with N_m.
        x = 1-np.cos(th)
        two = (d[1]*x[2]**2-d[2]*x[1]**2)/(x[1]*x[2]**2-x[2]*x[1]**2)
        cubic = np.linalg.lstsq(np.array([x,x*x,x**3]).T,d,rcond=None)[0][0]
        return two, cubic

    def vector(name):
        r = read('twist/theta_junction_results/' + name)
        th = np.array(sorted(v['theta'] for v in r['theta'].values()))
        E = {v['theta']: v['E'] for v in r['theta'].values()}
        if 'gamma_B' in r:
            g = r['gamma_B']; e0 = E[0][0]
            d = np.array([(E[t][0]-e0)/g for t in th])
            tilt = (E[0][1]-e0)/g
        else:
            g = r['unit']['dphi']['E']-r['unit']['phi']['E']; e0 = E[0]
            d = np.array([(E[t]-e0)/g for t in th])
            tilt = (r['tilt']['E']-e0)/g
        c, cub = curvature(th, d)
        return dict(N=r['N'], tilt=tilt, Cangle=c, Cangle_cubic=cub, r=d[-1]/(2*c), wall=d[-1],
                    angles=len(th))

    def o4_dmrg_scan(name, ward):
        # DMRG angle scan (theta = 0 from dmrg_hq.py, theta = pi from dmrg_wall.py, same h, b, chi);
        # energies are stored without a tilt level, which comes from the Ward run at the same h.
        r = read('twist/theta_junction_results/' + name)
        th = np.array(sorted(v['theta'] for v in r['theta'].values()))
        E = {v['theta']: v['E'] for v in r['theta'].values()}
        d = np.array([(E[t]-E[0])/r['unit_u'] for t in th])
        c, cub = curvature(th, d)
        tilt = read('twist/ct_ward_results/' + ward)['tilt']
        return dict(N=r['N'], tilt=tilt, Cangle=c, Cangle_cubic=cub, r=d[-1]/(2*c), wall=d[-1], angles=len(th))

    def power_fit(n,y):
        # Three-point fits have no residual degrees of freedom; never use covariance
        # as an uncertainty. These are recorded only as ansatz sensitivity.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            p = curve_fit(lambda n,a,b,p:a+b*n**(-p),np.asarray(n,float),y,
                          p0=[float(y[-1]),1.,2.],maxfev=20000)[0]
        return p.tolist()

    def analyze():
        names=[f'o2wf_N{n}_b8.json' for n in range(6,10)]
        names += [f'dmrg_o2wf_N{n}_chi{c}_b8.json' for n,c in
                  [(10,700),(11,800),(12,1000),(13,800),(14,900)]]
        data={'O2':[vector(n) for n in names],
              'O3':[vector(n) for n in ['N5_b8.json','o3wf_N6_b8.json','N7_b8.json','N8_b8.json']],
              'O4':[vector(n) for n in ['o4wf_diag_N6_b8.json','o4wf_diag_N7_b8_red.json','o4wf_diag_N8_b8_red.json']]}
        data['spin1']=[]
        for n in range(6,11):
            d=read(f'twist/theta_junction_results/dey_N{n}_b1024.json')
            th=np.array(sorted(float(k) for k in d['theta']))*np.pi
            E={float(k)*np.pi:v for k,v in d['theta'].items()}
            prof=np.array([(E[t]-E[0.0])/d['gamma'] for t in th])
            c,cub=curvature(th,prof)
            data['spin1'].append(dict(N=n,tilt=d['tilt'],Cangle=c,Cangle_cubic=cub,r=d['Delta_pi']/(2*c),
                wall=d['Delta_pi'],HQ=(d['theta0_levels'][1]*d['theta0_Qamp'][1])**2/4,
                angles=len(d['theta']),profile_wall=float(prof[-1])))
        for n,c in [(12,700),(14,900)]:           # spin-one DMRG (twist/dmrg_dey.py): angles 0, pi/8, pi/4, pi and the tilt state
            d=read(f'twist/theta_junction_results/dmrg_dey_N{n}_chi{c}_b1024.json')
            T,g=d['theta'],d['gamma']
            prof=np.array([(T[k]['E']-T['0.0000']['E'])/g for k in ('0.0000','0.1250','0.2500','1.0000')])
            cc=curvature(np.array([0,.125,.25])*np.pi,prof[:3])[0]
            lev=(d['tilt']['E1']-T['0.0000']['E'])/g
            data['spin1'].append(dict(N=n,tilt=lev,Cangle=cc,Cangle_cubic=None,r=prof[-1]/(2*cc),wall=float(prof[-1]),
                HQ=(lev*d['tilt']['Qamp'])**2/4,angles=4))
        ed3=list(data['O3'])
        for n,c in [(9,600),(10,800),(11,800),(12,1000),(13,1200),(14,1200)]:
            d=read(f'flow/dmrg_pin_out/N{n}_chi{c}_b8.json')
            row=dict(N=n,tilt=d['tilt_bulk'],wall=d['Delta_pm_bulk'])
            if n in (10,12):
                # DMRG angle scan at eta = 0, pi/8, pi/4 with the same h, b and chi as the wall run (E_0 agrees to 1e-7);
                # bulk unit from the wall run.  Three angles fix the two-angle curvature; no global cubic.
                s=read(f'twist/theta_junction_results/dmrg_o3wf_N{n}_chi{c}_b8.json')['theta']
                g=(d['states']['pm']['E']-d['states']['pp']['E'])/d['Delta_pm_bulk']
                th=np.array(sorted(v['theta'] for v in s.values())); E={v['theta']:v['E'] for v in s.values()}
                c,_=curvature(th,np.array([(E[t]-E[0])/g for t in th]))
                row.update(Cangle=c,Cangle_cubic=None,r=d['Delta_pm_bulk']/(2*c),angles=len(th))
            data['O3'].append(row)
        data['O4'].append(o4_dmrg_scan('dmrg_o4wf_diag_N9_chi800_b8.json','hq_o4wf_diag_N9_chi800.json'))
        for n in (10,11):
            d=read(f'twist/theta_junction_results/wall_o4wf_diag_N{n}_chi900.json')
            data['O4'].append(dict(N=n,tilt=d['tilt'],wall=d['Delta_pi']))
        windows={'O2':[6,10,11], 'spin1':[6,8], 'O3':[5,9,11], 'O4':[6,7]}
        fits={}
        for model,rows in data.items():
            fits[model]=[]
            for first in windows[model]:
                ds=[r for r in rows if r['N']>=first]
                n=np.array([r['N'] for r in ds],float)
                y=np.array([r['wall'] for r in ds]);t=np.array([r['tilt']-1 for r in ds])
                for name,x in [('1/N',1/n),('tilt_excess',t)]:
                    for degree in ([1,2] if len(ds)>=4 else [1]):
                        p=np.polyfit(x,y,degree)
                        fits[model].append(dict(first=first,last=int(n[-1]),variable=name,
                            degree=degree,coefficients=p.tolist(),limit=float(p[-1]),
                            rms=float(np.sqrt(np.mean((np.polyval(p,x)-y)**2)))))
        # An additional fast-correction sensitivity for the O(4) sequence, over all sizes and the last three.
        o4_fast=[]
        for first in (6,8):
            ds=[r for r in data['O4'] if r['N']>=first]
            n=np.array([r['N'] for r in ds],float)
            p=np.polyfit(n**-2,[r['wall'] for r in ds],1)
            o4_fast.append(dict(variable='1/N^2',first=first,last=int(n[-1]),limit=float(p[-1]),coefficients=p.tolist()))
        hq={}
        for model,cfg in [('O2','o2wf'),('O3','o3wf'),('O4','o4wf_diag')]:
            hq[model]=[dict(N=n,HQ=read(f'twist/tilt_test_results/{cfg}_N{n}.json')['C_t_HQ'])
                       for n in (range(6,11) if model=='O2' else range(6,9))]
        for n,c in [(10,800),(12,1000)]:
            d=read(f'twist/ct_ward_results/hq_o3wf_N{n}_chi{c}.json')
            hq['O3'].append(dict(N=n,HQ=d['C_t_HQ']))
        for n,c in [(9,800),(10,900),(11,900)]:
            d=read(f'twist/ct_ward_results/hq_o4wf_diag_N{n}_chi{c}.json')
            hq['O4'].append(dict(N=n,HQ=d['C_t_HQ']))
        for n in (12,14):
            d=read(f'twist/ct_ward_results/hq_o2wf_N{n}_chi1000.json')
            hq['O2'].append(dict(N=n,HQ=d['C_t_HQ']))
        hq['spin1']=[dict(N=d['N'],HQ=d['HQ']) for d in data['spin1']]
        hq_fits={}
        for model,rows in hq.items():
            n=np.array([r['N'] for r in rows],float); y=np.array([r['HQ'] for r in rows])
            hq_fits[model]=dict(power=power_fit(n,y),linear_N2=np.polyfit(n**-2,y,1).tolist(),
                                linear_N3=np.polyfit(n**-3,y,1).tolist())
        shape={}
        for model,rows in data.items():
            rows=[r for r in rows if 'r' in r]
            if model=='O2':rows=rows[-4:]
            t=np.array([r['tilt']-1 for r in rows]); y=[r['r'] for r in rows]
            shape[model]={'linear':np.polyfit(t,y,1).tolist(), 'sizes':[r['N'] for r in rows]}
            if model=='spin1':shape[model]['quadratic']=np.polyfit(t,y,2).tolist()
        # The forecast is fitted to the exact-diagonalization scans at N_m = 5-8 only, so later sizes stay out of sample.
        p=power_fit([r['N'] for r in ed3],[r['Cangle'] for r in ed3])
        q=np.polyfit([r['tilt']-1 for r in ed3],[r['r'] for r in ed3],1).tolist(); prediction=[]
        for r in data['O3'][4:]:
            pred=2*(p[0]+p[1]*r['N']**(-p[2]))*np.polyval(q,r['tilt']-1)
            prediction.append(dict(N=r['N'],measured_tilt=r['tilt'],predicted=float(pred),
                                   direct=r['wall'],residual=float(r['wall']-pred)))
        coupling=[]
        for n in [6,7,8]:
            d=read(f'flow/dflow_c_ope_N{n}.json');rows=list(d['flow'].values())
            coupling.append(dict(N=n,uv=min(rows,key=lambda x:abs(x['g']))['b'],
                ir=min(rows,key=lambda x:abs(x['g']+.09))['b'],o2=d['o2']['b']))
        d=read('flow/dflow_c_endpoint.json')
        coupling.append(dict(N=9,uv=d['o3_N9_g-0.000']['b'],ir=d['o3_N9_g-0.090']['b'],o2=d['o2_N9']['b']))
        cfits={}
        for power in [1.,.5]:
            n=np.array([r['N'] for r in coupling[1:]],float)
            cfits[str(power)]={key:float(np.polyfit(n**(-power),[r[key] for r in coupling[1:]],1)[-1])
                               for key in ['uv','ir','o2']}
        ward_ranges={}
        for model,fit in hq_fits.items():
            cs=[fit['power'][0],fit['linear_N2'][-1],fit['linear_N3'][-1]]
            rs=[shape[model]['linear'][-1]]
            if model=='spin1':rs.append(shape[model]['quadratic'][-1])
            ward_ranges[model]=dict(stiffness=[min(cs),max(cs)],
                                   wall=[2*min(cs)*min(rs),2*max(cs)*max(rs)])
        # Correction-law test of the measured wall in the best-converged regulator (spin one), and a common-limit fit with
        # vector O(2) at N >= 8 (separate correction amplitudes per regulator).  Slow terms: the reflection-even neutral line
        # operator (Dh ~ 1.52, N^-0.26) and the bulk s' (N^-0.40).  Forced W = 1 shows what a marginal wall costs in residual.
        # The form with both slow terms is ill-conditioned: it records the slow amplitudes (they nearly cancel), the
        # column-normalized condition number, and the line-exponent range 0.20-0.35 allowed by the dimension of phi_1.
        def lsq(A,y):
            c=np.linalg.lstsq(A,y,rcond=None)[0]; return c,float(np.sqrt(np.mean((y-A@c)**2)))
        two_slow=lambda p:(lambda n:[n**-p,n**-0.40,1/n,1/n**2])
        forms={'slow0.26':lambda n:[n**-0.26,1/n,1/n**2],'slow0.40':lambda n:[n**-0.40,1/n,1/n**2],
               'fast3':lambda n:[1/n,1/n**2,1/n**3],'slow0.26+0.40':two_slow(0.26)}
        slow_exps={'slow0.26':[0.26],'slow0.40':[0.40],'fast3':[],'slow0.26+0.40':[0.26,0.40]}
        sp1=data['spin1']; n1=np.array([r['N'] for r in sp1],float); w1=np.array([r['wall'] for r in sp1])
        def vec_of(nmin):
            v=[r for r in data['O2'] if r['N']>=nmin]
            return np.array([r['N'] for r in v],float), np.array([r['wall'] for r in v])
        mid=(n1[1:]+n1[:-1])/2; inc=np.diff(w1)/np.diff(n1)
        law=dict(increment_exponents=[float(np.log(inc[i]/inc[i+1])/np.log(mid[i+1]/mid[i])) for i in range(len(inc)-1)])
        for name,f in forms.items():
            A=np.vstack([np.ones_like(n1)]+f(n1)).T; c,rms=lsq(A,w1); c1,rms1=lsq(A[:,1:],w1-1.0); k=A.shape[1]-1
            ex=slow_exps[name]
            law[name]=dict(W_inf=float(c[0]),rms=rms,rms_forced_1=rms1,
                           slow_term_at_14=float(sum(c[1+i]*14**-e for i,e in enumerate(ex))) if ex else None)
            if len(ex)==2:
                law[name].update(slow_amplitudes=[float(c[1]),float(c[2])],slow_amplitudes_forced_1=[float(c1[0]),float(c1[1])],
                                 condition=float(np.linalg.cond(A/np.linalg.norm(A,axis=0))))
                for pl in (0.20,0.35):
                    Ap=np.vstack([np.ones_like(n1)]+two_slow(pl)(n1)).T; cp,rp=lsq(Ap,w1); _,rp1=lsq(Ap[:,1:],w1-1.0)
                    law[name][f'line_exponent_{pl:.2f}']=dict(W_inf=float(cp[0]),rms=rp,rms_forced_1=rp1)
            for nmin in (8,6):
                nv,wv=vec_of(nmin)
                Aj=np.zeros((len(n1)+len(nv),1+2*k)); Aj[:,0]=1; Aj[:len(n1),1:1+k]=np.vstack(f(n1)).T; Aj[len(n1):,1+k:]=np.vstack(f(nv)).T
                yj=np.concatenate([w1,wv]); cj,rj=lsq(Aj,yj); cj1,rj1=lsq(Aj[:,1:],yj-1.0)
                law[name][f'joint_with_vector_N{nmin}']=dict(W_inf=float(cj[0]),rms=rj,rms_forced_1=rj1)
        out=dict(data=data,direct_fits=fits,o4_fast_correction=o4_fast,hq=hq,hq_fits=hq_fits,spin1_correction_law=law,
                 ward_ranges=ward_ranges,
                 shape_fits=shape,prediction=dict(curvature_fit=p,shape_fit=q,
                 limit=2*p[0]*q[-1],rows=prediction),couplings=coupling,coupling_fits=cfits,
                 interpretation='Unweighted fit sensitivity, not a confidence interval or bound on omitted corrections.')
        store.put('fun/paper_2026_10_24_physics/figure_sources/JUNCTION_ROUTE_AUDIT.json', out)
        pass   # was: the input manifest of sha256 sums of the frozen copies
        return out

    def main():
        r=analyze()
        for name,fits in r['direct_fits'].items():
            v=[f['limit'] for f in fits]
            print(name,'direct fit limits',min(v),max(v),'Ward power-fit stiffness',r['hq_fits'][name]['power'][0])
        print('O4 fast-correction limits',[(f['first'],f['last'],f['limit']) for f in r['o4_fast_correction']])
        print('O3 forecast residuals',[(r['N'],r['residual']) for r in r['prediction']['rows']])

    def cli(argv):
        with fuzzy.logged('fun/paper_2026_10_24_physics/figure_sources/junction_analysis.log'):
            main()
    return namespace(locals(), globals())


STEPS = {'theta_on': theta_on, 'theta_dey': theta_dey, 'dmrg_pin': dmrg_pin, 'dmrg_theta': dmrg_theta, 'dmrg_dey': dmrg_dey, 'dmrg_wall': dmrg_wall, 'o4_package': o4_package, 'junction_analysis': junction_analysis}
REPORT = [('junction_analysis',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
