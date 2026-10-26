"""What is the stiffness of a pinned line, and why does the small-angle curvature converge so slowly?

The tilt normalization C_tilt is measured with an exact broken-generator Ward identity, C^HQ = |<t|[H, Q_12]|0>|^2/4 gamma_B^2
(eq:tiltHQ), by exact diagonalization and DMRG. The small-angle curvature of the junction is the tilt propagator between
the two poles; the tower relation C^angle = C^HQ + w_0 (2 - Delta_tilt - Q) (eq:towerS) explains why it approaches C_tilt
only as the lattice tilt family becomes conformal, and a global cubic fit overstates it (paper, Sec. 5.2 and App. E,
"The tilt normalization").

The Ward runs take minutes (ED) to 7 hours (DMRG, O(4) N_m = 11); their outputs are stored with the junction scans.

usage: python ward_stiffness.py <step> [args]      (python ward_stiffness.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def tilt_test():
    """Is the lattice tilt state of the pinned O(N) line a clean conformal primary?  A test with the exact broken generator.

Q = Q_12 = sum_m i (c+_{m1} c_{m2} - c+_{m2} c_{m1}) commutes with the bulk Hamiltonian and not with the pin along e_1, so
[H, Q] is exactly the tilt-direction pin at the two poles.  In the continuum, i[H, Q] = -(t(N) + t(S)) and the pole
insertions create the tilt and its parallel descendants P^k |t> with amplitudes (1 + (-1)^k) sqrt((2 Delta_t)_k / k!).
With Delta_t = 1:
  |<k|[H,Q]|0_D>| = 2 sqrt(C_t) sqrt(k + 1)  (k even, zero for k odd),   |<k|Q|0_D>| = 2 sqrt(C_t) / sqrt(k + 1),
so the level-2 descendant carries 1/3 of the tilt's Q-weight, and |<t|Q|0_D>|^2 = 4 C_t gives C_t from the exact generator
and the lattice tilt state alone (unit-free).  The P2-odd, K = 0 spectrum (parities of flavours 3.. even) is resolved
level by level; levels are in bulk units relative to the pinned ground state.
usage: python tilt_test.py <cfg> N [b_pin=8] [nlev=8]   -> tilt_test_results/<cfg>_N{N}.json
"""
    import os
    import sys
    import json
    from types import SimpleNamespace
    import numpy as np
    import scipy.sparse as sp
    HERE = 'twist'
    sys.path.insert(0, os.path.join(HERE, 'spectator_pinning'))
    M = fuzzy.onmodel()   # was: import onmodel as M
    S = fuzzy.question('02_rg_flow/easy_axis').scan()   # was: import scan as S
    _fs_fockfast = fuzzy; popcount = _fs_fockfast.popcount   # was: from fs.fockfast import popcount
    _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
    _theta_on = fuzzy.question('04_pinned_lines/junctions').theta_on(); bulk = _theta_on.bulk   # was: from theta_on import bulk
    _pin_flow = fuzzy.question('04_pinned_lines/monodromy_pinning').pin_flow(); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix

    OUT = os.path.join(HERE, 'tilt_test_results')


    def run(cfgname, N, b_pin=8.0, nlev=8):
        pass   # was os.makedirs(...): the store needs no output directory
        cfg = CONFIGS[cfgname]
        Nv = cfg['Nv']
        hc, u = bulk(cfgname, N)
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], Nv, N, 0.0, cfg['U'], hc, pairs=())
        secs = [M.Basis(fl, orbs, koff, N, [], 0, (p1, p2) + (1,) * (Nv - 2)) for p1 in (1, -1) for p2 in (1, -1)]
        st = np.sort(np.concatenate([x.states for x in secs]))
        bas = SimpleNamespace(states=st, dim=len(st), nmodes=secs[0].nmodes)
        H = (M.build_H(bas, two, mu) - b_pin * (S.hopping(st, N - 1, 2 * N - 1) + S.hopping(st, 0, N))).tocsr()
        P2 = 1 - 2 * (popcount(st & np.int64(((1 << N) - 1) << (2 * N))) % 2)
        ev, od = np.flatnonzero(P2 > 0), np.flatnonzero(P2 < 0)
        w0, v0, _ = S.eigenpairs(H[ev][:, ev].tocsr(), k=1)
        w1, v1, _ = S.eigenpairs(H[od][:, od].tocsr(), k=nlev)
        E0 = w0[0]
        A = op_matrix(st, st, [(N + m, 2 * N + m, 1.0) for m in range(N)])          # sum_m c+_{m1} c_{m2}
        A = (A - A.T).tocsr()                                                         # Q = i A
        g0 = np.zeros(len(st)); g0[ev] = v0[:, 0]
        Ag = A @ g0
        amp = np.abs(v1.T @ Ag[od])                                                   # |<n|Q|0_D>|
        lev = (w1 - E0) / u
        hq = amp * (w1 - E0) / u                                                      # |<n|[H,Q]|0_D>| / gamma_B
        norm = float(np.sum(Ag[od] ** 2))                                             # ||Q|0_D>||^2 (all states)
        res = dict(cfg=cfgname, N=N, b_pin=b_pin, u=u, levels=lev.tolist(), Q_amp=amp.tolist(), HQ_amp=hq.tolist(),
                   Q_norm2=norm, C_t_Q=float(amp[0] ** 2 / 4), C_t_HQ=float(hq[0] ** 2 / 4))
        store.put(os.path.join(OUT, f'{cfgname}_N{N}.json'), res)
        print(f"{cfgname} N={N}: tilt {lev[0]:.4f}  C_t from |<t|Q|0>|^2/4 = {res['C_t_Q']:.4f}, from |<t|[H,Q]|0>|^2/4 (bulk units) = "
              f"{res['C_t_HQ']:.4f};  share of ||Q|0>||^2 on the tilt {amp[0] ** 2 / norm:.3f}", flush=True)
        for i in range(nlev):
            print(f"    level {lev[i]:.4f}: |<n|Q|0>|^2/|<t|Q|0>|^2 = {(amp[i] / amp[0]) ** 2:.4f}   |<n|[H,Q]|0>|/|<t|[H,Q]|0>| = {hq[i] / hq[0]:.4f}",
                  flush=True)
        return res

    def main():
        run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 8.0,
            int(sys.argv[4]) if len(sys.argv) > 4 else 8)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/tilt_test_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def dmrg_hq():
    """DMRG estimate of the tilt normalization from the exact broken generator (tilt_test.py beyond ED), with the density
Ward estimate (dmrg_ward_ct.py) from the same states.

Pinned ground state |0_D> and tilt |t> (first excited state; N_e and K conserved) of dmrg_theta.build at theta = 0.
With A_a = sum_m (c+_{m1} c_{ma} - c+_{ma} c_{m1}) (Q_1a = i A_a), the exact identity [H, Q_1a] = tilt pin at the poles gives
    C_t^HQ = (E_t - E_0)^2 sum_{a>=2} |<t|A_a|0_D>|^2 / (4 gamma_B^2),
primary-only and independent of how the O(N-1) tilt multiplet mixes; gamma_B = E(d phi) - E(phi) of the untwisted bulk at
the same h (charge-basis DMRG, computed here).  For o4wf_diag beyond the stored tunings h_c is held at its N = 7-8 value
(it is flat to 1e-5 over N = 6-8).
usage: python dmrg_hq.py <cfg> N chi [b=8]   -> ct_ward_results/hq_<cfg>_N{N}_chi{chi}.json
"""
    import os
    import sys
    import json
    import time
    import glob
    from math import pi
    import numpy as np
    HERE = 'twist'
    sys.path.insert(0, HERE)
    OUT = os.path.join(HERE, 'ct_ward_results')


    def h_value(cfgname, N):
        _dmrg_theta = fuzzy.question('04_pinned_lines/junctions').dmrg_theta(); h_of = _dmrg_theta.h_of   # was: from dmrg_theta import h_of
        if cfgname == 'o4wf_diag' and not store.exists(os.path.join(HERE, 'on_results', f'o4wf_diag_N{N}.json')):
            return float(np.mean([store.get(os.path.join(HERE, 'on_results', f'o4wf_diag_N{n}.json'))['h_c'] for n in (7, 8)]))
        return h_of(cfgname, N)[0]


    def generator_mpo(model, order, N, a):
        from tenpy.models.model import CouplingModel
        lat = model.lat
        pos = {k: i for i, k in enumerate(order)}
        cm = CouplingModel(lat)
        li = lambda p: list(lat.mps2lat_idx(p))
        for m in range(N):
            k1, ka = N + m, a * N + m
            cm.add_local_term(1.0, [('Cd', li(pos[k1])), ('C', li(pos[ka]))])
            cm.add_local_term(-1.0, [('Cd', li(pos[ka])), ('C', li(pos[k1]))])
        return cm.calc_H_MPO()


    def bulk_unit(cfgname, N, h, chi):
        _dmrg_jobs = __import__('dmrg'); on_problem = _dmrg_jobs.on_problem   # was: from dmrg_jobs import on_problem
        _dmrg_twist = __import__('dmrg'); build_model, run_dmrg = _dmrg_twist.build_model, _dmrg_twist.run_dmrg   # was: from dmrg_twist import build_model, run_dmrg
        _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
        cfg = CONFIGS[cfgname]
        npair, npar = len(cfg['pairs']), cfg['Nv'] - 2 * len(cfg['pairs'])
        E = {}
        for K in (0, 1):
            mode_m, charges, names, mods, terms, onsite, occ = on_problem(cfgname, N, 0.0, [[1] + [0] * (npair - 1), K, [1] * npar], h)
            model, order = build_model(mode_m, charges, names, terms, onsite, mods)
            E[K] = float(run_dmrg(model, order, occ, chi_max=chi, sweeps=30)[0])
        return E[1] - E[0]


    def run(cfgname, N, chi, b=8.0):
        _dmrg_theta = fuzzy.question('04_pinned_lines/junctions').dmrg_theta(); build = _dmrg_theta.build   # was: from dmrg_theta import build
        _dmrg_twist = __import__('dmrg'); run_dmrg = _dmrg_twist.run_dmrg   # was: from dmrg_twist import run_dmrg
        _dmrg_ward_ct = globals()['dmrg_ward_ct'](); density_mpo, matrix_element = _dmrg_ward_ct.density_mpo, _dmrg_ward_ct.matrix_element   # was: from dmrg_ward_ct import density_mpo, matrix_element
        _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
        pass   # was os.makedirs(...): the store needs no output directory
        t0 = time.time()
        h = h_value(cfgname, N)
        Nv = CONFIGS[cfgname]['Nv']
        model, order, occ = build(cfgname, N, h, 0.0, b)
        E0, psi0, eng0 = run_dmrg(model, order, occ, chi_max=chi, sweeps=30)
        E1, psi1, eng1 = run_dmrg(model, order, occ, chi_max=chi, sweeps=30, orthogonal_to=[psi0])
        gen = [abs(matrix_element(psi1, generator_mpo(model, order, N, a), psi0)) for a in range(2, Nv + 1)]
        one = matrix_element(psi0, density_mpo(model, order, N, 1), psi0).real
        dens = [abs(matrix_element(psi1, density_mpo(model, order, N, a), psi0)) for a in range(2, Nv + 1)]
        u = bulk_unit(cfgname, N, h, chi)
        hq = (E1 - E0) * float(np.sqrt(sum(x * x for x in gen)))
        res = dict(cfg=cfgname, N=N, chi=chi, b=b, h=h, E0=float(E0), E1=float(E1), u=u, tilt=(E1 - E0) / u,
                   gen_overlaps=gen, C_t_HQ=(hq / u) ** 2 / 4, one_point=one, density_overlaps=dens,
                   C_t_Ward=(one / (pi * float(np.sqrt(sum(x * x for x in dens))))) ** 2,
                   trunc=[float(eng0.sweep_stats['max_trunc_err'][-1]), float(eng1.sweep_stats['max_trunc_err'][-1])], seconds=time.time() - t0)
        store.put(os.path.join(OUT, f'hq_{cfgname}_N{N}_chi{chi}.json'), res)
        print(f"{cfgname} N={N} chi={chi}: tilt {res['tilt']:.4f}  C_t^HQ {res['C_t_HQ']:.5f}  C_t^Ward {res['C_t_Ward']:.5f}  "
              f"trunc {res['trunc'][0]:.0e}/{res['trunc'][1]:.0e}  [{res['seconds']:.0f}s]", flush=True)
        return res

    def main():
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 8.0)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/dmrg_hq_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def junction_pt2():
    """Exact small-angle curvature of the pinned-line junction and its decomposition over intermediate states (2026-10-25).

Same Hamiltonian as theta_on.py: H(theta) = H_O(Nv) - b [V_N + cos(theta) S_1 + sin(theta) S_2], with V_N the north pin and
S_a = c+_{S0} c_{Sa} + h.c. at the south pole.  At theta = 0, with |0> the pinned ground state,
    E''(0) = <0|V''|0> - 2 <0|V' (H - E_0)^{-1} V'|0>,   V' = -b S_2,  V'' = +b S_1,
so C_curv = E''(0)/gamma_B is the exact lattice curvature (Delta_j = C theta^2/2), free of the angular-grid fit behind C^ang.
V'|0> is odd under psi_2 -> -psi_2, so the resolvent is taken in the odd sector (positive definite there; conjugate gradients).
Decomposition: for the lowest odd states n, the single-pole weight w_n = |<n|V'|0>|^2/((E_n - E_0) gamma_B) (in the continuum
w = C_t for EVERY member of the tilt family, primary and descendants) and the Ward weight W_n = |<n|[H,Q]|0>|^2/(4 gamma_B^2),
Q = generator of rotations in the (1,2) plane (continuum: W = C_t (k+1) at even level k, 0 at odd k; C^HQ = W of the tilt).
Also: the lowest few states that are even under psi_2 -> -psi_2 (the neutral spectrum), whose lowest excitation sets the
leading irrelevant correction of the junction energies.
usage: python junction_pt2.py <cfg> N [b=8] [nlow=8]   -> theta_junction_results/pt2_<cfg>_N{N}_b{b}.json
"""
    import os
    import sys
    import json
    import time
    import itertools
    from types import SimpleNamespace
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    HERE = 'twist'
    sys.path.insert(0, os.path.join(HERE, 'spectator_pinning'))
    M = fuzzy.onmodel()   # was: import onmodel as M
    S = fuzzy.question('02_rg_flow/easy_axis').scan()   # was: import scan as S
    _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
    _theta_on = fuzzy.question('04_pinned_lines/junctions').theta_on(); bulk = _theta_on.bulk   # was: from theta_on import bulk
    _pin_flow = fuzzy.question('04_pinned_lines/monodromy_pinning').pin_flow(); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix

    OUT = os.path.join(HERE, 'theta_junction_results')


    def run(cfgname, N, b=8.0, nlow=8):
        t0 = time.time()
        cfg = CONFIGS[cfgname]; Nv = cfg['Nv']
        hc, u = bulk(cfgname, N)
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], Nv, N, 0.0, cfg['U'], hc, pairs=())
        pars = [p[:2] + (1,) * (Nv - 2) for p in itertools.product((1, -1), repeat=2)]      # flavours 3.. even
        secs = [M.Basis(fl, orbs, koff, N, [], 0, par) for par in pars]
        states = np.sort(np.concatenate([x.states for x in secs]))
        basis = SimpleNamespace(states=states, dim=len(states), nmodes=secs[0].nmodes)
        H0 = M.build_H(basis, two, mu).tocsr()
        north = S.hopping(states, N - 1, N + N - 1)
        s1, s2 = S.hopping(states, 0, N), S.hopping(states, 0, 2 * N)
        H = (H0 - b * (north + s1)).tocsr()
        # psi_2 parity of each basis state
        f2 = np.zeros(len(states), int)
        for j in range(N):
            f2 += ((states >> np.int64(2 * N + j)) & 1).astype(int)
        odd = np.flatnonzero(f2 % 2 == 1); even = np.flatnonzero(f2 % 2 == 0)
        He, Ho = H[even][:, even].tocsr(), H[odd][:, odd].tocsr()
        we, ve = S.eigenpairs(He, k=nlow)[:2]
        wo, vo = S.eigenpairs(Ho, k=nlow)[:2]
        E0 = we[0]; g = np.zeros(len(states)); g[even] = ve[:, 0]
        Vp = -b * s2; Vpp = b * s1
        vg = (Vp @ g)[odd]
        A = (Ho - E0 * sp.identity(len(odd))).tocsr()
        x, info = sla.cg(A, vg, rtol=1e-12, maxiter=20000)
        assert info == 0, info
        E2 = g @ (Vpp @ g) - 2 * vg @ x
        Ccurv = E2 / u
        # generator A12 = sum_m (c+_{m1} c_{m2} - c+_{m2} c_{m1}), Q = i A12, [H,Q]|0> -> i (E_n - E_0) <n|A12|0>
        terms = [(N + j, 2 * N + j, 1.0) for j in range(N)] + [(2 * N + j, N + j, -1.0) for j in range(N)]
        A12 = op_matrix(states, states, terms)
        ag = (A12 @ g)[odd]
        levels = []
        for k in range(len(wo)):
            dE = wo[k] - E0
            a1 = vo[:, k] @ vg
            levels.append(dict(gap=float(dE / u), w=float(a1 ** 2 / (dE * u)), W=float((dE * (vo[:, k] @ ag)) ** 2 / (4 * u ** 2))))
        out = dict(cfg=cfgname, N=N, b=b, gamma_B=u, C_curv=float(Ccurv), contact=float(g @ (Vpp @ g) / u),
                   odd=levels, even_gaps=[float((w - E0) / u) for w in we[1:]], seconds=time.time() - t0)
        store.put(os.path.join(OUT, f'pt2_{cfgname}_N{N}_b{b:g}.json'), out)
        print(f'{cfgname} N={N}: C_curv = {Ccurv:.5f}  (contact {out["contact"]:.4f}); odd states (gap, w, W):', flush=True)
        for lv in levels:
            print(f'    {lv["gap"]:.4f}  w {lv["w"]:.5f}  W {lv["W"]:.5f}')
        print('  lowest even (neutral) gaps:', ' '.join(f'{x:.4f}' for x in out['even_gaps']), f'  [{out["seconds"]:.0f}s]', flush=True)
        return out

    def main():
        run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 8.0,
            int(sys.argv[4]) if len(sys.argv) > 4 else 8)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/junction_pt2_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def junction_spectral():
    """Spectral side of the Ward vs angle stiffness gap: the junction curvature as a pole-to-pole tilt propagator (2026-10-25).

Same Hamiltonian as junction_pt2.py.  Rotating BOTH pins costs nothing (O(N) symmetry), and the reflection m -> N-1-m exchanges
them.  With G = (H - E_0)^{-1} on the psi_2-odd sector and V'_S = -b S_2, V'_N = -b N_2 (north pin rotated the same way),
    E''_glob = <V''_N + V''_S> - 2 <(V'_N+V'_S) G (V'_N+V'_S)> = 0   and   <V''_N> = <V''_S>, <V'_N G V'_N> = <V'_S G V'_S>
eliminate the contact term:
    E''(0) = <V''_S> - 2 <V'_S G V'_S> = 2 <0|V'_N G V'_S|0>,   i.e.   C_curv = 2 sum_n P_n w_n,
with w_n = |<n|V'_S|0>|^2 / ((E_n - E_0) gamma_B) and P_n = <n|V'_N|0>/<n|V'_S|0> = +-1 the reflection parity.  States localized
at one pole cancel in pairs, so the sum is carried by the delocalized line spectrum.  In the continuum every member of the tilt
family has w = C_t and P = (-1)^k, so the sum is 2 C_t (1 - 1 + 1 - ...) = C_t (Abel), while the Ward stiffness
C^HQ = W_0 = w_0 Delta_0 uses the primary alone.
usage: python junction_spectral.py <cfg> N [b=8] [nlow=30]   -> theta_junction_results/spec_<cfg>_N{N}_b{b}.json
"""
    import os
    import sys
    import json
    import time
    import itertools
    from types import SimpleNamespace
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    HERE = 'twist'
    sys.path.insert(0, os.path.join(HERE, 'spectator_pinning'))
    M = fuzzy.onmodel()   # was: import onmodel as M
    S = fuzzy.question('02_rg_flow/easy_axis').scan()   # was: import scan as S
    _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
    _theta_on = fuzzy.question('04_pinned_lines/junctions').theta_on(); bulk = _theta_on.bulk   # was: from theta_on import bulk

    OUT = os.path.join(HERE, 'theta_junction_results')


    def run(cfgname, N, b=8.0, nlow=30):
        t0 = time.time()
        cfg = CONFIGS[cfgname]; Nv = cfg['Nv']
        hc, u = bulk(cfgname, N)
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], Nv, N, 0.0, cfg['U'], hc, pairs=())
        pars = [p[:2] + (1,) * (Nv - 2) for p in itertools.product((1, -1), repeat=2)]
        secs = [M.Basis(fl, orbs, koff, N, [], 0, par) for par in pars]
        states = np.sort(np.concatenate([x.states for x in secs]))
        H0 = M.build_H(SimpleNamespace(states=states, dim=len(states), nmodes=secs[0].nmodes), two, mu).tocsr()
        north, n2 = S.hopping(states, N - 1, 2 * N - 1), S.hopping(states, N - 1, 3 * N - 1)
        s1, s2 = S.hopping(states, 0, N), S.hopping(states, 0, 2 * N)
        H = (H0 - b * (north + s1)).tocsr()
        f2 = np.zeros(len(states), int)
        for j in range(N):
            f2 += ((states >> np.int64(2 * N + j)) & 1).astype(int)
        odd = np.flatnonzero(f2 % 2 == 1); even = np.flatnonzero(f2 % 2 == 0)
        He, Ho = H[even][:, even].tocsr(), H[odd][:, odd].tocsr()
        we, ve = S.eigenpairs(He, k=1)[:2]
        E0 = we[0]; g = np.zeros(len(states)); g[even] = ve[:, 0]
        vS, vN = (-b * s2 @ g)[odd], (-b * n2 @ g)[odd]
        A = (Ho - E0 * sp.identity(len(odd))).tocsr()
        xS, info = sla.cg(A, vS, rtol=1e-12, maxiter=20000)
        assert info == 0, info
        C_contact = (g @ (b * s1 @ g) - 2 * vS @ xS) / u                  # junction_pt2's C_curv
        C_cross = 2 * vN @ xS / u                                         # pole-to-pole propagator
        wo, vo = S.eigenpairs(Ho, k=nlow)[:2]
        levels, partial = [], 0.0
        for k in range(len(wo)):
            dE = wo[k] - E0
            aS, aN = vo[:, k] @ vS, vo[:, k] @ vN
            cross = 2 * aS * aN / (dE * u)
            partial += cross
            levels.append(dict(gap=float(dE / u), w=float(aS ** 2 / (dE * u)), P=float(aN / aS) if abs(aS) > 1e-14 else 0.0,
                               cross=float(cross), partial=float(partial)))
        out = dict(cfg=cfgname, N=N, b=b, gamma_B=u, C_contact=float(C_contact), C_cross=float(C_cross),
                   C_HQ=float(levels[0]['w'] * levels[0]['gap']), levels=levels, seconds=time.time() - t0)
        store.put(os.path.join(OUT, f'spec_{cfgname}_N{N}_b{b:g}.json'), out)
        print(f'{cfgname} N={N}: C_curv contact form {C_contact:.6f}  pole-to-pole form {C_cross:.6f}  C^HQ {out["C_HQ"]:.5f}'
              f'  [{out["seconds"]:.0f}s]', flush=True)
        for lv in levels:
            if abs(lv['cross']) > 2e-4:
                print(f'    gap {lv["gap"]:.4f}  w {lv["w"]:.5f}  P {lv["P"]:+.3f}  2Pw {lv["cross"]:+.5f}  partial {lv["partial"]:.5f}')
        return out

    def main():
        run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 8.0,
            int(sys.argv[4]) if len(sys.argv) > 4 else 30)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/junction_spectral_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def ct_ward():
    """Tilt normalization C_t of the O(N) pinned line from the rotation Ward identity, independent of the angle scan.

Rotating the pinning direction by d(theta) in the (1, a) plane changes <phi_a(x)> by d(theta) a_phi / r^Delta_phi, and also
by d(theta) int dtau <phi_a(x) t_a(tau)> = d(theta) sqrt(C_t) b pi / r^Delta_phi, where b is the bulk-to-defect coefficient of
the unit-normalized phi_a into the unit-normalized tilt (Delta_t = 1).  Hence (Girault, Paulos, van Vliet 2509.26561, eq. 3.16)
    C_t = (a_phi / (pi b))^2.
On the fuzzy sphere, with the lattice density V_a(Omega) = psi_0^+ psi_a + h.c. and the pinned ground state |0_D> (north and
south pole along e_1, hopping b_pin, as theta_on.py at theta = 0) and the tilt |t> (lowest state odd under psi_2 -> -psi_2):
    a_phi(vartheta) = <0_D|V_1(vartheta)|0_D> sin(vartheta)^Delta_phi / c,   b(vartheta) = <t|V_2(vartheta)|0_D> sin(vartheta)^(Delta_phi-1) / c,
with c = <phi_1|V_1|0> in the bulk.  At the equator c and the radial factors cancel in C_t.  Only m = m' terms enter (K = 0).
Compared with Delta''(0) of the angle scan (theta_junction_results), which is in bulk units.
usage: python ct_ward.py <cfg> N [b_pin=8]   -> ct_ward_results/<cfg>_N{N}.json

Also supplies orbital_weight to dmrg_ward_ct.
"""
    import os
    import sys
    import json
    import itertools
    from math import lgamma, log, exp, pi, sin
    from types import SimpleNamespace
    import numpy as np
    HERE = 'twist'
    sys.path.insert(0, os.path.join(HERE, 'spectator_pinning'))
    M = fuzzy.onmodel()   # was: import onmodel as M
    S = fuzzy.question('02_rg_flow/easy_axis').scan()   # was: import scan as S
    _fs_fockfast = fuzzy; popcount = _fs_fockfast.popcount   # was: from fs.fockfast import popcount
    _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
    _theta_on = fuzzy.question('04_pinned_lines/junctions').theta_on(); bulk = _theta_on.bulk   # was: from theta_on import bulk

    OUT = os.path.join(HERE, 'ct_ward_results')
    DPHI = {'o2wf': 0.519088, 'o3wf': 0.518942, 'o4wf_diag': 0.51812}


    def orbital_weight(N, m, vt):
        j = (N - 1) / 2
        ln = lgamma(2 * j + 2) - log(4 * pi) - lgamma(j + m + 1) - lgamma(j - m + 1)
        return exp(ln) * np.cos(vt / 2) ** (2 * (j + m)) * np.sin(vt / 2) ** (2 * (j - m))


    def run(cfgname, N, b_pin=8.0, thetas=(0.3, 0.4, 0.5)):
        pass   # was os.makedirs(...): the store needs no output directory
        cfg = CONFIGS[cfgname]
        Nv = cfg['Nv']
        hc, u = bulk(cfgname, N)
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], Nv, N, 0.0, cfg['U'], hc, pairs=())
        # parities of flavours 3..Nv fixed even; flavours 1 and 2 summed (the pin breaks P1, P2 labels the tilt)
        secs = [M.Basis(fl, orbs, koff, N, [], 0, (p1, p2) + (1,) * (Nv - 2)) for p1 in (1, -1) for p2 in (1, -1)]
        st = np.sort(np.concatenate([x.states for x in secs]))
        bas = SimpleNamespace(states=st, dim=len(st), nmodes=secs[0].nmodes)
        H0 = M.build_H(bas, two, mu).tocsr()
        P2 = 1 - 2 * (popcount(st & np.int64(((1 << N) - 1) << (2 * N))) % 2)        # parity of flavour 2
        P1 = 1 - 2 * (popcount(st & np.int64(((1 << N) - 1) << N)) % 2)
        hop = lambda a, m: S.hopping(st, m, a * N + m)                              # psi_0 - psi_a at orbital m
        pin = hop(1, N - 1) + hop(1, 0)
        def lowest_by(Hm, mask):
            """Lowest state inside the parity block `mask` (diagonalized in the block, then embedded)."""
            idx = np.flatnonzero(mask)
            w, v, _ = S.eigenpairs(Hm[idx][:, idx].tocsr(), k=1)
            full = np.zeros(len(mask))
            full[idx] = v[:, 0]
            return w[0], full
        # bulk: vacuum (P1 = P2 = +) and phi_1 (P1 = -, P2 = +)
        E0, v0 = lowest_by(H0, (P1 > 0) & (P2 > 0))
        Ep, vp = lowest_by(H0, (P1 < 0) & (P2 > 0))
        # defect: ground (P2 = +) and tilt (P2 = -)
        HD = (H0 - b_pin * pin).tocsr()
        ED, vD = lowest_by(HD, P2 > 0)
        Et, vt_ = lowest_by(HD, P2 < 0)
        res = dict(cfg=cfgname, N=N, b_pin=b_pin, h_c=hc, u=u, tilt=(Et - ED) / u, phi=(Ep - E0) / u, theta={})
        for x in thetas:
            th = pi * x
            w = np.array([orbital_weight(N, m - (N - 1) / 2, th) for m in range(N)])
            V1 = sum(w[m] * hop(1, m) for m in range(N))
            V2 = sum(w[m] * hop(2, m) for m in range(N))
            c = float(vp @ (V1 @ v0))
            one = float(vD @ (V1 @ vD))
            tt = float(vt_ @ (V2 @ vD))
            s = sin(th)
            a_phi = one * s ** DPHI[cfgname] / c
            bcoef = tt * s ** (DPHI[cfgname] - 1) / c
            res['theta'][f'{x:.2f}'] = dict(c=c, one_point=one, tilt_overlap=tt, a_phi=abs(a_phi), b=abs(bcoef),
                                            C_t=(a_phi / (pi * bcoef)) ** 2)
        store.put(os.path.join(OUT, f'{cfgname}_N{N}.json'), res)
        eq = res['theta']['0.50']
        print(f"{cfgname} N={N}: tilt {res['tilt']:.4f}  a_phi {eq['a_phi']:.4f}  b {eq['b']:.4f}  C_t(Ward) {eq['C_t']:.4f}  "
              f"[vartheta/pi=0.3,0.4: C_t " + ' '.join(f"{res['theta'][k]['C_t']:.4f}" for k in ('0.30', '0.40')) + ']', flush=True)
        return res

    def main():
        run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 8.0)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/ct_ward_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def dmrg_ward_ct():
    """DMRG version of ct_ward.py: the Ward-identity tilt normalization C_t = [<0_D|V_1|0_D> / (pi |<t|V_perp|0_D>|)]^2 at the
equator, beyond ED.

Same Hamiltonian as dmrg_theta.py at theta = 0 (pins along e_1 at both poles, hopping b, N_e and K conserved).  The tilt
|t> is the first excited state (orthogonal to the pinned ground state in the same charge structure, so the lattice
densities V_a = sum_m w_m (c+_{m0} c_{ma} + h.c.) are charge-neutral MPOs).  For O(3) the tilt is a doublet; the
quadrature sum over a = 2, 3 makes the estimate independent of how the doublet mixes.
usage: python dmrg_ward_ct.py <cfg> N chi [b=8]   -> ct_ward_results/dmrg_<cfg>_N{N}_chi{chi}.json

Also supplies density_mpo and matrix_element to dmrg_hq.
"""
    import os
    import sys
    import json
    import time
    from math import pi
    import numpy as np
    HERE = 'twist'
    sys.path.insert(0, HERE)
    OUT = os.path.join(HERE, 'ct_ward_results')


    def density_mpo(model, order, N, a):
        from tenpy.models.model import CouplingModel, MPOModel
        _ct_ward = globals()['ct_ward'](); orbital_weight = _ct_ward.orbital_weight   # was: from ct_ward import orbital_weight
        lat = model.lat
        pos = {k: i for i, k in enumerate(order)}
        cm = CouplingModel(lat)
        li = lambda p: list(lat.mps2lat_idx(p))
        for m in range(N):
            w = orbital_weight(N, m - (N - 1) / 2, pi / 2)
            k0, ka = m, a * N + m
            cm.add_local_term(w, [('Cd', li(pos[k0])), ('C', li(pos[ka]))])
            cm.add_local_term(w, [('Cd', li(pos[ka])), ('C', li(pos[k0]))])
        return cm.calc_H_MPO()


    def matrix_element(bra, mpo, ket):
        from tenpy.networks.mpo import MPOEnvironment
        return complex(MPOEnvironment(bra, mpo, ket).full_contraction(0))


    def run(cfgname, N, chi, b=8.0):
        _dmrg_theta = fuzzy.question('04_pinned_lines/junctions').dmrg_theta(); build, h_of = _dmrg_theta.build, _dmrg_theta.h_of   # was: from dmrg_theta import build, h_of
        _dmrg_twist = __import__('dmrg'); run_dmrg = _dmrg_twist.run_dmrg   # was: from dmrg_twist import run_dmrg
        _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
        pass   # was os.makedirs(...): the store needs no output directory
        t0 = time.time()
        h, _ = h_of(cfgname, N)
        model, order, occ = build(cfgname, N, h, 0.0, b)
        E0, psi0, eng0 = run_dmrg(model, order, occ, chi_max=chi, sweeps=30)
        E1, psi1, eng1 = run_dmrg(model, order, occ, chi_max=chi, sweeps=30, orthogonal_to=[psi0])
        Nv = CONFIGS[cfgname]['Nv']
        one = matrix_element(psi0, density_mpo(model, order, N, 1), psi0).real
        tr = [abs(matrix_element(psi1, density_mpo(model, order, N, a), psi0)) for a in range(2, Nv + 1)]
        bt = float(np.sqrt(sum(x * x for x in tr)))
        res = dict(cfg=cfgname, N=N, chi=chi, b=b, h=h, E0=float(E0), E1=float(E1), one_point=one, tilt_overlaps=tr,
                   C_t=(one / (pi * bt)) ** 2, trunc=[float(eng0.sweep_stats['max_trunc_err'][-1]), float(eng1.sweep_stats['max_trunc_err'][-1])],
                   seconds=time.time() - t0)
        store.put(os.path.join(OUT, f'dmrg_{cfgname}_N{N}_chi{chi}.json'), res)
        print(f"{cfgname} N={N} chi={chi}: <V1> {one:.6f}  |<t|V_perp|0>| {bt:.6f}  C_t(Ward) {res['C_t']:.5f}  "
              f"E1-E0 {E1 - E0:.6f}  trunc {res['trunc'][0]:.0e}/{res['trunc'][1]:.0e}  [{res['seconds']:.0f}s]", flush=True)
        return res

    def main():
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 8.0)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/dmrg_ward_ct_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def curvature_bias():
    """Issue 4: is the Ward / angle stiffness gap real?  The paper's C^ang is the linear coefficient of a cubic polynomial in
x = 1 - cos(eta) fitted to the whole angular profile.  A two-angle estimator, D = C x + u2 x^2 through the two smallest
angles (pi/8, pi/4), reproduces the exact second-order curvature (CCS pt2, N_m <= 9) to < 1e-4, while the cubic fit
overstates it by a bias that grows with N_m.  Applied to every saved scan, it gives the curvature at the DMRG sizes too.
Inputs are the frozen junction scans of the paper (figure_sources/data) and the Ward values of JUNCTION_ROUTE_AUDIT.json.
usage: python curvature_bias.py   -> prints the comparison, writes curvature_bias.json
"""
    import json
    from pathlib import Path
    import numpy as np
    from scipy.optimize import curve_fit
    HERE = Path(__file__).resolve().parent
    FS = HERE.parent / 'fun' / 'paper_2026_10_24_physics' / 'figure_sources'
    DATA = FS / 'data' / 'twist' / 'theta_junction_results'
    AUDIT = store.get('fun/paper_2026_10_24_physics/figure_sources/JUNCTION_ROUTE_AUDIT.json')   # written by junctions.py

    SCANS = {'O2': {6: 'o2wf_N6_b8.json', 7: 'o2wf_N7_b8.json', 8: 'o2wf_N8_b8.json', 9: 'o2wf_N9_b8.json',
                    10: 'dmrg_o2wf_N10_chi700_b8.json', 11: 'dmrg_o2wf_N11_chi800_b8.json', 12: 'dmrg_o2wf_N12_chi1000_b8.json',
                    13: 'dmrg_o2wf_N13_chi800_b8.json', 14: 'dmrg_o2wf_N14_chi900_b8.json'},
             'O3': {5: 'N5_b8.json', 6: 'o3wf_N6_b8.json', 7: 'N7_b8.json', 8: 'N8_b8.json'},
             'O4': {6: 'o4wf_diag_N6_b8.json', 7: 'o4wf_diag_N7_b8_red.json', 8: 'o4wf_diag_N8_b8_red.json',
                    9: 'dmrg_o4wf_diag_N9_chi800_b8.json'},
             'spin1': {n: f'dey_N{n}_b1024.json' for n in range(6, 11)}}
    PT2 = {'O2': 'o2wf', 'O3': 'o3wf', 'O4': 'o4wf_diag'}


    def profile(model, name):
        r = store.get('twist/theta_junction_results/' + name)   # the frozen scans of the paper, as stored
        if model == 'spin1':
            th = np.array(sorted(float(k) for k in r['theta'])) * np.pi
            E = {float(k) * np.pi: v for k, v in r['theta'].items()}
            d = np.array([(E[t] - E[0.0]) / r['gamma'] for t in th])
            return th, d, r['tilt']
        th = np.array(sorted(v['theta'] for v in r['theta'].values()))
        E = {v['theta']: v['E'] for v in r['theta'].values()}
        if 'gamma_B' in r:
            g = r['gamma_B']; e0 = E[0][0]; d = np.array([(E[t][0] - e0) / g for t in th]); tilt = (E[0][1] - e0) / g
        elif 'unit_u' in r:
            d = np.array([(E[t] - E[0]) / r['unit_u'] for t in th]); tilt = None
        else:
            g = r['unit']['dphi']['E'] - r['unit']['phi']['E']; e0 = E[0]
            d = np.array([(E[t] - e0) / g for t in th]); tilt = (r['tilt']['E'] - e0) / g
        return th, d, tilt


    def estimators(th, d):
        x = 1 - np.cos(th)
        cubic = np.linalg.lstsq(np.vstack([x, x * x, x ** 3]).T, d, rcond=None)[0][0]
        x1, x2, d1, d2 = x[1], x[2], d[1], d[2]
        two = (d1 * x2 ** 2 - d2 * x1 ** 2) / (x1 * x2 ** 2 - x2 * x1 ** 2)
        return cubic, two, d[-1]


    def powfit(n, y):
        p = curve_fit(lambda n, a, b, c: a + b * n ** (-c), np.asarray(n, float), y, p0=[y[-1], 1.0, 2.0], maxfev=40000)[0]
        return p

    def main():
        out = {}
        for model, scans in SCANS.items():
            hq = {r['N']: r['HQ'] for r in AUDIT['hq'][model]}
            tilt_a = {r['N']: r['tilt'] for r in AUDIT['data'][model] if 'tilt' in r}
            rows = []
            for N, name in sorted(scans.items()):
                th, d, tilt = profile(model, name)
                cubic, two, wall = estimators(th, d)
                exact = None
                if model in PT2:
                    f = f'twist/theta_junction_results/pt2_{PT2[model]}_N{N}_b8.json'
                    if store.exists(f):
                        exact = store.get(f)['C_curv']
                rows.append(dict(N=N, cubic=cubic, two=two, exact=exact, HQ=hq.get(N), wall=wall,
                                 tilt=tilt if tilt is not None else tilt_a.get(N)))
            out[model] = rows
            print(f'{model}:  N   cubic    two-angle  exact(pt2)  HQ       r_cubic  r_two')
            for r in rows:
                ex = f"{r['exact']:.5f}" if r['exact'] else '   ---   '
                hqs = f"{r['HQ']:.5f}" if r['HQ'] else '  ---  '
                print(f"     {r['N']:2d}  {r['cubic']:.5f}  {r['two']:.5f}   {ex}  {hqs}  {r['wall'] / (2 * r['cubic']):.4f}  {r['wall'] / (2 * r['two']):.4f}")
            v = [r for r in rows if r['exact']]
            if v:
                print('   max |two-angle - exact| =', f"{max(abs(r['two'] - r['exact']) for r in v):.1e}",
                      ' cubic - exact:', ' '.join(f"{r['cubic'] - r['exact']:.4f}" for r in v))
            g = [(r['N'], r['two'] - r['HQ']) for r in rows if r['HQ']]
            loc = [np.log(g[i][1] / g[i + 1][1]) / np.log(g[i + 1][0] / g[i][0]) for i in range(len(g) - 1)]
            print('   gap (two-angle - HQ):', ' '.join(f'{n}:{x:.4f}' for n, x in g), '| local exponents', ' '.join(f'{x:.2f}' for x in loc))
            n = np.array([r['N'] for r in rows], float)
            if len(n) >= 4:
                p = powfit(n, np.array([r['two'] for r in rows]))
                print(f'   two-angle C continuation (free power, all sizes): {p[0]:.4f} (exponent {p[2]:.2f});  cubic: {powfit(n, np.array([r["cubic"] for r in rows]))[0]:.4f}')
            t = np.array([r['tilt'] - 1 for r in rows]); rt = np.array([r['wall'] / (2 * r['two']) for r in rows]); rc = np.array([r['wall'] / (2 * r['cubic']) for r in rows])
            k = 4 if model == 'O2' else len(rows)
            print(f'   shape factor, linear in tilt excess (last {k}): two-angle {np.polyfit(t[-k:], rt[-k:], 1)[-1]:.4f}   cubic {np.polyfit(t[-k:], rc[-k:], 1)[-1]:.4f}')
        store.put('twist/curvature_bias.json', out)

    def cli(argv):
        with fuzzy.logged('twist/curvature_bias.log'):
            main()
    return namespace(locals(), globals())


STEPS = {'tilt_test': tilt_test, 'dmrg_hq': dmrg_hq, 'junction_pt2': junction_pt2, 'junction_spectral': junction_spectral, 'ct_ward': ct_ward, 'dmrg_ward_ct': dmrg_ward_ct, 'curvature_bias': curvature_bias}
REPORT = [('curvature_bias',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
