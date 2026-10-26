"""Which correction law controls the extrapolation of the wall?

The lowest neutral level of a straight pinned line is the restriction phi_1 of the order parameter, at Delta ~ 1.5; it is
even under the reflection that exchanges the poles, so it corrects every level by a slow N_m^{-(Delta - 1)/2} term. The
steps identify it (neutral_parity), measure which observables it moves by adding the pin on the next orbital (the wall as
N^-0.31, the protected levels as N^-0.68), and fit the best-converged wall sequence, the spin-one regulator, with and
without the slow terms (paper, App. E, "The measured wall and its factorization").

usage: python wall_correction.py <step> [args]      (python wall_correction.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def neutral_parity():
    """Reflection parity of the neutral pinned-line spectrum, and which neutral operators the pin can generate (2026-10-25).

The straight pinned line (theta = 0) is invariant under P = R_y(pi): orbital m -> -m with phase (-1)^(j-m), which swaps the
poles and reverses the line, t -> -t.  A neutral line primary odd under P cannot appear in the effective line action
int dt g O(t), so only P-even neutral primaries control the finite-size corrections (O ~ N_m^{-(Dh-1)/2}).
For the lowest psi_2-even (neutral) states this computes the relative parity <n|P|n>/<0|P|0> and the weight of the pin
operator itself, |<n|V_pin|0>|^2/(E_n-E_0) in gamma_B units (V_pin = north + south pin terms, the operator conjugate to b);
descendants of P-odd primaries are P-even but are total derivatives.  Cross-check: for psi_2-odd states the same P must
reproduce the sign of a_N/a_S used in junction_spectral.py.
usage: python neutral_parity.py <cfg> N [b=8] [nlow=12]   -> theta_junction_results/npar_<cfg>_N{N}_b{b}.json
"""
    import os
    import sys
    import json
    import time
    import itertools
    from types import SimpleNamespace
    import numpy as np
    import scipy.sparse as sp
    HERE = 'twist'
    sys.path.insert(0, os.path.join(HERE, 'spectator_pinning'))
    M = fuzzy.onmodel()   # was: import onmodel as M
    S = fuzzy.question('02_rg_flow/easy_axis').scan()   # was: import scan as S
    _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
    _theta_on = fuzzy.question('04_pinned_lines/junctions').theta_on(); bulk = _theta_on.bulk   # was: from theta_on import bulk

    OUT = os.path.join(HERE, 'theta_junction_results')


    def reflection(states, N, nflav):
        """Sparse matrix of R_y(pi): c+_{a,i} -> (-1)^(N-1-i) c+_{a,N-1-i}, i = orbital index (m = i - j)."""
        new = np.zeros_like(states); phase = np.zeros(len(states), int); reorder = np.zeros(len(states), int)
        for a in range(nflav):
            k = np.zeros(len(states), int)
            for i in range(N):
                occ = (states >> np.int64(a * N + i)) & 1
                new |= occ << np.int64(a * N + N - 1 - i)
                phase += occ * (N - 1 - i); k += occ
            reorder += k * (k - 1) // 2
        dest = np.searchsorted(states, new)
        assert np.all(dest < len(states)) and np.array_equal(states[dest], new)
        sign = 1.0 - 2.0 * ((phase + reorder) % 2)
        return sp.csr_matrix((sign, (dest, np.arange(len(states)))), shape=(len(states), len(states)))


    def run(cfgname, N, b=8.0, nlow=12):
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
        Vpin = (north + s1).tocsr()
        H = (H0 - b * Vpin).tocsr()
        P = reflection(states, N, Nv + 1)
        assert abs(P @ Vpin - Vpin @ P).max() < 1e-12 and abs(P @ H - H @ P).max() < 1e-9
        f2 = np.zeros(len(states), int)
        for j in range(N):
            f2 += ((states >> np.int64(2 * N + j)) & 1).astype(int)
        odd = np.flatnonzero(f2 % 2 == 1); even = np.flatnonzero(f2 % 2 == 0)
        He, Ho = H[even][:, even].tocsr(), H[odd][:, odd].tocsr()
        Pe, Po = P[even][:, even].tocsr(), P[odd][:, odd].tocsr()
        we, ve = S.eigenpairs(He, k=nlow)[:2]
        wo, vo = S.eigenpairs(Ho, k=4)[:2]
        E0, g = we[0], ve[:, 0]
        p0 = g @ (Pe @ g)
        Ve = Vpin[even][:, even]
        neutral = []
        for k in range(1, len(we)):
            v = ve[:, k]; dE = we[k] - E0
            neutral.append(dict(gap=float(dE / u), P=float(v @ (Pe @ v) / p0), w_pin=float((v @ (Ve @ g)) ** 2 / (dE * u))))
        gfull = np.zeros(len(states)); gfull[even] = g
        aS, aN = (s2 @ gfull)[odd], (n2 @ gfull)[odd]
        check = [dict(gap=float((wo[k] - E0) / u), P=float(vo[:, k] @ (Po @ vo[:, k]) / p0),
                      sign_aN_over_aS=float(np.sign((vo[:, k] @ aN) * (vo[:, k] @ aS)))) for k in range(len(wo))]
        out = dict(cfg=cfgname, N=N, b=b, gamma_B=u, P0=float(p0), neutral=neutral, odd_check=check, seconds=time.time() - t0)
        store.put(os.path.join(OUT, f'npar_{cfgname}_N{N}_b{b:g}.json'), out)
        print(f'{cfgname} N={N}: <0|P|0> = {p0:+.6f}   [{out["seconds"]:.0f}s]', flush=True)
        for lv in neutral:
            print(f'    neutral gap {lv["gap"]:.4f}  P {lv["P"]:+.3f}  pin weight {lv["w_pin"]:.3e}')
        print('  odd-sector check (gap, P, sign a_N/a_S):', ' '.join(f'({c["gap"]:.3f},{c["P"]:+.2f},{c["sign_aN_over_aS"]:+.0f})'
                                                           for c in check), flush=True)
        return out

    def main():
        run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 8.0,
            int(sys.argv[4]) if len(sys.argv) > 4 else 12)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/neutral_parity_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def junction_firstorder():
    """First-order response of pinned-line levels to symmetric pin perturbations (2026-10-25).

H(theta) = H_O(N_v) - b [V_N + R_theta V_S] as in theta_on.py.  Perturbations, both L_z-conserving, reflection-even and
O(N-1)-symmetric about the local pin direction:
  V_b : the pin itself (derivative in b; checks against the pin-strength scan, dW/db = 0.0165/b^2 at b >> 1),
  V_2 : the same pin on the next orbital at each pole (m = +-(j-1)), rotated with the south pin.
For each, the first-order shift of a level relative to the theta = 0 ground state, in gamma_B units, is
    d_n = -( <n|V|n> - <0|V|0> ) / gamma_B          (H -> H - eps V)
for n = tilt (lowest psi_2-odd state at theta = 0), the lowest neutral excitation (Dh ~ 1.52), and the wall (ground state at
theta = pi).  If the leading irrelevant line operator O (Dh ~ 1.52) dominates a perturbation, every d_n scales as
N_m^{-(Dh-1)/2} ~ N_m^{-0.26} and d_wall/d_tilt = C_{wall wall O}/C_{t t O} is universal (independent of the perturbation).
usage: python junction_firstorder.py <cfg> N [b=8]   -> theta_junction_results/fo_<cfg>_N{N}_b{b}.json
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
    _theta_on = fuzzy.question('04_pinned_lines/junctions').theta_on(); bulk = _theta_on.bulk   # was: from theta_on import bulk

    OUT = os.path.join(HERE, 'theta_junction_results')


    def run(cfgname, N, b=8.0):
        t0 = time.time()
        cfg = CONFIGS[cfgname]; Nv = cfg['Nv']
        hc, u = bulk(cfgname, N)
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], Nv, N, 0.0, cfg['U'], hc, pairs=())
        pars = [p[:2] + (1,) * (Nv - 2) for p in itertools.product((1, -1), repeat=2)]
        secs = [M.Basis(fl, orbs, koff, N, [], 0, par) for par in pars]
        states = np.sort(np.concatenate([x.states for x in secs]))
        H0 = M.build_H(SimpleNamespace(states=states, dim=len(states), nmodes=secs[0].nmodes), two, mu).tocsr()
        north, s1 = S.hopping(states, N - 1, 2 * N - 1), S.hopping(states, 0, N)
        north2, s12 = S.hopping(states, N - 2, 2 * N - 2), S.hopping(states, 1, N + 1)
        f2 = np.zeros(len(states), int)
        for j in range(N):
            f2 += ((states >> np.int64(2 * N + j)) & 1).astype(int)
        odd = np.flatnonzero(f2 % 2 == 1); even = np.flatnonzero(f2 % 2 == 0)
        res = {}
        for sgn, tag in ((1.0, '0'), (-1.0, 'pi')):                     # south pin along +e1 (theta = 0) or -e1 (theta = pi)
            Vb = (north + sgn * s1).tocsr(); V2 = (north2 + sgn * s12).tocsr()
            H = (H0 - b * Vb).tocsr()
            for sec, idx, k in (('even', even, 3), ('odd', odd, 2)):
                Hs = H[idx][:, idx].tocsr()
                w, v = S.eigenpairs(Hs, k=k)[:2]
                Vbs, V2s = Vb[idx][:, idx], V2[idx][:, idx]
                res[(tag, sec)] = [(w[i], v[:, i] @ (Vbs @ v[:, i]), v[:, i] @ (V2s @ v[:, i])) for i in range(len(w))]
        E0, b0, c0 = res[('0', 'even')][0]
        lev = {'tilt': res[('0', 'odd')][0], 'neutral': res[('0', 'even')][1],
               'wall': min(res[('pi', 'even')][0], res[('pi', 'odd')][0])}
        out = dict(cfg=cfgname, N=N, b=b, gamma_B=u, levels={})
        for name, (E, vb, v2) in lev.items():
            out['levels'][name] = dict(Delta=float((E - E0) / u), d_b=float(-(vb - b0) / u), d_2=float(-(v2 - c0) / u))
        out['seconds'] = time.time() - t0
        store.put(os.path.join(OUT, f'fo_{cfgname}_N{N}_b{b:g}.json'), out)
        print(f'{cfgname} N={N}:  ' + '  '.join(f'{k}: D {x["Delta"]:.4f} d_b {x["d_b"]:+.3e} d_2 {x["d_2"]:+.4f}'
                                                for k, x in out['levels'].items()) + f'  [{out["seconds"]:.0f}s]', flush=True)
        return out

    def main():
        run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 8.0)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/junction_firstorder_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def junction_firstorder_angles():
    """usage: python junction_firstorder_angles.py <cfg> N   (2026-10-25)
First-order V_2 response of the junction energy Delta(theta) at several angles: is delta Delta(theta) proportional to
Delta(theta)?  If so the leading irrelevant line operator only rescales the stiffness and leaves the shape factor r fixed.
"""
    import os, sys, itertools
    from types import SimpleNamespace
    import numpy as np
    pass   # was: T = '<research tree>/twist'; sys.path.insert(0, T); ...; os.chdir(T)  (imports rewired below, logs go to the store)
    M = fuzzy.onmodel()   # was: import onmodel as M
    S = fuzzy.question('02_rg_flow/easy_axis').scan()   # was: import scan as S
    _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
    _theta_on = fuzzy.question('04_pinned_lines/junctions').theta_on(); bulk = _theta_on.bulk   # was: from theta_on import bulk

    def main():
        cfgname, N, b = sys.argv[1], int(sys.argv[2]), 8.0
        cfg = CONFIGS[cfgname]; Nv = cfg['Nv']; hc, u = bulk(cfgname, N)
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], Nv, N, 0.0, cfg['U'], hc, pairs=())
        pars = [p[:2] + (1,) * (Nv - 2) for p in itertools.product((1, -1), repeat=2)]
        secs = [M.Basis(fl, orbs, koff, N, [], 0, par) for par in pars]
        st = np.sort(np.concatenate([x.states for x in secs]))
        H0 = M.build_H(SimpleNamespace(states=st, dim=len(st), nmodes=secs[0].nmodes), two, mu).tocsr()
        north, s1, s2 = S.hopping(st, N - 1, 2 * N - 1), S.hopping(st, 0, N), S.hopping(st, 0, 2 * N)
        north2, t1, t2 = S.hopping(st, N - 2, 2 * N - 2), S.hopping(st, 1, N + 1), S.hopping(st, 1, 2 * N + 1)
        rows = []
        for th in (0.0, np.pi / 8, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi):
            c, s = np.cos(th), np.sin(th)
            H = (H0 - b * (north + c * s1 + s * s2)).tocsr(); V2 = (north2 + c * t1 + s * t2).tocsr()
            w, v = S.eigenpairs(H, k=1)[:2]
            rows.append((th, w[0], v[:, 0] @ (V2 @ v[:, 0])))
        E0, c0 = rows[0][1], rows[0][2]
        out = []
        for th, E, vv in rows[1:]:
            D = (E - E0) / u; dD = -(vv - c0) / u; x = 1 - np.cos(th)
            out.append(f'th/pi={th / np.pi:.3f}: D {D:.4f} dD {dD:+.4f} dD/D {dD / D:.3f}')
        print(f'{cfgname} N={N}: ' + ' | '.join(out), flush=True)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/junction_firstorder_angles_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def junction_firstorder_spin1():
    """usage: python junction_firstorder_spin1.py <cfg> N   (2026-10-25)
First-order V_2 / V_b responses of the transverse-spin-1 neutral levels (displacement candidate) at theta = 0.
"""
    import os, sys, json, itertools
    from types import SimpleNamespace
    import numpy as np
    pass   # was: T = '<research tree>/twist'; sys.path.insert(0, T); ...; os.chdir(T)  (imports rewired below, logs go to the store)
    M = fuzzy.onmodel()   # was: import onmodel as M
    S = fuzzy.question('02_rg_flow/easy_axis').scan()   # was: import scan as S
    _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
    _theta_on = fuzzy.question('04_pinned_lines/junctions').theta_on(); bulk = _theta_on.bulk   # was: from theta_on import bulk

    def main():
        cfgname, N, b = sys.argv[1], int(sys.argv[2]), 8.0
        cfg = CONFIGS[cfgname]; Nv = cfg['Nv']; hc, u = bulk(cfgname, N)
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], Nv, N, 0.0, cfg['U'], hc, pairs=())
        pars = [p[:2] + (1,) * (Nv - 2) for p in itertools.product((1, -1), repeat=2)]
        out = {}
        for K in (0, 1):
            secs = [M.Basis(fl, orbs, koff, N, [], K, par) for par in pars]
            st = np.sort(np.concatenate([x.states for x in secs]))
            H0 = M.build_H(SimpleNamespace(states=st, dim=len(st), nmodes=secs[0].nmodes), two, mu).tocsr()
            Vb = (S.hopping(st, N - 1, 2 * N - 1) + S.hopping(st, 0, N)).tocsr()
            V2 = (S.hopping(st, N - 2, 2 * N - 2) + S.hopping(st, 1, N + 1)).tocsr()
            f2 = np.zeros(len(st), int)
            for j in range(N):
                f2 += ((st >> np.int64(2 * N + j)) & 1).astype(int)
            ev = np.flatnonzero(f2 % 2 == 0)
            H = (H0 - b * Vb).tocsr()[ev][:, ev].tocsr()
            w, v = S.eigenpairs(H, k=1 if K == 0 else 4)[:2]
            out[K] = [(w[i], v[:, i] @ (Vb[ev][:, ev] @ v[:, i]), v[:, i] @ (V2[ev][:, ev] @ v[:, i])) for i in range(len(w))]
        E0, b0, c0 = out[0][0]
        print(f'{cfgname} N={N} s=1 neutral levels (Delta, d_b, d_2):  ' + '  '.join(
            f'({(E - E0) / u:.4f}, {-(vb - b0) / u:+.2e}, {-(v2 - c0) / u:+.4f})' for E, vb, v2 in out[1]), flush=True)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/junction_firstorder_spin1_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def slow_response():
    """Separating the two slow finite-size corrections of the pinned O(N) line by response (2026-10-25).

Two irrelevant couplings give the slowest corrections at the sizes reached: the leading neutral line operator (Dh ~ 1.52,
levels shift as N_m^{-0.26}) and the leading bulk singlet s' (Delta ~ 3.79, N_m^{-0.40}).  Over N_m <= 14 the two powers are
nearly degenerate in fits, so we measure how each observable responds to each coupling separately:
  bulk : U -> U + dU with h retuned to the critical surface (same Delta_phi condition as run_on_defect.py),
  line : H -> H - eps V_2, the same pin on the next orbital at both poles (V_2 rotated with the south pin).
Observables, in units of the bulk unit gamma_B = E(d phi) - E(phi) at the same (U, h):
  bulk protected : Delta_J (lowest q=0, K=1: the O(2) current, 2) and Delta_T (lowest q=0, K=2: T and dJ, 3),
  line           : tilt level, lowest neutral level, displacement (lowest psi_2-even K=1 level, 2), wall Delta(pi),
                   Ward stiffness C^HQ = |<t|[H,Q_12]|0>|^2 / 4 gamma_B^2, two-angle curvature C^ang (eta = pi/8, pi/4).
usage: python slow_response.py <cfg> N [b=8] [dU=0.01] [eps=0.002]   -> theta_junction_results/slow_<cfg>_N{N}.json
"""
    import os
    import sys
    import json
    import time
    import itertools
    from types import SimpleNamespace
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import brentq
    HERE = 'twist'
    sys.path.insert(0, os.path.join(HERE, 'spectator_pinning'))
    M = fuzzy.onmodel()   # was: import onmodel as M
    S = fuzzy.question('02_rg_flow/easy_axis').scan()   # was: import scan as S
    _run_on_defect = fuzzy.run_on_defect(); CONFIGS = _run_on_defect.CONFIGS   # was: from run_on_defect import CONFIGS
    _pin_flow = fuzzy.question('04_pinned_lines/monodromy_pinning').pin_flow(); op_matrix = _pin_flow.op_matrix   # was: from pin_flow import op_matrix

    OUT = os.path.join(HERE, 'theta_junction_results')


    def tune(cfg, N, U):
        """h_c(U) and the bulk levels in the pair-charge basis, exactly as run_on_defect.py tunes."""
        kw = dict(pairs=cfg['pairs'], twist=cfg['twist'])
        fl, orbs, koff, two, mu0 = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, U, 0.0, **kw)
        mu1 = M.build_terms(cfg['model'], cfg['Nv'], N, 0.0, U, 1.0, **kw)[4]
        dmu = mu1 - mu0
        npair, npar = len(cfg['pairs']), cfg['Nv'] - 2 * len(cfg['pairs'])
        z, p, q1 = (0,) * npair, (+1,) * npar, (1,) + (0,) * (npair - 1)
        secs = {'vac': (z, 0), 'phi': (q1, 0), 'dphi': (q1, 1), 'J': (z, 1), 'T': (z, 2)}
        B = {k: M.Basis(fl, orbs, koff, N, list(q), K, p) for k, (q, K) in secs.items()}
        H0 = {k: M.build_H(B[k], two, mu0) for k in secs}
        nh = {k: sum(dmu[i] * ((B[k].states >> np.int64(i)) & 1) for i in range(len(dmu)) if dmu[i] != 0) for k in secs}
        E = lambda h, k, n=1: M.lowest(H0[k] + sp.diags(h * nh[k]), n)
        tgt = cfg['dphi'] / (1 + cfg['dphi'])
        f = lambda h: (E(h, 'phi')[0] - E(h, 'vac')[0]) / (E(h, 'dphi')[0] - E(h, 'vac')[0]) - tgt
        hs = np.linspace(-0.2, 0.0, 21); vals = [f(h) for h in hs]
        br = [(hs[i], hs[i + 1]) for i in range(20) if vals[i] * vals[i + 1] < 0]
        hc = brentq(f, *br[-1], xtol=1e-10)
        ev, ep, ed = E(hc, 'vac')[0], E(hc, 'phi')[0], E(hc, 'dphi')[0]
        u = ed - ep
        return hc, u, dict(J=float((E(hc, 'J')[0] - ev) / u), T=float((E(hc, 'T')[0] - ev) / u))


    def line(cfg, N, U, hc, u, b, eps):
        Nv = cfg['Nv']
        fl, orbs, koff, two, mu = M.build_terms(cfg['model'], Nv, N, 0.0, U, hc, pairs=())
        pars = [p[:2] + (1,) * (Nv - 2) for p in itertools.product((1, -1), repeat=2)]
        out = {}
        for K in (0, 1):
            secs = [M.Basis(fl, orbs, koff, N, [], K, par) for par in pars]
            st = np.sort(np.concatenate([x.states for x in secs]))
            H0 = M.build_H(SimpleNamespace(states=st, dim=len(st), nmodes=secs[0].nmodes), two, mu).tocsr()
            north, s1, s2 = S.hopping(st, N - 1, 2 * N - 1), S.hopping(st, 0, N), S.hopping(st, 0, 2 * N)
            north2, t1, t2 = S.hopping(st, N - 2, 2 * N - 2), S.hopping(st, 1, N + 1), S.hopping(st, 1, 2 * N + 1)
            f2 = np.zeros(len(st), int)
            for j in range(N):
                f2 += ((st >> np.int64(2 * N + j)) & 1).astype(int)
            ev, od = np.flatnonzero(f2 % 2 == 0), np.flatnonzero(f2 % 2 == 1)
            Hth = lambda c, s: (H0 - b * (north + c * s1 + s * s2) - eps * (north2 + c * t1 + s * t2)).tocsr()
            if K == 1:
                H = Hth(1.0, 0.0)
                out['D_abs'] = S.eigenpairs(H[ev][:, ev].tocsr(), k=1)[0][0]
                continue
            H = Hth(1.0, 0.0)
            we, ve = S.eigenpairs(H[ev][:, ev].tocsr(), k=2)[:2]
            wo, vo = S.eigenpairs(H[od][:, od].tocsr(), k=1)[:2]
            E0 = we[0]
            terms = [(N + j, 2 * N + j, 1.0) for j in range(N)] + [(2 * N + j, N + j, -1.0) for j in range(N)]
            A12 = op_matrix(st, st, terms)
            g = np.zeros(len(st)); g[ev] = ve[:, 0]
            amp = vo[:, 0] @ (A12 @ g)[od]
            out.update(E0=E0, tilt=(wo[0] - E0) / u, neutral=(we[1] - E0) / u,
                       CHQ=((wo[0] - E0) * amp) ** 2 / (4 * u ** 2))
            Hp = Hth(-1.0, 0.0)
            out['wall'] = (min(S.eigenpairs(Hp[ev][:, ev].tocsr(), k=1)[0][0],
                               S.eigenpairs(Hp[od][:, od].tocsr(), k=1)[0][0]) - E0) / u
            d = []
            for th in (np.pi / 8, np.pi / 4):
                d.append((S.eigenpairs(Hth(np.cos(th), np.sin(th)), k=1)[0][0] - E0) / u)
            x1, x2 = 1 - np.cos(np.pi / 8), 1 - np.cos(np.pi / 4)
            out['Cang'] = (d[0] * x2 ** 2 - d[1] * x1 ** 2) / (x1 * x2 ** 2 - x2 * x1 ** 2)
        out['D'] = (out.pop('D_abs') - out['E0']) / u
        out.pop('E0')
        return {k: float(v) for k, v in out.items()}


    def run(cfgname, N, b=8.0, dU=0.01, eps=0.002):
        t0 = time.time()
        cfg = CONFIGS[cfgname]; U0 = cfg['U']
        res = dict(cfg=cfgname, N=N, b=b, U0=U0, dU=dU, eps=eps, points={})
        for tag, U, e in (('base', U0, 0.0), ('U+', U0 + dU, 0.0), ('U-', U0 - dU, 0.0), ('e+', U0, eps), ('e-', U0, -eps)):
            hc, u, bulkl = tune(cfg, N, U)
            rec = dict(U=U, eps=e, h_c=hc, gamma_B=u, **bulkl)
            rec.update(line(cfg, N, U, hc, u, b, e))
            res['points'][tag] = rec
            print(f'{cfgname} N={N} {tag}: ' + ' '.join(f'{k} {v:.5f}' for k, v in rec.items() if k not in ('U', 'eps')),
                  f'[{time.time() - t0:.0f}s]', flush=True)
        P = res['points']
        keys = [k for k in P['base'] if k not in ('U', 'eps', 'h_c', 'gamma_B')]
        res['dU'] = {k: (P['U+'][k] - P['U-'][k]) / (2 * dU) for k in keys}
        res['deps'] = {k: (P['e+'][k] - P['e-'][k]) / (2 * eps) for k in keys}
        store.put(os.path.join(OUT, f'slow_{cfgname}_N{N}.json'), res)
        print('  d/dU  :', ' '.join(f'{k} {v:+.4f}' for k, v in res['dU'].items()))
        print('  d/deps:', ' '.join(f'{k} {v:+.4f}' for k, v in res['deps'].items()), flush=True)
        return res

    def main():
        a = sys.argv
        run(a[1], int(a[2]), float(a[3]) if len(a) > 3 else 8.0, float(a[4]) if len(a) > 4 else 0.01,
            float(a[5]) if len(a) > 5 else 0.002)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('twist/slow_response_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def dey_series():
    """Spin-one O(2) junction series: ED N_m = 6-10 (theta_dey.py) and DMRG N_m >= 11 (dmrg_dey.py), Phase B item 1.
Small-angle curvature (two smallest angles), Ward stiffness, wall, shape factor, the convergence exponent q of the wall
increments (increments ~ N^-q; a power-law correction converges only for q > 1), and the paper's fits.
usage: python dey_series.py
"""
    import json, glob, os
    import numpy as np
    from scipy.optimize import curve_fit
    HERE = 'twist'; R = os.path.join(HERE, 'theta_junction_results')


    def two_angle(th, d):
        x = 1 - np.cos(th)
        return (d[1] * x[2] ** 2 - d[2] * x[1] ** 2) / (x[1] * x[2] ** 2 - x[2] * x[1] ** 2)

    def main():
        rows = {}
        for n in range(6, 11):
            d = store.get(os.path.join(R, f'dey_N{n}_b1024.json'))
            th = np.array(sorted(float(k) for k in d['theta'])) * np.pi; E = {float(k) * np.pi: v for k, v in d['theta'].items()}
            prof = np.array([(E[t] - E[0.0]) / d['gamma'] for t in th])
            rows[n] = dict(src='ED', tilt=d['tilt'], C=two_angle(th, prof), HQ=(d['theta0_levels'][1] * d['theta0_Qamp'][1]) ** 2 / 4, wall=prof[-1])
        for f in sorted(store.glob(os.path.join(R, 'dmrg_dey_N*_chi*_b1024.json'))):
            d = store.get(f); T, g = d['theta'], d['gamma']
            if not all(k in T for k in ('0.0000', '0.1250', '0.2500', '1.0000')) or 'tilt' not in d:
                continue
            th = np.array([0, .125, .25]) * np.pi; prof = np.array([(T[k]['E'] - T['0.0000']['E']) / g for k in ('0.0000', '0.1250', '0.2500')])
            lev = (d['tilt']['E1'] - T['0.0000']['E']) / g
            rows[d['N']] = dict(src=f"DMRG chi {d['chi']}", tilt=lev, C=two_angle(th, prof), HQ=(lev * d['tilt']['Qamp']) ** 2 / 4,
                                wall=(T['1.0000']['E'] - T['0.0000']['E']) / g,
                                trunc=max(T[k]['trunc'] for k in T), trunc1=d['tilt']['trunc1'])
        Ns = np.array(sorted(rows), float)
        get = lambda k: np.array([rows[int(n)][k] for n in Ns])
        tilt, C, HQ, W = get('tilt'), get('C'), get('HQ'), get('wall'); r = W / (2 * C); x = tilt - 1
        print(' N   src           tilt     C(2-angle)  C^HQ     wall     r')
        for i, n in enumerate(Ns):
            print(f'{int(n):2d}  {rows[int(n)]["src"]:12s}  {tilt[i]:.5f}  {C[i]:.5f}    {HQ[i]:.5f}  {W[i]:.5f}  {r[i]:.4f}')
        inc = np.diff(W); mid = (Ns[1:] + Ns[:-1]) / 2
        print('wall increment exponents q:', ' '.join(f'{-np.log(inc[i + 1] / inc[i]) / np.log(mid[i + 1] / mid[i]):.2f}' for i in range(len(inc) - 1)))
        pw = lambda n, y: curve_fit(lambda n, a, b, c: a + b * n ** (-c), n, y, p0=[y[-1], 1.0, 2.0], maxfev=40000)[0]
        ch = [pw(Ns, HQ)[0], np.polyfit(Ns ** -2.0, HQ, 1)[-1], np.polyfit(Ns ** -3.0, HQ, 1)[-1]]
        print(f'C^HQ fits: power {ch[0]:.4f}  N^-2 {ch[1]:.4f}  N^-3 {ch[2]:.4f};  C (2-angle) power fit {pw(Ns, C)[0]:.4f}')
        wf = [np.polyfit(1 / Ns[k:], W[k:], dg)[-1] for k in (0, 1) for dg in (1, 2)] + [np.polyfit(x[k:], W[k:], dg)[-1] for k in (0, 1) for dg in (1, 2)]
        print('measured wall fits (1/N lin, quad; x_t lin, quad over all and from the 2nd size):', ' '.join(f'{v:.4f}' for v in wf))
        rf = [np.polyfit(x, r, 1)[-1], np.polyfit(x, r, 2)[-1], np.polyfit(x[-4:], r[-4:], 1)[-1]]
        print('shape factor fits (x_t lin all, quad all, lin last 4):', ' '.join(f'{v:.4f}' for v in rf))
        print(f'factorized 2 C^HQ r: {2 * min(ch) * min(rf):.4f} - {2 * max(ch) * max(rf):.4f}')

    def cli(argv):
        with fuzzy.logged('twist/dey_series.log'):
            main()
    return namespace(locals(), globals())


STEPS = {'neutral_parity': neutral_parity, 'junction_firstorder': junction_firstorder, 'junction_firstorder_angles': junction_firstorder_angles, 'junction_firstorder_spin1': junction_firstorder_spin1, 'slow_response': slow_response, 'dey_series': dey_series}
REPORT = [('dey_series',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
