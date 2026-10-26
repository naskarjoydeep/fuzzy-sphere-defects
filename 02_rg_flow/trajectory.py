"""Where is the critical surface along the anisotropy flows?

H(h, g) = H_WF(h) + g W with W = sum_m (n_m1 + n_m2 - 2 n_m3) in the O(3) model (and the O(4) analogue). At every
(N, g) the field h is retuned so that the light order parameter has Delta_phi = 0.518942 in units of its own
descendant gap. Results: the critical trajectory on both branches (O(3) -> O(2) for g < 0, O(3) -> Ising for g > 0),
the O(4) -> O(3) trajectory, and the controls of App. C.1 (thermal amplitudes, the size-crossing and stress-tensor
prescriptions, criteria that do not use Delta_phi, and the crossover functions).

usage: python trajectory.py <step> [args]      (python trajectory.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def partB_flow():
    """Part B: the retuned flow O(3) -> XY (g < 0, easy-plane) and O(3) -> Ising (g > 0, easy-axis).

At each (N, g), h is tuned so that the lighter order-parameter component (phi_3 for g > 0, phi_+- for
g < 0) has Delta = Delta_IR in units of its own d(phi) descendant gap u (Delta_IR = 0.51815 Ising,
0.51909 XY; O(3) 0.51894 at g = 0 -- the three agree to 0.2%, so one criterion follows the critical
surface along the whole flow).  Then the low spectrum of every sector is stored in units of u, with
l = 0 (l = 1) levels separated by comparison with the K = 1 (K = 2) sector.

usage: python partB_flow.py N g1,g2,...   (both signs are run from g = 0 outward, resumable)
"""
    HERE = 'flow'
    import sys
    import os
    import json
    import time
    import numpy as np
    F = fuzzy.o3flow()

    HC0 = {4: -0.09794238, 5: -0.09776263, 6: -0.09755398, 7: -0.09736940, 8: -0.09721605,
           9: -0.09709057}


    def l_split(lo, hi, tol=1e-6):
        """entries of lo not degenerate with any entry of hi (the extra, lower-spin states)."""
        return [e for e in lo if all(abs(e - f) > tol for f in hi)]


    def measure(fm, h, g, comp, k=5, light=False):
        E = {}
        ks = ((('S0', 6), ('S1', 5), ('S2', 2)) + tuple((nm, 3 if nm.endswith('0') else 4) for nm in
              ('P0', 'P1', 'V0', 'V1', 'A0', 'A1', 'B0', 'B1'))) if light else \
             (('S0', 8), ('S1', 7), ('S2', 4), ('P0', k), ('P1', k + 1), ('V0', k), ('V1', k + 1),
              ('A0', k), ('A1', k + 1), ('B0', k), ('B1', k + 1))
        for nm, kk in ks:
            E[nm] = fm.levels(nm, h, g, k=kk)
        e0 = E['S0'][0]
        u = E[comp + '1'][0] - E[comp + '0'][0]
        D = {nm: ((v - e0) / u).tolist() for nm, v in E.items()}
        r = dict(u=float(u), D=D)
        r['S_l0'] = l_split(D['S0'][1:], D['S1'])
        r['S_l1'] = l_split(D['S1'], D['S2'])
        r['P_l0'] = l_split(D['P0'], D['P1'])
        r['V_l0'] = l_split(D['V0'], D['V1'])
        r['A_l0'] = l_split(D['A0'], D['A1'])
        r['B_l0'] = l_split(D['B0'], D['B1'])
        r['T'] = D['S2'][0]
        return r


    def run(N, gs, kfile=None, const=False):
        path = kfile or os.path.join(HERE, f'partB_const_N{N}.json' if const else f'partB_N{N}.json')
        res = store.get(path) if store.exists(path) else {'N': N, 'points': {}}
        cache = tuple(F.SECTORS) if N <= 8 else ('S0', 'P0', 'P1', 'V0', 'V1')   # N=9 sectors are ~1 GB each
        fm = F.FlowModel(N, cache=cache)
        for sign in (+1, -1):
            h_prev = HC0[N]
            hist = []                                     # (g, h_c) along this branch, for extrapolation
            for g0 in [0.0] + sorted(gs):
                g = sign * g0
                key = f"{g:+.5f}"
                if key in res['points'] and res['points'][key].get('h_c') is not None:
                    h_prev = res['points'][key]['h_c']
                    hist.append((g, h_prev))
                    continue
                if len(hist) >= 2:                        # h_c(g) is smooth: extrapolate linearly in g
                    (ga, ha), (gb, hb) = hist[-2], hist[-1]
                    h_prev = hb + (hb - ha) / (gb - ga) * (g - gb)
                comp = 'V' if g0 == 0 else ('P' if g > 0 else 'V')
                target = F.TARGET['o3'] if (g0 == 0 or const) else (F.TARGET['ising'] if g > 0 else F.TARGET['xy'])
                t0 = time.time()
                hc, fail = None, None
                if g0 != 0 or N >= 9:
                    hc, nit = fm.tune_newton(g, h_prev, comp, target)
                if hc is None:
                    for span in (0.01, 0.03, 0.08, 0.2):
                        hc, _, fail = fm.tune(g, h_prev, comp=comp, target=target, span=span, n=7)
                        if hc is not None:
                            break
                if hc is None:
                    print(f"N={N} g={g:+.4f}: NO critical point found (last scan {np.round(fail[1], 4)})",
                          flush=True)
                    res['points'][key] = dict(g=g, h_c=None)
                    store.put(path, res)
                    break
                r = measure(fm, hc, g, comp, light=N >= 9)
                r.update(g=g, h_c=hc, comp=comp, dh=hc - HC0[N], time=time.time() - t0)
                res['points'][key] = r
                store.put(path, res)
                h_prev = hc
                hist.append((g, hc))
                heavy = r['V_l0'][0] if comp == 'P' else r['P_l0'][0]
                light = r['P_l0'][0] if comp == 'P' else r['V_l0'][0]
                print(f"N={N} g={g:+.4f}: h_c={hc:.6f} (dh {hc-HC0[N]:+.2e}) light {light:.4f} heavy {heavy:.4f} "
                      f"S_l0 {np.round(r['S_l0'][:2], 4)} B {np.round(r['B_l0'][:1], 4)} "
                      f"A {np.round(r['A_l0'][:1], 4)} J {np.round(r['S_l1'][:1], 4)} T {r['T']:.4f}  "
                      f"[{r['time']:.0f}s]", flush=True)
        return res


    def cli(argv):
        N = int(argv[1])
        gs = [float(x) for x in argv[2].split(',')]
        run(N, gs, const=len(argv) > 3 and argv[3] == 'const')
    return namespace(locals(), globals())


@section
def flow_run():
    """Driver for the flows of gflow.py (ED).

Stage A (g = 0): tune h so the order parameter has the UV dimension in units of its descendant gap, then store
the low spectrum of every sector with first-order data: energies, and the eigenvalues of W and n_h inside each
(near-)degenerate group, i.e. degenerate first-order perturbation theory.  From these: Wigner-Eckart ratios, the
raw response Phi = |<phi_l|W|phi_l>|/u, and kappa = dh/dg along the tuned line for each sign of g.
Stage B (scan): the tuned trajectory with the constant UV target (as partB_const), both signs as given by the
spec, h_c(g) extrapolated along each branch; spectra of every sector in units of the light descendant gap.

usage: python flow_run.py FLOW A N1 N2 ...          -> gf_FLOW_A.json (h_c per N, resumable)
       python flow_run.py FLOW B N g1,g2,... [lite]  -> gf_FLOW_B_N{N}.json (resumable)
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    G = fuzzy.gflow()

    HERE = 'flow'
    KA = {'S0': 6, 'S1': 4, 'S2': 3}


    def l_split(lo, hi, tol=1e-6):
        return [e for e in lo if all(abs(e - f) > tol for f in hi)]


    def groups(E, tol=1e-7):
        out, cur = [], [0]
        for i in range(1, len(E)):
            if E[i] - E[cur[-1]] < tol:
                cur.append(i)
            else:
                out.append(cur)
                cur = [i]
        out.append(cur)
        return out


    def stage_a(flow, Ns):
        path = os.path.join(HERE, f'gf_{flow}_A.json')
        res = store.get(path) if store.exists(path) else {'flow': flow, 'N': {}}
        spec = G.SPECS[flow]
        uv = G.DPHI[spec['uv']]
        for N in Ns:
            if str(N) in res['N']:
                continue
            t0 = time.time()
            big = N >= (10 if flow == 'o2' else 8)
            huge = N >= (11 if flow == 'o2' else 9)          # ~2M states per sector: build on the fly, minimal sectors
            fm = G.GFlow(flow, N, cache=() if huge else (('S0', 'L0', 'L1') if big else None), verbose=False)
            done = sorted(int(n) for n in res['N'])
            if len(done) >= 2:
                h1, h2 = res['N'][str(done[-2])]['h_c'], res['N'][str(done[-1])]['h_c']
                guess = h2 + (h2 - h1) / (done[-1] - done[-2]) * (N - done[-1])
            elif done:
                guess = res['N'][str(done[-1])]['h_c']
            else:
                guess = None
            hc = None
            if guess is not None:
                hc, _ = fm.tune_newton(0.0, guess, 'L', uv)
            if hc is None:
                for span in (0.02, 0.06):
                    hc, _ = fm.tune_scan(0.0, guess if guess is not None else -0.095, 'L', uv, span=span, n=13)
                    if hc is not None:
                        break
            if hc is None:
                print(f"{flow} N={N}: no critical point", flush=True)
                continue
            sec = {}
            for nm in (('S0', 'S2', 'L0', 'L1', 'H0', 'H1') if huge else spec['sectors']):
                k = KA.get(nm, 3)
                E, U = fm.levels(nm, hc, 0.0, k=k, vecs=True)
                s = fm.sector(nm)
                rows = []
                for gidx in groups(E):
                    Ug = U[:, gidx]
                    Wb = Ug.T @ (s['nw'][:, None] * Ug)
                    Hb = Ug.T @ (s['nh'][:, None] * Ug)
                    wev = np.linalg.eigvalsh(0.5 * (Wb + Wb.T))
                    rows.append(dict(E=float(np.mean(E[gidx])), deg=len(gidx), W=wev.tolist(),
                                     nh=float(np.trace(Hb) / len(gidx))))
                sec[nm] = rows
                if big and nm not in ('S0', 'L0', 'L1'):
                    fm.cache.pop(nm, None)
            e0 = sec['S0'][0]['E']
            u = sec['L1'][0]['E'] - sec['L0'][0]['E']
            out = dict(h_c=hc, u=u, e0=e0, sectors=sec, time=time.time() - t0)
            # kappa for each sign: tuning on the light component's (phi, dphi) pair
            for sgn in spec['signs']:
                c = spec['light'][sgn]
                v, p, d = sec['S0'][0], sec[c + '0'][0], sec[c + '1'][0]
                a, b = p['E'] - v['E'], d['E'] - v['E']
                wp, wd = (min(p['W']) if sgn > 0 else max(p['W'])), (min(d['W']) if sgn > 0 else max(d['W']))
                # the light state is the one pushed DOWN by g W: lowest W eigenvalue for g > 0, highest for g < 0
                dfg = ((wp - v['W'][0]) * b - a * (wd - v['W'][0])) / b ** 2
                dfh = ((p['nh'] - v['nh']) * b - a * (d['nh'] - v['nh'])) / b ** 2
                out[f'kappa{sgn:+d}'] = -dfg / dfh
                out[f'Phi{sgn:+d}'] = abs(wp) / u
            res['N'][str(N)] = out
            store.put(path, res)
            D = lambda nm, i: (sec[nm][i]['E'] - e0) / u
            print(f"{flow} N={N}: h_c {hc:.8f}  u {u:.5f}  S0 {[round(D('S0', i), 4) for i in range(1, len(sec['S0']))]}  "
                  f"T {D('S2', 0):.4f}  " + "  ".join(f"kappa{s:+d} {out[f'kappa{s:+d}']:+.4f} Phi{s:+d} {out[f'Phi{s:+d}']:.4f}"
                                                      for s in spec['signs']) + f"  [{out['time']:.0f}s]", flush=True)
        return res


    def measure(fm, h, g, lite):
        spec = fm.spec
        E = {}
        for nm in spec['sectors']:
            if nm == 'S0':
                k = 6 if lite else 8
            elif nm == 'S1':
                k = 5 if lite else 6
            elif nm == 'S2':
                k = 2 if lite else 4
            else:
                k = (3 if lite else 4) + (1 if nm.endswith('1') else 0)
            E[nm] = fm.levels(nm, h, g, k=k)
            if lite and nm not in fm.cache_names:
                fm.cache.pop(nm, None)
        e0 = E['S0'][0]
        lc = fm.light(g)
        u = E[lc + '1'][0] - E[lc + '0'][0]
        D = {nm: ((v - e0) / u).tolist() for nm, v in E.items()}
        r = dict(u=float(u), D=D, S_l0=l_split(D['S0'][1:], D['S1']), S_l1=l_split(D['S1'], D['S2']), T=D['S2'][0])
        for base in sorted({nm[:-1] for nm in spec['sectors'] if nm[0] != 'S'}):
            if base + '0' in D and base + '1' in D:
                r[base + '_l0'] = l_split(D[base + '0'], D[base + '1'])
        return r


    def stage_b(flow, N, gs, lite=False):
        A = store.get(os.path.join(HERE, f'gf_{flow}_A.json'))['N']
        hc0 = A[str(N)]['h_c']
        path = os.path.join(HERE, f'gf_{flow}_B_N{N}.json')
        res = store.get(path) if store.exists(path) else {'flow': flow, 'N': N, 'points': {}}
        spec = G.SPECS[flow]
        uv = G.DPHI[spec['uv']]
        cache = None if not lite else ('S0', 'L0', 'L1', 'H0', 'H1')
        fm = G.GFlow(flow, N, cache=cache, verbose=False)
        for sgn in spec['signs']:
            hist = [(0.0, hc0)]
            for g0 in [0.0] + sorted(gs):
                g = sgn * g0
                key = f"{g:+.5f}"
                if key in res['points'] and res['points'][key].get('h_c') is not None:
                    hist.append((g, res['points'][key]['h_c']))
                    continue
                if key in res['points']:
                    break                                    # a recorded failure ends this branch
                if len(hist) >= 2:
                    (ga, ha), (gb, hb) = hist[-2], hist[-1]
                    guess = hb + (hb - ha) / (gb - ga) * (g - gb) if gb != ga else hb
                else:
                    guess = hist[-1][1]
                t0 = time.time()
                comp = fm.light(g)
                if g0 == 0:
                    hc = hc0
                else:
                    hc, _ = fm.tune_newton(g, guess, comp, uv)
                    if hc is None:
                        for span in (0.005, 0.015, 0.04):
                            hc, _ = fm.tune_scan(g, guess, comp, uv, span=span, n=7)
                            if hc is not None:
                                break
                if hc is None:
                    print(f"{flow} N={N} g={g:+.5f}: no critical point", flush=True)
                    res['points'][key] = dict(g=g, h_c=None)
                    store.put(path, res)
                    break
                r = measure(fm, hc, g, lite)
                r.update(g=g, h_c=hc, comp=comp, time=time.time() - t0)
                res['points'][key] = r
                store.put(path, res)
                hist.append((g, hc))
                hv = fm.heavy(g)
                print(f"{flow} N={N} g={g:+.5f}: h_c {hc:.7f} light {r[comp + '_l0'][0]:.4f} heavy {r[hv + '_l0'][0]:.4f} "
                      f"S_l0 {np.round(r['S_l0'][:3], 4)} T {r['T']:.4f}  [{r['time']:.0f}s]", flush=True)
        return res


    def cli(argv):
        flow, stage = argv[1], argv[2]
        if stage == 'A':
            stage_a(flow, [int(x) for x in argv[3:]])
        else:
            N = int(argv[3])
            gs = [float(x) for x in argv[4].split(',')]
            stage_b(flow, N, gs, lite=len(argv) > 5 and argv[5] == 'lite')
    return namespace(locals(), globals())


@section
def traj_validate():
    """Task 1: is the tuned RG trajectory of the O(3) -> Ising / XY flow independent of the tuning prescription?

Prescriptions (all hold the bulk on the critical surface of the light order-parameter component):
  A    production (partB_flow.py const): the light ratio f = (E_phi - E_0)/(E_dphi - E_0) = D/(1 + D), D = 0.518942.
       Files partB_const_N*.json (and partB_N*.json, if present, are A with the IR targets 0.518149 / 0.519088).
  B    target-free size crossing: f_N(h, g) = f_{N+1}(h, g) for the light component (phenomenological
       renormalization). Never uses a bootstrap number. One h per size pair (N, N+1), spectra stored for size N.
  C    stress-tensor calibration: (E_phi - E_0)/(E_T - E_0) = D/3, with T the lowest S2 (Q = 0, K = 2, P = +) level.
Residual thermal detuning between prescriptions: at each (N, g),
       tau(P) = (h_P - h_A) * |<eps_IR|n_h|0>| / u ,
with eps_IR the lowest L = 0 singlet above the vacuum and u the light descendant gap, i.e. the change of the
dimensionless coupling of the IR thermal operator, the analogue of the scaling variable x for W.
usage: python traj_validate.py A N            -> traj_A_thermal_N{N}.json   (amplitudes along the production points)
       python traj_validate.py B N [auto|g1,g2,...]  -> traj_B_N{N}.json    (crossing of N and N+1; N+1 <= 9)
       python traj_validate.py C N [auto|g1,g2,...]  -> traj_C_N{N}.json
'auto' (default) takes the |g| grid of partB_const_N{N}.json, so B and C are evaluated at the same g as A.
B and C run both signs of g outward from g = 0, resumable, with points in the partB_const format plus 'thermal'.
"""
    import os
    import sys
    import json
    import time
    import numpy as np
    F = fuzzy.o3flow()
    PB = partB_flow()

    HERE = 'flow'
    D_PHI = F.TARGET['o3']


    def production(N):
        P = store.get(os.path.join(HERE, f'partB_const_N{N}.json'))['points']
        return {round(p['g'], 6): p for p in P.values() if p.get('h_c') is not None}


    def comp_of(g):
        return 'P' if g > 0 else 'V'


    def thermal(fm, h, g, comp):
        """|<eps_IR|n_h|0>|/u, the h-response of the light ratio, and the identification of eps_IR."""
        E, U = fm.levels('S0', h, g, k=4, vecs=True)
        E1 = fm.levels('S1', h, g, k=4)
        j = next((i for i in range(1, len(E)) if np.all(np.abs(E1 - E[i]) > 1e-6 * max(1, abs(E[i])))), None)
        if j is None:
            return dict(amp=float('nan'), eps_IR=float('nan'), u=float('nan'), f=float('nan'), dfdh=float('nan'))
        nh = fm.sector('S0')['nh']
        amp = float(U[:, j] @ (nh * U[:, 0]))
        e_light = fm.levels(comp + '0', h, g)[0]
        e_desc = fm.levels(comp + '1', h, g)[0]
        u = e_desc - e_light
        f, dfdh = fm.f_and_dfdh(h, g, comp, 0.0)
        return dict(amp=abs(amp) / u, eps_IR=float((E[j] - E[0]) / u), u=float(u), f=float(f), dfdh=float(dfdh))


    def tc_f_and_dfdh(fm, h, g, comp, tgt):
        """stress-tensor-calibrated tuning function (E_phi - E_0)/(E_T - E_0) - tgt and its Hellmann-Feynman h-derivative."""
        out = []
        for nm in ('S0', comp + '0', 'S2'):
            E, U = fm.levels(nm, h, g, k=1, vecs=True)
            out.append((E[0], float(np.abs(U[:, 0]) ** 2 @ fm.sector(nm)['nh'])))
        (e0, d0), (e1, d1), (e2, d2) = out
        a, b = e1 - e0, e2 - e0
        return a / b - tgt, ((d1 - d0) * b - a * (d2 - d0)) / b ** 2


    def newton(fun, h, tol=1e-10, maxit=12, max_step=0.02):
        for it in range(maxit):
            f, fp = fun(h)
            if not np.isfinite(fp) or fp == 0 or abs(f / fp) > max_step:
                return None
            h -= f / fp
            if abs(f) < tol:
                return h
        return None


    def bracket(fun, h0, spans=(0.005, 0.02, 0.06)):
        from scipy.optimize import brentq
        g = lambda h: fun(h)[0]
        for s in spans:
            hs = np.linspace(h0 - s, h0 + s, 9)
            v = [g(x) for x in hs]
            br = [(hs[i], hs[i + 1]) for i in range(8) if v[i] * v[i + 1] < 0]
            if br:
                i = min(range(len(br)), key=lambda k: abs(br[k][0] - h0))
                return brentq(g, *br[i], xtol=1e-10)
        return None


    def mode_A(N):
        fm = F.FlowModel(N, cache=('S0', 'S1', 'P0', 'P1', 'V0', 'V1'), verbose=False)
        out = dict(N=N, points={})
        for g, p in sorted(production(N).items()):
            t0 = time.time()
            out['points'][f'{g:+.5f}'] = dict(g=g, h_c=p['h_c'], **thermal(fm, p['h_c'], g, comp_of(g) if g else 'V'))
            print(f"A N={N} g={g:+.4f}: |<eps|n_h|0>|/u {out['points'][f'{g:+.5f}']['amp']:.4f}  [{time.time()-t0:.0f}s]",
                  flush=True)
            store.put(os.path.join(HERE, f'traj_A_thermal_N{N}.json'), out)


    def run_branchwise(N, gs, kind):
        path = os.path.join(HERE, f'traj_{kind}_N{N}.json')
        res = store.get(path) if store.exists(path) else dict(N=N, kind=kind, points={})
        A = production(N)
        cache = tuple(F.SECTORS) if N <= 8 else ('S0', 'S1', 'P0', 'P1', 'V0', 'V1') + (('S2',) if kind == 'C' else ())
        fm = F.FlowModel(N, cache=cache, verbose=False)
        fm2 = None
        if kind == 'B':
            if N + 1 > 9:
                sys.exit('B needs N+1 <= 9 (ED)')
            A2 = production(N + 1)
            fm2 = F.FlowModel(N + 1, cache=('S0', 'P0', 'P1', 'V0', 'V1'), verbose=False)
        for sign in (+1, -1):
            hist = []
            for g0 in [0.0] + sorted(gs):
                g = sign * g0
                key = f'{g:+.5f}'
                comp = comp_of(g) if g0 else 'V'
                if key in res['points'] and res['points'][key].get('h_c') is not None:
                    hist.append((g, res['points'][key]['h_c']))
                    continue
                gk = round(g, 6)
                if kind == 'B':
                    if gk not in A or gk not in A2:
                        continue
                    guess = 0.5 * (A[gk]['h_c'] + A2[gk]['h_c'])
                    fun = lambda h: tuple(np.subtract(fm.f_and_dfdh(h, g, comp, 0.0), fm2.f_and_dfdh(h, g, comp, 0.0)))
                else:
                    if gk not in A:
                        continue
                    guess = A[gk]['h_c']
                    fun = lambda h: tc_f_and_dfdh(fm, h, g, comp, D_PHI / 3)
                if len(hist) >= 2:
                    (ga, ha), (gb, hb) = hist[-2], hist[-1]
                    guess = hb + (hb - ha) / (gb - ga) * (g - gb) if abs(guess - hb) > 0.02 else guess
                t0 = time.time()
                hc = newton(fun, guess)
                if hc is None:
                    hc = bracket(fun, guess)
                if hc is None:
                    print(f'{kind} N={N} g={g:+.4f}: no solution near {guess:.5f}', flush=True)
                    res['points'][key] = dict(g=g, h_c=None)
                    store.put(path, res)
                    break
                r = PB.measure(fm, hc, g, comp, light=N >= 9)
                r.update(g=g, h_c=hc, comp=comp, h_A=A[gk]['h_c'], dh_vs_A=hc - A[gk]['h_c'],
                         thermal=thermal(fm, hc, g, comp), time=time.time() - t0)
                if kind == 'B':
                    r['f_N'] = fm.f_and_dfdh(hc, g, comp, 0.0)[0]
                    r['f_N1'] = fm2.f_and_dfdh(hc, g, comp, 0.0)[0]
                res['points'][key] = r
                store.put(path, res)
                hist.append((g, hc))
                print(f"{kind} N={N} g={g:+.4f}: h={hc:.7f} (A {A[gk]['h_c']:.7f}, dh {hc - A[gk]['h_c']:+.2e})  "
                      f"light {r['P_l0' if comp == 'P' else 'V_l0'][0]:.4f}  T {r['T']:.4f}  [{r['time']:.0f}s]", flush=True)


    def cli(argv):
        kind, N = argv[1], int(argv[2])
        if kind == 'A':
            mode_A(N)
        else:
            arg = argv[3] if len(argv) > 3 else 'auto'
            gs = sorted({abs(g) for g in production(N)} - {0.0}) if arg == 'auto' else [float(x) for x in arg.split(',')]
            run_branchwise(N, gs, kind)
    return namespace(locals(), globals())


@section
def imp_extract():
    """Crossover functions from a family of scans (production partB_const or improved imp_scan), same method as
crossover_extrap.py v2 except the x -> 0 anchor: Q(x) = (O(x) - O(0))/x is anchored by a linear fit to its three
smallest-x points instead of the exact Taylor coefficients (so it runs on any scan). sigma is the first-order
heavy - light slope per x_eff, taken from the smallest-x point (it is constant in N to 0.1%).
usage: python imp_extract.py prod|imp      -> imp_cf_{prod|imp}.json
"""
    import os
    import sys
    import glob
    import json
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    HERE = 'flow'
    YT, C, OM = 1.79, 0.89, 0.85
    WS, WINDOWS = (0.5, 0.85, 1.0), ((5, 9), (6, 9))
    fN = lambda N: N ** (YT / 2) * (1 + C * N ** (-OM))


    def load(kind):
        pat = 'partB_const_N*.json' if kind == 'prod' else 'imp_scan_N*.json'
        B = {}
        for f in sorted(store.glob(os.path.join(HERE, pat))):
            r = store.get(f)
            B[r['N']] = [p for p in r['points'].values() if p.get('h_c') is not None]
        return B


    def first(lst, i=0):
        return lst[i] if len(lst) > i else np.nan


    def branch(P, N, sign):
        Q = sorted([p for p in P if p['g'] == 0 or np.sign(p['g']) == sign], key=lambda p: abs(p['g']))
        seen, R = set(), []
        for p in Q:
            if abs(p['g']) in seen:
                continue
            seen.add(abs(p['g'])); R.append(p)
        light, heavy = ('P_l0', 'V_l0') if sign > 0 else ('V_l0', 'P_l0')
        g = np.array([abs(p['g']) for p in R])
        return dict(x=g * fN(N), heavy=np.array([first(p[heavy]) for p in R]), light=np.array([first(p[light]) for p in R]),
                    d0=np.array([first(p['S_l0']) for p in R]), s=np.array([first(p['S_l0'], 1) for p in R]))


    def at(d, obs, xe):
        O0 = d[obs][0]
        if xe == 0:
            return O0
        m = (d['x'] > 0) & np.isfinite(d[obs])
        x, v = d['x'][m], d[obs][m]
        if len(x) < 3 or xe > x.max():
            return np.nan
        Qv = (v - O0) / x
        c1, c0 = np.polyfit(x[:3], Qv[:3], 1)
        xa = x.min() * np.array([0.125, 0.25, 0.5])
        if xe <= xa[-1]:
            return float(O0 + xe * (c0 + c1 * xe))
        X = np.concatenate([xa, x]); Qa = np.concatenate([c0 + c1 * xa, Qv])
        return float(O0 + xe * PchipInterpolator(np.log(X), Qa)(np.log(xe)))


    def extrap(Ns, vals):
        out = []
        for w in WS:
            for lo, hi in WINDOWS:
                m = (Ns >= lo) & (Ns <= hi) & np.isfinite(vals)
                if m.sum() < 3:
                    continue
                A = np.vstack([np.ones(m.sum()), Ns[m] ** (-w)]).T
                out.append(np.linalg.lstsq(A, vals[m], rcond=None)[0][0])
        return (float(np.median(out)), float((max(out) - min(out)) / 2)) if out else (np.nan, np.nan)


    def main(kind):
        B = load(kind)
        Ns = np.array(sorted(B))
        res = {}
        for sign, lab in ((+1, 'Ising'), (-1, 'XY')):
            br = {N: branch(B[N], N, sign) for N in Ns}
            sigN = []
            for N in Ns:                                  # intercept of the linear anchor of (heavy - light)/x
                d = br[N]
                m = d['x'] > 0
                x, q = d['x'][m], (d['heavy'][m] - d['light'][m]) / d['x'][m]
                sigN.append(np.polyfit(x[:3], q[:3], 1)[1])
            sigN = np.array(sigN)
            sigma = extrap(Ns, sigN)[0]
            rows = []
            for xe in np.concatenate([[0.0], np.geomspace(0.002, 0.42, 26)]):
                row = dict(x_eff=float(xe), X=float(sigma * xe))
                for obs in ('heavy', 'd0', 's', 'light'):
                    vals = np.array([at(br[N], obs, xe) for N in Ns])
                    row[obs], row[obs + '_err'] = extrap(Ns, vals)
                    row[obs + '_N'] = vals.tolist()
                rows.append(row)
            res[lab] = dict(sigma=sigma, sigma_N=sigN.tolist(), Ns=Ns.tolist(), rows=rows)
            small = [r for r in rows if 0 < r['X'] <= 0.22 and np.isfinite(r['heavy'])]
            X = np.array([r['X'] for r in small]); A = np.vstack([X, X ** 2, X ** 3]).T
            fit = {}
            for obs in ('heavy', 'd0', 's'):
                y = np.array([r[obs] for r in small]) - (0.518942 if obs == 'heavy' else rows[0][obs])
                fit[obs] = np.linalg.lstsq(A, y, rcond=None)[0].tolist()
            res[lab]['smallX_fit'] = fit
            Xg = np.array([r['X'] for r in rows[1:]])
            res[lab]['at_X'] = {}
            for X0 in (0.1, 0.2, 0.3, 0.5):
                hv = np.interp(np.log(X0), np.log(Xg), [r['heavy'] for r in rows[1:]])
                dv = np.interp(np.log(X0), np.log(Xg), [r['d0'] for r in rows[1:]])
                res[lab]['at_X'][str(X0)] = dict(hl_over_X=(hv - 0.518942) / X0, d0=dv)
            print(f"   {lab} at X: " + "  ".join(f"X={k}: (h-l)/X {v['hl_over_X']:.4f} d0 {v['d0']:.4f}" for k, v in res[lab]['at_X'].items()))
            print(f"{kind} {lab}: sigma {sigma:.4f} (per N {np.round(sigN, 4)}); small-X b1, b2: " +
                  "  ".join(f"{o} {fit[o][0]:+.3f} {fit[o][1]:+.3f}" for o in fit), flush=True)
        store.put(os.path.join(HERE, f'imp_cf_{kind}.json'), res)


    def cli(argv):
        main(argv[1])
    return namespace(locals(), globals())


@section
def traj_analyze():
    """Analysis of task 1 (traj_validate.py): do the tuning prescriptions give the same continuum crossover?

For each prescription (A production, A_IR = partB_N*.json if present, B size crossing, C stress-tensor calibration)
the crossover functions of the heavy and light order-parameter levels and of the two lowest L = 0 singlets are
extracted exactly as in imp_extract.py (interpolation of (O(x) - O(0))/x at fixed x_eff, then O_inf + a N^-w over the
same exponents), with the size windows that each prescription has (B stops at N = 8).
Outputs, per branch (Ising g > 0, XY g < 0):
  * the extrapolated curves and their spread, and the differences P - A with combined spreads;
  * the residual thermal detuning tau(P; N, x_eff) = (h_P - h_A) |<eps_IR|n_h|0>|/u (traj_*_N*.json 'thermal'),
    and its local exponent in N at fixed x_eff: tau -> 0 with N means the prescriptions share the continuum trajectory.
usage: python traj_analyze.py   -> traj_summary.json, traj_report.txt, figs/traj_validate.png
"""
    import os
    import glob
    import json
    import numpy as np
    IE = imp_extract()

    HERE = 'flow'
    OBS = ('heavy', 'light', 'd0', 's')
    XEFF = np.concatenate([[0.0], np.geomspace(0.002, 0.42, 18)])


    def load(pattern):
        B = {}
        for f in sorted(store.glob(os.path.join(HERE, pattern))):
            r = store.get(f)
            B[r['N']] = [p for p in r['points'].values() if p.get('h_c') is not None]
        return B


    def curves(B):
        Ns = np.array(sorted(B))
        if len(Ns) < 3:
            return None
        hi = int(Ns.max())
        IE.WINDOWS = ((int(Ns.min()), hi), (int(Ns.min()) + 1, hi))
        out = {}
        for sign, lab in ((+1, 'Ising'), (-1, 'XY')):
            br = {N: IE.branch(B[N], N, sign) for N in Ns}
            rows = []
            for xe in XEFF:
                row = dict(x_eff=float(xe))
                for o in OBS:
                    vals = np.array([IE.at(br[N], o, xe) for N in Ns])
                    row[o], row[o + '_err'] = IE.extrap(Ns, vals)
                    row[o + '_N'] = vals.tolist()
                rows.append(row)
            out[lab] = dict(Ns=Ns.tolist(), rows=rows)
        return out


    def detuning(kind):
        T = {}
        for f in sorted(store.glob(os.path.join(HERE, f'traj_{kind}_N*.json'))):
            r = store.get(f)
            T[r['N']] = sorted([dict(g=p['g'], x=abs(p['g']) * IE.fN(r['N']), tau=p['dh_vs_A'] * p['thermal']['amp'],
                                     dh=p['dh_vs_A']) for p in r['points'].values() if p.get('h_c') is not None],
                               key=lambda q: q['g'])
        return T


    def main():
        sets = {'A': load('partB_const_N*.json'), 'A_IR': load('partB_N*.json'), 'B': load('traj_B_N*.json'),
                'C': load('traj_C_N*.json')}
        C = {k: curves(v) for k, v in sets.items() if v}
        C = {k: v for k, v in C.items() if v}
        L, summ = [], dict(curves=C, detuning={}, diffs={})
        for k in ('B', 'C'):
            T = detuning(k)
            summ['detuning'][k] = T
            for sign, lab in ((+1, 'Ising'), (-1, 'XY')):
                for x0 in (0.02, 0.05, 0.1, 0.2):
                    vals = []
                    for N, pts in sorted(T.items()):
                        q = [p for p in pts if np.sign(p['g']) == sign or p['g'] == 0]
                        if len(q) < 2:
                            continue
                        xs = np.array([p['x'] for p in q]); ts = np.array([p['tau'] for p in q])
                        o = np.argsort(xs)
                        vals.append((N, float(np.interp(x0, xs[o], ts[o]))))
                    if vals:
                        Ns = np.array([v[0] for v in vals]); tv = np.array([v[1] for v in vals])
                        loc = (np.log(np.abs(tv[1:]) / np.abs(tv[:-1])) / np.log(Ns[1:] / Ns[:-1])).round(2).tolist() \
                            if len(vals) > 1 and np.all(tv != 0) else []
                        L.append(f'{k} {lab} x_eff={x0}: tau(N) ' + ' '.join(f'{n}:{t:+.2e}' for n, t in vals) +
                                 f'   local exponents {loc}')
        if 'A' in C:
            for k in [q for q in C if q != 'A']:
                summ['diffs'][k] = {}
                for lab in ('Ising', 'XY'):
                    ra, rk = C['A'][lab]['rows'], C[k][lab]['rows']
                    d = [{o: (b[o] - a[o], float(np.hypot(a[o + '_err'], b[o + '_err']))) for o in OBS}
                         | dict(x_eff=a['x_eff']) for a, b in zip(ra, rk)]
                    summ['diffs'][k][lab] = d
                    for r in d[::4]:
                        L.append(f'{k}-A {lab} x_eff={r["x_eff"]:.3f}: ' +
                                 '  '.join(f'{o} {r[o][0]:+.4f}+-{r[o][1]:.4f}' for o in OBS))
        store.put(os.path.join(HERE, 'traj_summary.json'), summ)
        store.put(os.path.join(HERE, 'traj_report.txt'), '\n'.join(L) + '\n')
        print('\n'.join(L) if L else 'no B/C data yet')
        if len(C) > 1:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(14, 4))
            for j, lab in enumerate(('Ising', 'XY')):
                for k, st in zip(C, ('-', '--', ':', '-.')):
                    rows = C[k][lab]['rows'][1:]
                    x = [r['x_eff'] for r in rows]
                    for o, col in (('heavy', 'C0'), ('d0', 'C1'), ('s', 'C2')):
                        axs[j].plot(x, [r[o] for r in rows], st, color=col, label=f'{k} {o}' if j == 0 else None)
                axs[j].set_xscale('log'); axs[j].set_xlabel('x_eff'); axs[j].set_title(f'O(3) -> {lab}')
            axs[0].legend(fontsize=6, frameon=False, ncol=2)
            for k, st in (('B', 'o-'), ('C', 's--')):
                for N, pts in sorted(summ['detuning'].get(k, {}).items()):
                    axs[2].plot([p['g'] for p in pts], [p['tau'] for p in pts], st, ms=2, label=f'{k} N={N}')
            axs[2].axhline(0, color='0.5', lw=0.8); axs[2].set_xlabel('g'); axs[2].set_ylabel('tau (IR thermal units)')
            axs[2].legend(fontsize=6, frameon=False)
            fig.tight_layout(); fig.savefig(figpath('traj_validate.png'), dpi=150)


    def cli(argv):
        main()
    return namespace(locals(), globals())


@section
def retune_test():
    """Does tuning h on Delta_phi bias the y_t estimators?  ED test at N = 4..9, g = 0 (env: quantum or pytorch).

The production sequence tunes h_c(N) so that the light order-parameter ratio f = (E_phi - E_0)/(E_dphi - E_0)
equals Delta_phi/(1 + Delta_phi) with the bootstrap Delta_phi.  Here h is tuned instead on the O(3) singlet s,
the operator that h couples to, with two criteria that never use Delta_phi:
  A: Delta_s = (E_s - E_0)/(E_dphi - E_phi) = 1.59488            (unit: the phi_3 descendant spacing)
  B: (E_s - E_0)/(E_t - E_0) = 1.59488/1.20954                    (a ratio of two gaps, no unit at all)
s is the lowest S0 excited level that is degenerate neither with t (the Q = 2 component, sector B0) nor with an
L_z = 1 level (sector S1), i.e. the lowest L = 0 singlet.  Newton steps use Hellmann-Feynman dE/dh = <n_h>.
At each tuned h the same estimators as rg_eig.py are evaluated: cal = |df/dg|, raw = |<W>_phi - <W>_0| / u,
spec = (3 - Delta_t)/2 with u = E_dphi - E_phi.
usage: python retune_test.py N [N ...]
"""
    HERE = 'flow'
    import sys
    import json
    import numpy as np
    F = fuzzy.o3flow()

    DS, DT = 1.59488, 1.20954


    def states(fm, h):
        """energies and <W>, <n_h> of vac, s, t, phi3, dphi3 at (h, g = 0); also the S1 levels for the s check."""
        out = {}
        E, U = fm.levels('S0', h, 0.0, k=5, vecs=True)
        Et_, Ut = fm.levels('B0', h, 0.0, k=1, vecs=True)
        Et = Et_[0]
        E1 = fm.levels('S1', h, 0.0, k=4)
        s0 = fm.sector('S0')
        def obs(sec, u):
            p = np.abs(u) ** 2
            return float(p @ sec['nw']), float(p @ sec['nh'])
        out['vac'] = (E[0],) + obs(s0, U[:, 0])
        cand = [i for i in range(1, len(E)) if abs(E[i] - Et) > 1e-7 and np.min(np.abs(E1 - E[i])) > 1e-7]
        if not cand:
            raise RuntimeError('no L = 0 singlet found among the S0 levels')
        i = cand[0]
        out['s'] = (E[i],) + obs(s0, U[:, i])
        out['t'] = (Et,) + obs(fm.sector('B0'), Ut[:, 0])
        for nm, sec in (('phi3', 'P0'), ('dphi3', 'P1')):
            Ex, Ux = fm.levels(sec, h, 0.0, k=1, vecs=True)
            out[nm] = (Ex[0],) + obs(fm.sector(sec), Ux[:, 0])
        return out


    def criterion(st, which):
        """value and h-derivative of the tuning condition (Hellmann-Feynman)."""
        E = {k: v[0] for k, v in st.items()}
        D = {k: v[2] for k, v in st.items()}
        if which == 'A':
            num, den = E['s'] - E['vac'], E['dphi3'] - E['phi3']
            dnum, dden = D['s'] - D['vac'], D['dphi3'] - D['phi3']
            tgt = DS
        else:
            num, den = E['s'] - E['vac'], E['t'] - E['vac']
            dnum, dden = D['s'] - D['vac'], D['t'] - D['vac']
            tgt = DS / DT
        return num / den - tgt, (dnum * den - num * dden) / den ** 2


    def estimators(st):
        E = {k: v[0] for k, v in st.items()}
        W = {k: v[1] for k, v in st.items()}
        a, b = E['phi3'] - E['vac'], E['dphi3'] - E['vac']
        u = b - a
        dfdg = ((W['phi3'] - W['vac']) * b - a * (W['dphi3'] - W['vac'])) / b ** 2
        return dict(cal=abs(dfdg), raw=abs(W['phi3'] - W['vac']) / u, spec=(3 - (E['t'] - E['vac']) / u) / 2,
                    Delta_phi=a / u, Delta_s=(E['s'] - E['vac']) / u, Delta_t=(E['t'] - E['vac']) / u)


    def tune(fm, h0, which, tol=1e-11, maxit=12):
        h = h0
        for it in range(maxit):
            st = states(fm, h)
            f, fp = criterion(st, which)
            step = -f / fp
            h += step
            if abs(step) < tol:
                break
        return h, states(fm, h), it + 1


    def cli(argv):
        for N in map(int, argv[1:]):
            fm = F.FlowModel(N, cache=('S0', 'S1', 'P0', 'P1', 'B0'), verbose=False)
            h_phi = store.get(os.path.join(HERE, f'partA_N{N}.json'))['h_c']
            res = dict(N=N, h_phi=h_phi, phi=estimators(states(fm, h_phi)))
            for which in ('A', 'B'):
                h, st, its = tune(fm, h_phi, which)
                res[which] = dict(h=h, its=its, **estimators(st))
            store.put(os.path.join(HERE, f'retune_N{N}.json'), res)
            print(f"N={N}: h_phi {h_phi:.8f}  h_A {res['A']['h']:.8f}  h_B {res['B']['h']:.8f}  |  "
                  + "  ".join(f"{k}: cal {res[k]['cal']:.6f} raw {res[k]['raw']:.6f} spec {res[k]['spec']:.5f} "
                              f"Dphi {res[k]['Delta_phi']:.5f} Ds {res[k]['Delta_s']:.5f}" for k in ('phi', 'A', 'B')),
                  flush=True)
    return namespace(locals(), globals())


@section
def crossover_extrap():
    """Universal crossover functions of O(3) -> Ising (g > 0) and O(3) -> XY (g < 0) from the Delta_phi-tuned
scans partB_const_N{5..9}.json (env: quantum or pytorch).

Scaling variable: x_eff = |g| N^{y_t/2} (1 + c N^{-omega}), y_t = 1.79, with the correction (c, omega) =
(0.89, 0.85) measured from the g = 0 raw response (README 4b/6b).  At fixed x_eff each level is interpolated
in log x_eff for every N and extrapolated as O(N) = O_inf + a N^{-w}; the spread over w in {0.5, 0.85, 1.0}
and over the windows N = 5..9 and 6..9 is the quoted uncertainty.  Levels are (E - E_0)/u with u the
descendant spacing of the light component, so the light level is 0.518942 by construction of the tuning.
The universal variable is X = sigma x_eff, with sigma the N -> infinity slope of heavy - light at small x_eff,
so that heavy - light = X + O(X^2) on both sides.

v2 (2026-10-20): each level is interpolated through Q(x) = (O(x) - O(0))/x, which is smooth, instead of O(x)
in log x. Interpolating O itself in log x across the sparse N = 9 grid was log-convex and pulled the small-x
values down by about 3%, which the N -> infinity fit amplified (v1 gave sigma = 9.15 and heavy - light =
0.9 X). Q is anchored at x -> 0 by the exact Taylor coefficients of cpt_lattice.py / cpt_scheme.py, and
sigma is now the exact first-order slope |a1_hl(N)|/f_N extrapolated in N (8.74). v1 output is kept in
crossover_functions_v1.json.
usage: python crossover_extrap.py      -> crossover_functions.json (then fig_crossover.py)
"""
    import os
    import glob
    import json
    import numpy as np
    from scipy.interpolate import PchipInterpolator


    def cli(argv):

        HERE = 'flow'
        YT, C, OM = 1.79, 0.89, 0.85
        WS, WINDOWS = (0.5, 0.85, 1.0), ((5, 9), (6, 9))
        CFT = dict(t=1.20954, s=1.59488, phi=0.518942, eps=1.41263, sigma=0.518149, sxy=1.51136, phixy=0.519088)
        B = {}
        for f in sorted(store.glob(os.path.join(HERE, 'partB_const_N*.json'))):
            r = store.get(f)
            B[r['N']] = [p for p in r['points'].values() if p.get('h_c') is not None]


        def first(lst, i=0):
            return lst[i] if len(lst) > i else np.nan


        def branch(N, sign):
            P = sorted([p for p in B[N] if p['g'] == 0 or np.sign(p['g']) == sign], key=lambda p: abs(p['g']))
            seen, Q = set(), []
            for p in P:                                   # g = +0 and -0 both appear; keep one
                if abs(p['g']) in seen:
                    continue
                seen.add(abs(p['g'])); Q.append(p)
            light, heavy = ('P_l0', 'V_l0') if sign > 0 else ('V_l0', 'P_l0')
            g = np.array([abs(p['g']) for p in Q])
            return dict(x=g * N ** (YT / 2) * (1 + C * N ** (-OM)),
                        heavy=np.array([first(p[heavy]) for p in Q]), light=np.array([first(p[light]) for p in Q]),
                        d0=np.array([first(p['S_l0']) for p in Q]), s=np.array([first(p['S_l0'], 1) for p in Q]))


        SC = store.get(os.path.join(HERE, 'cpt_scheme.json'))
        TAY = {'d0': 't0', 's': 's'}


        def fN(N):
            return N ** (YT / 2) * (1 + C * N ** (-OM))


        def taylor(N, sign, obs):
            """exact (Q0, Q1) with (O - O0)/x_eff = Q0 + Q1 x_eff + ..., from the g = 0 lattice derivatives."""
            s = SC[str(N)]['Ising' if sign > 0 else 'XY']
            sig = abs(s['a1_hl']) / fN(N)
            if obs == 'light':
                return 0.0, 0.0
            key = TAY.get(obs, 'phipm' if sign > 0 else 'phi3')
            return s[key]['b1'] * sig, s[key]['b2'] * sig ** 2


        def at(N, sign, obs, xe):
            d = branch(N, sign)
            O0 = d[obs][0]
            if xe == 0:
                return O0
            m = (d['x'] > 0) & np.isfinite(d[obs])
            x, v = d['x'][m], d[obs][m]
            if xe > x.max():
                return np.nan
            q0, q1 = taylor(N, sign, obs)
            xa = x.min() * np.array([0.125, 0.25, 0.5])
            if xe <= xa[-1]:
                return float(O0 + xe * (q0 + q1 * xe))
            X = np.concatenate([xa, x])
            Q = np.concatenate([q0 + q1 * xa, (v - O0) / x])
            return float(O0 + xe * PchipInterpolator(np.log(X), Q)(np.log(xe)))


        def extrap(Ns, vals):
            out = []
            for w in WS:
                for lo, hi in WINDOWS:
                    m = (Ns >= lo) & (Ns <= hi) & np.isfinite(vals)
                    if m.sum() < 3:
                        continue
                    A = np.vstack([np.ones(m.sum()), Ns[m] ** (-w)]).T
                    coef, *_ = np.linalg.lstsq(A, vals[m], rcond=None)
                    out.append(coef[0])
            if not out:
                return np.nan, np.nan
            return float(np.median(out)), float((max(out) - min(out)) / 2)


        Ns = np.array(sorted(B))
        res = {}
        for sign, lab in ((+1, 'Ising'), (-1, 'XY')):
            xg = np.concatenate([[0.0], np.geomspace(0.002, 0.42, 26)])
            rows = []
            for xe in xg:
                row = dict(x_eff=float(xe))
                for obs in ('heavy', 'd0', 's', 'light'):
                    vals = np.array([at(N, sign, obs, xe) for N in Ns])
                    row[obs], row[obs + '_err'] = extrap(Ns, vals)
                    row[obs + '_N'] = vals.tolist()
                rows.append(row)
            # universal normalization: the exact first-order slope of heavy - light, extrapolated in N
            sigN = np.array([abs(SC[str(N)]['Ising' if sign > 0 else 'XY']['a1_hl']) / fN(N) for N in Ns])
            sigma, sig_err = extrap(Ns, sigN)
            for r in rows:
                r['X'] = sigma * r['x_eff']
            res[lab] = dict(sigma=sigma, sigma_err=sig_err, sigma_N=sigN.tolist(), rows=rows)
            print(f"\n{lab}: sigma = {sigma:.4f} +- {sig_err:.4f} (per N {np.round(sigN, 4)})")
            print("      X    heavy           lightest singlet   s                light")
            for r in rows[::2]:
                f = lambda k: f"{r[k]:.4f}+-{r[k + '_err']:.4f}" if np.isfinite(r[k]) else "      --       "
                print(f"  {r['X']:6.3f}  {f('heavy')}  {f('d0')}  {f('s')}  {f('light')}")
        store.put(os.path.join(HERE, 'crossover_functions.json'), dict(yt=YT, c=C, omega=OM, CFT=CFT, flows=res))
    return namespace(locals(), globals())


STEPS = {'o3flow': fuzzy.o3flow, 'gflow': fuzzy.gflow, 'partB_flow': partB_flow, 'flow_run': flow_run, 'traj_validate': traj_validate, 'imp_extract': imp_extract, 'traj_analyze': traj_analyze, 'retune_test': retune_test, 'crossover_extrap': crossover_extrap}
REPORT = [('traj_analyze',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
