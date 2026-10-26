"""Does conformal perturbation theory account for the response of the bulk to the anisotropy?

At the O(3) point the anisotropy couples to a component of the rank-two tensor t. Results: the first-order slopes
and the curvature on the lattice, the retuned CPT predictions from bootstrap data, the descendant amplitudes r_k and
the family sums, the lattice amplitudes of the nearly null descendant of phi and their convergence to N = 16 (DMRG),
and the rank-three family.

usage: python cpt.py <step> [args]      (python cpt.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def partA_slopes():
    """Part A: leading splitting of O(3) multiplets under the spin-2 anisotropy, at the O(3) critical point.

sigma_n = <n|W|n> / u0 is d Delta_n / dg at g = 0 (Hellmann-Feynman), in units of the d(phi)
descendant gap u0.  Wigner-Eckart fixes the ratios inside a multiplet (phi: +1 : -2; t: 2 : -1 : -2);
the growth with N gives the RG eigenvalue: sigma ~ N^{y_t/2}, y_t = 3 - Delta_t ~ 1.7905 (bootstrap).
"""
    HERE = 'flow'
    import sys
    import json
    import time
    import numpy as np
    F = fuzzy.o3flow()

    HC_STORED = {6: -0.09755397824438677, 7: -0.09736940424839385, 8: -0.09721604550411893,
                 9: -0.09709057245043519}


    def expW(fm, name, h, k):
        E, U = fm.levels(name, h, 0.0, k=k, vecs=True)
        nw = fm.sector(name)['nw']
        return E, np.einsum('ij,i,ij->j', U.conj(), nw, U).real


    def run(N):
        t0 = time.time()
        fm = F.FlowModel(N, cache=tuple(F.SECTORS))
        hguess = HC_STORED.get(N, -0.0975)
        if N >= 9:                                   # stored h_c reproduced to 1e-8 at N = 6, 7, 8
            hc = hguess
        else:
            hc, comp, fail = fm.tune(0.0, hguess, comp='V', target=F.TARGET['o3'], span=0.03, n=13)
        E0, W0 = expW(fm, 'S0', hc, 4)
        out = dict(N=N, h_c=hc, h_c_stored=HC_STORED.get(N))
        e0 = E0[0]
        lev = {}
        for name, k in (('S0', 4), ('S1', 3), ('P0', 2), ('P1', 2), ('V0', 2), ('V1', 2), ('A0', 2),
                        ('A1', 2), ('B0', 2), ('B1', 2), ('S2', 2)):
            E, W = expW(fm, name, hc, k)
            lev[name] = dict(E=(E - e0).tolist(), W=W.tolist())
        u0 = lev['V1']['E'][0] - lev['V0']['E'][0]
        out['u0'] = u0
        out['W_vac'] = lev['S0']['W'][0]
        D = lambda nm, i: lev[nm]['E'][i] / u0
        S = lambda nm, i: (lev[nm]['W'][i] - out['W_vac']) / u0
        out['Delta'] = dict(phi_pm=D('V0', 0), phi_3=D('P0', 0), t2=D('B0', 0), t1=D('A0', 0), t0=D('S0', 1),
                            s=D('S0', 2), T=D('S2', 0), J12=D('S1', 0))
        out['sigma'] = dict(phi_pm=S('V0', 0), phi_3=S('P0', 0), t2=S('B0', 0), t1=S('A0', 0), t0=S('S0', 1),
                            s=S('S0', 2))
        out['levels'] = lev
        out['time'] = time.time() - t0
        print(f"N={N}: h_c={hc:.8f} (stored {HC_STORED.get(N)}), u0={u0:.6f}, W_vac={out['W_vac']:.2e}  "
              f"[{out['time']:.0f}s]", flush=True)
        print("   Delta: " + ", ".join(f"{k}={v:.4f}" for k, v in out['Delta'].items()), flush=True)
        print("   sigma: " + ", ".join(f"{k}={v:+.5f}" for k, v in out['sigma'].items()), flush=True)
        sg = out['sigma']
        print(f"   W-E: phi_pm/phi_3 = {sg['phi_pm']/sg['phi_3']:+.6f} (-0.5); t1/t2 = {sg['t1']/sg['t2']:+.6f} "
              f"(-0.5); t0/t2 = {sg['t0']/sg['t2']:+.6f} (-1)", flush=True)
        store.put(os.path.join(HERE, f'partA_N{N}.json'), out)
        return out


    def cli(argv):
        for N in map(int, argv[1:]):
            run(N)
    return namespace(locals(), globals())


@section
def cpt_lattice():
    """Exact second-order response at the O(3) point: the lattice side of second-order conformal perturbation
theory (CPT) for the Delta_phi-tuned crossover (ED, env: quantum).

At g = 0, h = h_c(N) (partA_N{N}.json) this records, for the states the scheme uses,
    vac, t0, s (lowest three L = 0 levels of S0), phi_3 (P0), dphi_3 (P1), phi_+- (V0), dphi_+- (V1),
the energy and its derivatives E_g, E_h (Hellmann-Feynman) and E_gg, E_gh, E_hh. The second derivatives are
fourth-order central differences of the Hellmann-Feynman expectation values on the lines g = +-d, +-2d
(h = h_c) and h = h_c +- e, +-2e (g = 0). E_gh is measured twice (d<n_h>/dg and d<W>/dh) as a check.

It also records the operator data at g = 0:
  * <vac|W|t0>, <vac|n_h|s>, <s|W|t0>, the normalizations of t and s;
  * a Lanczos spectral decomposition of Q W|phi> and Q n_h|phi> (Q projects out phi), for phi_3 and phi_+-.
    These are the intermediate states of the second-order sums for the order-parameter levels. They live
    in the L = 0 part of the phi sector, because W and n_h commute with the rotations.
usage: python cpt_lattice.py N [N ...]        -> cpt_lattice_N{N}.json
"""
    HERE = 'flow'
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    F = fuzzy.o3flow()

    KS0 = 6                       # S0 levels per solve (vac, t0, s plus L > 0 members)


    def lzero(E0, E1, tol=1e-7):
        return [i for i in range(1, len(E0)) if np.min(np.abs(E1 - E0[i])) > tol]


    def hf(fm, name, h, g, k):
        """energies and the Hellmann-Feynman derivatives <W>, <n_h> of the k lowest levels; also the vectors."""
        E, U = fm.levels(name, h, g, k=k, vecs=True)
        s = fm.sector(name)
        P = np.abs(U) ** 2
        return E, P.T @ s['nw'], P.T @ s['nh'], U


    def d1(fm2, fm1, fp1, fp2, d):
        """fourth-order central first derivative from f(-2d), f(-d), f(d), f(2d)."""
        return (8 * (fp1 - fm1) - (fp2 - fm2)) / (12 * d)


    def lanczos(Hm, b, nsteps, orth):
        """Lanczos with full reorthogonalization, started from b projected off the vectors in orth.
    Returns Ritz values, amplitudes <ritz|b>, residual norms, and (V, S) to rebuild Ritz vectors."""
        for q in orth:
            b = b - (q @ b) * q
        nb = np.linalg.norm(b)
        V = np.zeros((nsteps, b.size))
        a, be = np.zeros(nsteps), np.zeros(nsteps)
        V[0] = b / nb
        m = nsteps
        for j in range(nsteps):
            w = Hm @ V[j]
            a[j] = V[j] @ w
            w -= V[:j + 1].T @ (V[:j + 1] @ w)
            w -= V[:j + 1].T @ (V[:j + 1] @ w)
            for q in orth:
                w -= (q @ w) * q
            be[j] = np.linalg.norm(w)
            if j + 1 < nsteps:
                if be[j] < 1e-12 * nb:
                    m = j + 1
                    break
                V[j + 1] = w / be[j]
        T = np.diag(a[:m]) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
        th, S = np.linalg.eigh(T)
        return th, nb * S[0], be[m - 1] * np.abs(S[-1]), V[:m], S


    def spectral(fm, name, h, E_n, u, psi, nsteps, orth=None):
        """Second-order data of the state psi of sector `name` (energy E_n, unit u), at g = 0.
    W part: Lanczos on Q W|psi>, Q = 1 - |psi><psi|; its Ritz pairs give the intermediate states and weights.
    Mixed part: the overlaps <ritz|n_h|psi> of the W Ritz vectors (reliable for converged Ritz pairs) and,
    for a sector ground state, the exact mixed sum from a CG solve (H - E_n) x = Q n_h|psi>."""
        s = fm.sector(name)
        Hm = fm.H(name, h, 0.0)
        orth = [psi] if orth is None else orth
        th, amp, res, V, S = lanczos(Hm, s['nw'] * psi, nsteps, orth)
        bh = s['nh'] * psi
        for q in orth:
            bh = bh - (q @ bh) * q
        amp_h = (V @ bh) @ S
        out = dict(ritz=((th - E_n) / u).tolist(), amp_W=amp.tolist(), amp_nh=amp_h.tolist(), resid=res.tolist(),
                   second_W=float(np.sum(amp ** 2 / (E_n - th))), mixed_ritz=float(np.sum(amp * amp_h / (E_n - th))))
        if len(orth) == 1:                                   # ground state of its sector: exact solves
            shift = (Hm - E_n * sp.identity(Hm.shape[0], format='csr')).tocsr()
            x, info = sla.cg(shift, bh, rtol=1e-12, maxiter=6000)
            x = x - (psi @ x) * psi
            bw = s['nw'] * psi
            bw = bw - (psi @ bw) * psi
            out.update(mixed=float(-(bw @ x)), second_nh=float(-(bh @ x)), cg_info=int(info))
        return out


    def run(N, nsteps=60):
        t0 = time.time()
        fm = F.FlowModel(N, cache=('S0', 'S1', 'P0', 'P1', 'V0', 'V1'), verbose=False)
        hc = store.get(os.path.join(HERE, f'partA_N{N}.json'))['h_c']
        dg = 0.002 * N ** (-1.7905 / 2)
        dh = 3 * dg
        # ----- centre: identification and operator data
        E0, W0, H0, U0 = hf(fm, 'S0', hc, 0.0, KS0)
        E1 = fm.levels('S1', hc, 0.0, k=5)
        lz = lzero(E0, E1)
        idx = dict(vac=0, t0=lz[0], s=lz[1])
        s0 = fm.sector('S0')
        me = lambda op, i, j: float(U0[:, i] @ (s0[op] * U0[:, j]))
        ope = dict(W_vac_t0=me('nw', 0, idx['t0']), nh_vac_s=me('nh', 0, idx['s']), nh_vac_t0=me('nh', 0, idx['t0']),
                   W_s_t0=me('nw', idx['s'], idx['t0']), W_vac_s=me('nw', 0, idx['s']),
                   nh_s_t0=me('nh', idx['s'], idx['t0']))
        # ----- stencil
        pts = {'c': (0.0, hc)}
        for k, f in (('gm2', -2), ('gm1', -1), ('gp1', 1), ('gp2', 2)):
            pts[k] = (f * dg, hc)
        for k, f in (('hm2', -2), ('hm1', -1), ('hp1', 1), ('hp2', 2)):
            pts[k] = (0.0, hc + f * dh)
        rec = {}
        for nm, k, sel in (('S0', KS0, idx), ('P0', 1, dict(phi3=0)), ('P1', 1, dict(dphi3=0)),
                           ('V0', 1, dict(phipm=0)), ('V1', 1, dict(dphipm=0))):
            vals = {}
            for key, (g, h) in pts.items():
                E, W, Hh, _ = hf(fm, nm, h, g, k)
                vals[key] = (E, W, Hh)
            for st, i in sel.items():
                f = lambda key, j: vals[key][j][i]
                rec[st] = dict(E=float(f('c', 0)), Eg=float(f('c', 1)), Eh=float(f('c', 2)),
                               Egg=float(d1(f('gm2', 1), f('gm1', 1), f('gp1', 1), f('gp2', 1), dg)),
                               Egh=float(d1(f('gm2', 2), f('gm1', 2), f('gp1', 2), f('gp2', 2), dg)),
                               Egh_alt=float(d1(f('hm2', 1), f('hm1', 1), f('hp1', 1), f('hp2', 1), dh)),
                               Ehh=float(d1(f('hm2', 2), f('hm1', 2), f('hp1', 2), f('hp2', 2), dh)),
                               Egg_2nd=float(((f('gp1', 1) - f('gm1', 1)) / (2 * dg))))
            print(f"  N={N} {nm} done [{time.time() - t0:.0f}s]", flush=True)
        out = dict(N=N, h_c=hc, dg=dg, dh=dh, idx=idx, ope=ope, states=rec)
        # ----- spectral decomposition: order-parameter components (sector ground states) and t0 (excited)
        out['spectral'] = {}
        for nm, st, dn in (('P0', 'phi3', 'dphi3'), ('V0', 'phipm', 'dphipm')):
            E, U = fm.levels(nm, hc, 0.0, k=1, vecs=True)
            u = rec[dn]['E'] - rec[st]['E']
            out['spectral'][st] = spectral(fm, nm, hc, E[0], u, U[:, 0], nsteps)
            print(f"  N={N} spectral {st} done [{time.time() - t0:.0f}s]", flush=True)
        u = rec['dphi3']['E'] - rec['phi3']['E']
        out['spectral']['t0'] = spectral(fm, 'S0', hc, E0[idx['t0']], u, U0[:, idx['t0']], nsteps,
                                         orth=[U0[:, idx['t0']]])
        print(f"  N={N} spectral t0 done [{time.time() - t0:.0f}s]", flush=True)
        out['time'] = time.time() - t0
        store.put(os.path.join(HERE, f'cpt_lattice_N{N}.json'), out)
        r = rec
        print(f"N={N}: Egh check phi3 {r['phi3']['Egh']:.6e} vs {r['phi3']['Egh_alt']:.6e}; "
              f"Egg phi3 {r['phi3']['Egg']:.6e} (2nd-order fd {r['phi3']['Egg_2nd']:.6e}); "
              f"Lanczos 2*sum {2 * out['spectral']['phi3']['second_W']:.6e}  [{out['time']:.0f}s]", flush=True)
        return out


    def cli(argv):
        for N in map(int, argv[1:]):
            run(N)
    return namespace(locals(), globals())


@section
def cpt_scheme():
    """Taylor coefficients of the Delta_phi-tuned crossover at g = 0 from the exact lattice derivatives
(cpt_lattice_N{N}.json), in the universal normalization of conformal perturbation theory.

Along the scheme h = h_c + kappa g + kappa_2 g^2, every level O_n = (E_n - E_0)/u, with u the descendant
spacing of the light component, is expanded to second order in g. The tuning O_light = Delta_phi fixes
kappa (first order) and kappa_2 (second order). X is the first-order heavy - light splitting, so
    O_n = O_n(0) + b1 X + b2 X^2 + O(X^3),   b1 = a1/a1_hl,  b2 = a2/a1_hl^2,
with a1, a2 the g-Taylor coefficients and a1_hl the slope of heavy - light. No fitted parameters.
The Ising scheme tunes phi_3 (light for g > 0), the XY scheme tunes phi_+- (light for g < 0).
usage: python cpt_scheme.py            -> table + cpt_scheme.json
"""
    import os
    import glob
    import json
    import numpy as np

    HERE = 'flow'
    DPHI = 0.518942


    def ser(st, kap, kap2, ref=None):
        """[c0, c1, c2] of E(g) along h = h_c + kap g + kap2 g^2 (relative to ref if given)."""
        c = np.array([st['E'], st['Eg'] + kap * st['Eh'],
                      0.5 * (st['Egg'] + 2 * kap * st['Egh'] + kap ** 2 * st['Ehh']) + kap2 * st['Eh']])
        return c - ser(ref, kap, kap2) if ref is not None else c


    def div(a, b):
        """series a/b to second order."""
        q0 = a[0] / b[0]
        q1 = (a[1] - q0 * b[1]) / b[0]
        q2 = (a[2] - q0 * b[2] - q1 * b[1]) / b[0]
        return np.array([q0, q1, q2])


    def scheme(S, light, heavy):
        dl, l = 'd' + light, light
        vac = S['vac']

        def O(name, kap, kap2):
            u = ser(S[dl], kap, kap2, ref=S[l])
            return div(ser(S[name], kap, kap2, ref=vac), u)
        # kappa: O_l first-order coefficient is linear in kappa
        f1 = lambda k: O(l, k, 0.0)[1]
        kap = -f1(0.0) / (f1(1.0) - f1(0.0))
        f2 = lambda k2: O(l, kap, k2)[2]
        kap2 = -f2(0.0) / (f2(1.0) - f2(0.0))
        res = {nm: O(nm, kap, kap2).tolist() for nm in ('t0', 's', heavy, l)}
        hl = np.array(res[heavy]) - np.array(res[l])
        out = dict(kappa=kap, kappa2=kap2, a1_hl=hl[1], a2_hl=hl[2], b2_hl=hl[2] / hl[1] ** 2)
        for nm in ('t0', 's', heavy):
            c = res[nm]
            out[nm] = dict(O0=c[0], b1=c[1] / hl[1], b2=c[2] / hl[1] ** 2)
        out['light_check'] = res[l]
        # pieces of b2_hl: E-difference curvature and the u term
        u = ser(S[dl], kap, 0.0, ref=S[l])
        e_hl = ser(S[heavy], kap, 0.0, ref=S[l])
        out['b2_hl_parts'] = dict(energy=e_hl[2] / u[0] / hl[1] ** 2 * 1.0, unit=-e_hl[1] * u[1] / u[0] ** 2 / hl[1] ** 2)
        return out


    def cli(argv):
        rows = {}
        for f in sorted(store.glob(os.path.join(HERE, 'cpt_lattice_N*.json')), key=lambda s: int(s.split('_N')[1][:-5])):
            r = store.get(f)
            S = r['states']
            N = r['N']
            rows[N] = dict(Ising=scheme(S, 'phi3', 'phipm'), XY=scheme(S, 'phipm', 'phi3'))
        for lab in ('Ising', 'XY'):
            print(f"\n{lab} scheme: b1, b2 of the levels in units of X (first-order heavy - light)")
            print("   N   kappa      a1_hl     b2_hl (energy, unit)       t0: O0  b1      b2        s: O0   b1      b2")
            for N, rr in rows.items():
                o = rr[lab]
                p = o['b2_hl_parts']
                print(f"  {N:2d}  {o['kappa']:+.4f}  {o['a1_hl']:+9.3f}  {o['b2_hl']:+.4f} ({p['energy']:+.4f}, {p['unit']:+.4f})"
                      f"   {o['t0']['O0']:.4f} {o['t0']['b1']:+.4f} {o['t0']['b2']:+.4f}"
                      f"   {o['s']['O0']:.4f} {o['s']['b1']:+.4f} {o['s']['b2']:+.4f}")
        store.put(os.path.join(HERE, 'cpt_scheme.json'), rows)
    return namespace(locals(), globals())


@section
def cpt_kin():
    """Conformal kinematics for second-order conformal perturbation theory on S^2 x R (d = 3, radius 1).

Perturbation V_O = int_{S^2} dOmega O(n), O a unit-normalized scalar primary of dimension a. For scalar
primaries n (ket) and m (bra) with <m|O(n)|n> = C at |x| = 1, the L = 0 descendants |m;k> ~ (P^2)^k|m>
(normalized) have
    <m;k|V_O|n> = 4 pi C r_k,   r_k = F_k(b) sqrt((2k+1)! / (4^k (D_m)_k (D_m - 1/2)_k)),
    F_k(b) = binom(b, 2k+1)/b,  b = 2 - (D_m + a - D_n),
from the sphere average of the three-point function, sum_k F_k rho^{2k} = [(1+rho)^b - (1-rho)^b]/(2 rho b),
and the norms ||(P^2)^k|D>||^2 = 4^k (2k+1)! (D)_k (D - 1/2)_k.  r_0 = 1, and r_k^2 ~ k^{2a - 2 D_n - 3}.
For the level-one descendant of a scalar D, <dD|V_O|dD> / <D|V_O|D> = 1 + a(a - 3)/(6 D).
Checks (run this file): the series against the closed form, the free scalar (D = 1/2, a = 1, ratio 1/3 =
omega_0/omega_1 for a mass term), and the vacuum row <O;k|V_O|0> against the direct expansion of V|0>.
"""
    HERE = 'flow'
    import numpy as np
    from scipy.special import gammaln


    def log_absF(b, k):
        """log|binom(b, 2k+1)/b| and its sign, for an array of k (b need not be an integer)."""
        k = np.asarray(k)
        n = 2 * k + 1
        kmax = int(k.max())
        j = np.arange(1, 2 * kmax + 1)                       # factors (b - j)/(j + 1) for j = 1..2k
        f = (b - j) / (j + 1.0)
        lf = np.concatenate([[0.0], np.cumsum(np.log(np.abs(f)))])
        sg = np.concatenate([[1.0], np.cumprod(np.sign(f))])
        # binom(b, n)/b = prod_{j=1}^{n-1} (b - j)/(j + 1)   (the j = 0 factor b/1 cancels the 1/b)
        return lf[n - 1], sg[n - 1]


    def r_k(Dn, Dm, a, K):
        """r_k for k = 0..K (array), signed."""
        k = np.arange(K + 1)
        b = 2.0 - (Dm + a - Dn)
        lF, sF = log_absF(b, k)
        lN = (gammaln(2 * k + 2) - k * np.log(4.0) - (gammaln(Dm + k) - gammaln(Dm))
              - (gammaln(Dm - 0.5 + k) - gammaln(Dm - 0.5)))
        return sF * np.exp(lF + 0.5 * lN)


    def rho1(D, a):
        """<dD|V_O|dD>/<D|V_O|D> for the level-one descendant of a scalar D."""
        return 1.0 + a * (a - 3.0) / (6.0 * D)


    def family_sum(Dn, Dm, a1, a2, gap0, K=200000, kmin=0):
        """sum_{k >= kmin} r_k(a1) r_k(a2) / (Dn - Dm - 2k)  with a power-law tail beyond K.
    gap0 = Dn - Dm.  The k = 0 term (the primary) is included when kmin = 0."""
        r1, r2 = r_k(Dn, Dm, a1, K), r_k(Dn, Dm, a2, K)
        k = np.arange(K + 1)
        t = r1 * r2 / (gap0 - 2.0 * k)
        s = t[kmin:].sum()
        # tail: t_k ~ c k^{-p}, p = 1 + (3 + 2 Dn - a1 - a2) ; fit c on the last decade
        p = 1.0 + (3.0 + 2.0 * Dn - a1 - a2)
        kk = k[K // 2:]
        c = np.mean(t[K // 2:] * kk ** p)
        tail = c * (K + 0.5) ** (1 - p) / (p - 1) if p > 1 else np.nan
        return float(s + tail), float(tail)


    def cli(argv):
        a, D = 1.20954, 0.518942
        rho = 0.37
        b = 2.0 - a
        ks = np.arange(80)
        lF, sF = log_absF(b, ks)
        ser = np.sum(sF * np.exp(lF) * rho ** (2 * ks))
        print('generating function: series %.15f closed %.15f' % (ser, ((1 + rho) ** b - (1 - rho) ** b) / (2 * rho * b)))
        print('free scalar mass term: rho1(1/2, 1) =', rho1(0.5, 1.0), '(exact 1/3)')
        # vacuum row: <O;k|V|0> = 4 pi sqrt(4^k (a)_k (a-1/2)_k / (2k+1)!)
        K = 6
        k = np.arange(K + 1)
        direct = np.exp(0.5 * (k * np.log(4.0) + gammaln(a + k) - gammaln(a) + gammaln(a - 0.5 + k) - gammaln(a - 0.5)
                               - gammaln(2 * k + 2)))
        print('vacuum row: formula', np.round(r_k(0.0, a, a, K), 10), '\n            direct ', np.round(direct, 10))
        print('phi self-family r_k (O = t):', np.round(r_k(D, D, a, 5), 5))
        for K in (20000, 200000):
            print(K, 'S_tt (phi, t, t):', family_sum(D, D, a, a, 0.0, K=K, kmin=1))
    return namespace(locals(), globals())


@section
def cpt_compare():
    """Second-order conformal perturbation theory (CPT) for the Delta_phi-tuned crossover, against the exact
lattice Taylor coefficients (cpt_lattice_N*.json via cpt_scheme.py).

Units: X = first-order heavy - light splitting, so heavy - light = X + b2_hl X^2 + ...
CFT data: Delta_phi, Delta_s, Delta_t and lambda_{phi phi s}, lambda_{phi phi t}, lambda_{sss},
lambda_{tts}, lambda_{ttt} from the O(3) bootstrap (Chester et al., arXiv:2011.14647). Conventions, for
unit-normalized t(A) = t_ij A_ij (tr A^2 = 1) and t_W along W = diag(1, 1, -2)/sqrt 6:
  <phi_i phi_j t(A)> = sqrt2 lambda_{phi phi t} A_ij   (their <phi phi phi phi> T-structure is
                       d_ik d_jl + d_il d_jk - (2/3) d_ij d_kl = 2P, so the free value is lambda = 1),
  <t(A) t(B) t(C)>  = sqrt2 lambda_{ttt} tr(ABC),   <t(A) t(B) s> = lambda_{tts} tr(AB),
  <phi_i phi_j s>    = lambda_{phi phi s} d_ij.
The ttt and tts factors are fixed by the lattice: c_t/c_phi = <W>_t0/<W>_phi3 runs 0.958 -> 0.898 (N = 4..13)
toward lambda_ttt/(2 lambda_phiphit) = 0.857, and the first-order slopes need C_tts/C_phiphis = 1.876.
The absolute normalization c_phi = lambda_{phi phi t}/sqrt3 = 0.505 is confirmed by <phi|W|phi>/<0|W|t0>,
whose 1/N extrapolation over N = 6..9 gives 2 c_phi = 1.00 (+-0.02).
Then <phi_3|t_W|phi_3> = -2 c_phi and <t0|t_W|t0> = -2 c_t with c_t = lambda_ttt/(2 sqrt3).

First order (exact in CPT):
  b1(t0) = (2/3)[-c_t/c_phi + r C_tts/C_phiphis + 0.0213]      (Ising)
         = (1/3)[2 c_t/c_phi + r C_tts/C_phiphis + 0.0213]     (XY)
  b1(s)  = (2/3) or (1/3) x [r C_sss/C_phiphis + (D_s/D_phi)(r alpha_s - alpha_t)]
  with alpha_O = a(3 - a)/6 from <dphi|V_O|dphi>/<phi|V_O|phi> = 1 - alpha_O/Delta_phi, r = (1+alpha_t)/(1+alpha_s).
Second order, heavy - light (UV finite, low-lying states dominate):
  b2_hl = (T_tt + 2 kappa T_ts)/(t_h - t_l)^2 - u1/(t_h - t_l)
  self family (phi descendants): Ising (S_tt - 4 r S_ts)/3, XY -(S_tt + 2 r S_ts)/3 -- kinematics only;
  rank-3 tensor t3:   -+ (1/45) (C3/c_phi)^2 |Sigma_3|  (Ising -, XY +... signs below);
  j = 2 states (q = 1 only) and phi' (j = 1) from the lattice couplings.
usage: python cpt_compare.py
"""
    import os
    import json
    import numpy as np
    r_k = cpt_kin().r_k
    family_sum = cpt_kin().family_sum

    HERE = 'flow'
    Dp, Dt, Ds = 0.518942, 1.20954, 1.59488
    LAM = dict(phiphis=0.524261, phiphit=0.87451, sss=0.5055, tts=0.98348, ttt=1.49957)
    cphi = LAM['phiphit'] / np.sqrt(3)
    ct = LAM['ttt'] / (2 * np.sqrt(3))
    al = lambda a: a * (3 - a) / 6
    r = (1 + al(Dt)) / (1 + al(Ds))
    du = al(Dt) - r * al(Ds)
    CG2 = {1: {0: 2 / 5, 1: 1 / 10}, 2: {0: 0.0, 1: 1 / 2}, 3: {0: 3 / 5, 1: 2 / 5}}   # <1 q; 2 0|j q>^2
    CQ = {0: -2, 1: 1}                                                                   # W-E factor of phi_q


    def first_order():
        A, B, C = ct / cphi, LAM['tts'] / LAM['phiphis'], LAM['sss'] / LAM['phiphis']
        d0 = dict(Ising=(2 / 3) * (-A + r * B - (Dt / Dp) * du), XY=(1 / 3) * (2 * A + r * B - (Dt / Dp) * du))
        s = dict(Ising=(2 / 3) * (r * C - (Ds / Dp) * du), XY=(1 / 3) * (r * C - (Ds / Dp) * du))
        return d0, s


    def b2_hl_cpt(C3=None, D3=2.04, extra=None):
        """CPT heavy - light curvature, split by family. C3 = reduced phi-t-t3 coupling / c_phi."""
        Stt = -family_sum(Dp, Dp, Dt, Dt, 0.0, kmin=1)[0]
        Sts = -family_sum(Dp, Dp, Dt, Ds, 0.0, kmin=1)[0]
        out = {}
        for lab, ql, qh in (('Ising', 0, 1), ('XY', 1, 0)):
            cl, ch = CQ[ql], CQ[qh]
            parts = dict(self_tt=-(ch + cl) * Stt / (ch - cl), self_ts=2 * cl * r * Sts / (ch - cl),
                         unit=cl * du / (Dp * (ch - cl)))
            if C3 is not None:
                S3 = family_sum(Dp, D3, Dt, Dt, Dp - D3)[0]                    # sum_k r_k^2/(Dp - D3 - 2k) < 0
                parts['t3'] = (CG2[3][qh] - CG2[3][ql]) * C3 ** 2 * S3 / 9
            for k, v in (extra or {}).items():
                parts[k] = v[lab]
            out[lab] = dict(total=sum(parts.values()), **parts)
        return out, dict(S_tt=Stt, S_ts=Sts)


    def lattice_parts(N):
        """b2_hl of the lattice split into intermediate states (Ritz pairs of the W Lanczos run)."""
        L = store.get(os.path.join(HERE, f'cpt_lattice_N{N}.json'))
        S = L['states']
        sc = store.get(os.path.join(HERE, 'cpt_scheme.json'))[str(N)]
        out = {}
        for lab, l, h in (('Ising', 'phi3', 'phipm'), ('XY', 'phipm', 'phi3')):
            kap = sc[lab]['kappa']
            u = S['d' + l]['E'] - S[l]['E']
            a1 = sc[lab]['a1_hl']
            norm = u * a1 ** 2
            fam = {}
            for st, sgn in ((h, +1), (l, -1)):
                sp = L['spectral'][st]
                th, aW, ah = np.array(sp['ritz']), np.array(sp['amp_W']), np.array(sp['amp_nh'])
                den = -th * u
                q = 1 if st == 'phipm' else 0
                for t, w, hh, d in zip(th, aW, ah, den):
                    if abs(w) < 1e-6 * abs(S[st]['Eg']):
                        continue
                    key = f"{t:.2f}"
                    fam.setdefault(key, [0.0, 0.0, 0.0])
                    fam[key][0] += sgn * w ** 2 / d / norm
                    fam[key][1] += sgn * 2 * kap * w * hh / d / norm
                tot_tt = sp['second_W'] / norm * sgn
                tot_ts = 2 * kap * sp['mixed'] / norm * sgn
                fam.setdefault('_total', [0.0, 0.0, 0.0])
                fam['_total'][0] += tot_tt
                fam['_total'][1] += tot_ts
            out[lab] = fam
        return out


    def cli(argv):
        d0, s = first_order()
        print(f"inputs: c_phi {cphi:.6f}  c_t {ct:.6f}  c_t/c_phi {ct / cphi:.5f}  C_tts/C_phiphis "
              f"{LAM['tts'] / LAM['phiphis']:.5f}  C_sss/C_phiphis {LAM['sss'] / LAM['phiphis']:.5f}  r {r:.6f}  "
              f"alpha_t - r alpha_s {du:+.6f}")
        print(f"\nFIRST ORDER (CPT, bootstrap data):  lightest singlet b1: Ising {d0['Ising']:.4f}  XY {d0['XY']:.4f}   "
              f"second singlet b1: Ising {s['Ising']:.4f}  XY {s['XY']:.4f}")
        t3 = family_sum(Dp, 2.04, Dt, Dt, Dp - 2.04)[0]
        print(f"\nt3 family sum Sigma_3(D3 = 2.04) = {t3:.5f};  D3 = 2.10: {family_sum(Dp, 2.10, Dt, Dt, Dp - 2.10)[0]:.5f}")
        for C3 in (2.45, 2.8, 3.2):
            res, SS = b2_hl_cpt(C3=C3)
            print(f"C3/c_phi = {C3}:  " + "   ".join(f"{k}: total {v['total']:+.4f} (self_tt {v['self_tt']:+.4f}, "
                                                       f"self_ts {v['self_ts']:+.4f}, t3 {v['t3']:+.4f}, unit {v['unit']:+.4f})"
                                                       for k, v in res.items()))
        print('S sums', SS)
        for N in (6, 8, 9):
            P = lattice_parts(N)
            print(f"\nN={N} lattice b2_hl by intermediate state (Ritz value: tt, ts):")
            for lab, fam in P.items():
                items = sorted([(k, v) for k, v in fam.items() if k != '_total'], key=lambda z: float(z[0]))
                big = [(k, v) for k, v in items if abs(v[0]) + abs(v[1]) > 2e-3]
                ssum = np.array([v for k, v in items]).sum(0)
                print(f"  {lab}: total tt {fam['_total'][0]:+.4f} ts {fam['_total'][1]:+.4f} | resolved tt {ssum[0]:+.4f} "
                      f"ts {ssum[1]:+.4f} | " + ", ".join(f"{k}: ({v[0]:+.4f}, {v[1]:+.4f})" for k, v in big))
    return namespace(locals(), globals())


@section
def imp_ops():
    """Lattice operators for the improvement study (GZWH model, ED sectors).
channel_terms(N, weights, G0, G1): terms of sum_a w_a int (psi^+ A_a psi)[G0 delta + G1 lap delta](psi^+ A_a psi),
A_a = E(0,a) + E(a,0), in the onmodel (k1,k2,k3,k4,v) + one-body format (ordering term kept, as in onmodel).
weights {1,2,3}: (1,1,-2) is the spin-2 counterterm W2L, (1,1,1) the singlet S2L (the GZWH n.n channel itself)."""
    HERE = 'flow'
    import numpy as np
    F = fuzzy.o3flow()

    M = F.M


    def channel_terms(N, weights, G0, G1):
        cfg = F.CFG
        fl = M.Flavours(cfg['Nv'], cfg['pairs'], cfg['twist'])
        s = (N - 1) / 2
        orbs, koff = zip(*[M.orbitals(N, 0.0, fl.qtw[k]) for k in range(fl.F)])
        off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])])
        Fn = fl.F
        E = lambda a, b: np.eye(Fn)[:, [a]] @ np.eye(Fn)[[b], :]
        index = [{round(m, 9): i for i, m in enumerate(o)} for o in orbs]
        two, one, cache = {}, {}, {}

        def I(m1, m2, m3, m4):
            key = (round(m1, 9), round(m2, 9), round(m3, 9), round(m4, 9))
            if key not in cache:
                cache[key] = M.four_orbital(s, m1, m2, m3, m4)
            return cache[key]
        for ach, wt in weights.items():
            A = fl.rot(E(0, ach) + E(ach, 0))
            A[np.abs(A) < 1e-14] = 0
            nz = [(a, b) for a in range(Fn) for b in range(Fn) if A[a, b] != 0]
            for (a, b) in nz:
                for (c, d) in nz:
                    wgt = wt * A[a, b] * A[c, d]
                    for i1, m1 in enumerate(orbs[a]):
                        for i2, m2 in enumerate(orbs[c]):
                            for i3, m3 in enumerate(orbs[d]):
                                i4 = index[b].get(round(m1 + m2 - m3, 9))
                                if i4 is None:
                                    continue
                                Id, Il = I(m1, m2, m3, m1 + m2 - m3)
                                v = wgt * (G0 * Id + G1 * Il)
                                if abs(v) < 1e-13:
                                    continue
                                key = (off[a] + i1, off[c] + i2, off[d] + i3, off[b] + i4)
                                two[key] = two.get(key, 0) + v
                for (c, d) in nz:
                    if c != b:
                        continue
                    wgt = wt * A[a, b] * A[b, d]
                    for i1, m1 in enumerate(orbs[a]):
                        i3 = index[d].get(round(m1, 9))
                        if i3 is None:
                            continue
                        v = wgt * sum(G0 * I(m1, m2, m1, m2)[0] + G1 * I(m1, m2, m1, m2)[1] for m2 in orbs[b])
                        one[(off[a] + i1, off[d] + i3)] = one.get((off[a] + i1, off[d] + i3), 0) + v
        vals = np.array(list(two.values()) + list(one.values()))
        assert np.abs(vals.imag).max() < 1e-10
        mu = np.zeros(off[-1])
        for (k, l), v in one.items():
            assert k == l or abs(v) < 1e-10
            if k == l:
                mu[k] += v.real
        return [(k[0], k[1], k[2], k[3], v.real) for k, v in two.items() if abs(v) > 1e-13], mu


    def sector_op(fm, name, terms):
        t, mu = terms
        Q, K, P = F.SECTORS[name]
        b = M.Basis(fm.fl, fm.orbs, fm.koff, fm.N, [Q], K, (P,))
        return M.build_H(b, t, mu).tocsr()
    return namespace(locals(), globals())


@section
def imp_thermal():
    """Improved lattice operators at the O(3) point (ED, GZWH, g = 0): spin-2 W(beta) = W1 + beta W2L and thermal
n'(gamma) = n_h + gamma S2L, with the stress-tensor admixture tau H of n' removed.

Notation: d_n(O) = <n|O|n> - <0|O|0>, Delta E_n = E_n - E_0, R_n = Delta E_n / Delta E_phi (lattice).
* beta: level-one ratio of W(beta) = rho_1(D_phi, D_t) = 0.30447. Test: level-two ratio against rho_2 = 1.12941,
  vacuum ratio <0|W|t0>/<phi|W|phi> against 1/(2 c_phi) = 0.99029, box coupling against 0.52181.
* gamma: tau-free conditions on dR_n = d_n - R_n d_phi (tau H drops out of these exactly):
    (B, kinematic)  dR_box / dR_dphi = (rho_2 - 1 - 2/D_phi)/(rho_1 - 1 - 1/D_phi) at a = D_s,
    (A, bootstrap)  dR_dphi / dR_t0 = (rho_1 - 1 - 1/D_phi)/(C_tts/C_phiphis - D_t/D_phi).
  Agreement of the two gamma's is a consistency test of the improvement.
* tau: level-one ratio (d'_dphi - tau dE_dphi)/(d'_phi - tau dE_phi) = rho_1(D_phi, D_s) = 0.28027.
  Tests: level-two ratio against rho_2 = 1.84981, and the tau-free box coupling <box|n'|phi>/(d'_phi - tau dE_phi)
  against 1.95335.
usage: python imp_thermal.py N [N ...]    -> imp_thermal_N{N}.json
"""
    HERE = 'flow'
    import sys
    import json
    import numpy as np
    import scipy.sparse as sp
    F = fuzzy.o3flow()
    channel_terms = imp_ops().channel_terms
    sector_op = imp_ops().sector_op
    lanczos = cpt_lattice().lanczos

    Dp, Dt, Ds = 0.518942, 1.20954, 1.59488
    r1 = lambda D, a: (6 * D + a ** 2 - 3 * a) / (6 * D)
    r2 = lambda D, a: (24 * D ** 2 + 8 * D * a ** 2 - 24 * D * a - 12 * D + a ** 4 - 6 * a ** 3 + 7 * a ** 2 + 6 * a) / \
        (12 * D * (2 * D - 1))
    CFT = dict(rho1_t=r1(Dp, Dt), rho2_t=r2(Dp, Dt), rho1_s=r1(Dp, Ds), rho2_s=r2(Dp, Ds), vac=0.99029, box_t=0.52181,
               box_s=1.95335, QB=(r2(Dp, Ds) - 1 - 2 / Dp) / (r1(Dp, Ds) - 1 - 1 / Dp),
               QA=(r1(Dp, Ds) - 1 - 1 / Dp) / (0.98348 / 0.524261 - Dt / Dp))


    def run(N):
        fm = F.FlowModel(N, cache=('S0', 'S1', 'P0', 'P1'), verbose=False)
        hc = store.get(os.path.join(HERE, f'partA_N{N}.json'))['h_c']
        T2 = channel_terms(N, {1: 1.0, 2: 1.0, 3: -2.0}, 0.0, 1.0 / N ** 2)
        TS = channel_terms(N, {1: 1.0, 2: 1.0, 3: 1.0}, 0.0, 1.0 / N ** 2)
        ops, st = {}, {}
        for nm in ('S0', 'P0', 'P1'):
            sec = fm.sector(nm)
            ops[nm] = dict(W1=sp.diags(sec['nw']).tocsr(), W2=sector_op(fm, nm, T2), NH=sp.diags(sec['nh']).tocsr(),
                           S2=sector_op(fm, nm, TS))
        E0, U0 = fm.levels('S0', hc, 0.0, k=6, vecs=True)
        E1 = fm.levels('S1', hc, 0.0, k=5)
        lz = [i for i in range(1, len(E0)) if np.min(np.abs(E1 - E0[i])) > 1e-7]
        st['vac'] = ('S0', E0[0], U0[:, 0])
        st['t0'] = ('S0', E0[lz[0]], U0[:, lz[0]])
        Ep, Up = fm.levels('P0', hc, 0.0, k=1, vecs=True)
        st['phi'] = ('P0', Ep[0], Up[:, 0])
        Ed, Ud = fm.levels('P1', hc, 0.0, k=1, vecs=True)
        st['dphi'] = ('P1', Ed[0], Ud[:, 0])
        th, amp, res, V, S = lanczos(fm.H('P0', hc, 0.0), fm.sector('P0')['nh'] * Up[:, 0], 60, [Up[:, 0]])
        jb = min([i for i in range(len(th)) if abs(amp[i]) > 1e-3 * np.abs(amp).max()], key=lambda i: th[i])
        box = V.T @ S[:, jb]
        box /= np.linalg.norm(box)
        st['box'] = ('P0', th[jb], box)
        diag = {n: {o: float(v @ (ops[sec][o] @ v)) for o in ops[sec]} for n, (sec, E, v) in st.items()}
        dE = {n: st[n][1] - st['vac'][1] for n in st}
        offbox = {o: float(box @ (ops['P0'][o] @ Up[:, 0])) for o in ops['P0']}
        offvac = {o: float(U0[:, 0] @ (ops['S0'][o] @ U0[:, lz[0]])) for o in ('W1', 'W2')}
        out = dict(N=N, h_c=hc, E={n: st[n][1] for n in st}, diag=diag, offbox=offbox, offvac=offvac, CFT=CFT)
        lin = lambda f, o1, o2, g: f(o1) + g * f(o2)
        # ---- spin-2: beta from rho_1, tests
        Wd = lambda n, o: diag[n][o]
        beta = (CFT['rho1_t'] * Wd('phi', 'W1') - Wd('dphi', 'W1')) / (Wd('dphi', 'W2') - CFT['rho1_t'] * Wd('phi', 'W2'))
        Wb = lambda n: Wd(n, 'W1') + beta * Wd(n, 'W2')
        out['W'] = dict(beta=beta, rho1=Wb('dphi') / Wb('phi'), rho2=Wb('box') / Wb('phi'),
                        vac=abs((offvac['W1'] + beta * offvac['W2']) / Wb('phi')),
                        box=abs((offbox['W1'] + beta * offbox['W2']) / Wb('phi')),
                        rho2_W1=Wd('box', 'W1') / Wd('phi', 'W1'))
        # ---- thermal: gamma from the tau-free conditions
        R = {n: dE[n] / dE['phi'] for n in st}
        d = lambda n, o: diag[n][o] - diag['vac'][o]
        dR = lambda n, o: d(n, o) - R[n] * d('phi', o)
        def solve(n1, n2, Q):          # (dR_n1(NH) + g dR_n1(S2)) = Q (dR_n2(NH) + g dR_n2(S2))
            return (Q * dR(n2, 'NH') - dR(n1, 'NH')) / (dR(n1, 'S2') - Q * dR(n2, 'S2'))
        gB = solve('box', 'dphi', CFT['QB'])
        gA = solve('dphi', 't0', CFT['QA'])
        out['gamma'] = dict(B=gB, A=gA, Q_B_at0=dR('box', 'NH') / dR('dphi', 'NH'), Q_A_at0=dR('dphi', 'NH') / dR('t0', 'NH'))
        for lab, g in (('g0', 0.0), ('gB', gB), ('gA', gA)):
            dp = lambda n: d(n, 'NH') + g * d(n, 'S2')
            tau = (dp('dphi') - CFT['rho1_s'] * dp('phi')) / (dE['dphi'] - CFT['rho1_s'] * dE['phi'])
            den = dp('phi') - tau * dE['phi']
            out[lab] = dict(gamma=g, tau=tau, rho1=(dp('dphi') - tau * dE['dphi']) / den,
                            rho2=(dp('box') - tau * dE['box']) / den,
                            box=abs((offbox['NH'] + g * offbox['S2']) / den),
                            box_raw=abs((offbox['NH'] + g * offbox['S2']) / dp('phi')))
        store.put(os.path.join(HERE, f'imp_thermal_N{N}.json'), out)
        w = out['W']
        print(f"N={N}: W(beta={w['beta']:.3f}): rho2 {w['rho2']:.4f} (CFT {CFT['rho2_t']:.4f}; W1 alone {w['rho2_W1']:.4f}) "
              f"vac {w['vac']:.4f} (0.9903) box {w['box']:.4f} (0.5218)", flush=True)
        print(f"      gamma_B {gB:+.4f} gamma_A {gA:+.4f} (Q_B {out['gamma']['Q_B_at0']:.4f}->{CFT['QB']:.4f}, "
              f"Q_A {out['gamma']['Q_A_at0']:.4f}->{CFT['QA']:.4f})", flush=True)
        for lab in ('g0', 'gB', 'gA'):
            o = out[lab]
            print(f"      {lab}: tau {o['tau']:+.5f}  rho2 {o['rho2']:.4f} (CFT {CFT['rho2_s']:.4f})  box tau-free {o['box']:.4f} "
                  f"(CFT 1.9534; raw {o['box_raw']:.4f})", flush=True)
        return out


    def cli(argv):
        for N in map(int, argv[1:]):
            run(N)
    return namespace(locals(), globals())


@section
def imp_admixture():
    """Test of a two-admixture theory of the lattice spin-2 field W (ED, GZWH, N = 5-9).
  W  propto  int t + e2(N) int t2_rr + d(N) int d_tau^2 t,   Delta(t2) = 3 + delta (delta global, a CFT number)
t2 = rank-2-flavour, spin-2 Lorentz primary (twist ~ 2 Delta_phi); its radial-component kinematics from imp_spin2_kin
(rho1 = rho1_0 + 1/D, r1 sqrt(24D(D-1/2)) = (Do-3)(Do+2), rho2 closed form); d_tau^2 t leaves diagonal elements alone
and multiplies off-diagonal ones by (Delta E)^2 (4 for box phi - phi, Delta_t^2 for vacuum - t0).
Observables per N: rho1, rho2 (diagonal level-one/two ratios), vac = |<0|W|t0>/<phi|W|phi>|, box = |<box|W|phi>/<phi|W|phi>|.
usage: python imp_admixture.py
"""
    HERE = 'flow'
    import json
    import numpy as np
    from scipy.optimize import least_squares


    def cli(argv):

        D, Dt = 0.518942, 1.20954
        nrm = np.sqrt(24 * D * (D - 0.5))
        r1_0 = lambda a: 1 + a * (a - 3) / (6 * D)
        r2_0 = lambda a: (24*D**2 + 8*D*a**2 - 24*D*a - 12*D + a**4 - 6*a**3 + 7*a**2 + 6*a) / (12*D*(2*D - 1))
        b_0 = lambda a: a * (a - 1) / nrm
        r1_2 = lambda a: r1_0(a) + 1 / D
        r2_2 = lambda a: (24*D**2 + 8*D*a**2 - 24*D*a + 36*D + a**4 - 6*a**3 - 5*a**2 + 42*a - 24) / (12*D*(2*D - 1))
        b_2 = lambda a: (a - 3) * (a + 2) / nrm
        VAC = 0.99029
        Ns = [5, 6, 7, 8, 9]
        data = {}
        for N in Ns:
            o = store.get(os.path.join(HERE, f'imp_thermal_N{N}.json'))
            dg, ob, ov = o['diag'], o['offbox'], o['offvac']
            p = dg['phi']['W1']
            data[N] = dict(rho1=dg['dphi']['W1'] / p, rho2=dg['box']['W1'] / p, vac=abs(ov['W1'] / p), box=abs(ob['W1'] / p))


        def model(dl, e2, d):
            a2 = 3 + dl
            den = 1 + e2
            return dict(rho1=(r1_0(Dt) + e2 * r1_2(a2)) / den, rho2=(r2_0(Dt) + e2 * r2_2(a2)) / den,
                        vac=VAC * (1 + d * Dt ** 2) / den, box=(b_0(Dt) * (1 + 4 * d) + e2 * b_2(a2)) / den)


        keys = ('rho1', 'rho2', 'vac', 'box')
        def resid(p):
            dl = p[0]; out = []
            for i, N in enumerate(Ns):
                m = model(dl, p[1 + 2 * i], p[2 + 2 * i])
                out += [(m[k] - data[N][k]) / max(0.05, abs(data[N][k])) for k in keys]
            return np.array(out)


        best = None
        for dl0 in (-0.05, 0.0, 0.02, 0.05, 0.1, 0.2):
            p0 = [dl0] + [0.3, -0.1] * len(Ns)
            r = least_squares(resid, p0)
            if best is None or r.cost < best.cost:
                best = r
        p = best.x
        print(f"global delta = Delta(t2) - 3 = {p[0]:+.4f}   (fit cost {best.cost:.2e}, 20 data, 11 parameters)")
        print(" N    e2       d      | rho1 lat/model   rho2 lat/model   vac lat/model   box lat/model")
        E2, Dd = [], []
        for i, N in enumerate(Ns):
            e2, d = p[1 + 2 * i], p[2 + 2 * i]; E2.append(e2); Dd.append(d)
            m = model(p[0], e2, d)
            print(f"{N:2d}  {e2:+.4f}  {d:+.4f}  | " + "  ".join(f"{data[N][k]:.4f}/{m[k]:.4f}" for k in keys))
        lnN = np.log(Ns)
        for lab, arr, th in (('e2', E2, (3 + p[0] - Dt) / 2), ('d', Dd, 1.0)):
            arr = np.array(arr)
            if np.all(arr > 0) or np.all(arr < 0):
                sl = -np.polyfit(lnN, np.log(np.abs(arr)), 1)[0]
                print(f"{lab}(N) ~ N^-{sl:.3f}   (theory: {th:.3f})")
        store.put(os.path.join(HERE, 'imp_admixture.json'), dict(delta=p[0], e2=E2, d=Dd, data=data))
    return namespace(locals(), globals())


@section
def lsq_penalty():
    """L^2 = L_- L_+ (exact in the L_z = 0 sector) as one- and two-body terms in the onmodel/dmrg_jobs mode convention:
mode k = f N + jj, orbital m = jj - s, s = (N - 1)/2, L_+ = sum_{f, jj} sqrt((N-1-jj)(jj+1)) c+_{jj+1,f} c_{jj,f}.
L_- L_+ = sum ell_jj^2 n_{jj,f} - sum ell_jj ell_jj' c+_{jj,f} c+_{jj'+1,f'} c_{jj+1,f} c_{jj',f'}.
Two-body terms use the (k1, k2, k3, k4, v) = v c+_k1 c+_k2 c_k3 c_k4 format of onmodel.build_terms."""
    HERE = 'flow'
    import numpy as np


    def lsq_terms(N, F=4, lam=1.0):
        ell = np.sqrt([(N - 1 - jj) * (jj + 1) for jj in range(N - 1)])
        one = np.zeros(F * N)
        two = []
        for f in range(F):
            for jj in range(N - 1):
                one[f * N + jj] += lam * ell[jj] ** 2
                for f2 in range(F):
                    for jj2 in range(N - 1):
                        a, c, b, d = f * N + jj, f2 * N + jj2 + 1, f * N + jj + 1, f2 * N + jj2
                        if a == c or b == d:
                            continue
                        two.append((a, c, b, d, -lam * ell[jj] * ell[jj2]))
        return one, two
    return namespace(locals(), globals())


@section
def dmrg_box():
    """DMRG for the near-null descendant box(phi_3): the third L = 0 level of the P0 sector (env: quantum).

The sector (Q = 0, L_z = 0, P = -) holds phi_3, dphi_3 (L = 1), t3, the L = 2 member of the level-two
descendants, box phi_3, ...  A penalty lam L^2 = lam L_- L_+ (exact at L_z = 0, lsq_penalty.py; it commutes
with H to 1e-14) lifts every L > 0 level by lam L(L+1) and leaves the L = 0 spectrum alone. So the L = 0
levels phi_3 < t3 < box phi_3 are the three lowest states, found in sequence with orthogonal_to. As in
dmrg_jobs.run_job, every mode is shifted by -2 so the projected-out directions (eigenvalue 0) stay above.
For each state: E, <L^2>, <W>, <n_h>, and the couplings <state|W|phi_3>, <state|n_h|phi_3>.
CFT values (cpt_kin.r_k): box phi: <W>/<W>_phi = 0.52181, n_h/(<n_h>_phi - <n_h>_0) = 1.95335.
usage: python dmrg_box.py N h chi [lam] [sweeps] [tag] [nstates]   -> dmrg_box_out/N{N}_chi{chi}{tag}.json
(nstates = 2 stops after t3, which is all the phi-t-t3 coupling needs)
"""
    import os
    import sys
    import json
    import time
    import numpy as np

    HERE = 'flow'
    TW = os.path.join(HERE, '..', 'twist')
    OUT = os.path.join(HERE, 'dmrg_box_out')
    SHIFT = 2.0


    def main(N, h, chi, lam=0.04, sweeps=30, tag='', nstates=3):
        from dmrg import on_problem
        from dmrg import build_model, compress_mpo, run_dmrg
        from tenpy.models.model import MPOModel
        from tenpy.networks.mps import MPSEnvironment
        F = fuzzy.o3flow()
        lsq_terms = lsq_penalty().lsq_terms
        t0 = time.time()
        mode_m, charges, names, mods, terms, onsite, occ = on_problem('o3wf', N, 0.0, [[0], 0, [-1]], h)
        s = (N - 1) / 2
        assert all(abs(mode_m[f * N + jj] - (jj - s)) < 1e-12 for f in range(4) for jj in range(N))
        one, two = lsq_terms(N, lam=lam)
        model, order = build_model(mode_m, charges, names, list(terms) + two,
                                   [v + lam_one - SHIFT for v, lam_one in zip(onsite, one)], mods=mods)
        Hc = compress_mpo(model.H_MPO)
        mc = MPOModel(model.lat, Hc)
        one1, two1 = lsq_terms(N, lam=1.0)
        lmod, _ = build_model(mode_m, charges, names, two1, list(one1), mods=mods)
        Lc = compress_mpo(lmod.H_MPO)
        fm = F.FlowModel(N, cache=(), verbose=False)
        w, dmu = fm.wmode, fm.dmu
        tb = time.time() - t0
        print(f"N={N}: MPO chi {max(Hc.chi)} (L^2 MPO {max(Lc.chi)}), built in {tb:.0f}s", flush=True)

        def modes(arr):
            out = np.zeros(len(order))
            for pos, k in enumerate(order):
                out[k] = arr[pos]
            return out

        states, res = [], {}
        for lab in ('phi3', 't3', 'box')[:nstates]:
            t1 = time.time()
            E, psi, eng = run_dmrg(mc, order, occ, chi_max=chi, sweeps=sweeps, orthogonal_to=list(states))
            Ne = len(occ)
            n_d = modes(np.array(psi.expectation_value('N')))
            L2 = float(Lc.expectation_value(psi))
            r = dict(E=float(E) + SHIFT * Ne, L2=L2, W=float(w @ n_d), nh=float(dmu @ n_d),
                     trunc=float(eng.sweep_stats['max_trunc_err'][-1]), chi=int(max(psi.chi)),
                     sweeps=len(eng.sweep_stats['E']), t=time.time() - t1)
            if states:
                env = MPSEnvironment(psi, states[0])                     # <state| n_k |phi_3>
                n_o = modes(np.array(env.expectation_value('N')))
                r.update(W_phi=float(w @ n_o), nh_phi=float(dmu @ n_o),
                         ovl=[abs(complex(psi.overlap(q))) for q in states])
            res[lab] = r
            states.append(psi)
            print(f"  {lab}: E {r['E']:.10f} L2 {L2:.2e} W {r['W']:+.6f} nh {r['nh']:.6f} "
                  + (f"W_phi {r['W_phi']:+.6f} nh_phi {r['nh_phi']:+.6f} ovl {max(r['ovl']):.1e} " if 'W_phi' in r else '')
                  + f"trunc {r['trunc']:.1e} chi {r['chi']} sweeps {r['sweeps']} [{r['t']:.0f}s]", flush=True)
        out = dict(N=N, h=h, chi=chi, lam=lam, mpo_chi=int(max(Hc.chi)), states=res, t=time.time() - t0)
        None
        store.put(os.path.join(OUT, f'N{N}_chi{chi}{tag}.json'), out)
        return out


    def cli(argv):
        N, h, chi = int(argv[1]), float(argv[2]), int(argv[3])
        lam = float(argv[4]) if len(argv) > 4 else 0.04
        sweeps = int(argv[5]) if len(argv) > 5 else 30
        tag = argv[6] if len(argv) > 6 else ''
        nst = int(argv[7]) if len(argv) > 7 else 3
        main(N, h, chi, lam, sweeps, tag, nst)
    return namespace(locals(), globals())


@section
def dmrg_box_analyze():
    """Couplings of box(phi_3) and t3 to phi_3 vs N: ED (cpt_lattice_N4..9, Lanczos Ritz pairs) and DMRG
(dmrg_box_out/N*_chi*.json, with <n_h>_0 and the descendant spacing u from dmrg_out at the same h).
Ratios: W coupling / <W>_phi, n_h coupling / (<n_h>_phi - <n_h>_0); CFT 0.52181 and 1.95335 for box phi.
usage: python dmrg_box_analyze.py       -> table + dmrg_box_summary.json
"""
    import os
    import re
    import glob
    import json
    import numpy as np

    HERE = 'flow'
    CFT = dict(W=0.52181, nh=1.95335)


    def ed_rows():
        rows = {}
        for N in range(4, 10):
            L = store.get(os.path.join(HERE, f'cpt_lattice_N{N}.json'))
            S, sp = L['states'], L['spectral']['phi3']
            th, aW, ah = np.array(sp['ritz']), np.array(sp['amp_W']), np.array(sp['amp_nh'])
            dn = S['phi3']['Eh'] - S['vac']['Eh']
            rW, rh = aW / S['phi3']['Eg'], ah / dn
            j1 = [j for j in range(len(th)) if abs(rh[j]) > 1e-3]
            jb = min(j1, key=lambda j: th[j])
            j3 = [j for j in range(len(th)) if abs(rW[j]) > 1e-3 and abs(rh[j]) < 1e-3]
            jt = min(j3, key=lambda j: th[j])
            rows[N] = dict(src='ED', box_W=abs(rW[jb]), box_nh=abs(rh[jb]), box_prod=rW[jb] * rh[jb],
                           D_box=0.518942 + th[jb], t3_W=abs(rW[jt]), D_t3=0.518942 + th[jt])
        return rows


    def dmrg_rows():
        """per N: box(phi) couplings from the highest-chi run that has the box state, t3 coupling from the
    highest-chi run that has t3 (the chi = 1200 runs stop after t3). Also every t3 value by chi."""
        runs = {}
        for f in store.glob(os.path.join(HERE, 'dmrg_box_out', 'N*_chi*.json')):
            b = os.path.basename(f)
            if '_lam' in b and '_lam0.04' not in b:          # penalty tests at N = 8: keep the production lam
                continue
            r = store.get(f)
            runs.setdefault(r['N'], []).append(r)
        rows = {}
        for N, rs in runs.items():
            ref = {}
            for g in store.glob(os.path.join(HERE, 'dmrg_out', f"N{N}_h{rs[0]['h']:.9f}_*_chi*.json")):
                m = re.search(r'_([a-z0-9]+)_chi(\d+)\.json', g)
                ref.setdefault(m.group(1), {})[int(m.group(2))] = store.get(g)
            pick = lambda st: ref[st][max(ref[st])] if st in ref else None
            vac, dphi = pick('vac'), pick('dphi3')
            withbox = [r for r in rs if 'box' in r['states']]
            rt = max(rs, key=lambda r: r['chi'])
            row = dict(src=f"DMRG chi {rt['chi']}", chi=rt['chi'],
                       t3_by_chi={r['chi']: abs(r['states']['t3']['W_phi'] / r['states']['phi3']['W']) for r in rs},
                       t3_W=abs(rt['states']['t3']['W_phi'] / rt['states']['phi3']['W']))
            u = (dphi['E'] - rt['states']['phi3']['E']) if dphi else np.nan
            row['D_t3'] = (rt['states']['t3']['E'] - vac['E']) / u if vac else np.nan
            row['D_phi'] = (rt['states']['phi3']['E'] - vac['E']) / u if vac else np.nan
            row['L2'] = [rt['states'][k]['L2'] for k in rt['states']]
            row['trunc'] = [rt['states'][k]['trunc'] for k in rt['states']]
            row['W_phi3_check'] = (rt['states']['phi3']['W'], pick('phi3')['W'] if pick('phi3') else np.nan)
            if withbox and vac is not None:
                rb = max(withbox, key=lambda r: r['chi'])
                p, b = rb['states']['phi3'], rb['states']['box']
                dn = p['nh'] - vac['nh']
                row.update(box_chi=rb['chi'], box_W=abs(b['W_phi'] / p['W']), box_nh=abs(b['nh_phi'] / dn),
                           box_prod=(b['W_phi'] / p['W']) * (b['nh_phi'] / dn),
                           D_box=(b['E'] - vac['E']) / (dphi['E'] - p['E']) if dphi else np.nan)
            else:
                row.update(box_chi=None, box_W=np.nan, box_nh=np.nan, box_prod=np.nan, D_box=np.nan)
            rows[N] = row
        return rows


    def cli(argv):
        E, D = ed_rows(), dmrg_rows()
        print(" N  source         box: W/CFT  nh/CFT  prod/CFT  Delta     t3: C3/c_phi Delta    checks")
        allrows = {}
        for N in sorted(set(E) | set(D)):
            for r in ([E[N]] if N in E else []) + ([D[N]] if N in D else []):
                c3 = r['t3_W'] * 2 / np.sqrt(3 / 5)
                chk = '' if r['src'] == 'ED' else (f"L2 {max(r['L2']):.0e} trunc {max(r['trunc']):.0e} D_phi {r['D_phi']:.5f} "
                                                   f"W_phi3 {r['W_phi3_check'][0]:.5f}/{r['W_phi3_check'][1]:.5f} "
                                                   + 'C3 by chi ' + ', '.join(f"{c}: {v * 2 / np.sqrt(3 / 5):.4f}"
                                                                              for c, v in sorted(r['t3_by_chi'].items())))
                print(f"{N:2d}  {r['src']:13s}  {r['box_W'] / CFT['W']:.4f}  {r['box_nh'] / CFT['nh']:.4f}  "
                      f"{r['box_prod'] / (CFT['W'] * CFT['nh']):.4f}   {r['D_box']:.4f}    {c3:.4f}   {r['D_t3']:.4f}   {chk}")
                allrows.setdefault(N, []).append(r)
        store.put(os.path.join(HERE, 'dmrg_box_summary.json'), {str(k): v for k, v in allrows.items()})
    return namespace(locals(), globals())


@section
def box_converge():
    """Task 2a: converge the individual box(phi) matrix elements of W and of n_h (not only their product F).

Data: ED N = 4-9 (cpt_lattice_N*.json via dmrg_box_analyze.ed_rows) and DMRG N >= 10 (dmrg_box_out, dmrg_out via
dmrg_box_analyze.dmrg_rows). Each element is a fraction of its conformal value (W: 0.52181, n_h: 1.95335).
mode 'jobs': writes dmrg_box_out/run_box_converge.sh, the DMRG runs that extend the full three-state box data to
    N = 14-16 at chi = 1200 and 1600, plus the vac / phi3 / dphi3 references (dmrg_resp.py) at the same h. h(N) for
    N >= 10 continues the ED differences geometrically (ratio 0.8182), as in rg_eig.HEXT. The script waits for AC
    power and is NOT started by this file.
mode 'analyze': chi convergence per N, then N -> infinity for W, n_h and their product, with
    (a) O_inf + A N^-w, w in {0.5, 0.85, 1}, windows N >= 6 and N >= 8;
    (b) the mixing-model shape of the W element, (1 + 4 d(N)) b0 + e2(N) b2 over (1 + e2(N)), with e2, d continued
        as power laws fitted at N = 5-9 (imp_admixture.json) -- a prediction, not a fit, for N >= 10.
usage: python box_converge.py jobs | analyze    -> run_box_converge.sh | box_converge_summary.json
"""
    import os
    import sys
    import json
    import numpy as np
    DBA = dmrg_box_analyze()

    HERE = 'flow'
    CFT = dict(W=0.52181, nh=1.95335)
    HEXT = {10: -0.09698791, 11: -0.09690392, 12: -0.0968352, 13: -0.09677897}
    RATIO = 0.8182


    def h_of(N):
        h = dict(HEXT)
        for n in range(14, N + 1):
            h[n] = h[n - 1] + (h[n - 1] - h[n - 2]) * RATIO
        return round(h[N], 8)


    def jobs():
        lines = ['#!/bin/zsh',
                 '# Task 2a: full three-state box(phi) DMRG at N = 14-16 and their references. AC power only.',
                 f'cd {HERE}', 'PY=~/anaconda3/envs/quantum/bin/python',
                 'onac() { pmset -g batt | grep -q "AC Power"; }',
                 'export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 VECLIB_MAXIMUM_THREADS=5 MKL_NUM_THREADS=5']
        for N in (14, 15, 16):
            h = h_of(N)
            for chi in (1200, 1600):
                lines.append('until onac; do sleep 30; done')
                lines.append(f'$PY dmrg_resp.py run {N} {h} {chi} vac,phi3,dphi3 > dmrg_out/ref_N{N}_chi{chi}.log 2>&1')
                lines.append(f'$PY dmrg_box.py {N} {h} {chi} 0.04 30 _full 3 > dmrg_box_out/N{N}_chi{chi}_full.log 2>&1')
        lines.append('echo "$(date "+%F %T") box_converge done" >> dmrg_box_out/run_box_converge.log')
        p = os.path.join(HERE, 'dmrg_box_out', 'run_box_converge.sh')
        store.put(p, '\n'.join(lines) + '\n')
        os.chmod(p, 0o755)
        print(f'wrote {p} (not started); h(N): ' + ', '.join(f'{N}: {h_of(N)}' for N in (14, 15, 16)))


    def extrap(Ns, v, lo):
        out = []
        for w in (0.5, 0.85, 1.0):
            m = (Ns >= lo) & np.isfinite(v)
            if m.sum() < 3:
                continue
            A = np.vstack([np.ones(m.sum()), Ns[m] ** (-w)]).T
            out.append(np.linalg.lstsq(A, v[m], rcond=None)[0][0])
        return (float(np.median(out)), float((max(out) - min(out)) / 2)) if out else (np.nan, np.nan)


    def model_W(Ns):
        A = store.get(os.path.join(HERE, 'imp_admixture.json'))
        D, Dt = 0.518942, 1.20954
        nrm = np.sqrt(24 * D * (D - 0.5))
        b0, b2 = Dt * (Dt - 1) / nrm, (3 + A['delta'] - 3) * (3 + A['delta'] + 2) / nrm
        n0 = np.arange(5, 10)
        e2, d = np.array(A['e2']), np.array(A['d'])
        pe = -np.polyfit(np.log(n0), np.log(np.abs(e2)), 1)[0]
        pd = -np.polyfit(np.log(n0), np.log(np.abs(d)), 1)[0]
        e2N = lambda N: e2[-1] * (9 / N) ** pe if N > 9 else e2[N - 5]
        dN = lambda N: d[-1] * (9 / N) ** pd if N > 9 else d[N - 5]
        return {int(N): float(((1 + 4 * dN(N)) * b0 + e2N(N) * b2) / (1 + e2N(N)) / b0) for N in Ns if N >= 5}


    def analyze():
        rows = {**DBA.dmrg_rows(), **DBA.ed_rows()}           # exact ED wins where both exist (N = 8, 9)
        Ns = np.array(sorted(rows))
        tab = {o: np.array([rows[N]['box_' + o] / CFT[o] if o != 'prod' else rows[N]['box_prod'] / (CFT['W'] * CFT['nh'])
                            for N in Ns]) for o in ('W', 'nh', 'prod')}
        out = dict(N=Ns.tolist(), fractions={o: v.tolist() for o, v in tab.items()},
                   chi={int(N): rows[N].get('box_chi') for N in Ns}, model_W=model_W(Ns), extrap={})
        for o, v in tab.items():
            out['extrap'][o] = {f'N>={lo}': extrap(Ns, np.abs(v), lo) for lo in (6, 8)}
        store.put(os.path.join(HERE, 'box_converge_summary.json'), out)
        print(' N   chi    W/CFT    n_h/CFT  prod/CFT   model W/CFT')
        for i, N in enumerate(Ns):
            print(f"{N:2d}  {str(out['chi'][int(N)]):>5}  {tab['W'][i]:.4f}   {tab['nh'][i]:.4f}   {tab['prod'][i]:.4f}    "
                  f"{out['model_W'].get(int(N), float('nan')):.4f}")
        for o, e in out['extrap'].items():
            print(f'{o}: ' + '  '.join(f'{k}: {v[0]:.4f} +- {v[1]:.4f}' for k, v in e.items()))


    def cli(argv):
        jobs() if argv[1] == 'jobs' else analyze()
    return namespace(locals(), globals())


@section
def imp_spin2_kin_core():
    """imp_spin2_kin_core"""
    HERE = 'flow'
    import sympy as sp
    D, Do = sp.symbols('D Do', positive=True)
    y = sp.symbols('y1:4'); w = sp.symbols('w1:4'); x = sp.symbols('x1:4'); t = sp.symbols('t')
    dot = lambda u, v: sum(p * q for p, q in zip(u, v))


    def ser(expr, order):
        return sp.expand(sp.series(expr, t, 0, order + 1).removeO().subs(t, 1))


    def deg_trunc(expr, ymax, wmax):
        P = sp.Poly(sp.expand(expr), *y, *w)
        return sum(c * sp.prod([v ** e for v, e in zip(list(y) + list(w), m)]) for m, c in P.terms()
                   if sum(m[:3]) <= ymax and sum(m[3:]) <= wmax)


    def sphere_avg(expr):
        P = sp.Poly(sp.expand(expr), *x)
        out = 0
        df = lambda m: sp.factorial2(2 * m - 1) if m > 0 else 1
        for m, c in P.terms():
            if any(e % 2 for e in m):
                continue
            n = [e // 2 for e in m]
            out += c * df(n[0]) * df(n[1]) * df(n[2]) / sp.factorial2(2 * sum(n) + 1)
        return sp.simplify(out)


    def kin(l, Dval=None, Doval=None):
        nonlocal D, Do
        D0, Do0 = D, Do
        if Dval is not None:
            D, Do = Dval, Doval
        try:
            return _kin(l)
        finally:
            D, Do = D0, Do0


    def _kin(l):
        yt, wt = [t * c for c in y], [t * c for c in w]
        # expansions (x on the unit sphere): each to second order in its vector
        Ay = ser((1 - 2 * dot(x, yt) + dot(yt, yt)) ** (-(Do - l) / 2), 2)
        Aw = ser((1 - 2 * dot(x, wt) + dot(wt, wt)) ** (-(Do - l) / 2), 2)
        Qp = ser((1 - 2 * dot(yt, w) + dot(yt, yt) * dot(w, w)) ** (-(2 * D - Do + l) / 2), 2)
        Qp = deg_trunc(Qp, 2, 2)
        if l == 0:
            S = 1
        else:
            iy = ser((1 - 2 * dot(x, yt) + dot(yt, yt)) ** (-1), 2)
            iw = ser((1 - 2 * dot(x, wt) + dot(wt, wt)) ** (-1), 2)
            Z = [deg_trunc((y[i] - x[i] * dot(y, y)) * iy - (w[i] - x[i]) * iw, 2, 2) for i in range(3)]
            xZ = dot(x, Z)
            S = deg_trunc(sp.expand(xZ ** 2 - dot(Z, Z) / 3), 2, 2)
        prod = deg_trunc(sp.expand(deg_trunc(sp.expand(S * Ay), 2, 2) * Aw), 2, 2)
        prod = deg_trunc(sp.expand(prod * Qp), 2, 2)
        zero = {**{v: 0 for v in y}, **{v: 0 for v in w}}
        G0 = sphere_avg(prod.subs(zero))
        c11 = sphere_avg(sp.diff(prod, y[2], w[2]).subs(zero))
        lap = lambda f, vs: sum(sp.diff(f, v, 2) for v in vs)
        c22 = sphere_avg(lap(lap(prod, y), w).subs(zero))
        c20 = sphere_avg(lap(prod.subs({v: 0 for v in w}), y).subs({v: 0 for v in y}))
        rho1 = sp.factor(sp.simplify(c11 / (2 * D) / G0))
        rho2 = sp.factor(sp.simplify(c22 / (24 * D * (D - sp.Rational(1, 2))) / G0))
        r1 = sp.factor(sp.simplify(c20 / G0))                  # still to be divided by sqrt(24 D (D - 1/2))
        return G0, rho1, rho2, r1
    return namespace(locals(), globals())


@section
def amp_kin_extra():
    """Descendant kinematics for the withheld-observable test of task 2: the helicity-n member (P_+)^n |D> of level n.

V = int_{S^2} dOmega x^mu x^nu ... O (radial component of a spin-l primary, l = 0 or 2; l = 0 is a scalar), dimension Do.
With the null vectors y' = c (1, -i, 0) and w = c' (1, i, 0) one has y'.K = c K_-, w.P = c' P_+, y'^2 = w^2 = 0, and
    G(y', w) = <D| e^{c K_-} V e^{c' P_+} |D> / <D|V|D> = sum_n (c c')^n / (n!)^2 <D|K_-^n V P_+^n|D> / <D|V|D>,
because V commutes with L_z. The same function for V -> 1 is (1 - 4 c c')^(-D), so the normalized diagonal element is
    ratio_n = [c^n c'^n] G  /  [(c c')^n] (1 - 4 c c')^(-D) = [c^n c'^n] G / ((D)_n 4^n / n!).
n = 1 is the level-one (L = 1) ratio and must equal rho1 of imp_spin2_kin_core.kin (rotation invariance), which is the
built-in check. n = 2 is the spin-two level-two descendant, never used in any fit (helicity 2 exists only at L = 2).
usage: python amp_kin_extra.py      -> self-checks, then the n = 2 values for the O(3) inputs
"""
    HERE = 'flow'
    import sympy as sp
    KC = imp_spin2_kin_core()

    c, cp = sp.symbols('c cp')


    def generating(l, Dval, Doval):
        """the truncated product of _kin (degree <= 2 in y' and in w), with numeric D, Do."""
        y, w, x, t = KC.y, KC.w, KC.x, KC.t
        dot = KC.dot
        D, Do = Dval, Doval
        yt, wt = [t * q for q in y], [t * q for q in w]
        Ay = KC.ser((1 - 2 * dot(x, yt) + dot(yt, yt)) ** (-sp.Rational(1, 1) * (Do - l) / 2), 2)
        Aw = KC.ser((1 - 2 * dot(x, wt) + dot(wt, wt)) ** (-sp.Rational(1, 1) * (Do - l) / 2), 2)
        Qp = KC.deg_trunc(KC.ser((1 - 2 * dot(yt, w) + dot(yt, yt) * dot(w, w)) ** (-(2 * D - Do + l) / 2), 2), 2, 2)
        if l == 0:
            S = 1
        else:
            iy = KC.ser((1 - 2 * dot(x, yt) + dot(yt, yt)) ** (-1), 2)
            iw = KC.ser((1 - 2 * dot(x, wt) + dot(wt, wt)) ** (-1), 2)
            Z = [KC.deg_trunc((y[i] - x[i] * dot(y, y)) * iy - (w[i] - x[i]) * iw, 2, 2) for i in range(3)]
            S = KC.deg_trunc(sp.expand(dot(x, Z) ** 2 - dot(Z, Z) / 3), 2, 2)
        prod = KC.deg_trunc(sp.expand(KC.deg_trunc(sp.expand(S * Ay), 2, 2) * Aw), 2, 2)
        return KC.deg_trunc(sp.expand(prod * Qp), 2, 2)


    def helicity_ratio(l, Dval, Doval, n):
        y, w = KC.y, KC.w
        prod = generating(l, Dval, Doval)
        zero = {**{v: 0 for v in y}, **{v: 0 for v in w}}
        G0 = KC.sphere_avg(prod.subs(zero))
        sub = {y[0]: c, y[1]: -sp.I * c, y[2]: 0, w[0]: cp, w[1]: sp.I * cp, w[2]: 0}
        poly = sp.Poly(sp.expand(prod.subs(sub)), c, cp)
        coef = KC.sphere_avg(sp.expand(poly.coeff_monomial(c ** n * cp ** n)))
        norm = sp.rf(Dval, n) * 4 ** n / sp.factorial(n)
        return complex(sp.N(coef / G0 / norm))


    def selfcheck(D=0.518942):
        """n = 1 against rho1 of kin(); Do -> 0 (identity) gives 1."""
        out = []
        for l, Do in ((0, 1.20954), (0, 1.59488), (2, 3.0148), (2, 3.0)):
            h1 = helicity_ratio(l, D, Do, 1)
            r1 = complex(sp.N(KC.kin(l, D, Do)[1]))
            out.append((l, Do, h1, r1))
        out.append((0, 1e-9, helicity_ratio(0, D, 1e-9, 2), 1.0))
        return out


    def cli(argv):
        for l, Do, a, b in selfcheck():
            print(f'l={l} Do={Do}: helicity ratio {a.real:+.8f}{a.imag:+.1e}i   reference {b.real:+.8f}')
        D, Dt, delta = 0.518942, 1.20954, 0.014785
        print('n = 2, t (l = 0, Do = Dt):', helicity_ratio(0, D, Dt, 2).real)
        print('n = 2, t2 (l = 2, Do = 3 + delta):', helicity_ratio(2, D, 3 + delta, 2).real)
    return namespace(locals(), globals())


@section
def curv_families():
    """Task 3b: the heavy - light curvature b2_hl of the production field W split by intermediate conformal family, with a
measured (not assumed) remainder.

mode 'run': at g = 0, h_c(N), the Lanczos decomposition of Q W|phi> and the overlaps with Q n_h|phi>, for phi_3 (P0)
    and phi_+ (V0), with more steps than cpt_lattice.py (default 160), plus the exact second-order totals from CG
    solves (cpt_lattice.spectral). -> curv_ritz_N{N}.json
mode 'analyze': each converged Ritz state n (residual < 1e-6, |<n|W|phi>| > 1e-6 |<phi|W|phi>|) is labelled by
  * flavour spin j of its O(3) multiplet, from Wigner-Eckart: the same multiplet appears in the phi_3 and the phi_+
    decompositions with |a_+|^2/|a_3|^2 = CG2[j][1]/CG2[j][0] = 1/4 (j = 1), 2/3 (j = 3); j = 2 has no phi_3 partner;
    n_h, a singlet, can only reach j = 1 (a consistency check);
  * family and level k: ladders theta = theta_primary + 2 k s, with the lattice spacing s from box(phi) (j = 1) and
    new primaries opened by the lowest unassigned state;
  and its tt and ts contributions use the normalization of cpt_compare.lattice_parts (kappa, a1_hl from cpt_scheme.json).
  Per scheme (Ising: light phi_3; XY: light phi_+) it reports the phi family by k against the CPT terms
  r_k(Dt)^2/(2k) and r_k(Dt) r_k(Ds)/(2k) (cpt_kin), the t3 family (compare c3_extract), the j = 2 family and other
  primaries, and remainder = exact total - assigned sum.
usage: python curv_families.py run N [nsteps]      -> curv_ritz_N{N}.json
       python curv_families.py analyze             -> curv_families_summary.json, table
"""
    import os
    import sys
    import json
    import glob
    import numpy as np

    HERE = 'flow'
    CG = {1: 0.25, 3: 2 / 3}                                # CG2[j][1] / CG2[j][0]


    def run(N, nsteps=160):
        F = fuzzy.o3flow()
        CL = cpt_lattice()
        fm = F.FlowModel(N, cache=('S0', 'P0', 'P1', 'V0', 'V1'), verbose=False)
        hc = store.get(os.path.join(HERE, f'partA_N{N}.json'))['h_c']
        E0, U0 = fm.levels('S0', hc, 0.0, k=1, vecs=True)
        nh0 = float(U0[:, 0] @ (fm.sector('S0')['nh'] * U0[:, 0]))
        out = dict(N=N, h_c=hc, nsteps=nsteps, nh_vac=nh0)
        for st, s0, s1 in (('phi3', 'P0', 'P1'), ('phipm', 'V0', 'V1')):
            E, U = fm.levels(s0, hc, 0.0, k=1, vecs=True)
            E1 = fm.levels(s1, hc, 0.0, k=1)
            psi, u = U[:, 0], float(E1[0] - E[0])
            sp_ = CL.spectral(fm, s0, hc, float(E[0]), u, psi, nsteps)
            sec = fm.sector(s0)
            out[st] = dict(E=float(E[0]), u=u, Eg=float(psi @ (sec['nw'] * psi)), Eh=float(psi @ (sec['nh'] * psi)), **sp_)
            print(f"N={N} {st}: {len(sp_['ritz'])} Ritz pairs, second_W {sp_['second_W']:+.6e}, mixed {sp_.get('mixed', 0):+.6e}",
                  flush=True)
        store.put(os.path.join(HERE, f'curv_ritz_N{N}.json'), out)


    def states(R, key):
        S = R[key]
        th, aW, ah, rs = (np.array(S[k]) for k in ('ritz', 'amp_W', 'amp_nh', 'resid'))
        keep = (rs < 1e-6) & (np.abs(aW) > 1e-6 * abs(S['Eg']))
        return [dict(theta=float(t), aW=float(w), ah=float(h)) for t, w, h in zip(th[keep], aW[keep], ah[keep])]


    def classify(P, V, tol=2e-3):
        """flavour spin j for every state of the phi_3 (P) and phi_+ (V) lists."""
        for s in P + V:
            s['j'] = None
        for p in P:
            partners = [v for v in V if abs(v['theta'] - p['theta']) < tol * max(1, p['theta'])]
            if not partners:
                p['j'] = 'unmatched'
                continue
            v = min(partners, key=lambda q: abs(q['theta'] - p['theta']))
            R = (v['aW'] / p['aW']) ** 2
            j = min(CG, key=lambda jj: abs(np.log(R / CG[jj])))
            p['j'] = v['j'] = j if abs(np.log(R / CG[j])) < 0.3 else f'ambiguous(R={R:.3f})'
            p['R'] = v['R'] = float(R)
        for v in V:
            if v['j'] is None:
                v['j'] = 2
        return P, V


    def ladders(states_j, s_spacing, start=None, tol=0.15):
        """assign (family index, k) along theta = theta_prim + 2 k s; start = theta of a known primary (phi: 0)."""
        fams = [] if start is None else [start]
        for s in sorted(states_j, key=lambda q: q['theta']):
            best = None
            for fi, t0 in enumerate(fams):
                k = round((s['theta'] - t0) / (2 * s_spacing))
                if k >= 0 and abs(s['theta'] - t0 - 2 * k * s_spacing) < tol:
                    best = (fi, k)
                    break
            if best is None:
                fams.append(s['theta'])
                best = (len(fams) - 1, 0)
            s['fam'], s['k'] = best
        return fams


    def analyze():
        CC = cpt_compare()
        r_k = cpt_kin().r_k
        sch = store.get(os.path.join(HERE, 'cpt_scheme.json'))
        out = {}
        for fn in sorted(store.glob(os.path.join(HERE, 'curv_ritz_N*.json'))):
            R = store.get(fn); N = R['N']
            P, V = classify(states(R, 'phi3'), states(R, 'phipm'))
            j1 = [s for s in P if s['j'] == 1]
            box = min(j1, key=lambda s: s['theta']) if j1 else None
            sp = box['theta'] / 2 if box else 1.0
            for lst in (P, V):
                ladders([s for s in lst if s['j'] == 1], sp, start=0.0)
                ladders([s for s in lst if s['j'] == 3], sp)
                ladders([s for s in lst if s['j'] == 2], sp)
                for s in lst:
                    s.setdefault('fam', None); s.setdefault('k', None)
            nh_bad = [s for s in P + V if s['j'] != 1 and abs(s['ah']) > 1e-6 * max(1, abs(R['phi3']['Eh'] - R['nh_vac']))]
            res = dict(spacing=sp, n_states=dict(P=len(P), V=len(V)), nh_on_non_vectors=len(nh_bad), schemes={})
            for lab, light, heavy, ql, qh in (('Ising', 'phi3', 'phipm', 0, 1), ('XY', 'phipm', 'phi3', 1, 0)):
                kap, a1 = sch[str(N)][lab]['kappa'], sch[str(N)][lab]['a1_hl']
                lst = {'phi3': P, 'phipm': V}
                parts, tot = {}, np.zeros(2)
                for st, sgn in ((heavy, +1), (light, -1)):
                    u = R[st]['u']; norm = u * a1 ** 2
                    tot += sgn * np.array([R[st]['second_W'] / norm, 2 * kap * R[st].get('mixed', R[st]['mixed_ritz']) / norm])
                    for s in lst[st]:
                        den = -s['theta'] * u
                        key = f"j{s['j']}_f{s['fam']}_k{s['k']}"
                        c = parts.setdefault(key, np.zeros(2))
                        c += sgn * np.array([s['aW'] ** 2 / den / norm, 2 * kap * s['aW'] * s['ah'] / den / norm])
                assigned = np.sum([v for v in parts.values()], axis=0) if parts else np.zeros(2)
                cl, ch = CC.CQ[ql], CC.CQ[qh]
                rt, rs_ = r_k(CC.Dp, CC.Dp, CC.Dt, 6), r_k(CC.Dp, CC.Dp, CC.Ds, 6)
                cpt_k = {k: dict(tt=float(-(ch + cl) * rt[k] ** 2 / (2 * k) / (ch - cl)),
                                 ts=float(2 * cl * CC.r * rt[k] * rs_[k] / (2 * k) / (ch - cl))) for k in range(1, 7)}
                res['schemes'][lab] = dict(parts={k: v.tolist() for k, v in parts.items()}, total=tot.tolist(),
                                           assigned=assigned.tolist(), remainder=(tot - assigned).tolist(), cpt_phi_k=cpt_k)
            out[N] = res
            print(f"N={N}: spacing s {sp:.3f}; states P {len(P)} V {len(V)}; n_h on non-vectors {len(nh_bad)}")
            for lab, r in res['schemes'].items():
                top = sorted(r['parts'].items(), key=lambda kv: -abs(kv[1][0]) - abs(kv[1][1]))[:6]
                print(f"   {lab}: total tt {r['total'][0]:+.4f} ts {r['total'][1]:+.4f} | remainder tt {r['remainder'][0]:+.2e} "
                      f"ts {r['remainder'][1]:+.2e} | " + ', '.join(f"{k} ({v[0]:+.4f},{v[1]:+.4f})" for k, v in top))
                print('      CPT phi family per k (tt, ts): ' +
                      ', '.join(f"k{k} ({v['tt']:+.4f},{v['ts']:+.4f})" for k, v in list(r['cpt_phi_k'].items())[:4]))
        store.put(os.path.join(HERE, 'curv_families_summary.json'), out)


    def cli(argv):
        if argv[1] == 'run':
            run(int(argv[2]), int(argv[3]) if len(argv) > 3 else 160)
        else:
            analyze()
    return namespace(locals(), globals())


@section
def amp_withheld():
    """Task 2b: test the operator-mixing model of W on an observable it was not fitted to.

The model (imp_admixture.py, fitted at N = 5-9 to rho1, rho2, vac and box of W) is
    W propto int t + e2(N) int t2_rr + d(N) int d_tau^2 t,   Delta(t2) = 3 + delta,
and its d_tau^2 t piece leaves diagonal elements unchanged. The withheld observable is the diagonal ratio of the
spin-two level-two descendant, K = 2 member (P_+)^2|phi>, the lowest state of the sector (Q, K, P) = (0, 2, -) for
phi_3 and (1, 2, +) for phi_+:
    R2 = <dd phi|W|dd phi> / <phi|W|phi>,  model: (h2(t) + e2 h2(t2)) / (1 + e2),  CFT: h2(t),
with h2 from amp_kin_extra.helicity_ratio(l, Dphi, Do, 2). The phi_+ ratio must equal the phi_3 one (Wigner-Eckart;
the admixtures are rank-2 tensors too), and the K = 1 ratio must reproduce the fitted rho1 (checks).
usage: python amp_withheld.py measure N     -> amp_withheld_N{N}.json  (ED at g = 0, h_c from partA_N{N}.json)
       python amp_withheld.py predict       -> amp_withheld_summary.json, table
"""
    import os
    import sys
    import json
    import numpy as np
    F = fuzzy.o3flow()

    HERE = 'flow'
    F.SECTORS.setdefault('P2', (0, 2, -1))
    F.SECTORS.setdefault('V2', (1, 2, +1))
    D_PHI, D_T = 0.518942, 1.20954


    def measure(N):
        fm = F.FlowModel(N, cache=('S0', 'P0', 'P1', 'P2', 'V0', 'V1', 'V2'), verbose=False)
        hc = store.get(os.path.join(HERE, f'partA_N{N}.json'))['h_c']
        e0 = fm.levels('S0', hc, 0.0)[0]
        out = dict(N=N, h_c=hc)
        for comp in ('P', 'V'):
            st = {}
            for k in (0, 1, 2):
                E, U = fm.levels(f'{comp}{k}', hc, 0.0, k=2, vecs=True)
                v = U[:, 0]
                st[k] = dict(E=float(E[0] - e0), E2=float(E[1] - e0), W=float(v @ (fm.sector(f'{comp}{k}')['nw'] * v)))
            u = st[1]['E'] - st[0]['E']
            out[comp] = dict(states=st, u=u, R1=st[1]['W'] / st[0]['W'], R2=st[2]['W'] / st[0]['W'],
                             level2_spacing=(st[2]['E'] - st[0]['E']) / u)
            print(f"N={N} {comp}: R1 {out[comp]['R1']:.5f}  R2 {out[comp]['R2']:.5f}  "
                  f"(E_dd - E_phi)/u {out[comp]['level2_spacing']:.4f} (2 if the K = 2 ground is dd phi)", flush=True)
        store.put(os.path.join(HERE, f'amp_withheld_N{N}.json'), out)


    def predict():
        AK = amp_kin_extra()
        A = store.get(os.path.join(HERE, 'imp_admixture.json'))
        dl = A['delta']
        h2_t = AK.helicity_ratio(0, D_PHI, D_T, 2).real
        h2_t2 = AK.helicity_ratio(2, D_PHI, 3 + dl, 2).real
        h1_t = AK.helicity_ratio(0, D_PHI, D_T, 1).real
        h1_t2 = AK.helicity_ratio(2, D_PHI, 3 + dl, 1).real
        rows = {}
        for i, N in enumerate((5, 6, 7, 8, 9)):
            p = os.path.join(HERE, f'amp_withheld_N{N}.json')
            if not store.exists(p):
                continue
            M = store.get(p)
            e2 = A['e2'][i]
            rows[N] = dict(R2_lat=M['P']['R2'], R2_lat_V=M['V']['R2'], R2_model=(h2_t + e2 * h2_t2) / (1 + e2), R2_cft=h2_t,
                           R1_lat=M['P']['R1'], R1_model=(h1_t + e2 * h1_t2) / (1 + e2),
                           level2_spacing=M['P']['level2_spacing'], e2=e2)
        out = dict(kin=dict(h2_t=h2_t, h2_t2=h2_t2, h1_t=h1_t, h1_t2=h1_t2, delta=dl), rows=rows)
        store.put(os.path.join(HERE, 'amp_withheld_summary.json'), out)
        print(f'kinematics: h2(t) {h2_t:.5f}  h2(t2) {h2_t2:.5f}  h1(t) {h1_t:.5f} h1(t2) {h1_t2:.5f}')
        print(' N   R2 lattice (phi3 / phi+)   model    CFT     | R1 lattice  model (fitted, check) | spacing')
        for N, r in rows.items():
            print(f"{N:2d}   {r['R2_lat']:.5f} / {r['R2_lat_V']:.5f}   {r['R2_model']:.5f}  {r['R2_cft']:.5f}  | "
                  f"{r['R1_lat']:.5f}  {r['R1_model']:.5f} | {r['level2_spacing']:.3f}")


    def cli(argv):
        if argv[1] == 'measure':
            for N in map(int, argv[2:]):
                measure(N)
        else:
            predict()
    return namespace(locals(), globals())


STEPS = {'o3flow': fuzzy.o3flow, 'gflow': fuzzy.gflow, 'partA_slopes': partA_slopes, 'cpt_lattice': cpt_lattice, 'cpt_scheme': cpt_scheme, 'cpt_kin': cpt_kin, 'cpt_compare': cpt_compare, 'imp_ops': imp_ops, 'imp_thermal': imp_thermal, 'imp_admixture': imp_admixture, 'lsq_penalty': lsq_penalty, 'dmrg_box': dmrg_box, 'dmrg_box_analyze': dmrg_box_analyze, 'box_converge': box_converge, 'imp_spin2_kin_core': imp_spin2_kin_core, 'amp_kin_extra': amp_kin_extra, 'curv_families': curv_families, 'amp_withheld': amp_withheld}
REPORT = [('cpt_scheme',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
