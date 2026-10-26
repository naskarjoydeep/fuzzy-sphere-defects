"""What is the Z2 twist line of the 3d Ising CFT?

The Ising model of Zhu et al. (V0, V1, h) = (4.75, 1, 3.16) in the basis psi_pm = (psi_up +- psi_dn)/sqrt2, with
half a unit of flux through the poles for psi_- only, so every Z2-odd operator is antiperiodic around the line.
Results: the defect spectrum by ED (N <= 14) and DMRG (N <= 20) in bulk and displacement units, the ratio of the
lowest even scalar to the lowest odd mode, the bulk-to-defect coefficient of sigma and the defect three-point
coefficients from the pole estimator.

usage: python ising_twist.py <step> [args]      (python ising_twist.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def run_defect():
    """Defect spectrum of the Z2 twist line on the fuzzy sphere, calibrated by the bulk stress tensor."""
    HERE = 'twist'
    import sys
    import time
    import numpy as np
    import scipy.sparse.linalg as sla
    from fuzzy import couplings_for, TwistBasis, interaction_terms, build_H_fast as build_H


    def cli(argv):

        V0, V1, h = 4.75, 1.0, 3.16          # Ising point of 2210.13482, pipeline normalisation
        def low(N, twisted, twoLz, z2, terms, k):
            b = TwistBasis(N, twisted, twoLz, z2)
            H = build_H(b, terms, h)
            if b.dim <= 2500:
                w = np.linalg.eigvalsh(H.toarray())[:k]
            else:
                w = np.sort(sla.eigsh(H, k=min(k, b.dim - 2), which='SA', tol=1e-12)[0])
            return w, b.dim

        Ns = [int(x) for x in argv[1:]] or [8, 10, 12]
        for N in Ns:
            t0 = time.time()
            G0, G1, _ = couplings_for(N, V0, V1)
            tb = interaction_terms(N, False, G0, G1)
            wv, _ = low(N, False, 0, +1, tb, 2)
            wt, _ = low(N, False, 4, +1, tb, 2)
            E0, gam = wv[0], (wt[0] - wv[0]) / 3            # T = lowest Z2-even state at L_z = 2
            td = interaction_terms(N, True, G0, G1)
            sect = {}
            dims = {}
            for (z2, tl) in ((+1, 0), (+1, 2), (+1, 4), (-1, 1), (-1, 3), (-1, 5)):
                sect[(z2, tl)], dims[(z2, tl)] = low(N, True, tl, z2, td, 8)
            ED = sect[(+1, 0)][0]
            D = {k: (v - ED) / gam for k, v in sect.items()}
            print(f"\nN={N}  (bulk gamma={gam:.4f}; defect E0-bulk E0 = {(ED-E0)/gam:.4f} in units of gamma;"
                  f"  {time.time()-t0:.0f}s; dims {dims})")
            lab = {(+1, 0): 'Z2+ s=0  ', (+1, 2): 'Z2+ s=1  ', (+1, 4): 'Z2+ s=2  ',
                   (-1, 1): 'Z2- s=1/2', (-1, 3): 'Z2- s=3/2', (-1, 5): 'Z2- s=5/2'}
            for k in lab:
                print(f"   {lab[k]}: " + ' '.join(f"{x:7.4f}" for x in D[k]), flush=True)
    return namespace(locals(), globals())


@section
def ising_levels():
    """ED level table of the Ising half-twist (V0, V1, h) = (4.75, 1, 3.16) for the level-diagram figure:
lowest levels in each (Z2, 2Lz) sector, and the mirror parity R_yz (psi_m -> psi_{-m}, i.e. spacetime parity 0+/0-)
in the s = 0 sector.  usage: python ising_levels.py N nlev  -> zgh_results/ising_levels_N{N}.json"""
    HERE = 'twist'
    import sys
    import json
    import time
    import numpy as np
    import scipy.sparse.linalg as sla
    from fuzzy import couplings_for, TwistBasis, interaction_terms, build_H_fast
    from fuzzy import mirror


    def cli(argv):

        N, k = int(argv[1]), int(argv[2])
        G0, G1, _ = couplings_for(N, 4.75, 1.0)
        td = interaction_terms(N, True, G0, G1)
        out = {}
        t0 = time.time()
        for two in range(0, 8):
            z2 = +1 if two % 2 == 0 else -1
            b = TwistBasis(N, True, two, z2)
            H = build_H_fast(b, td, 3.16)
            nk = k + 6 if two == 0 else k
            if b.dim <= 3000:
                w, v = np.linalg.eigh(H.toarray()); w, v = w[:nk], v[:, :nk]
            else:
                w, v = sla.eigsh(H, k=nk, which='SA', tol=1e-11); o = np.argsort(w); w, v = w[o], v[:, o]
            rec = {'E': w.tolist(), 'dim': b.dim}
            if two == 0:
                idx, sign = mirror(b)
                Rv = np.zeros_like(v); Rv[idx] = sign[:, None] * v
                # degenerate levels can mix R; diagonalize R inside near-degenerate clusters
                Rm = v.T @ Rv
                rec['R'] = np.round(np.diag(Rm), 4).tolist()
            out[str(two)] = rec
            print(f'N={N} 2Lz={two} dim {b.dim} [{time.time()-t0:.0f}s]', flush=True)
        store.put(os.path.join(HERE, f'zgh_results/ising_levels_N{N}.json'), out)
        E0 = out['0']['E'][0]; D = out['2']['E'][0] - E0
        for two, r in out.items():
            print(two, ' '.join(f'{2*(e-E0)/D:.3f}' for e in r['E']), ' R:', r.get('R', ''))
    return namespace(locals(), globals())


@section
def ising_amplitudes():
    """Ising half-twist amplitudes against Monte Carlo and bootstrap (lessons from arXiv:2610.31269).

Model: twist.py (normal ordered, (V0, V1, h) = (4.75, 1, 3.16)), psi_- twisted.  Density n^z = psi_+^+ psi_- + h.c.
(leading content sigma), evaluated at azimuth 0.
  b_{sigma psi_s} = |<psi_s| n^z(pi/2) |0_D>| / c_sigma,   c_sigma = |<sigma| n^z(theta) |0>| in the untwisted bulk
  (theta independent for a scalar primary; checked).  Exact free value b_{1/2} = sqrt(2/pi).
  Pole coefficients (arXiv:2610.31269 Eq. (C3)) on theta0 in {pi/12, pi/6, pi/4}, displacement-calibrated dimensions:
    C_{psi psi Dbar} = |<D| n^z |psi_1/2>| / |<psi_1/2| n^z |0_D>| (cos theta0)^{-(D_D - 2 D_psi)},   bootstrap 1.345
    C_{psi psibar s} = |<s| n^z |psi_1/2>| / |<psi_-1/2| n^z |0_D>| (cos theta0)^{-(D_s - 2 D_psi)}, bootstrap 0.95
usage: python ising_amplitudes.py N   -> analysis/ising_amplitudes/N<N>.json
"""
    import os
    import sys
    import json
    import time
    from math import pi
    import numpy as np

    HERE = 'fun/combined_2026_10_22/analysis'
    TW = os.path.join(HERE, '..', '..', '..', 'twist')
    from fuzzy import couplings_for, TwistBasis, interaction_terms, build_H_fast, orbital_ms
    from fuzzy import onebody, orb, lowest

    TH0 = np.array([pi / 12, pi / 6, pi / 4])
    EQ = np.array([pi / 3, pi / 2, 2 * pi / 3])


    def nz_melt(N, twisted, ba, va, bb, vb, dtwo, th):
        """<a| n^z(theta, 0) |b> where <a| has 2L_z larger by dtwo than |b>."""
        s = (N - 1) / 2
        mp, mm = orbital_ms(N, False), orbital_ms(N, twisted)
        out = np.zeros_like(th)
        for i, m in enumerate(mp):
            for j, mpr in enumerate(mm):
                if abs(2 * (m - mpr) - dtwo) < 1e-9:              # c+_{+,m} c_{-,m'} raises L_z by m - m'
                    out += onebody(ba, va, bb, vb, i, N + j) * orb(s, m, th) * orb(s, mpr, th)
                if abs(2 * (mpr - m) - dtwo) < 1e-9:              # c+_{-,m'} c_{+,m}
                    out += onebody(ba, va, bb, vb, N + j, i) * orb(s, mpr, th) * orb(s, m, th)
        return np.abs(out)


    def main(N):
        t0 = time.time()
        G0, G1, _ = couplings_for(N, 4.75, 1.0)
        tb, td = interaction_terms(N, False, G0, G1), interaction_terms(N, True, G0, G1)
        b0 = TwistBasis(N, False, 0, 1); bs = TwistBasis(N, False, 0, -1)
        e0, v0 = lowest(b0, build_H_fast(b0, tb, 3.16), 1); es, vs = lowest(bs, build_H_fast(bs, tb, 3.16), 1)
        csig = nz_melt(N, False, bs, vs[:, 0], b0, v0[:, 0], 0, EQ)
        S, V, E = {}, {}, {}
        for two, z2, k in ((0, 1, 2), (1, -1, 1), (-1, -1, 1), (3, -1, 1), (2, 1, 1)):
            b = TwistBasis(N, True, two, z2); w, v = lowest(b, build_H_fast(b, td, 3.16), k)
            S[two], V[two], E[two] = b, v, w
        E0, Dg = E[0][0], E[2][0] - E[0][0]
        dim = dict(psi12=2 * (E[1][0] - E0) / Dg, psi32=2 * (E[3][0] - E0) / Dg, s=2 * (E[0][1] - E0) / Dg)
        vac = V[0][:, 0]
        b12 = nz_melt(N, True, S[1], V[1][:, 0], S[0], vac, 1, np.array([pi / 2]))[0] / csig[1]
        b32 = nz_melt(N, True, S[3], V[3][:, 0], S[0], vac, 3, np.array([pi / 2]))[0] / csig[1]
        num = nz_melt(N, True, S[2], V[2][:, 0], S[1], V[1][:, 0], 1, TH0)
        den = nz_melt(N, True, S[1], V[1][:, 0], S[0], vac, 1, TH0)
        rD = num / den * np.cos(TH0) ** (-(2 - 2 * dim['psi12']))
        num = nz_melt(N, True, S[0], V[0][:, 1], S[1], V[1][:, 0], -1, TH0)
        den = nz_melt(N, True, S[-1], V[-1][:, 0], S[0], vac, -1, TH0)
        rs = num / den * np.cos(TH0) ** (-(dim['s'] - 2 * dim['psi12']))
        out = dict(N=N, csig=csig.tolist(), dims=dim, b_sigma_psi12=b12, b_sigma_psi32=b32,
                   C_psipsiD=dict(ratio=rD.tolist(), C=float(rD.mean()), err=float((rD.max() - rD.min()) / 2)),
                   C_psipsibar_s=dict(ratio=rs.tolist(), C=float(rs.mean()), err=float((rs.max() - rs.min()) / 2)))
        None
        store.put(os.path.join(HERE, 'ising_amplitudes', f'N{N}.json'), out)
        print(f"Ising N={N}: c_sigma flat {csig.min():.5f}-{csig.max():.5f}  b_sigma_psi1/2 {b12:.4f}  b_sigma_psi3/2 {b32:.4f}  "
              f"C_psipsiD {rD.mean():.3f}({(rD.max()-rD.min())/2:.3f})  C_psipsibar_s {rs.mean():.3f}({(rs.max()-rs.min())/2:.3f})  "
              f"dims psi {dim['psi12']:.4f} s {dim['s']:.3f} [{time.time()-t0:.0f}s]", flush=True)


    def cli(argv):
        main(int(argv[1]))
    return namespace(locals(), globals())


@section
def dmrg_tables():
    """Every number in the monodromy note's ED+DMRG tables and fits, read from saved data.

Sources: ../run_N8_12.log, ../run_N14.log (Ising ED, T-calibrated, 4 decimals); ../dmrg_results/*.json (DMRG);
o2_ed_precise.json (O(2) ED at full precision, from o2_ed_precise.py; the ../o2_*.log files round gaps to 3
decimals, which is too coarse for the fits).
"""
    import json
    import glob
    import os
    import re
    import numpy as np
    from scipy.optimize import curve_fit

    HERE = 'twist/note'
    TW = os.path.join(HERE, '..')
    R = [store.get(f) for f in store.glob(os.path.join(TW, 'dmrg_results', '*.json'))]
    ED2 = store.get(os.path.join(HERE, 'o2_ed_precise.json'))


    def get(kind, N, alpha, sec):
        c = [r for r in R if r['kind'] == kind and r['N'] == N and abs(r['alpha'] - alpha) < 1e-9
             and tuple(r['sector']) == tuple(sec)]
        return max(c, key=lambda r: r['chi']) if c else None


    # ------------------------------------------------------------------ Ising
    def ising_ed():
        """{N: {(Z2, 2s): [levels]}} in units of the bulk stress-tensor calibration."""
        out = {}
        for log in ('run_N8_12.log', 'run_N14.log'):
            N = None
            for line in store.text(os.path.join(TW, log)).splitlines(True):
                m = re.match(r"N=(\d+)", line)
                if m:
                    N = int(m.group(1)); out[N] = {}; continue
                m = re.match(r"\s+Z2([+-]) s=(\S+)\s*:\s*(.*)", line)
                if m and N:
                    s = m.group(2); twos = int(s[:-2]) if s.endswith('/2') else 2 * int(s)
                    out[N][(m.group(1), twos)] = [float(x) for x in m.group(3).split()]
        return out


    def ising_series():
        """T-calibrated series psi, psi32, psi52, D, s, po, tp on N = 8..20 (ED to 14, DMRG beyond)."""
        ed = ising_ed()
        S = {k: {} for k in ('psi', 'psi32', 'psi52', 'D', 's', 'po', 'tp')}
        for N, L in ed.items():
            S['psi'][N], S['psi32'][N], S['psi52'][N] = L[('-', 1)][0], L[('-', 3)][0], L[('-', 5)][0]
            S['D'][N], S['s'][N], S['po'][N], S['tp'][N] = L[('+', 2)][0], L[('+', 0)][1], L[('+', 0)][2], L[('+', 4)][0]
        meta = {}
        for N in sorted({r['N'] for r in R if r['kind'] == 'ising'}):
            E0 = get('ising', N, 0.5, (0, 1)); B0, BT = get('ising', N, 0, (0, 1)), get('ising', N, 0, (4, 1))
            if not (E0 and B0 and BT):
                continue
            gam = (BT['E'] - B0['E']) / 3
            for k, sec in (('psi', (1, -1)), ('psi32', (3, -1)), ('D', (2, 1)), ('tp', (4, 1))):
                r = get('ising', N, 0.5, sec)
                if r:
                    val = (r['E'] - E0['E']) / gam
                    if N in S[k] and abs(S[k][N] - val) > 2e-4:
                        raise RuntimeError(f"DMRG/ED mismatch {k} N={N}: {val} vs {S[k][N]}")
                    S[k].setdefault(N, val)
            trs = [x['trunc'] for x in R if x['kind'] == 'ising' and x['N'] == N]
            meta[N] = dict(chi=E0['chi'], mpo=E0['mpo_chi'], trunc=max(trs))
        return S, meta


    def ising_fits(S):
        N = np.array(sorted(S['D'])); D = np.array([S['D'][n] for n in N])
        fits = {}
        for nmin in (12, 14):
            m = N >= nmin
            (b, p), cov = curve_fit(lambda n, b, p: 2 + b * n**-p, N[m], D[m], p0=[1, 0.6])
            fits[nmin] = (p, np.sqrt(cov[1, 1]), np.abs(2 + b * N[m]**-p - D[m]).max())
        p = fits[12][0]
        ext = {}
        for k, ser in S.items():
            n = np.array(sorted(ser)); y = np.array([ser[x] for x in n]); d = np.array([S['D'][x] for x in n])
            for cal, yy in (('T', y), ('D', 2 * y / d)):
                x = n**-p
                ext[(k, cal)] = yy[-1] - (yy[-2] - yy[-1]) / (x[-2] - x[-1]) * x[-1]
        return fits, p, ext


    # ------------------------------------------------------------------ O(2)
    def o2E(N, a, q, k):
        if N <= 9 or (N == 10 and a == 0.0):
            return ED2.get(f"{N}_{a}_{q}_{k}")
        r = get('o2', N, a, (q, k))
        return r['E'] if r else None


    def o2_dcal(N, a, q, k):
        e0, ed, e = o2E(N, a, 0, 0), o2E(N, a, 0, 1), o2E(N, a, q, k)
        return 2 * (e - e0) / (ed - e0) if None not in (e0, ed, e) else np.nan


    def o2_stab(Ns=(6, 7, 8, 9, 10, 12, 14)):
        rows = {}
        for N in Ns:
            b0, bs, bd, bj = o2E(N, 0.0, 0, 0), o2E(N, 0.0, 1, 0), o2E(N, 0.0, 1, 1), o2E(N, 0.0, 0, 1)
            e0, edd, ep = o2E(N, 0.5, 0, 0), o2E(N, 0.5, 0, 1), o2E(N, 0.5, 1, 0)
            g = bd - bs
            rows[N] = dict(D_sig=(edd - e0) / g, P_sig=(ep - e0) / g, P_D=2 * (ep - e0) / (edd - e0),
                           j=(bj - b0) / g if bj else np.nan, sigma=(bs - b0) / g)
        return rows


    def o2_alpha_c(Ns=(8, 9, 10, 12, 14)):
        out = {}
        for N in Ns:
            al = [0.35, 0.4, 0.45]; y = np.array([o2_dcal(N, a, 1, -1) for a in al])
            c = np.polyfit(al, y - 1, 2)
            out[N] = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and 0.3 < x.real < 0.5][0]
        return out


    def cli(argv):
        S, meta = ising_series()
        fits, p, ext = ising_fits(S)
        Ns = sorted(S['D'])
        print("% ISING (T-cal rows, then D-cal rows); DMRG meta:", meta)
        for k in ('psi', 'psi32', 'psi52', 'D', 's', 'po', 'tp'):
            print(f"{k:6s} T: " + ' & '.join(f"{S[k][n]:.4f}" if n in S[k] else '--' for n in Ns)
                  + f"   ext(p={p:.3f}) {ext[(k, 'T')]:.4f}")
        for k in ('psi', 'psi32', 's', 'tp'):
            print(f"{k:6s} D: " + ' & '.join(f"{2 * S[k][n] / S['D'][n]:.4f}" if n in S[k] else '--' for n in Ns)
                  + f"   ext {ext[(k, 'D')]:.4f}")
        for nmin, (pp, dp, res) in fits.items():
            print(f"D_T = 2 + b N^-p, N>={nmin}: p = {pp:.3f} +- {dp:.3f} (max resid {res:.1e}) -> implied Dhat_s = {1 + 2 * pp:.3f}")
        print("% O(2) stability at alpha=1/2")
        st = o2_stab()
        for key in ('D_sig', 'P_sig', 'P_D', 'j', 'sigma'):
            print(f"{key:6s}: " + ' & '.join(f"{st[N][key]:.4f}" for N in st))
        Nn = np.array(list(st)); P = np.array([st[N]['P_D'] for N in st])
        (a, b, pp), cov = curve_fit(lambda n, a, b, p: a + b * n**-p, Nn, P, p0=[0.88, -0.05, 1], maxfev=50000)
        print(f"Psi_1/2 (D-cal) fit a+bN^-p: a={a:.4f} p={pp:.2f}+-{np.sqrt(cov[2,2]):.2f} resid {np.abs(a+b*Nn**-pp-P).max():.1e};"
              f" 1/N two-point (12,14): {P[-1] - (P[-2]-P[-1])/(1/12-1/14)/14:.4f}")
        print("% O(2) D-cal table (DMRG; N=10 also ED)")
        for al in (0.1, 0.25, 0.35, 0.4, 0.45, 0.5):
            print(f"{al:.2f} & " + ' & '.join(f"{o2_dcal(N, al, 1, 0):.4f}" for N in (10, 12, 14)) + ' & '
                  + ' & '.join(f"{o2_dcal(N, al, 1, -1):.4f}" for N in (10, 12, 14)))
        ac = o2_alpha_c()
        print("alpha_c:", {N: round(v, 4) for N, v in ac.items()})
        Na = np.array(list(ac)); A = np.array(list(ac.values()))
        (a, b, pp), cov = curve_fit(lambda n, a, b, p: a + b * n**-p, Na, A, p0=[0.41, -0.2, 1], maxfev=50000)
        lin = A[-1] - (A[-2] - A[-1]) / (1 / Na[-2] - 1 / Na[-1]) / Na[-1]
        print(f"alpha_c extrap: free fit a={a:.4f} (p={pp:.2f}); 1/N from N={Na[-2]},{Na[-1]}: {lin:.4f}")
        mx = {k: max(r['trunc'] for r in R if r['kind'] == k and r['N'] >= 12) for k in ('ising', 'o2')}
        mpo = {k: sorted({r['mpo_chi'] for r in R if r['kind'] == k and r['N'] >= 10}) for k in ('ising', 'o2')}
        print("max discarded weight (N>=12):", mx, " MPO bond dims:", mpo)
    return namespace(locals(), globals())


STEPS = {'run_defect': run_defect, 'ising_levels': ising_levels, 'ising_amplitudes': ising_amplitudes, 'dmrg_tables': dmrg_tables}
REPORT = [('dmrg_tables',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
