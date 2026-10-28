"""Can an ordered line of pinned monodromy lines disorder into the unpinned line? Walls against entropy at large N.

An ordered line D_+ (+) D_- has g = 2 g_pin. By the g-theorem the unpinned monodromy line can flow into it only if
log g_u - log g_pin > log 2, and its walls are relevant where Delta_+- < 1 (paper, Sec. 5.4, Table 8, and App. E.8,
Table 20). At large N, with every component twisted in complex pairs, the entropy released by pinning is
(N/2) Delta(alpha) + Delta^(1)(alpha) + O(1/N):
  entropy_window   the leading coefficient Delta(alpha), from the coincident deficit A_alpha(varsigma), and the leading
                   entropic and wall thresholds;
  pinned_det       the term of order N^0 at alpha = 0: the sigma fluctuation determinant of the ordinary pinned line,
                   with the mixing of sigma and the condensate;
  twisted_det      the same determinant on the unpinned and on the pinned line at alpha > 0;
  twisted_summary  Delta^(1) at alpha = 0-0.5 and the next-order entropic thresholds of Table 8.
A trace takes about 10 minutes (pinned_det) to an hour (twisted_det, per alpha). Their regulated sums are stored; their
(lam, mode) tables are rebuilt into pinned_entropy_files/, which git ignores, and no other step reads them.

usage: python pinned_entropy.py <step> [args]      (python pinned_entropy.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace          # noqa: E402,F401


@section
def entropy_window():
    """Defect entropy of the pinned monodromy line at large N, and the entropic condition for an ordered line.

Physics. On S^1 x H^2 the large-N line has a constant background mass varsigma. Per complex component of twist alpha,
    d log g / d varsigma = -A_alpha(varsigma),
    A_alpha(s) = sum_{s in Z+alpha} [dg(1/2 + sqrt(s^2 + varsigma)) - log|s|] + log(2 sin pi alpha)   (0 < alpha < 1),
    A_0(s)     = sum_{n in Z} [dg(1/2 + sqrt(n^2 + varsigma)) - dg(1/2 + |n|)],
the coincident-point deficit of the draft (A_alpha(0) = (pi/2)(1 - 2 alpha) tan pi alpha), from dF/dvarsigma = (1/2) Vol
sum <phi phi> with Vol(S^1 x H^2) = -4 pi^2. Hence l_alpha(varsigma) = log g_free(alpha) - int_0^varsigma A_alpha.
Full twist (N/2 complex fields, all twisted; the draft's large-N counterpart of the O(2) data):
  unpinned: the gap equation A_alpha(varsigma_u) = 0, log g_u = (N/2) l_alpha(varsigma_u);
  pinned:   the classical field sqrt(N) phi_* e^{i alpha theta} forces varsigma_p = 1/4 - alpha^2, and its quadratic action
            vanishes there, so log g_p = (N/2) l_alpha(varsigma_p) + O(1)   [phi_*^2 = A_alpha(varsigma_p)/4 pi^2].
Partial twist (one twisted complex field, N - 2 spectators), pinned: log g_p = l_alpha(vp) + ((N-2)/2) l_0(vp) + O(1),
  unpinned: log g_u = log g_free(alpha) + O(1/N).
Entropic condition. An ordered line D_+ (+) D_- has g = 2 g_p. The g-theorem lets the plain (unpinned) monodromy line
flow into it only if log g_u - log g_p > log 2. At leading order this reads (N/2) Delta(alpha) > log 2.
Since l_alpha is maximal where A_alpha = 0, log g_u >= log g_p automatically (pinning lowers g, as a line flow must).
Gates: (G0) A_alpha(0) closed form; (G1) log g_free against Table (tab:entropy); (G2) varsigma_u(1/2) = -0.168374 and
log g_u per complex = 0.1617 (App. D, full twist); (G3) phi_*^2 = A_alpha(1/4 - alpha^2)/4 pi^2 against tab:monopinlargeN.
usage: python entropy_window.py   -> entropy_window.json
"""
    import json
    import mpmath as mp

    mp.mp.dps = 25
    LOG2 = mp.log(2)


    def A(alpha, v):
        alpha, v = mp.mpf(alpha), mp.mpf(v)
        if alpha == 0:
            f = lambda n: mp.digamma(0.5 + mp.sqrt(n * n + v)) - mp.digamma(0.5 + n)
            return f(0) + 2 * mp.nsum(f, [1, mp.inf])
        pos = mp.nsum(lambda n: mp.digamma(0.5 + mp.sqrt((n + alpha) ** 2 + v)) - mp.log(n + alpha), [0, mp.inf])
        neg = mp.nsum(lambda n: mp.digamma(0.5 + mp.sqrt((n - alpha) ** 2 + v)) - mp.log(n - alpha), [1, mp.inf])
        return pos + neg + mp.log(2 * mp.sin(mp.pi * alpha))


    def loggfree(alpha):
        if alpha == 0:
            return mp.mpf(0)
        a = mp.mpf(alpha)
        f = lambda nu: nu * mp.tanh(mp.pi * nu) * mp.log((1 - 2 * mp.exp(-2 * mp.pi * nu) * mp.cos(2 * mp.pi * a)
                                                             + mp.exp(-4 * mp.pi * nu)) / (1 - mp.exp(-2 * mp.pi * nu)) ** 2)
        return mp.quad(f, [0, 0.5, 2, mp.inf])


    def intA(alpha, v0, v1):
        return mp.quad(lambda v: A(alpha, v), [v0, v1])


    def sigma_u(alpha):
        if alpha == 0:
            return mp.mpf(0)
        lo = -min(alpha, 1 - alpha) ** 2 * (1 - 1e-9)
        return mp.findroot(lambda v: A(alpha, v), (lo * 0.999, mp.mpf('-1e-8')), solver='anderson')


    def main():
        out = dict(gates={}, rows=[])
        table_lgf = {0.1: 0.009441, 0.2: 0.028660, 0.3: 0.048057, 0.4: 0.061800, 0.5: 0.066717}
        table_phi = {0.0: 0.044413, 0.1: 0.043815, 0.2: 0.041969, 0.3: 0.038688, 0.4: 0.033535, 0.5: 0.025330}
        table_wall = {0.0: 0.2756, 0.1: 0.2706, 0.2: 0.2554, 0.3: 0.2290, 0.4: 0.1892, 0.5: 0.1312}   # Delta_+-/N, leading order
        g0 = max(abs(A(a, 0) - mp.pi / 2 * (1 - 2 * a) * mp.tan(mp.pi * a)) for a in (0.1, 0.25, 0.4, 0.5))
        out['gates']['G0_A0_closed_form'] = float(g0)
        out['gates']['G1_loggfree'] = {str(a): [float(loggfree(a)), t] for a, t in table_lgf.items()}
        su = sigma_u(0.5)
        lu = loggfree(0.5) - intA(0.5, 0, su)
        out['gates']['G2_full_twist_half'] = dict(sigma_u=float(su), logg_u_per_complex=float(lu), target=[-0.168374, 0.1617])
        print('G0', out['gates']['G0_A0_closed_form'])
        print('G1', out['gates']['G1_loggfree'])
        print('G2', out['gates']['G2_full_twist_half'], flush=True)
        for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            vp = mp.mpf(0.25) - a * a
            su = sigma_u(a)
            lgf = loggfree(a)
            phi2 = A(a, vp) / (4 * mp.pi ** 2)
            Dfull = intA(a, su, vp)                       # per complex component, = (2/N)(log g_u - log g_p)
            Dsp = intA(0.0, 0, vp)                        # per spectator complex component
            Dtw = intA(a, 0, vp)                          # O(1) part of the partial twist
            row = dict(alpha=a, sigma_u=float(su), sigma_p=float(vp), loggfree=float(lgf), phi2=float(phi2), phi2_table=table_phi[a],
                       logg_u_per_complex=float(lgf - intA(a, 0, su)), logg_p_per_complex=float(lgf - intA(a, 0, vp)),
                       Delta_full_per_complex=float(Dfull), N_entropic_full=float(2 * LOG2 / Dfull) if Dfull > 0 else None,
                       wall_over_N=table_wall[a], N_wall=1 / table_wall[a],
                       Delta_partial_spectator_per_complex=float(Dsp), Delta_partial_O1=float(Dtw),
                       N_entropic_partial=float(2 + 2 * (LOG2 - Dtw) / Dsp) if Dsp > 0 else None)
            out['rows'].append(row)
            print(f"alpha={a}: s_u {row['sigma_u']:+.6f} s_p {row['sigma_p']:.4f}  phi2 {row['phi2']:.6f} (table {table_phi[a]})  "
                  f"Delta_full/complex {row['Delta_full_per_complex']:.5f} -> N_ent {row['N_entropic_full']}  N_wall {row['N_wall']:.2f}  "
                  f"| partial: spectator {row['Delta_partial_spectator_per_complex']:.5f}, O(1) {row['Delta_partial_O1']:.5f}, "
                  f"N_ent {row['N_entropic_partial']}", flush=True)
        store.put('analytic/pinned_entropy/entropy_window.json', out)

    def cli(argv):
        mp.mp.dps = 25
        main()
    return namespace(locals(), globals())


@section
def pinned_det():
    """O(N^0) defect entropy of the ordinary pinned line at large N: the sigma fluctuation determinant with condensate mixing.

Physics. Integrating out the N real fields exactly at fixed sigma, with the pinning source in phi_1, gives
N s[sigma] = N [ (1/2) Tr log(-d^2 + sigma) - (1/2) j G_sigma j ]. Its saddle is the pinned line (on S^1 x H^2: background
varsigma = 1/4, classical field sqrt(N) phi_* with phi_*^2 = A_0(1/4)/4 pi^2), and
    log g = N l_0 + log g^(1) + O(1/N),   log g^(1) = -(1/2) log det( s''_pin / s''_trivial ),
    s''(x, y) = -(1/2) G(x, y)^2 - phi_*^2 G(x, y),
with G the propagator of one component at varsigma = 1/4 (modes D_s = 1/2 + sqrt(s^2 + 1/4)). The term phi_*^2 G is the mixing of
sigma with the condensate. In the S^1 mode n and the H^2 spectral parameter lam (Plancherel lam tanh(pi lam)/2pi,
Vol(H^2) = -2pi):
    R_n(lam) = [ sum_{s+s'=n} B_{D_s D_s'}(lam) + 4 pi phi_*^2/(lam^2 + n^2 + 1/4) ] / U_n(lam),
    log g^(1) = (1/2) sum_n int_0^inf dlam lam tanh(pi lam) log R_n(lam),
where U_n is the trivial kernel (closed form, bubble.U_exact) and B the AdS_2 bubble (bubble.a). The pinned bubble sum is U_n plus
the pole-by-pole difference over (s, j), which converges much faster than either sum.
Gates. (G1) gap balance: lam^2 [sum B - U] -> -4 pi phi_*^2 at large lam, so the 1/lam^2 tail of the numerator cancels;
(G2) R > 0; (G3) truncation (S, J) and lam-grid convergence; (G4) the regulated trace is fitted in the cutoff Lambda of a
function of n^2 + lam^2 (two regulator shapes).
usage: python pinned_det.py gates | python pinned_det.py trace S J nmax lammax
"""
    import os
    FILES = os.path.join(fuzzy.ROOT, '04_pinned_lines', 'pinned_entropy_files')   # was: the working folder largeN/ (git-ignored here)
    os.makedirs(FILES, exist_ok=True)
    import sys
    import json
    import time
    import numpy as np
    import mpmath as mp
    _bubble = fuzzy.question('03_spectator_entropy/defect_entropy').bubble(); log_a, U_exact = _bubble.log_a, _bubble.U_exact   # was: from bubble import log_a, U_exact

    PHI2 = None


    def phi2():
        nonlocal PHI2   # was: global PHI2
        if PHI2 is None:
            f = lambda n: mp.digamma(0.5 + mp.sqrt(n * n + 0.25)) - mp.digamma(0.5 + n)
            A0 = f(0) + 2 * mp.nsum(f, [1, mp.inf])
            PHI2 = float(A0 / (4 * mp.pi ** 2))
        return PHI2


    def Dp(s):
        return 0.5 + np.sqrt(s * s + 0.25)


    def D0(s):
        return 0.5 + np.abs(s)


    def poles(n, S, J):
        """(A_p, M_p, A_0, M_0) over s in [-S, n + S], j in [0, J]."""
        s = np.arange(-S, n + S + 1, dtype=float)
        j = np.arange(0, J + 1, dtype=float)
        ss, jj = np.meshgrid(s, j, indexing='ij')
        d1p, d2p = Dp(ss), Dp(n - ss)
        d10, d20 = D0(ss), D0(n - ss)
        Ap = np.exp(log_a(d1p, d2p, jj)); Mp = d1p + d2p + 2 * jj - 0.5
        A0 = np.exp(log_a(d10, d20, jj)); M0 = d10 + d20 + 2 * jj - 0.5
        return Ap.ravel(), Mp.ravel(), A0.ravel(), M0.ravel()


    def diff_spectral(n, lam, S, J, chunk=200000):
        Ap, Mp, A0, M0 = poles(n, S, J)
        lam2 = np.asarray(lam, float)[:, None] ** 2
        out = np.zeros(len(lam))
        for k in range(0, len(Ap), chunk):
            sl = slice(k, k + chunk)
            out += (Ap[None, sl] / (lam2 + Mp[None, sl] ** 2) - A0[None, sl] / (lam2 + M0[None, sl] ** 2)).sum(1)
        return out


    def ratio(n, lam, S, J):
        U = U_exact(n, lam)
        num = U + diff_spectral(n, lam, S, J) + 4 * np.pi * phi2() / (np.asarray(lam) ** 2 + n * n + 0.25)
        return num / U


    def gates():
        p2 = phi2()
        print(f'phi_*^2 = {p2:.6f} (table 0.044413); 4 pi phi^2 = {4 * np.pi * p2:.6f}', flush=True)
        for (S, J) in ((200, 200), (400, 400), (800, 800)):
            t0 = time.time()
            lam = np.array([20.0, 40.0, 80.0])
            d = diff_spectral(0, lam, S, J)
            print(f'G1 n=0 S={S} J={J}: lam^2 * diff = ' + ' '.join(f'{x:.5f}' for x in lam ** 2 * d) +
                  f'   (target -{4 * np.pi * p2:.5f})  [{time.time() - t0:.1f}s]', flush=True)
        lam = np.array([0.01, 0.3, 1.0, 3.0, 10.0])
        for n in (0, 1, 3, 10):
            r = ratio(n, lam, 600, 600)
            print(f'G2 n={n}: R = ' + ' '.join(f'{x:.6f}' for x in r), flush=True)


    def main_gates():
        if sys.argv[1] == 'gates':
            gates()


    def lam_grid(lmax, npts):
        """Gauss-Legendre nodes on [0, 2], [2, 10], [10, lmax] (dense near 0 for the tanh measure)."""
        xs, ws = [], []
        for (a, b, k) in ((0.0, 2.0, npts // 5), (2.0, 10.0, npts // 5), (10.0, lmax, npts - 2 * (npts // 5))):
            x, w = np.polynomial.legendre.leggauss(k)
            xs.append((b - a) / 2 * x + (a + b) / 2); ws.append((b - a) / 2 * w)
        return np.concatenate(xs), np.concatenate(ws)


    def table(S, J, nmax, lmax, npts):
        lam, w = lam_grid(lmax, npts)
        L = np.zeros((nmax + 1, len(lam)))
        t0 = time.time()
        for n in range(nmax + 1):
            L[n] = np.log(ratio(n, lam, S, J))
            if n % 20 == 0:
                print(f'  n={n} [{time.time() - t0:.0f}s]', flush=True)
        return lam, w, L


    def regulated(lam, w, L, Lam, shape):
        n = np.arange(L.shape[0])[:, None]
        p2 = n ** 2 + lam[None, :] ** 2
        reg = np.exp(-p2 / Lam ** 2) if shape == 'gauss' else (1 + p2 / Lam ** 2) ** -4
        mult = np.where(n == 0, 1.0, 2.0)                      # n and -n
        return 0.5 * float((mult * w[None, :] * lam[None, :] * np.tanh(np.pi * lam[None, :]) * L * reg).sum())


    def trace(S, J, nmax, lmax, npts=400):
        lam, w, L = table(S, J, nmax, lmax, npts)
        np.savez(os.path.join(FILES, f'pinned_det_S{S}_J{J}_n{nmax}_l{int(lmax)}.npz'), lam=lam, w=w, L=L)
        out = {}
        for shape in ('gauss', 'lor4'):
            rows = []
            for Lam in (4, 6, 8, 10, 12, 16, 20, 24, 28, 32):
                rows.append((Lam, regulated(lam, w, L, Lam, shape)))
            out[shape] = rows
            print(shape + ': ' + '  '.join(f'{a}:{b:.6f}' for a, b in rows), flush=True)
        # large-p behaviour: p^2 log R along rays
        for ang in (0.0, 0.25, 0.5, 0.75, 1.0):
            vals = []
            for p in (40, 80, 120):
                nn = int(round(p * np.sin(ang * np.pi / 2))); ll = p * np.cos(ang * np.pi / 2)
                if nn <= nmax and ll <= lmax:
                    vals.append((p, float((nn ** 2 + ll ** 2) * np.log(ratio(nn, np.array([max(ll, 1e-3)]), S, J))[0])))
            print(f'  ray angle {ang:.2f}*pi/2: p^2 log R = ' + ' '.join(f'{p}:{v:.5f}' for p, v in vals), flush=True)
        store.put(f'largeN/pinned_det_S{S}_J{J}_n{nmax}.json', out)


    def main_trace():
        if sys.argv[1] == 'trace':
            trace(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))

    def cli(argv):
        if argv[1] == 'trace':
            with fuzzy.logged('largeN/pinned_det_S' + argv[2] + '.log'):
                main_trace()
        else:
            main_gates()
    return namespace(locals(), globals())


@section
def twisted_det():
    """O(N^0) entropy released by pinning at twist alpha (full twist, every component twisted in complex pairs).

log g^(1)_line = (1/2) sum_k int dlam lam tanh(pi lam) log[ Num_k(lam) / U_k(lam) ],  U_k the trivial kernel (bubble.U_exact).
Twisted complex pairs at background v: modes D_s = 1/2 + sqrt(s^2 + v), s in Z + alpha. The sigma kernel per N is
(1/2)|G_alpha|^2 (+ the mixing on the pinned line), so in the S^1 mode k
  Num_k = sum_{s in Z+alpha} B_{D_s, D_{s-k}}(lam) [+ 4 pi phi^2 (1/2)(1/(lam^2+(k+alpha)^2+v) + 1/(lam^2+(k-alpha)^2+v))],
unpinned: v = v_u (zero of A_alpha), no mixing; pinned: v = 1/4 - alpha^2, phi^2 = A_alpha(v)/4 pi^2.
The bubble sums are U_k plus pole-by-pole differences against the untwisted massless poles, pairing s = n + alpha with n.
Gates: (G1) gap balance: lam^2 (Num_k - U_k) -> 0 at large lam on both lines; (G2) alpha = 0 reproduces pinned_det.py;
(G3) positivity. The entropy released by pinning at O(N^0) is Delta^(1) = log g^(1)_u - log g^(1)_pin.
usage: python twisted_det.py gates alpha | python twisted_det.py trace alpha S J nmax lmax
"""
    import os
    FILES = os.path.join(fuzzy.ROOT, '04_pinned_lines', 'pinned_entropy_files')   # was: the working folder largeN/ (git-ignored here)
    os.makedirs(FILES, exist_ok=True)
    import sys
    import json
    import time
    import numpy as np
    import mpmath as mp
    _bubble = fuzzy.question('03_spectator_entropy/defect_entropy').bubble(); log_a, U_exact = _bubble.log_a, _bubble.U_exact   # was: from bubble import log_a, U_exact

    EW = globals()['entropy_window']()   # was: sys.path.insert(0, '../analytic/pinned_entropy'); import entropy_window as EW

    C16 = None


    def background(alpha):
        vu = float(EW.sigma_u(alpha)) if alpha > 0 else 0.0
        vp = 0.25 - alpha ** 2
        phi2 = float(EW.A(alpha, vp) / (4 * mp.pi ** 2))
        return vu, vp, phi2


    def diff_spectral(k, lam, alpha, v, S, J, chunk=200000):
        n = np.arange(-S, k + S + 1, dtype=float)
        j = np.arange(0, J + 1, dtype=float)
        nn, jj = np.meshgrid(n, j, indexing='ij')
        s1, s2 = nn + alpha, nn + alpha - k
        d1, d2 = 0.5 + np.sqrt(s1 ** 2 + v), 0.5 + np.sqrt(s2 ** 2 + v)
        e1, e2 = 0.5 + np.abs(nn), 0.5 + np.abs(nn - k)
        Ap = np.exp(log_a(d1, d2, jj)).ravel(); Mp = (d1 + d2 + 2 * jj - 0.5).ravel()
        A0 = np.exp(log_a(e1, e2, jj)).ravel(); M0 = (e1 + e2 + 2 * jj - 0.5).ravel()
        lam2 = np.asarray(lam, float)[:, None] ** 2
        out = np.zeros(len(lam))
        for c in range(0, len(Ap), chunk):
            sl = slice(c, c + chunk)
            out += (Ap[None, sl] / (lam2 + Mp[None, sl] ** 2) - A0[None, sl] / (lam2 + M0[None, sl] ** 2)).sum(1)
        return out


    def ratio(k, lam, alpha, line, bg, S, J):
        vu, vp, phi2 = bg
        lam = np.asarray(lam, float)
        U = U_exact(k, lam)
        if line == 'u':
            num = U + diff_spectral(k, lam, alpha, vu, S, J)
        else:
            mix = 0.5 * (1 / (lam ** 2 + (k + alpha) ** 2 + vp) + 1 / (lam ** 2 + (k - alpha) ** 2 + vp))
            num = U + diff_spectral(k, lam, alpha, vp, S, J) + 4 * np.pi * phi2 * mix
        return num / U


    def gates(alpha):
        bg = background(alpha)
        print(f'alpha={alpha}: v_u={bg[0]:.6f} v_pin={bg[1]:.4f} phi^2={bg[2]:.6f}', flush=True)
        lam = np.array([20.0, 40.0, 80.0])
        for line in ('u', 'pin'):
            for SJ in (400, 800, 1600):
                vu, vp, phi2 = bg
                v = vu if line == 'u' else vp
                d = diff_spectral(0, lam, alpha, v, SJ, SJ)
                if line == 'pin':
                    d = d + 4 * np.pi * phi2 * 0.5 * (1 / (lam ** 2 + alpha ** 2 + vp) + 1 / (lam ** 2 + alpha ** 2 + vp))
                print(f'  G1 {line:3s} k=0 S=J={SJ}: lam^2 (Num - U) = ' + ' '.join(f'{x:+.5f}' for x in lam ** 2 * d), flush=True)
        for line in ('u', 'pin'):
            r = [ratio(k, np.array([0.01, 0.5, 3.0]), alpha, line, bg, 800, 800) for k in (0, 1, 3)]
            print(f'  G3 {line}: ' + ' | '.join(' '.join(f'{x:.5f}' for x in rr) for rr in r), flush=True)


    def lam_grid(lmax, npts):
        xs, ws = [], []
        for (a, b, k) in ((0.0, 2.0, npts // 5), (2.0, 10.0, npts // 5), (10.0, lmax, npts - 2 * (npts // 5))):
            x, w = np.polynomial.legendre.leggauss(k)
            xs.append((b - a) / 2 * x + (a + b) / 2); ws.append((b - a) / 2 * w)
        return np.concatenate(xs), np.concatenate(ws)


    def regulated(lam, w, L, ks, Lam, shape):
        k = np.asarray(ks, float)[:, None]
        p2 = k ** 2 + lam[None, :] ** 2
        reg = np.exp(-p2 / Lam ** 2) if shape == 'gauss' else (1 + p2 / Lam ** 2) ** -4
        return 0.5 * float((w[None, :] * lam[None, :] * np.tanh(np.pi * lam[None, :]) * L * reg).sum())


    def trace(alpha, S, J, nmax, lmax, npts=400):
        """Twisted kernels are not symmetric under k -> -k at generic alpha, so both signs of k are tabulated."""
        bg = background(alpha)
        lam, w = lam_grid(lmax, npts)
        ks = list(range(-nmax, nmax + 1))
        t0 = time.time()
        out = dict(alpha=alpha, background=bg, S=S, J=J, nmax=nmax, lmax=lmax, lines={})
        for line in ('u', 'pin'):
            L = np.zeros((len(ks), len(lam)))
            for i, k in enumerate(ks):
                L[i] = np.log(ratio(k, lam, alpha, line, bg, S, J))
            rows = {shape: [(Lam, regulated(lam, w, L, ks, Lam, shape)) for Lam in (8, 12, 16, 20, 24, 28, 32)]
                    for shape in ('gauss', 'lor4')}
            out['lines'][line] = rows
            np.savez(os.path.join(FILES, f'twisted_det_a{alpha:g}_{line}_S{S}.npz'), lam=lam, w=w, L=L, ks=ks)
            print(f'alpha={alpha} {line}: gauss ' + ' '.join(f'{a}:{b:+.6f}' for a, b in rows['gauss']) +
                  f'  [{time.time() - t0:.0f}s]', flush=True)
            print(f'alpha={alpha} {line}: lor4  ' + ' '.join(f'{a}:{b:+.6f}' for a, b in rows['lor4']), flush=True)
        store.put(f'largeN/twisted_det_a{alpha:g}_S{S}.json', out)


    def main():
        a = sys.argv
        if a[1] == 'gates':
            gates(float(a[2]))
        elif a[1] == 'trace':
            trace(float(a[2]), int(a[3]), int(a[4]), int(a[5]), float(a[6]))

    def cli(argv):
        if argv[1] == 'trace':
            with fuzzy.logged(f'largeN/twisted_det_a{argv[2]}.log'):
                main()
        else:
            main()
    return namespace(locals(), globals())


@section
def twisted_summary():
    """Collect the O(N^0) entropy released by pinning at every twist from twisted_det traces (S = J = 1000, |k|, lam <= 120).
Estimator: Gaussian Lambda = 12-28 and Lorentzian Lambda = 12-32, fits b + c/Lambda (+ d/Lambda^2); value = mean of the four
fits, error = half their spread. alpha = 0 uses pinned_det.py (S = J = 1600) for the table and the twisted_det run as a gate.
N_ent solves (N/2) D_alpha + Delta1 = log 2 with D_alpha the leading coefficient of entropy_window.json.
"""
    import json
    import numpy as np

    def main():
        EW = store.get('analytic/pinned_entropy/entropy_window.json')
        LO = {round(r['alpha'], 2): r for r in EW['rows']}


        def extrapolate(rows):
            est = []
            for shape in ('gauss', 'lor4'):
                L = np.array(rows[shape])
                sel = L[(L[:, 0] >= 12) & (L[:, 0] <= 28)] if shape == 'gauss' else L[L[:, 0] >= 12]
                for deg in (1, 2):
                    X = np.vstack([sel[:, 0] ** -p for p in range(deg + 1)]).T
                    est.append(np.linalg.lstsq(X, sel[:, 1], rcond=None)[0][0])
            return float(np.mean(est)), float(np.ptp(est) / 2)


        out = {}
        for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            d = store.get(f'largeN/twisted_det_a{a:g}_S1000.json')
            u, eu = extrapolate(d['lines']['u'])
            p, ep = extrapolate(d['lines']['pin'])
            if a == 0.0:
                gate = p
                p, ep, u, eu = -0.0294, 0.0001, 0.0, 0.0
            D = LO[a]['Delta_full_per_complex']
            d1 = u - p
            out[a] = dict(u=u, pin=p, Delta1=d1, err=float(np.hypot(eu, ep)), N_wall=LO[a]['N_wall'],
                          N_ent_LO=2 * np.log(2) / D, N_ent_NLO=2 * (np.log(2) - d1) / D)
            print(f'alpha={a}: u={u:+.4f} pin={p:+.4f} Delta1={d1:+.4f}+-{out[a]["err"]:.4f}  '
                  f'N_wall={LO[a]["N_wall"]:.2f} N_ent {out[a]["N_ent_LO"]:.2f} -> {out[a]["N_ent_NLO"]:.2f}')
        print(f'gate: twisted_det at alpha = 0 gives log g_pin^(1) = {gate:+.5f} (pinned_det: -0.0294(1))')
        store.put('largeN/twisted_det_summary.json', {str(k): v for k, v in out.items()})

    def cli(argv):
        main()
    return namespace(locals(), globals())


STEPS = {'entropy_window': entropy_window, 'pinned_det': pinned_det, 'twisted_det': twisted_det, 'twisted_summary': twisted_summary}
REPORT = [('entropy_window',), ('twisted_summary',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
