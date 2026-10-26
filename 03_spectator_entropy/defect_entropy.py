"""Does a spectator raise the defect entropy at large N, and why is the answer fixed by the conformal weight?

The O(N) model at large N with one twisted complex component and N - 2 untwisted spectators, on S^1 x H^2 (paper,
Sec. 4 and App. D). The steps tabulate the fluctuation trace sector by sector and sum it with rotationally symmetric
cutoffs (Table 3(a), Figs. 3 and 9); check the closed form c = (64/3 pi) h_free (eq:cresult) and the uniformity of the
tail; show numerically and symbolically that every fixed circle sector vanishes ("The fixed-sector identity"); reproduce
the full twist of Giombi et al.; and test the choice of regulator against dimensional regularization near d = 4 and at
d = 7/2 ("The analytic formula").

The tables largeN/c1_*.npz took hours each to compute. They are stored with the logs of the runs that made them
(c1_run tab, alpha tab); every other step runs in seconds to minutes.

usage: python defect_entropy.py <step> [args]      (python defect_entropy.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def bubble():
    """AdS_2 bubble functions and the s-sums of the large-N sigma kernel on S^1 x H^2.

Product of two AdS_2 propagators at the same pair of points (Fitzpatrick-Kaplan, d = 1):
    G_{D1}(u) G_{D2}(u) = sum_j a_j(D1, D2) G_{D1+D2+2j}(u),
    a_j = (1/2)_j (S+2j)_{1/2} (S+j)_j / [2 sqrt(pi) j! (D1+j)_{1/2} (D2+j)_{1/2} (S+j-1/2)_j],  S = D1 + D2,
so with G~_D(lambda) = 1/(lambda^2 + (D - 1/2)^2) the bubble is B_{D1 D2}(lambda) = sum_j a_j / (lambda^2 + (S + 2j - 1/2)^2).
s-sums. In the S^1 mode n the sigma kernel is sum_{s + s' = n} B_{1/2+|s|, 1/2+|s'|}. Grouping by the total dimension, every level
M_m = 1/2 + |n| + m receives finitely many (s, j), so the s-sum is exact level by level:
    untwisted (s in Z):    inside pairs 0 <= s <= n on even m, outside pairs on even m;
    twisted (s in Z+1/2):  inside pairs on even m, outside pairs on odd m.
"""
    import numpy as np
    from scipy.special import gammaln, digamma
    def log_a(D1, D2, j):
        D1, D2, j = np.broadcast_arrays(np.asarray(D1, float), np.asarray(D2, float), np.asarray(j, float))
        S = D1 + D2
        return (gammaln(0.5 + j) - gammaln(0.5) + gammaln(S + 2 * j + 0.5) - gammaln(S + j) - np.log(2 * np.sqrt(np.pi))
                - gammaln(j + 1) - gammaln(D1 + j + 0.5) + gammaln(D1 + j) - gammaln(D2 + j + 0.5) + gammaln(D2 + j)
                - gammaln(S + 2 * j - 0.5) + gammaln(S + j - 0.5))


    def a(D1, D2, j):
        return np.exp(log_a(D1, D2, j))


    def dlog_a(D1, D2, j):
        """d/dD1 of log a_j (analytic)."""
        D1, D2, j = np.broadcast_arrays(np.asarray(D1, float), np.asarray(D2, float), np.asarray(j, float))
        S = D1 + D2
        return (digamma(S + 2 * j + 0.5) - digamma(S + j) - digamma(S + 2 * j - 0.5) + digamma(S + j - 0.5)
                - digamma(D1 + j + 0.5) + digamma(D1 + j))


    def levels(n, eta, mmax, dims=None):
        """Coefficients c_m of 1/(lambda^2 + M_m^2), M_m = 1/2 + |n| + m (m = 0..mmax), of sum_{s in Z+eta, s+s'=n} B_{D_s D_s'}.
        dims(s) gives the line dimension (default 1/2 + |s|); it must satisfy D_s + D_{n-s} - 1/2 - |n| = |s| + |n-s| - |n|
        level-wise only when dims is the free one, so for general dims use levels_general."""
        n = abs(n)
        c = np.zeros(mmax + 1)
        # inside pairs: s = eta, eta+1, ..., within [0, n]; total S = 1 + n, m = 2j
        ins = np.arange(eta, n + 1e-9, 1.0) if not (eta == 0.5 and n == 0) else np.array([])
        js = np.arange(0, mmax // 2 + 1)
        for s in ins:
            c[2 * js] += a(0.5 + s, 0.5 + n - s, js)
        # outside pairs: s = -r (r = 1, 2, ... untwisted; r = 1/2, 3/2, ... twisted), mirrored by s = n + r: m = 2r + 2j
        r = (1.0 if eta == 0 else 0.5)
        while 2 * r <= mmax:
            jj = np.arange(0, int((mmax - 2 * r) // 2) + 1)
            m = (2 * r + 2 * jj).astype(int)
            c[m] += 2 * a(0.5 + r, 0.5 + n + r, jj)
            r += 1.0
        return c


    def spectral(c, M, lam, e=None):
        """sum_m c_m/(lam^2 + M_m^2) [+ e_m/(lam^2 + M_m^2)^2], for an array of lam."""
        lam = np.atleast_1d(lam)[:, None]
        den = lam ** 2 + M[None, :] ** 2
        out = (c[None, :] / den).sum(1)
        if e is not None:
            out += (e[None, :] / den ** 2).sum(1)
        return out


    def U_exact(n, lam):
        """H^2 transform of 2 pi x theta-Fourier of G0^2 = e^{-|n| u}/(8 sinh u):  (pi/8) |Gamma((|n|+1/2+i lam)/2) / Gamma((|n|+3/2+i lam)/2)|^2."""
        from scipy.special import loggamma
        n = abs(n)
        z1 = (n + 0.5 + 1j * np.asarray(lam)) / 2
        return np.pi / 8 * np.exp(2 * (loggamma(z1).real - loggamma(z1 + 0.5).real))


    def D_levels(n, mmax, h=1e-5):
        """d/dD of B_{D, 1/2+|n|} at D = 1/2: coefficients (c_m, e_m) of 1/(lam^2+M_m^2) and 1/(lam^2+M_m^2)^2, even m only."""
        n = abs(n)
        js = np.arange(0, mmax // 2 + 1)
        c = np.zeros(mmax + 1); e = np.zeros(mmax + 1)
        aj = a(0.5, 0.5 + n, js)
        c[2 * js] = aj * dlog_a(0.5, 0.5 + n, js)
        M = 0.5 + n + 2 * js
        e[2 * js] = -2 * M * aj
        return c, e
    return namespace(locals(), globals())


@section
def c1():
    """O(1/N) sigma-fluctuation term c1 of log g for the large-N O(N) model with a two-component half-twist.

log g = 0.0667172 + c1/N + O(1/N^2), from -1/2 Tr(Pi_0^{-1} dPi) with Pi_0 = (N/2) G0^2 and
dPi = (G_tw^2 - G0^2) + N G0 dG_u. The spectators' s = 0 line mode sits in the singular branch, D = 1/2 - sqrt(a),
sqrt(a) = 4/(pi^2 N), fixed by the gap equation 2(-1/(4 pi^2)) + N sqrt(a)/8 = 0. On S^1 x H^2 everything is diagonal in
(n, lambda), and with Tr = Vol(S^1) Vol_reg(H^2) (1/2pi) sum_n int rho = -2 pi sum_n int rho,
    c1 = 2 pi sum_n int_0^inf dlam rho(lam) Num_n(lam)/U_n(lam),  rho = lam tanh(pi lam)/(2 pi),
    Num_n = T_n - U_n - (4/pi^2) D_n,  T_n/U_n = twisted/untwisted s-sums of AdS_2 bubbles,  D_n = d_D B_{D,1/2+|n|} at D = 1/2.
Level structure (bubble.py): t_m -> 1 on odd m, u_m -> 1 on even m, so T - U = sum (-1)^{m+1}/(lam^2+M_m^2) + decaying;
the alternating constant part is summed in closed form with digamma.
"""
    import numpy as np
    from scipy.special import gammaln, digamma, polygamma
    B = globals()['bubble']()   # was: import bubble as B
    def levels_fast(n, eta, mmax):
        """Same as bubble.levels, fully vectorized over the (pair, j) triangle."""
        n = abs(n)
        c = np.zeros(mmax + 1)
        js = np.arange(0, mmax // 2 + 1)
        ins = np.arange(eta, n + 1e-9, 1.0) if not (eta == 0.5 and n == 0) else np.array([])
        if len(ins):
            S, J = np.meshgrid(ins, js, indexing='ij')
            np.add.at(c, (2 * J).ravel(), B.a(0.5 + S, 0.5 + n - S, J).ravel())
        r0 = 1.0 if eta == 0 else 0.5
        rs = np.arange(r0, mmax / 2 + 1e-9, 1.0)
        R, J = np.meshgrid(rs, js, indexing='ij')
        m = np.rint(2 * R + 2 * J).astype(int)
        ok = m <= mmax
        np.add.at(c, m[ok], 2 * B.a(0.5 + R[ok], 0.5 + n + R[ok], J[ok]))
        return c


    def S_even(cc, lam):
        """sum_{k>=0} 1/(lam^2 + (cc + 2k)^2), lam >= 0 (array)."""
        lam = np.asarray(lam, float)
        out = np.empty_like(lam)
        small = lam < 1e-8
        z = (cc + 1j * lam[~small]) / 2
        out[~small] = digamma(z).imag / (2 * lam[~small])
        out[small] = polygamma(1, cc / 2) / 4
        return out


    class Sector:
        """Num_n and U_n for one S^1 mode n, as functions of lambda."""

        def __init__(self, n, mmax, h=1e-5):
            self.n = n = abs(n)
            self.mmax = mmax
            m = np.arange(mmax + 1)
            self.M = 0.5 + n + m
            t = levels_fast(n, 0.5, mmax)
            u = levels_fast(n, 0.0, mmax)
            dc, de = B.D_levels(n, mmax, h)
            self.delta = (t - (m % 2 == 1)) - (u - (m % 2 == 0))      # decaying part of t - u
            self.dc, self.de = dc, de
            self.alt = lambda lam: S_even(0.5 + n + 1, lam) - S_even(0.5 + n, lam)   # sum (-1)^{m+1}/(lam^2 + M_m^2)

        def parts(self, lam):
            lam = np.atleast_1d(np.asarray(lam, float))
            TU = self.alt(lam) + B.spectral(self.delta, self.M, lam)
            D = B.spectral(self.dc, self.M, lam, self.de)
            return TU, D

        def num(self, lam):
            TU, D = self.parts(lam)
            return TU - 4 / np.pi ** 2 * D

        def U(self, lam):
            return B.U_exact(self.n, lam)


    def F2_even(cc, lam, kmax=4000):
        """sum_{k>=0} 1/(lam^2 + (cc + 2k)^2)^2."""
        lam = np.asarray(lam, float)
        out = np.empty_like(lam)
        big = lam >= 0.5
        z = (cc + 1j * lam[big]) / 2
        l = lam[big]
        out[big] = digamma(z).imag / (4 * l ** 3) - polygamma(1, z).real / (8 * l ** 2) if False else _f2(z, l)
        M = cc + 2 * np.arange(kmax)
        ls = lam[~big][:, None]
        out[~big] = (1 / (ls ** 2 + M[None, :] ** 2) ** 2).sum(1) + 1 / (6 * (M[-1] + 1) ** 3)
        return out


    def _f2(z, l):
        # -(1/(2 lam)) d/dlam [Im psi(z)/(2 lam)], z = (cc + i lam)/2; scipy has no complex polygamma, so use a
        # finite difference of the closed form for S_even in lam
        h = 1e-4 * np.maximum(l, 1)
        def S(x):
            return digamma((2 * z.real + 1j * x) / 2).imag / (2 * x)
        return -(S(l + h) - S(l - h)) / (2 * h) / (2 * l)


    class SectorT(Sector):
        """Sector with constant asymptotes summed in closed form and a half-weight last term for the alternating part."""

        def __init__(self, n, mmax, h=1e-5):
            super().__init__(n, mmax, h)
            m = np.arange(mmax + 1)
            nu = self.delta - 4 / np.pi ** 2 * self.dc
            eps = -4 / np.pi ** 2 * self.de
            self.eps_inf = 16 / np.pi ** 3
            self.deps = eps - self.eps_inf * (m % 2 == 0)
            w = np.ones(mmax + 1); w[-1] = 0.5; w[-2] = 0.75; w[-3] = 0.75 if False else 1.0
            self.nu_w = nu * np.where(m == mmax, 0.5, 1.0)
            self.nu_tail = 8 / np.pi ** 3                                      # nu_m -> (8/pi^3)/M_m^2 on even m

        def num(self, lam):
            lam = np.atleast_1d(np.asarray(lam, float))
            n, M = self.n, self.M
            out = self.alt(lam) + B.spectral(self.nu_w, M, lam, self.deps)
            out += self.eps_inf * F2_even(0.5 + n, lam)
            Mt = M[-1] + 1.0                                                  # tail of the 1/M^2 asymptote (even ladder)
            L2 = lam ** 2
            out += self.nu_tail / 2 * (1 / (L2 * Mt) - (np.pi / 2 - np.arctan(Mt / lam)) / L2 / lam)
            return out
    return namespace(locals(), globals())


@section
def c1_run():
    """Tabulate the c1 integrand r_n(lam) = lam tanh(pi lam) Num_n(lam)/U_n(lam) on a lambda grid for n = 0..nmax
(c1 = sum_{n in Z} int_0^inf dlam r_n), and the regulated sums with covariant (functions of p^2 = n^2 + lam^2) cutoffs.
usage: python c1_run.py tab nmax mmax [mmax2]   -> c1_tab_<nmax>_<mmax>.npz
       python c1_run.py sum file.npz
"""
    import sys
    import time
    import numpy as np
    c1 = globals()['c1']()   # was: import c1
    EDGES = [0, 2, 10, 40, 120, 240, 400]
    NODES = [40, 40, 60, 80, 60, 50]


    def lam_grid():
        xs, ws = [], []
        for (a, b), k in zip(zip(EDGES[:-1], EDGES[1:]), NODES):
            x, w = np.polynomial.legendre.leggauss(k)
            xs.append(a + (b - a) * (x + 1) / 2); ws.append((b - a) / 2 * w)
        return np.concatenate(xs), np.concatenate(ws)


    def tab(nmax, mmax, mmax2=None):
        lam, w = lam_grid()
        R = np.zeros((nmax + 1, len(lam))); R2 = np.zeros_like(R) if mmax2 else None
        t0 = time.time()
        for n in range(nmax + 1):
            s = c1.SectorT(n, mmax)
            R[n] = lam * np.tanh(np.pi * lam) * s.num(lam) / s.U(lam)
            if mmax2:
                s2 = c1.SectorT(n, mmax2)
                R2[n] = lam * np.tanh(np.pi * lam) * s2.num(lam) / s2.U(lam)
            if n % 10 == 0:
                print(f'n={n} [{time.time() - t0:.0f}s]', flush=True)
        out = dict(lam=lam, w=w, R=R, nmax=nmax, mmax=mmax)
        if mmax2:
            out.update(R2=R2, mmax2=mmax2)
        f = f'c1_tab_{nmax}_{mmax}' + (f'_{mmax2}' if mmax2 else '') + '.npz'
        fuzzy.npz_put('largeN/' + f, **out)
        return f


    REGS = {'gauss': lambda x: np.exp(-x), 'lorentz3': lambda x: (1 + x) ** -3, 'exp': lambda x: np.exp(-np.sqrt(x))}


    def regulated(lam, w, R, P, reg):
        n = np.arange(R.shape[0])[:, None]
        f = REGS[reg]((n ** 2 + lam[None, :] ** 2) / P ** 2)
        per_n = (R * f * w[None, :]).sum(1)
        return per_n[0] + 2 * per_n[1:].sum()


    def summarize(fn):
        d = fuzzy.npz_load('largeN/' + fn)
        lam, w = d['lam'], d['w']
        Rs = [('R', d['R'])] + ([('R2', d['R2']), ('rich', d['R2'] + (d['R2'] - d['R']) / 7)] if 'R2' in d else [])
        Ps = np.array([4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32])
        Ps = Ps[Ps <= d['nmax'] / 5]
        for lab, R in Rs:
            for reg in REGS:
                c = np.array([regulated(lam, w, R, P, reg) for P in Ps])
                A = np.stack([np.ones_like(Ps, float), 1 / Ps, 1 / Ps ** 2, 1 / Ps ** 3], 1)
                fit = np.linalg.lstsq(A[-6:], c[-6:], rcond=None)[0]
                fit3 = np.linalg.lstsq(A[-5:, :3], c[-5:], rcond=None)[0]
                print(f'{lab:4s} {reg:8s}: c1(P) ' + ' '.join(f'{P}:{x:+.5f}' for P, x in zip(Ps, c)) +
                      f'  -> extrapolated {fit[0]:+.5f} (cubic) {fit3[0]:+.5f} (quadratic)')

    def main():
        a = sys.argv[1:]
        if a[0] == 'tab':
            f = tab(int(a[1]), int(a[2]), int(a[3]) if len(a) > 3 else None)
            summarize(f)
        else:
            summarize(a[1])

    def cli(argv):
        sys.argv = list(argv)
        key = 'largeN/c1_run_' + ('_'.join(argv[2:]) if argv[1] == 'tab' else 'sum_' + argv[2].replace('.npz', '')) + '.log'
        with fuzzy.logged(key):
            main()
    return namespace(locals(), globals())


@section
def alpha():
    """Partial twist at a general angle alpha (one complex component, phase e^{2 pi i alpha}), to test the per-mode identity
int_0^inf dlam rho(lam) Num_n(lam)/U_n(lam) = 0 found at alpha = 1/2.
T^alpha_n = sum_{s in Z+alpha} B_{1/2+|s|, 1/2+|s-n|} (from |G_c|^2), on three exact ladders M = c + 2k:
inside 0 < s < n (c = 1/2 + n), s < 0 (c = 1/2 + n + 2 - 2 alpha), s > n (c = 1/2 + n + 2 alpha); T^alpha_{-n} = T^{1-alpha}_n.
Gap balance: 2 (-B_alpha/(4 pi^2)) + N sqrt(a)/8 = 0, B_alpha = [sum_{Z+alpha} psi(1/2+|s|) - sum_Z psi(1/2+|s|)]_reg,
so Num_n = T^alpha_n - U_n - (4 B_alpha/pi^2) D_n.

Steps: "alpha a kmax" is the per-mode check (alpha.py a 2000, logged as largeN/c1_alpha<a>_sectors.log);
"alpha tab a nmax kmax" is alpha.tab_alpha(a, nmax, kmax), the hours-long table of run_alpha_scan.sh
(logged as largeN/c1_alpha<a>.log; the research log of the a = 0.25 run was c1_alpha025.log).
"""
    import numpy as np
    from scipy.special import digamma
    B = globals()['bubble']()   # was: import bubble as B
    c1 = globals()['c1']()   # was: import c1
    def B_alpha(alpha, L=4000.0):
        vals = []
        for Lam in (L, 2 * L):
            j = np.arange(-int(8 * Lam), int(8 * Lam) + 1)
            s = j + alpha
            tw = np.sum(digamma(0.5 + np.abs(s)) * np.exp(-(s / Lam) ** 2))
            un = np.sum(digamma(0.5 + np.abs(j)) * np.exp(-(j / Lam) ** 2))
            vals.append(tw - un)
        return vals[1] + (vals[1] - vals[0]) / 3


    def ladders(n, alpha, kmax):
        """[(offset c, coefficients over k = 0..kmax)] for T^alpha_n, n >= 0."""
        ks = np.arange(kmax + 1)
        out = []
        c0 = np.zeros(kmax + 1)
        for j in range(n):                                   # inside s = j + alpha in (0, n)
            s = j + alpha
            c0 += B.a(0.5 + s, 0.5 + n - s, ks)
        out.append([0.5 + n, c0])
        for off, rs in ((0.5 + n + 2 - 2 * alpha, 1 - alpha), (0.5 + n + 2 * alpha, alpha)):
            r = rs + np.arange(kmax + 1)                     # r = |s| for s < 0, or s - n for s > n
            R, K = np.meshgrid(r, ks, indexing='ij')
            lev = np.rint(R - rs + K).astype(int)             # M = off + 2(r - rs) + 2k
            ok = lev <= kmax
            cc = np.zeros(kmax + 1)
            np.add.at(cc, lev[ok], B.a(0.5 + R[ok], 0.5 + n + R[ok], K[ok]))
            out.append([off, cc])
        return out


    def T_alpha(n, alpha, kmax, lam):
        """T^alpha_n(lam) - U_n(lam) with the constant asymptotes of each ladder summed in closed form."""
        lam = np.atleast_1d(lam)
        L = ladders(n, alpha, kmax)
        u = c1.levels_fast(n, 0.0, 2 * kmax)[0::2]           # untwisted coefficients live on the even ladder c = 1/2 + n
        L[0][1] = L[0][1] - u
        tot = np.zeros_like(lam, dtype=float)
        ks = np.arange(kmax + 1)
        M2 = 0.5 + n + 2 * kmax; M1 = M2 / 2                 # common smooth window in M for the decaying residuals, so the
        def W(M):                                            # ~ A_l/k residuals of the separate ladders (sum_l A_l = 0) cancel
            x = np.clip((M - M1) / (M2 - M1), 0, 1)
            return 0.5 * (1 + np.cos(np.pi * x))
        for off, cc in L:
            cinf = np.mean(cc[-20:])
            cinf = round(cinf * 2) / 2                       # asymptote: 1/2 (outside ladders) or -1 (inside minus U)
            M = off + 2 * ks
            tot += cinf * c1.S_even(off, lam) + B.spectral((cc - cinf) * W(M), M, lam)
        return tot


    def B2_alpha(alpha, L=3000.0):
        """[sum_{Z+alpha} s^2 psi(1/2+|s|) - sum_Z s^2 psi(1/2+|s|)]_reg: Re dG_c(theta, u=0) = -(1/(4pi^2))(B_alpha - B2 theta^2/2 + ...)."""
        vals = []
        for Lam in (L, 2 * L):
            j = np.arange(-int(8 * Lam), int(8 * Lam) + 1)
            s = j + alpha
            tw = np.sum(s ** 2 * digamma(0.5 + np.abs(s)) * np.exp(-(s / Lam) ** 2))
            un = np.sum(j ** 2 * digamma(0.5 + np.abs(j)) * np.exp(-(j / Lam) ** 2))
            vals.append(tw - un)
        return vals[1] + (vals[1] - vals[0]) / 3


    def c1_local(alpha):
        """c1(alpha) = -(32/(3 pi^2)) B2(alpha): the covariant trace when every per-mode trace vanishes."""
        return -32 / (3 * np.pi ** 2) * B2_alpha(alpha)


    def tab_alpha(alpha, nmax, kmax):
        c1_run = globals()['c1_run']()
        lam, w = c1_run.lam_grid()
        Ba = B_alpha(alpha)
        R = {}
        for n in range(-nmax, nmax + 1):
            TU = T_alpha(abs(n), alpha if n >= 0 else 1 - alpha, kmax, lam)
            s = c1.SectorT(abs(n), 2 * kmax)
            m = np.arange(2 * kmax + 1)
            L2 = lam ** 2; Mt = s.M[-1] + 1.0
            spec = (B.spectral(-4 / np.pi ** 2 * s.dc * np.where(m == 2 * kmax, 0.5, 1.0), s.M, lam, s.deps)
                    + s.eps_inf * c1.F2_even(0.5 + abs(n), lam)
                    + s.nu_tail / 2 * (1 / (L2 * Mt) - (np.pi / 2 - np.arctan(Mt / lam)) / L2 / lam))
            R[n] = lam * np.tanh(np.pi * lam) * (TU + Ba * spec) / s.U(lam)
            if n % 20 == 0:
                print(f'  tab alpha={alpha}: n={n}', flush=True)
        ns = np.arange(-nmax, nmax + 1)
        fuzzy.npz_put(f'largeN/c1_alpha{alpha}_{nmax}_{kmax}.npz', lam=lam, w=w, R=np.array([R[n] for n in ns]), ns=ns)
        Ps = np.array([4, 5, 6, 8, 10, 12, 14, 16, 20, 24])
        Ps = Ps[Ps <= nmax / 5]
        for reg, f in (('gauss', lambda x: np.exp(-x)), ('lorentz3', lambda x: (1 + x) ** -3)):
            c = np.array([sum((R[n] * f((n ** 2 + lam ** 2) / P ** 2) * w).sum() for n in ns) for P in Ps])
            A = np.stack([np.ones_like(Ps, float), 1 / Ps, 1 / Ps ** 2, 1 / Ps ** 3], 1)
            fit = np.linalg.lstsq(A[-6:], c[-6:], rcond=None)[0]
            print(f'alpha={alpha} {reg}: c1(P) ' + ' '.join(f'{P}:{x:+.5f}' for P, x in zip(Ps, c)) + f' -> {fit[0]:+.5f}', flush=True)
        print(f'alpha={alpha}: local prediction c1 = {c1_local(alpha):+.5f}')

    def main():
        import sys
        c1_run = globals()['c1_run']()
        alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
        kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
        lam, w = c1_run.lam_grid()
        Ba = B_alpha(alpha)
        print(f'alpha = {alpha}: B_alpha = {Ba:.8f}' + ('  (alpha=1/2 must give 1)' if alpha == 0.5 else ''))

        def tail_total(r):
            sel = lam > 150
            A = np.stack([lam[sel] ** -2, lam[sel] ** -4, lam[sel] ** -6], 1)
            co = np.linalg.lstsq(A, r[sel], rcond=None)[0]
            return np.sum(w * r) + co[0] / 400 + co[1] / (3 * 400 ** 3) + co[2] / (5 * 400 ** 5)

        for n in (0, 1, -1, 2, -2):
            TU = T_alpha(abs(n), alpha if n >= 0 else 1 - alpha, kmax, lam)
            s = c1.SectorT(abs(n), 2 * kmax)
            m = np.arange(2 * kmax + 1)
            L2 = lam ** 2; Mt = s.M[-1] + 1.0                  # -(4/pi^2) D with the tail corrections of c1.SectorT
            spec = (B.spectral(-4 / np.pi ** 2 * s.dc * np.where(m == 2 * kmax, 0.5, 1.0), s.M, lam, s.deps)
                    + s.eps_inf * c1.F2_even(0.5 + abs(n), lam)
                    + s.nu_tail / 2 * (1 / (L2 * Mt) - (np.pi / 2 - np.arctan(Mt / lam)) / L2 / lam))
            num = TU + Ba * spec
            U = s.U(lam); rho2 = lam * np.tanh(np.pi * lam)
            r = rho2 * num / U
            i = [np.argmin(abs(lam - x)) for x in (20, 100, 300)]
            print(f'  n={n:+d}: lam^2 (T-U) -> {lam[i[-1]]**2*TU[i[-1]]:+.5f} (expect {-Ba/np.pi:+.5f}); lam^2 Num at 20,100,300: '
                  + ' '.join(f'{lam[k]**2*num[k]:+.2e}' for k in i) + f' | per-n total {tail_total(r):+.3e}, scale int|r| {np.sum(w*abs(r)):.4f}',
                  flush=True)

    def cli(argv):
        sys.argv = list(argv)
        if len(argv) > 1 and argv[1] == 'tab':
            with fuzzy.logged(f'largeN/c1_alpha{argv[2]}.log'):
                tab_alpha(float(argv[2]), int(argv[3]), int(argv[4]))
        else:
            with fuzzy.logged(f'largeN/c1_alpha{argv[1]}_sectors.log'):
                main()
    return namespace(locals(), globals())


@section
def largeN_traces():
    """The direct large-N traces at six angles (Table 3(a), "direct trace" row; Fig. 3(b)): Gaussian-cutoff sums of the
tabulated integrand, extrapolated in 1/p_max. This is the large-N block of
fun/paper_2026_10_24_physics/figure_sources/revise_figures.py, which draws Fig. 3(b) from the same numbers.
"""
    import numpy as np
    REGS={'Gaussian':lambda x:np.exp(-x),'Lorentzian':lambda x:(1+x)**-3}
    def regulated(lam,w,R,P,fn,ns=None):
     ns=np.arange(R.shape[0]) if ns is None else ns
     per=(R*fn((ns[:,None]**2+lam[None,:]**2)/P**2)*w[None,:]).sum(1)
     return per.sum() if ns.min()<0 else per[0]+2*per[1:].sum()
    def extrap(d,R,Ps,ns=None):
     c=np.array([regulated(d['lam'],d['w'],R,P,REGS['Gaussian'],ns) for P in Ps])
     return np.linalg.lstsq(np.stack([Ps**-float(i) for i in range(4)],1),c,rcond=None)[0][0]

    def main():
        d=fuzzy.npz_load('largeN/c1_tab_160_4000_8000.npz');Rh=d['R2']+(d['R2']-d['R'])/7
        q=fuzzy.npz_load('largeN/c1_alpha0.25_120_2000.npz');Ps=np.array([8,10,12,14,16,20,24,28,32])
        pts=[]
        for a in ('0.1','0.2','0.25','0.3','0.4'):          # |kappa| <= 120, levels <= 2000
         qa=fuzzy.npz_load(f'largeN/c1_alpha{a}_120_2000.npz');pts.append((float(a),extrap(qa,qa['R'],Ps[Ps<=24],qa['ns'])))
        pts.append((.5,extrap(d,Rh,Ps)))                   # |kappa| <= 160, levels <= 8000
        print('largeM direct',pts)

    def cli(argv):
        with fuzzy.logged('largeN/largeN_traces.log'):
            main()
    return namespace(locals(), globals())


@section
def uv_uniformity():
    """Issue 7: is the large-(kappa, nu) expansion of the fluctuation-trace integrand uniform in direction?
The paper's Eq. (cresult) needs r_n(lam) = A lam (lam^2 - 2 n^2) / p^5 + o(p^-2) uniformly on the (n, lam) quarter plane,
p^2 = n^2 + lam^2 (the C_kk = -2 C_perp structure times the measure lam tanh(pi lam) ~ p).  Uses the saved half-twist table
largeN/c1_tab_160_4000_8000.npz (CCS's c1_run.py; read only).  For each shell in p we report the largest residual
|r - r_lead| p^2 over all directions, with A fitted on the outermost complete shell, and the angular profile of p^2 r.
usage: python uv_uniformity.py [table.npz]   (all six angles pass: A = C_perp to 4e-5, residual p^2 ~ p^-2)
"""
    import numpy as np
    import sys


    def main():
        f = sys.argv[1] if len(sys.argv) > 1 else 'c1_tab_160_4000_8000.npz'    # or c1_alpha<a>_120_2000.npz (rows labelled by ns)
        d = fuzzy.npz_load('largeN/' + f)
        lam = d['lam']
        Rt = d['R2'] + (d['R2'] - d['R']) / 7 if 'R2' in d else d['R']   # Richardson-improved in the mode truncation
        n = (d['ns'] if 'ns' in d else np.arange(Rt.shape[0]))[:, None].astype(float); L = lam[None, :]
        p = np.sqrt(n ** 2 + L ** 2); phi = np.arctan2(n, L)            # phi = 0: H^2 axis, phi = pi/2: circle axis
        shape = L * (L ** 2 - 2 * n ** 2) / p ** 5
        ok = (p > 5) & (L > 2)
        pm = min(n.max(), 160); shell = (p > 0.75 * pm) & (p < pm) & ok
        A = np.sum(Rt[shell] * shape[shell]) / np.sum(shape[shell] ** 2)
        print(f'fitted A = {A:.6f}; compare with C_perp = -16 h_free/pi (-2/(3 pi^2) = -0.067547 at alpha = 1/2)')
        res = (Rt - A * shape) * p ** 2
        for lo, hi in [(10, 20), (20, 40), (40, 80), (80, pm)]:
            m = (p >= lo) & (p < hi) & ok
            worst = np.argmax(np.abs(res[m])); ph = phi[m][worst]
            print(f'shell p in [{lo:3.0f},{hi:3.0f}): max |residual| p^2 = {np.abs(res[m]).max():.3e} at phi/(pi/2) = {ph / (np.pi / 2):.2f};'
                  f'  leading term p^2 r_lead ~ {np.abs(A * shape * p ** 2)[m].max():.3e}')

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('analytic/uv_uniformity' + ('_' + argv[1].replace('.npz', '') if len(argv) > 1 else '') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def fulltwist():
    """Full-twist case: all N components half-twisted, no spectators (cross-check of the c1 normalization).

Gap function (H^2 frame): delta G_tw(x,x) = -(1/(4 pi^2)) Bgap(sigma),
    Bgap(sigma) = [sum_{s in Z+1/2} psi(1/2 + sqrt(s^2 + sigma)) - sum_{s in Z} psi(1/2 + |s|)]_reg,
which must give Bgap(0) = 1 (the closed-form deficit -1/(4 pi^2)); the full-twist saddle is Bgap(sigma*) = 0.
Bubble normalization at general dimensions: lam^2 d_{D1} B_{D1 D2} -> sum_j d_{D1} a_j = -psi'(D1)/(2 pi).
Linearized full twist: Num_full = (T - U) + kappa d_sigma T, kappa = -1/Bgap'(0), the analogue of the partial-twist Num with the
twisted fields' own response in place of the spectators' singular branch.
"""
    import numpy as np
    from scipy.special import digamma, polygamma
    from scipy.optimize import brentq
    B = globals()['bubble']()   # was: import bubble as B
    def Bgap(sigma, Lam=4000.0):
        # the difference of the two divergent sums with a Gaussian cutoff in s, extrapolated in 1/Lam^2
        vals = []
        for L in (Lam, 2 * Lam):
            k = np.arange(0, int(8 * L))
            sh = k + 0.5
            tw = 2 * np.sum(digamma(0.5 + np.sqrt(sh ** 2 + sigma)) * np.exp(-(sh / L) ** 2))
            un = digamma(0.5) + 2 * np.sum(digamma(0.5 + k[1:]) * np.exp(-(k[1:] / L) ** 2))
            vals.append(tw - un)
        return vals[1] + (vals[1] - vals[0]) / 3


    def Bgap_shift(sigma, kmax=200000):
        """Bgap(sigma) - Bgap(0): absolutely convergent."""
        sh = np.arange(0, kmax) + 0.5
        d = digamma(0.5 + np.sqrt(sh ** 2 + sigma)) - digamma(0.5 + sh)
        tail = sigma / 2 * (1 / sh[-1])                     # sum_{s > smax} sigma/(2 s^2), both signs of s
        return 2 * np.sum(d) + tail


    def dsigma_levels(n, mmax, h=1e-6):
        """Levels (c_m, e_m) of d_sigma T = sum_{s in Z+1/2} d_{D1} B_{D_s, D_{n-s}} / |s| at sigma = 0."""
        n = abs(n)
        c = np.zeros(mmax + 1); e = np.zeros(mmax + 1)
        js = np.arange(0, mmax // 2 + 1)

        def add(D1, D2, wgt, m):
            aj = B.a(D1, D2, J)
            da = aj * B.dlog_a(D1, D2, J)
            np.add.at(c, m, (wgt * da)); np.add.at(e, m, wgt * (-2 * (0.5 + n + m)) * aj)

        ins = np.arange(0.5, n, 1.0)
        for s in ins:
            J = js; add(0.5 + s, 0.5 + n - s, 1 / s, 2 * js)
        rs = np.arange(0.5, mmax / 2 + 1e-9, 1.0)
        R, Jg = np.meshgrid(rs, js, indexing='ij')
        m = np.rint(2 * R + 2 * Jg).astype(int); ok = m <= mmax
        R, Jg, m = R[ok], Jg[ok], m[ok]
        J = Jg; add(0.5 + R, 0.5 + n + R, 1 / R, m)                 # s = -rho
        J = Jg; add(0.5 + n + R, 0.5 + R, 1 / (n + R), m)           # s = n + rho
        return c, e

    def main():
        print('Bgap(0) =', Bgap(0.0), '(must be 1)')
        dB = np.sum(polygamma(1, 1 + np.arange(0, 400000)) / (np.arange(0, 400000) + 0.5))
        print("Bgap'(0) =", dB)
        f = lambda sg: 1 + Bgap_shift(sg)
        ss = brentq(f, -0.2499999, -0.01)
        print(f'full-twist saddle sigma* = {ss:.6f}  (linearized -1/Bgap\'(0) = {-1/dB:.6f});  lowest line dimension '
              f'1/2 + sqrt(1/4 + sigma*) = {0.5 + np.sqrt(0.25 + ss):.6f}')
        print('bubble normalization  sum_j d_D1 a_j  vs  -psi\'(D1)/(2 pi):')
        for D1, D2 in ((0.5, 0.5), (0.5, 2.5), (1.0, 1.0), (0.8366, 1.7), (2.3, 0.6)):
            js = np.arange(0, 2000000)
            da = B.a(D1, D2, js) * B.dlog_a(D1, D2, js)
            tail = da[-1] * js[-1]                          # da ~ c/j^2
            print(f'  D1={D1} D2={D2}: {da.sum() + tail:+.7f} vs {-polygamma(1, D1) / (2 * np.pi):+.7f}')

    def cli(argv):
        with fuzzy.logged('largeN/fulltwist.log'):
            main()
    return namespace(locals(), globals())


@section
def largeN_full_twist():
    """Leading large-N saddle of the O(2N) model with the full monodromy e^{2 pi i v} on S^1 x H^2 (d = 3), after
Giombi, Helmreich, Jepsen, Khanchandani (arXiv:2102.11815), Eqs. (4.6)-(4.7).

Defect dimensions Delta_s = 1/2 + sqrt(sigma + s^2), s in Z + v. The saddle condition is the vanishing of the
Phi Phibar term in the bulk OPE, Gamma((3-d)/2) sum_s Gamma(Delta_s)/Gamma(3-d+Delta_s) = 0. Continuing in d at
d = 3 - delta, the large-|s| part gives 2[zeta'(0,v) + zeta'(0,1-v)] = -2 log(2 sin pi v), and the rest is finite:
    sum_{s in Z+v} [ psi(1/2 + sqrt(sigma + s^2)) - log|s| ] = -log(2 sin pi v).
Gate: GHJK Eq. (4.21) gives sigma* = -0.168, Delta_1/2 = 0.786, Delta_3/2 = 1.943 at v = 1/2.
"""
    from mpmath import mp, mpf, psi, log, sqrt, sin, pi, nsum, inf, findroot
    mp.dps = 20


    def lhs(sigma, v):
        term = lambda s: psi(0, mpf(1) / 2 + sqrt(sigma + s * s)) - log(abs(s))
        return nsum(lambda k: term(k + v), [0, inf]) + nsum(lambda k: term(-(k + 1) + v), [0, inf])


    def sigma_star(v, guess=None):
        lo = -min(v, 1 - v) ** 2 + mpf('1e-12')
        g = lambda sg: lhs(sg, v) + log(2 * sin(pi * v))
        return findroot(g, (lo, mpf(0)), solver='anderson')

    def main():
        v = mpf('0.5')
        sg = sigma_star(v)
        print('v=1/2: sigma*', float(sg), ' Delta_1/2', float(0.5 + sqrt(sg + 0.25)), ' Delta_3/2', float(0.5 + sqrt(sg + 2.25)))
        # angle at which the s = v - 1 mode reaches dimension 1:  sqrt(sigma + (1-v)^2) = 1/2
        f = lambda vv: sigma_star(vv) + (1 - vv) ** 2 - mpf(1) / 4
        for vv in ('0.30', '0.35', '0.38', '0.40', '0.45'):
            vv = mpf(vv); s_ = sigma_star(vv)
            print(f'v={float(vv):.2f}: sigma* {float(s_):+.5f}  Delta_v {float(0.5 + sqrt(s_ + vv**2)):.4f}  Delta_(v-1) {float(0.5 + sqrt(s_ + (1-vv)**2)):.4f}')
        vc = findroot(f, mpf('0.38'))
        print('large-N alpha_c (Delta_{alpha-1} = 1):', float(vc))

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/largeN_full_twist.log'):
            main()
    return namespace(locals(), globals())


@section
def sector_position():
    """Position-space form of the fixed-sector integral (large-M partial twist on S^1 x H^2).

AdS2 propagator G_D(u) = Q_{D-1}(cosh u)/(2 pi), H^2 transform 1/(nu^2 + (D-1/2)^2), Plancherel rho = nu tanh(pi nu)/(2 pi).
Sector kernel without defect (common normalization of Eq. spectral): U_k(u) = sum_{s in Z} G_{1/2+|s|} G_{1/2+|k-s|} =? e^{-|k|u}/(8 sinh u).
Claim: W_k = U_k^{-1} (H^2 convolution) = (2/pi^2) (1/sinh u) d/du (e^{-|k|u}/sinh u), so that
    I_k = int rho(nu) dPi_k(nu)/U_k(nu) dnu = int_{H^2} W_k dPi_k = (32/pi) int_0^inf du U_k'(u) dPi_k(u)   (analytic regularization).
Checks: (1) U_k closed form vs the s-sum; (2) spectral transform of U_k vs (pi/8)|Gamma((|k|+1/2+i nu)/2)/Gamma((|k|+3/2+i nu)/2)|^2.
"""
    from mpmath import mp, mpf, legenq, cosh, sinh, exp, pi, quad, gamma, inf, legenp, tanh, nsum, fabs


    def main():
        mp.dps = 20


        def G(D, u):
            return legenq(D - 1, 0, cosh(u), type=3).real / (2 * pi)


        def U_sum(k, u, smax=60):
            return sum(G(mpf(1) / 2 + abs(s), u) * G(mpf(1) / 2 + abs(k - s), u) for s in range(-smax, smax + abs(k) + 1))


        def U_closed(k, u):
            return exp(-abs(k) * u) / (8 * sinh(u))


        for k in (0, 1, 3):
            for u in (mpf('0.3'), mpf('1.0'), mpf('2.5')):
                print(f'k={k} u={float(u)}: s-sum {float(U_sum(k, u)):.12f}  closed {float(U_closed(k, u)):.12f}')

        # H^2 transform of a radial function F(u): F~(nu) = 2 pi int_0^inf sinh u du P_{-1/2+i nu}(cosh u) F(u)
        def transform(F, nu):
            return 2 * pi * quad(lambda u: sinh(u) * legenp(-mpf(1) / 2 + 1j * nu, 0, cosh(u), type=3).real * F(u), [0, 1, 5, 20, 60])


        def U_spec(k, nu):
            a = abs(k) + mpf(1) / 2
            return pi / 8 * fabs(gamma((a + 1j * nu) / 2) / gamma((a + 1 + 1j * nu) / 2)) ** 2


        for k in (0, 2):
            for nu in (mpf('0.5'), mpf('2.0')):
                print(f'k={k} nu={float(nu)}: transform {float(transform(lambda u: U_closed(k, u), nu)):.10f}  closed {float(U_spec(k, nu)):.10f}')

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/sector_position.log'):
            main()
    return namespace(locals(), globals())


@section
def sector_check2():
    """Check int rho F~/U_k dnu = (32/pi) int_0^inf du U_k'(u) F(u) (analytic regularization) for F = G_D - G_D'.
"""
    from mpmath import mp, mpf, legenq, cosh, sinh, exp, pi, quad, gamma, inf, tanh, fabs, psi


    def main():
        mp.dps = 30
        from mpmath import log, euler
        def G(D, u):
            if u < mpf('1e-12'):                       # coincident expansion, error O(u^2 log u)
                return -(log(u / 2) + euler + psi(0, D)) / (2 * pi)
            return legenq(D - 1, 0, cosh(u), type=3).real / (2 * pi)


        def Uspec(k, nu):
            a = abs(k) + mpf(1) / 2
            return pi / 8 * fabs(gamma((a + 1j * nu) / 2) / gamma((a + 1 + 1j * nu) / 2)) ** 2


        def lhs(k, D, Dp):
            f = lambda nu: nu * tanh(pi * nu) / (2 * pi) * (1 / (nu**2 + (D - 0.5)**2) - 1 / (nu**2 + (Dp - 0.5)**2)) / Uspec(k, nu)
            return quad(f, [0, 1, 5, 20, 100, inf])


        def Up(k, u):
            k = abs(k)
            return -exp(-k * u) * (k * sinh(u) + cosh(u)) / (8 * sinh(u)**2)


        def rhs(k, D, Dp):
            F = lambda u: G(D, u) - G(Dp, u)
            F0 = -(psi(0, D) - psi(0, Dp)) / (2 * pi)
            inner = quad(lambda u: Up(k, u) * F(u) + F0 / (8 * u**2), [mpf('1e-9'), mpf('1e-4'), mpf('0.01'), mpf('0.5'), 1])
            outer = quad(lambda u: Up(k, u) * F(u), [1, 5, 20, inf])
            return 32 / pi * (inner + outer + F0 / 8)


        for k in (0, 1, 2):
            for D, Dp in ((mpf('0.5'), mpf('1.5')), (mpf('0.8'), mpf('2.3'))):
                print(f'k={k} D={float(D)} Dp={float(Dp)}: spectral {float(lhs(k, D, Dp)):+.10f}  position {float(rhs(k, D, Dp)):+.10f}')

    def cli(argv):
        mp.dps = 30
        with fuzzy.logged('analytic/sector_check2.log'):
            main()
    return namespace(locals(), globals())


@section
def Jbubble():
    """J_k(D1, D2) = (32/pi) int_0^inf du U_k'(u) G_D1(u) G_D2(u), analytically regularized at u -> 0.
G_D(u) = Q_{D-1}(cosh u)/(2 pi) ~ -(L + g + psi(D))/(2 pi), L = log(u/2), g = Euler gamma.
U_k'(u) = -1/(8u^2) + O(1): the only singular part of the integrand is -(1/(32 pi^2 u^2)) (L+c1)(L+c2), c_i = g + psi(D_i),
whose continued integral over (0, a] is -(1/(32 pi^2)) * [-(L^2 + 2L + 2)/u - (c1+c2)(L+1)/u - c1 c2/u] at u = a.
"""
    from mpmath import mp, mpf, legenq, cosh, sinh, exp, pi, quad, log, euler, psi, inf
    mp.dps = 30


    def G(D, u):
        return legenq(D - 1, 0, cosh(u), type=3).real / (2 * pi)


    def Up(k, u):
        k = abs(k)
        return -exp(-k * u) * (k * sinh(u) + cosh(u)) / (8 * sinh(u)**2)


    def J(k, D1, D2, a=mpf('0.25')):
        c1, c2 = euler + psi(0, D1), euler + psi(0, D2)
        sing = lambda u: -((log(u / 2) + c1) * (log(u / 2) + c2)) / (32 * pi**2 * u**2)
        inner = quad(lambda u: Up(k, u) * G(D1, u) * G(D2, u) - sing(u), [mpf('1e-9'), mpf('1e-5'), mpf('1e-3'), mpf('0.05'), a])
        La = log(a / 2)
        cont = -(1 / (32 * pi**2)) * (-(La**2 + 2 * La + 2) / a - (c1 + c2) * (La + 1) / a - c1 * c2 / a)
        outer = quad(lambda u: Up(k, u) * G(D1, u) * G(D2, u), [a, 1, 4, 15, inf])
        return 32 / pi * (inner + cont + outer)

    def main():
        import itertools
        for k in (0, 1):
            print(f'--- k = {k}')
            rows = []
            for D1, D2 in [(0.5, 0.5), (0.5, 1.5), (1.0, 1.0), (0.7, 1.3), (0.5, 2.5), (1.5, 1.5), (0.9, 2.1), (2.0, 2.0), (1.2, 2.8)]:
                D1, D2 = mpf(D1), mpf(D2)
                val = J(k, D1, D2)
                rows.append((float(D1), float(D2), float(val), float(psi(0, D1) + psi(0, D2))))
                print(f'D1={float(D1):.2f} D2={float(D2):.2f} S={float(D1+D2):.2f}: J = {float(val):+.10f}   psi1+psi2 = {rows[-1][3]:+.6f}', flush=True)

    def cli(argv):
        mp.dps = 30
        with fuzzy.logged('analytic/Jbubble.log'):
            main()
    return namespace(locals(), globals())


@section
def Jprecise():
    """High-precision J_k(D1, D2) (analytic regularization), for integer-relation fits.
"""
    from mpmath import mp, mpf, legenq, cosh, sinh, exp, pi, quad, log, euler, psi, inf
    mp.dps = 50


    def G(D, u):
        if u < mpf('1e-13'):
            return -(log(u / 2) + euler + psi(0, D)) / (2 * pi)
        return legenq(D - 1, 0, cosh(u), type=3).real / (2 * pi)


    def Up(k, u):
        k = abs(k)
        return -exp(-k * u) * (k * sinh(u) + cosh(u)) / (8 * sinh(u)**2)


    def J(k, D1, D2, a=mpf('0.25')):
        c1, c2 = euler + psi(0, D1), euler + psi(0, D2)
        sing = lambda u: -((log(u / 2) + c1) * (log(u / 2) + c2)) / (32 * pi**2 * u**2)
        inner = quad(lambda u: Up(k, u) * G(D1, u) * G(D2, u) - sing(u),
                     [mpf('1e-16'), mpf('1e-12'), mpf('1e-8'), mpf('1e-5'), mpf('1e-3'), mpf('0.03'), a])
        La = log(a / 2)
        cont = -(1 / (32 * pi**2)) * (-(La**2 + 2 * La + 2) / a - (c1 + c2) * (La + 1) / a - c1 * c2 / a)
        outer = quad(lambda u: Up(k, u) * G(D1, u) * G(D2, u), [a, 1, 4, 15, 40, inf])
        return 32 / pi * (inner + cont + outer)

    def main():
        import sys, time
        t = time.time()
        for (k, D1, D2) in [(1, '1', '1'), (0, '0.5', '1.5'), (0, '0.5', '2.5'), (2, '1', '2'), (2, '1.5', '1.5')]:
            v = J(k, mpf(D1), mpf(D2))
            print(k, D1, D2, mp.nstr(v, 25), mp.nstr(v * pi**3 / 2, 20), f'[{time.time()-t:.0f}s]', flush=True)

    def cli(argv):
        mp.dps = 50
        with fuzzy.logged('analytic/Jprecise.log'):
            main()
    return namespace(locals(), globals())


@section
def tadpole_test():
    """Tadpole lemma test: T_k(n) = (32/pi) int_0^inf du U_k'(u) G_{1/2+|k-n|}(u) P_{|n|-1/2}(cosh u), analytic regularization.
F = G0 * h with h = e^{i n theta} P_{|n|-1/2}(cosh u), a regular source-free solution with h(0,0) = 1.
Spectator case n = 0 should give d_D1 J(1/2, 1/2+k) * (-4/pi)^-1 ... ; compare with -(4|k|/pi) * h(0) / (32/pi) scale.
"""
    from mpmath import mp, mpf, legenq, legenp, cosh, sinh, exp, pi, quad, log, euler, psi, inf
    mp.dps = 40


    def G(D, u):
        if u < mpf('1e-13'):
            return -(log(u / 2) + euler + psi(0, D)) / (2 * pi)
        return legenq(D - 1, 0, cosh(u), type=3).real / (2 * pi)


    def P(nu, u):
        return legenp(nu, 0, cosh(u), type=3).real


    def Up(k, u):
        k = abs(k)
        return -exp(-k * u) * (k * sinh(u) + cosh(u)) / (8 * sinh(u)**2)


    def T(k, n, a=mpf('0.25')):
        D = mpf(1) / 2 + abs(k - n)
        nu = abs(n) - mpf(1) / 2
        c = euler + psi(0, D)
        # singular part: -(1/(8u^2)) * (-(L + c)/(2pi)) * 1, with P(1) = 1
        sing = lambda u: (log(u / 2) + c) / (16 * pi * u**2)
        inner = quad(lambda u: Up(k, u) * G(D, u) * P(nu, u) - sing(u),
                     [mpf('1e-16'), mpf('1e-12'), mpf('1e-8'), mpf('1e-5'), mpf('1e-3'), mpf('0.03'), a])
        La = log(a / 2)
        cont = (1 / (16 * pi)) * (-(La + 1) / a - c / a)          # continued int_0^a (L + c)/u^2
        outer = quad(lambda u: Up(k, u) * G(D, u) * P(nu, u), [a, 1, 3, 8])
        return 32 / pi * (inner + cont + outer)

    def main():
        for k in (0, 1, 2, 3):
            for n in (0, 1, -1, 2):
                print(f'k={k} n={n:+d}: T = {mp.nstr(T(k, n), 14)}   (-|k|/(2pi) * 4/pi * pi/32 ... raw |k| ratio {mp.nstr(T(k, n) * 2 * pi / max(abs(k), 1), 10)})', flush=True)

    def cli(argv):
        mp.dps = 40
        with fuzzy.logged('analytic/tadpole_test.log'):
            main()
    return namespace(locals(), globals())


@section
def tadpole_proof():
    """Checks for the general-kappa proof of the tadpole identities (P1), (P2).

Antiderivative: for any p, q with (1/sinh)(sinh p')' = mu(mu+1) p, (1/sinh)(sinh q')' = lam(lam+1) q and
E = e^{-su}/sinh u with s = lam+mu+1,
  F = e^{-su}{ [2 lam mu sinh u + e^u(lam+mu+coth u)] pq + [(lam+1)e^u - lam e^{-u}] p'q
              + [(mu+1)e^u - mu e^{-u}] pq' + 2 sinh u p'q' }   satisfies   F' = E' pq.
Since the ODE only sees mu(mu+1), mu -> -mu-1 gives a second antiderivative, for s = lam-mu.
"""
    import sympy as sp
    from mpmath import mp, mpf, psi, pi, tan, nsum, inf
    # 1. F' = E' pq identically (symbolic p, q)

    def main():
        u, mu, lam = sp.symbols('u mu lam')
        p, q = sp.Function('p')(u), sp.Function('q')(u)
        s = lam + mu + 1
        E = sp.exp(-s*u)/sp.sinh(u)
        F = sp.exp(-s*u)*((2*lam*mu*sp.sinh(u) + sp.exp(u)*(lam + mu + sp.cosh(u)/sp.sinh(u)))*p*q
                          + ((lam+1)*sp.exp(u) - lam*sp.exp(-u))*p.diff(u)*q
                          + ((mu+1)*sp.exp(u) - mu*sp.exp(-u))*p*q.diff(u)
                          + 2*sp.sinh(u)*p.diff(u)*q.diff(u))
        ode = {p.diff(u, 2): mu*(mu+1)*p - sp.cosh(u)/sp.sinh(u)*p.diff(u),
               q.diff(u, 2): lam*(lam+1)*q - sp.cosh(u)/sp.sinh(u)*q.diff(u)}
        res = sp.simplify((F.diff(u).subs(ode) - E.diff(u)*p*q).rewrite(sp.exp))
        print('1. F_u - E_u pq =', res)

        # 2. closed form t(s) = pi^3 J_k(1/2+|s|, 1/2+|k-s|) against the independent quadrature table
        mp.dps = 30
        def t(k, x):
            return 2*k + (k - 2*x)*(psi(0, mpf(1)/2 + abs(x)) - psi(0, mpf(1)/2 + abs(k - x)))
        print('2. inside line, J/(2/pi^3) = t/2:')
        table = {(2, 0.25): 0.75619808, (2, 0.625): 1.72725372, (3, 0.125): -0.45633659, (3, 0.875): 2.46874790,
                 (1, 0.375): 0.94806330}
        for (k, x), v in table.items():
            print(f'   k={k} s={x}: closed {float(t(k, mpf(x))/2):+.8f}   quadrature {v:+.8f}')
        print('   outside line, J = t/pi^3:')
        for (k, D1), v in {(2, 0.6): -3.36e-2, (2, 1.3): -1.08e-2, (3, 0.6): -8.27e-2, (3, 1.3): -3.01e-2}.items():
            x = mpf(1)/2 - D1                      # s < 0 with D1 = 1/2 + |s|
            print(f'   k={k} J({D1},{D1+k}): closed {float(t(k, x)/pi**3):+.4e}   quadrature {v:+.4e}')

        # 3. lattice sums: sum_{Z+alpha} t = k pi (1-2 alpha) tan(pi alpha), sum_Z t = 0
        print('3. lattice sums')
        for k in [1, 2, 3, 5, 8]:
            for a in [mpf(0), mpf('0.1'), mpf('0.25'), mpf('0.4')]:
                S = nsum(lambda n: t(k, a + n), [-inf, inf])
                print(f'   k={k} alpha={float(a)}: sum = {mp.nstr(S, 15)}   k pi(1-2a)tan(pi a) = {mp.nstr(k*pi*(1-2*a)*tan(pi*a), 15)}')

        # 4. the rational remainder R(alpha) = -2k^2 exactly
        al = sp.symbols('alpha')
        for k in range(1, 13):
            R = sum((2*k-1-2*al-4*m)*sum(1/(sp.Rational(1, 2)+al+r) for r in range(m)) for m in range(k)) \
                + sum((2*k+1+2*al-4*m)*sum(1/(sp.Rational(1, 2)-al+r) for r in range(m)) for m in range(1, k+1))
            assert sp.simplify(R + 2*k**2) == 0
        print('4. R(alpha) = -2 k^2 for k = 1..12')

    def cli(argv):
        mp.dps = 30
        with fuzzy.logged('analytic/tadpole_proof.log'):
            main()
    return namespace(locals(), globals())


@section
def PQ_conjecture():
    """Conjecture: Phi(mu, lam) = int_0^inf du d_u[e^{-(lam+mu+1)u}/sinh u] P_mu(cosh u) Q_lam(cosh u) = mu - lam  (AR at u -> 0).
"""
    from mpmath import mp, mpf, legenq, legenp, cosh, sinh, exp, pi, quad, log, euler, psi, inf
    mp.dps = 40


    def Q(nu, u):
        if u < mpf('1e-13'):
            return -(log(u / 2) + euler + psi(0, nu + 1))
        return legenq(nu, 0, cosh(u), type=3).real


    def P(nu, u):
        if u < mpf('1e-13'):
            return mpf(1)
        return legenp(nu, 0, cosh(u), type=3).real


    def Phi(mu, lam, a=mpf('0.25')):
        s = lam + mu + 1
        dE = lambda u: -exp(-s * u) * (s * sinh(u) + cosh(u)) / sinh(u)**2
        c = euler + psi(0, lam + 1)
        sing = lambda u: (log(u / 2) + c) / u**2
        inner = quad(lambda u: dE(u) * P(mu, u) * Q(lam, u) - sing(u),
                     [mpf('1e-16'), mpf('1e-12'), mpf('1e-8'), mpf('1e-5'), mpf('1e-3'), mpf('0.03'), a])
        La = log(a / 2)
        cont = -(La + 1) / a - c / a
        outer = quad(lambda u: dE(u) * P(mu, u) * Q(lam, u), [a, 1, 4, 15, 40, inf])
        return inner + cont + outer

    def main():
        for mu, lam in [('-0.5', '1.5'), ('0.3', '0.9'), ('-0.2', '2.4'), ('1.1', '0.4'), ('0', '0'), ('0.7', '2.2')]:
            mu, lam = mpf(mu), mpf(lam)
            print(f'mu={float(mu)} lam={float(lam)}: Phi = {mp.nstr(Phi(mu, lam), 14)}   mu - lam = {float(mu - lam)}', flush=True)

    def cli(argv):
        mp.dps = 40
        with fuzzy.logged('analytic/PQ_conjecture.log'):
            main()
    return namespace(locals(), globals())


@section
def PQ_antiderivative():
    """Look for F = f1 pq + f2 p_u q + f3 p q_u + f4 p_u q_u with F_u = E_u pq, E = e^{-su}/sinh u, s = lam+mu+1.
Work in y = e^{-u}; f_i = y^{s+1} g_i(y), g_i = N_i(y)/(y^m (1-y^2)^K), N_i polynomial.
"""
    import sympy as sp


    def main():
        y, mu, lam = sp.symbols('y mu lam')
        s = lam + mu + 1
        a, b = mu*(mu+1), lam*(lam+1)
        Ds = lambda g: -(s+1)*g - y*sp.diff(g, y)            # y^{-(s+1)} d/du (y^{s+1} g)
        coth = (1+y**2)/(1-y**2)
        Eu = Ds(2/(1-y**2))
        for K, m, deg in [(1, 2, 6), (2, 2, 8), (2, 4, 10), (3, 4, 12)]:
            cs, gs = [], []
            for i in range(4):
                c = sp.symbols(f'c{i}_0:{deg+1}')
                cs += c
                gs.append(sum(c[j]*y**j for j in range(deg+1))/(y**m*(1-y**2)**K))
            g1, g2, g3, g4 = gs
            eqs = [Ds(g1) + a*g2 + b*g3 - Eu,
                   g1 + Ds(g2) - g2*coth + b*g4,
                   g1 + Ds(g3) - g3*coth + a*g4,
                   g2 + g3 + Ds(g4) - 2*coth*g4]
            lin = []
            for e in eqs:
                num = sp.expand(sp.numer(sp.together(e)))
                lin += sp.Poly(num, y).coeffs()
            sol = sp.solve(lin, cs, dict=True)
            print('K', K, 'm', m, 'deg', deg, 'solutions', len(sol), flush=True)
            for so in sol:
                for i, g in enumerate(gs):
                    print(f'  g{i+1} =', sp.factor(sp.simplify(g.subs(so))))
            if sol:
                break

    def cli(argv):
        with fuzzy.logged('analytic/PQ_antiderivative.log'):
            main()
    return namespace(locals(), globals())


@section
def P1_exact():
    """(P1) in exact arithmetic: lambda_k = d_D1 J_k(1/2, 1/2+k) should equal k/(2 pi).

J_k = S_F + sum_{j'} c_{k,j'} M_{j'} in analytic regularization, with
  S_F = (2/pi^2) g(D1) g(D2) on the inside line and 0 on the outside line, g(D) = Gamma(D)/Gamma(D-1/2);
  M_{j'}(D1, D2) = int_{H^2} G_D1 G_D2 P_{j'}(cosh u);  c_{k,j'} = -(4/pi^2)(2j'+1) Gamma(k-n)Gamma(n+3/2)/(n! Gamma(k-n-1/2)), j' = k-2-2n.
At the corner: d_D1 S_F = (1/2) d/ds S_F(1/2+s, 1/2+k-s)|_0 = sqrt(pi) Gamma(k+1/2)/(pi^2 Gamma(k)),
  d_D1 M_{j'} = -(pi/4) int P_{-1/2} P_{j'} G_{1/2+k} = -(pi/4) sum_D' w_D'/(D'-1/2)^2, with P_{j'}(z) G_D = sum w_D' G_D'
  from z G_D = [D G_{D+1} + (D-1) G_{D-1}]/(2D-1), and int P_{-1/2} G_D = 1/(D-1/2)^2.
"""
    import sympy as sp
    R = sp.Rational
    half = R(1, 2)


    def c_coef(k, n):
        return -4 / sp.pi**2 * (2 * (k - 2 - 2 * n) + 1) * sp.gamma(k - n) * sp.gamma(n + R(3, 2)) / (sp.factorial(n) * sp.gamma(k - n - half))


    def zpow_terms(D, poly_coeffs):
        """P(z) G_D as {D': weight} for P(z) = sum_m a_m z^m."""
        out = {}
        cur = {D: sp.Integer(1)}
        for m, a in enumerate(poly_coeffs):
            if m > 0:
                new = {}
                for d, w in cur.items():
                    new[d + 1] = new.get(d + 1, 0) + w * d / (2 * d - 1)
                    new[d - 1] = new.get(d - 1, 0) + w * (d - 1) / (2 * d - 1)
                cur = new
            if a != 0:
                for d, w in cur.items():
                    out[d] = out.get(d, 0) + a * w
        return out

    def main():
        z = sp.symbols('z')
        for k in range(1, 16):
            SF = sp.sqrt(sp.pi) * sp.gamma(k + half) / (sp.pi**2 * sp.gamma(k))
            cpart = 0
            for n in range((k - 2) // 2 + 1):
                jp = k - 2 - 2 * n
                if jp < 0:
                    continue
                coeffs = sp.Poly(sp.legendre(jp, z), z).all_coeffs()[::-1]
                terms = zpow_terms(half + k, coeffs)
                dM = -sp.pi / 4 * sum(w / (d - half)**2 for d, w in terms.items())
                cpart += c_coef(k, n) * dM
            lam = sp.nsimplify(sp.simplify(SF + cpart))
            print(f'k={k}: d_D1 J = {sp.simplify(lam)}   target k/(2 pi) = {sp.Rational(k, 2)}/pi   diff = {sp.simplify(lam - sp.Rational(k, 2) / sp.pi)}', flush=True)

    def cli(argv):
        with fuzzy.logged('analytic/P1_exact.log'):
            main()
    return namespace(locals(), globals())


@section
def paper_form_check():
    """Check the antiderivative exactly as written in the paper (dimensions Delta_1, Delta_2, exponent kappa),
both lines, and the spectator normalization d_Delta G_Delta|_{1/2} = -(pi/4) P_{-1/2}(cosh chi).
"""
    import sympy as sp
    from mpmath import mp, mpf, legenq, legenp, cosh, pi, diff


    def main():
        x, D1, D2 = sp.symbols('chi Delta_1 Delta_2')
        f1, f2 = sp.Function('f1')(x), sp.Function('f2')(x)


        def check(k, A, B1):
            """A = coefficient of sinh f1 f2, B1 = coefficient multiplying f1 f2'."""
            F = sp.exp(-k*x)*((A*sp.sinh(x) + sp.exp(x)*(k - 1 + sp.cosh(x)/sp.sinh(x)))*f1*f2
                              + (D2*sp.exp(x) - (D2-1)*sp.exp(-x))*f1.diff(x)*f2
                              + B1*f1*f2.diff(x) + 2*sp.sinh(x)*f1.diff(x)*f2.diff(x))
            ode = {f1.diff(x, 2): D1*(D1-1)*f1 - sp.cosh(x)/sp.sinh(x)*f1.diff(x),
                   f2.diff(x, 2): D2*(D2-1)*f2 - sp.cosh(x)/sp.sinh(x)*f2.diff(x)}
            return sp.simplify((F.diff(x).subs(ode) - (sp.exp(-k*x)/sp.sinh(x)).diff(x)*f1*f2).rewrite(sp.exp))


        print('inside  (k = D1+D2-1):', check(D1+D2-1, 2*(D1-1)*(D2-1), D1*sp.exp(x) - (D1-1)*sp.exp(-x)))
        print('outside (k = D2-D1):  ', check(D2-D1, -2*D1*(D2-1), (1-D1)*sp.exp(x) + D1*sp.exp(-x)))

        mp.dps = 30
        G = lambda D, c: legenq(D - 1, 0, c, type=3).real/(2*pi)
        for chi in [mpf('0.3'), mpf('1.7')]:
            c = cosh(chi)
            print(f'chi={float(chi)}: dG/dDelta = {mp.nstr(diff(lambda D: G(D, c), mpf(1)/2), 15)}'
                  f'   -(pi/4)P = {mp.nstr(-pi/4*legenp(-mpf(1)/2, 0, c, type=3).real, 15)}')

    def cli(argv):
        mp.dps = 30
        with fuzzy.logged('analytic/paper_form_check.log'):
            main()
    return namespace(locals(), globals())


@section
def sector_closed():
    """Fixed-sector identity in closed form (2026-10-24).

Building block (numerically identified, sector_psi.py / v_closed_form.py):
    V_k(L) = int rho G~_L / U_k = v_k - (2/pi) F_k(L) + sum_{j = k-2, k-4, ... >= 0} c_{k,j} / ((L-j-1)(L+j)),
    F_k(L) = Gamma((L+1+k)/2) Gamma((L+1-k)/2) / [Gamma((L+k)/2) Gamma((L-k)/2)],   c_{k,j} = (2/pi)(2j+1) Res_{L=j+1} F_k.
Bubble trace W_k(D1, D2) = sum_j e_j V_k(D1 + D2 + 2j), and on the two lattice lines (D1, D2) = (1/2+|s|, 1/2+|k-s|):
  * v-part: sum_j e_j = -(psi(D1)+psi(D2))/(2 pi) + const, cancelled by the spectator (gap equation);
  * F-part: outside [0,k] the sum is a very-well-poised 3F2(1) with regularized value 0; inside it is Dougall's 5F4(1)
    with d = 1, giving -(1/pi) g(1/2+s) g(1/2+k-s), g(D) = Gamma(D)/Gamma(D-1/2); so W_F = (2/pi^2) h_k(s) 1_[0,k] + const;
  * c-part: sum_j e_j/((L_j-j'-1)(L_j+j')) = int_{H^2} G_D1 G_D2 P_j'(cosh chi) (bubble at nu = i(j'+1/2)), reduced to
    digamma differences by z G_D = [D G_{D+1} + (D-1) G_{D-1}]/(2D-1) and <G_a, G_b> = [psi(b)-psi(a)]/(2 pi (b-a)(a+b-1)).
Identity:  [sum_{s in Z+alpha} - sum_{s in Z}] W(s) - (4 A_alpha/pi^2) dW/dD1 (1/2, 1/2+k) = 0,  A_alpha = (pi/2)(1-2 alpha) tan(pi alpha).
usage: python sector_closed.py kmax
"""
    import sys
    import numpy as np
    from numpy.polynomial import legendre as Leg
    from scipy.special import digamma, polygamma, gammaln, gamma
    pi = np.pi


    def A(a):
        return pi / 2 * (1 - 2 * a) * np.tan(pi * a)


    def F(k, L):
        return gamma((L + 1 + k) / 2) * gamma((L + 1 - k) / 2) / (gamma((L + k) / 2) * gamma((L - k) / 2))


    def c_coeffs(k, h=1e-7):
        """c_{k,j} = (2/pi)(2j+1) Res_{L=j+1} F_k for j = k-2, k-4, ..., >= 0 (residue by symmetric limit)."""
        out = {}
        for j in range(k - 2, -1, -2):
            L0 = j + 1.0
            res = 0.5 * h * (F(k, L0 + h) - F(k, L0 - h))
            out[j] = 2 / pi * (2 * j + 1) * res
        return out


    def pair(a, b):
        """int_{H^2} G_a G_b (Plancherel): [psi(b) - psi(a)] / (2 pi (b - a)(a + b - 1)), with the a = b limit."""
        if abs(a - b) < 1e-10:
            return polygamma(1, a) / (2 * pi * (2 * a - 1))
        return (digamma(b) - digamma(a)) / (2 * pi * (b - a) * (a + b - 1))


    def zpow(D, m):
        """z^m G_D as {D': weight}, by z G_D = [D G_{D+1} + (D-1) G_{D-1}]/(2D-1) (never applied at D = 1/2)."""
        terms = {D: 1.0}
        for _ in range(m):
            new = {}
            for d, w in terms.items():
                new[d + 1] = new.get(d + 1, 0.0) + w * d / (2 * d - 1)
                new[d - 1] = new.get(d - 1, 0.0) + w * (d - 1) / (2 * d - 1)
            terms = new
        return terms


    def moment(D1, D2, j):
        """int_{H^2} G_D1 G_D2 P_j(cosh chi) = sum_m coef_m int (z^m1 G_D1)(z^m2 G_D2), m1 + m2 = m, split so that the
        recurrence never steps through D = 1/2 (at most floor(D - 1/2) steps on a propagator with D - 1/2 an integer)."""
        coef = Leg.leg2poly(np.eye(j + 1)[j])          # P_j(z) = sum_m coef[m] z^m
        cap = lambda D: max(0, int(np.floor(D - 0.5 + 1e-12)))   # steps keep every intermediate D >= 3/2
        total = 0.0
        for m in range(j + 1):
            if coef[m] == 0:
                continue
            m1 = min(m, cap(D1))
            m2 = m - m1
            assert m2 <= cap(D2), (D1, D2, m)
            t1, t2 = zpow(D1, m1), zpow(D2, m2)
            total += coef[m] * sum(w1 * w2 * pair(a, b) for a, w1 in t1.items() for b, w2 in t2.items())
        return total


    def W_parts(k, s, cs):
        """F-part (up to a constant) and c-part of the bubble trace at twisted-mode label s (real)."""
        D1, D2 = 0.5 + abs(s), 0.5 + abs(k - s)
        if 0 <= s <= k:
            g = lambda D: np.exp(gammaln(D) - gammaln(D - 0.5)) if D > 0.5 + 1e-14 else 0.0
            wf = 2 / pi ** 2 * g(D1) * g(D2)
        else:
            wf = 0.0
        wc = sum(c * moment(D1, D2, j) for j, c in cs.items())
        return wf, wc


    def identity(k, a, S=4000, eps=1e-5):
        cs = c_coeffs(k)
        W = lambda s: sum(W_parts(k, s, cs))
        tw = sum(W(n + a) for n in range(-S, S + k + 1))
        un = sum(W(float(n)) for n in range(-S, S + k + 1))
        one = lambda sg: (4 * (W(sg * eps) - W(0.0)) - (W(2 * sg * eps) - W(0.0)) / 2) / (3 * eps)   # O(eps^2) one-sided
        dW = 0.5 * (one(1) + one(-1))
        return tw - un, -4 * A(a) / pi ** 2 * dW

    def main():
        kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        for k in range(kmax + 1):
            print(f'k={k}: c_kj = ' + ', '.join(f'{j}:{c * pi ** 2:+.4f}/pi^2' for j, c in c_coeffs(k).items()), flush=True)
            for a in (0.1, 0.25, 0.4, 0.5 - 1e-9):
                lat, spec = identity(k, a)
                print(f'   alpha={a:.3f}: lattice {lat:+.8f}  spectator {spec:+.8f}  sum {lat + spec:+.2e}', flush=True)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('largeN/sector_closed' + ('_k' + argv[1] if len(argv) > 1 else '') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def sector_psi():
    """Test of the reformulated fixed-sector identity (2026-10-24).

The sector-kappa trace of the twisted bubble is a shifted lattice sum over the twisted mode s of
    J_k(s) = int_0^inf dlam rho(lam) B_{1/2+|s|, 1/2+|k-s|}(lam) / U_k(lam),   rho = lam tanh(pi lam)/(2 pi),
and the identity holds for every alpha iff J_k(s) = lam_k [psi(1/2+|s|) + psi(1/2+|k-s|)] + mu_k for all real s
(the shifted-minus-unshifted lattice sum of the psi part is A_alpha, cancelled by the spectator term through the gap
equation; a function whose shifted lattice sums do not depend on alpha is a constant).  J_k(s) itself diverges, but the
UV tail of B_{D1 D2} is universal apart from a term proportional to psi(D1) + psi(D2), so the three-point combination
    K(s) = [J(s) - J(s0)] - r(s) [J(s2) - J(s0)],   r(s) = [Psi(s) - Psi(s0)] / [Psi(s2) - Psi(s0)],
    Psi(s) = psi(1/2+|s|) + psi(1/2+|k-s|),
converges, and the reformulated identity predicts K(s) = 0.
usage: python sector_psi.py k s0 s2 s1 [s1 ...]
"""
    import sys
    import numpy as np
    from scipy.special import digamma
    B = globals()['bubble']()   # was: import bubble as B
    c1_run = globals()['c1_run']()   # was: import c1_run
    def Btilde(D1, D2, lam, J=6000):
        """sum_j a_j/(lam^2 + mu_j^2), mu_j = D1 + D2 + 2j - 1/2, with the a_j ~ 1/(pi j) tail beyond J summed in closed form."""
        lam = np.atleast_1d(lam)
        j = np.arange(J + 1)
        aj = B.a(D1, D2, j)
        mu = D1 + D2 + 2 * j - 0.5
        out = np.zeros_like(lam)
        for i0 in range(0, J + 1, 1000):                      # chunked to bound memory
            sl = slice(i0, min(i0 + 1000, J + 1))
            out += (aj[None, sl] / (lam[:, None] ** 2 + mu[None, sl] ** 2)).sum(1)
        jj = np.arange(J // 2, J + 1)                          # a_j pi j = 1 + e1/j + e2/j^2 fitted on the last half
        e = np.linalg.lstsq(np.stack([1 / jj, 1 / jj ** 2], 1), aj[jj] * np.pi * jj - 1, rcond=None)[0]
        X = J + 0.5                                            # midpoint rule for sum_{j > J}
        c = D1 + D2 - 0.5
        Q = lam ** 2 + (c + 2 * X) ** 2
        al = 1 / (lam ** 2 + c ** 2)
        lead = al * (-np.log(2) - np.log(X) + 0.5 * np.log(Q) - (c / lam) * (np.pi / 2 - np.arctan((c + 2 * X) / lam)))
        out += (lead + e[0] / (X * Q)) / np.pi
        return out


    def J_integrand(k, s, lam):
        return lam * np.tanh(np.pi * lam) / (2 * np.pi) * Btilde(0.5 + abs(s), 0.5 + abs(k - s), lam) / B.U_exact(k, lam)


    def Psi(k, s):
        return digamma(0.5 + abs(s)) + digamma(0.5 + abs(k - s))


    def integrate(r, lam, w):
        sel = lam > 150                                        # tail beyond 400 from a fit in nu^-2, nu^-2 log nu, nu^-4, nu^-4 log nu
        L = lam[sel]
        A = np.stack([L ** -2, L ** -2 * np.log(L), L ** -4, L ** -4 * np.log(L)], 1)
        co = np.linalg.lstsq(A, r[sel], rcond=None)[0]
        X = 400.0
        tail = (co[0] / X + co[1] * (np.log(X) + 1) / X + co[2] / (3 * X ** 3) + co[3] * (3 * np.log(X) + 1) / (9 * X ** 3))
        return np.sum(w * r) + tail


    def K(k, s0, s2, s1s):
        lam, w = c1_run.lam_grid()
        f0, f2 = J_integrand(k, s0, lam), J_integrand(k, s2, lam)
        P0, P2 = Psi(k, s0), Psi(k, s2)
        out = []
        for s in s1s:
            f1 = J_integrand(k, s, lam)
            r = (Psi(k, s) - P0) / (P2 - P0)
            comb = (f1 - f0) - r * (f2 - f0)
            out.append((s, integrate(comb, lam, w), np.sum(w * np.abs(f1 - f0)), r))
        return out

    def main():
        k, s0, s2 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
        s1s = [float(x) for x in sys.argv[4:]]
        for s, val, scale, r in K(k, s0, s2, s1s):
            print(f'k={k} s0={s0} s2={s2} s={s:+.3f}: K = {val:+.3e}   (scale int|J(s)-J(s0)| to 400: {scale:.4f}, r = {r:+.4f})', flush=True)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('largeN/sector_psi_' + '_'.join(argv[1:]) + '.log'):
            main()
    return namespace(locals(), globals())


@section
def v_closed_form():
    """Check of the closed form of V_k(L) = int_0^inf dlam rho(lam) G~_L(lam)/U_k(lam) (2026-10-24), rho = lam tanh(pi lam)/(2 pi):
    V_k(L) - V_k(L0) = -(2/pi)[Ft_k(L) - Ft_k(L0)],   Ft_k(L) = F_k(L) - (pi/2) sum_{j = k-2, k-4, ..} c_{k,j}/((L-j-1)(L+j)),
    F_k(L) = Gamma((L+1+k)/2) Gamma((L+1-k)/2)/[Gamma((L+k)/2) Gamma((L-k)/2)],  c_{k,j} = (2/pi)(2j+1) Res_{L=j+1} F_k.
The c-terms remove the spurious poles of F_k at L = k-1, k-3, ...; only L >= 1 + k enters the bubble sums.
usage: python v_closed_form.py kmax
"""
    import sys
    import numpy as np
    c1_run = globals()['c1_run']()   # was: import c1_run
    B = globals()['bubble']()   # was: import bubble as B
    _sector_psi = sector_psi(); integrate = _sector_psi.integrate
    _sector_closed = sector_closed(); F, c_coeffs = _sector_closed.F, _sector_closed.c_coeffs
    lam, w = c1_run.lam_grid()
    rho = lam * np.tanh(np.pi * lam) / (2 * np.pi)


    def V_num(k, L, L0):
        return integrate(rho * (1 / (lam ** 2 + (L - 0.5) ** 2) - 1 / (lam ** 2 + (L0 - 0.5) ** 2)) / B.U_exact(k, lam), lam, w)


    def V_closed(k, L, L0):
        cs = c_coeffs(k)
        Ft = lambda x: F(k, x) - np.pi / 2 * sum(c / ((x - j - 1) * (x + j)) for j, c in cs.items())
        return -2 / np.pi * (Ft(L) - Ft(L0))

    def main():
        for k in range(int(sys.argv[1]) + 1 if len(sys.argv) > 1 else 8):
            L0 = k + 1.0
            Ls = [k + 1.3, k + 2.0, k + 3.5, k + 6.0, k + 10.0]
            print(f'k={k}: max |V_num - V_closed| over L in {Ls}: {max(abs(V_num(k, L, L0) - V_closed(k, L, L0)) for L in Ls):.1e}')

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('largeN/v_closed_form' + ('_k' + argv[1] if len(argv) > 1 else '') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def c0_probe_4d():
    """Probe for the covariant check: the coincident structure of the large-N fluctuation kernel near d = 4.
On S^1 x H^3 (unit radius) a conformally coupled real scalar has G0 = 1/(8 pi^2 (cosh chi - cos theta)), and a field
twisted by e^{2 pi i alpha} has G_alpha = S/(8 pi^2 sinh chi), S = sum_{s in Z+alpha} e^{i s theta - |s| chi}.
Along the circle (chi -> 0): Re dG(theta,0) = dg + a theta^2 + ...  The slab (one circle mode at a time) limit of the
kernel dPi is a/(2 pi^2) + dg^2 and the isotropic (dimensional-regularization) limit is dg^2 (see notes).
usage: python c0_probe_4d.py
"""
    import sympy as sp


    def main():
        th, al = sp.symbols('theta alpha', positive=True)
        z, w = sp.exp(sp.I*th), sp.exp(-sp.I*th)
        # -dS/dchi at chi = 0, i.e. sum_s |s| e^{i s theta} (Abel-summed), for s = n + alpha >= 0 and s = -(n + 1 - alpha) < 0
        first = sp.exp(sp.I*al*th)*(al/(1 - z) + z/(1 - z)**2)
        second = sp.exp(-sp.I*(1 - al)*th)*((1 - al)/(1 - w) + w/(1 - w)**2)
        G_al = -(first + second)/(8*sp.pi**2)          # G_alpha(theta, chi -> 0)
        G_0 = 1/(8*sp.pi**2*(1 - sp.cos(th)))
        dG = sp.simplify(sp.re(sp.expand_complex(G_al)) - G_0) if False else None
        ser = sp.series(sp.expand_complex(G_al - G_0), th, 0, 3).removeO()
        ser = sp.simplify(sp.re(sp.expand_complex(ser)))
        c0 = sp.factor(sp.simplify(ser.subs(th, 0)))
        a = sp.factor(sp.simplify(sp.diff(ser, th, 2).subs(th, 0)/2))
        print('dg =', c0, '   expected -alpha(1-alpha)/(8 pi^2)')
        print('a  =', a)
        dg = -al*(1 - al)/(8*sp.pi**2)
        print('slab - iso   = a/(2 pi^2)         =', sp.factor(a/(2*sp.pi**2)))
        print('iso (DR)     = dg^2               =', sp.factor(dg**2))
        print('slab         = a/(2 pi^2) + dg^2  =', sp.factor(sp.simplify(a/(2*sp.pi**2) + dg**2)))
        print('c_iso near 4d = 16 pi^5 eps dg^2 =', sp.factor(16*sp.pi**5*dg**2), '* eps   (expected pi/4 a^2(1-a)^2)')
        print('stress-tensor term, -16 pi^5 eps a/(2 pi^2) =', sp.factor(-16*sp.pi**5*a/(2*sp.pi**2)), '* eps')

    def cli(argv):
        with fuzzy.logged('analytic/covariant_entropy/c0_probe_4d.log'):
            main()
    return namespace(locals(), globals())


@section
def c1_general_d():
    """General-d stress-tensor term of the large-N partial-twist entropy coefficient, and its two endpoints.
Momentum space near coincident points is flat: one circle direction p1 = kappa and d-1 hyperbolic directions.
The traceless term of the kernel, dPi_T = G0 * t (y1^2 - y_perp^2/(d-1)), gives the ratio R = K (p1^2 - p_perp^2/(d-1))/p^(d+2),
  K = -8 t Gamma(d-2) / (g_d^2 pi^(d/2) 2^(4-d) Gamma(2-d/2)),  g_d = Gamma(d/2-1)/(4 pi^(d/2)),
and a spherical cutoff minus the slab (one circle mode at a time) gives V_sph - V_slab = -K Omega_{d-1} J(d)/(2 pi)^d,
  J(d) = -2 int_0^1 (1-c^2)^((d-3)/2) (c^2 - (1-c^2)/(d-1)) log c dc,  Omega_{d-1} = area of S^(d-2).
The entropy coefficient of Ftilde = sin(pi(d-2)/2) log<D> is c_T = -sin(pi(d-2)/2) * 2 pi Vol(H^(d-1)) * (V_sph - V_slab).
Endpoints: d = 3 with t = -h/(2 pi) (paper eq. Gexpansion) must give 64 h/(3 pi); d = 4 - eps with t = 2a,
a = -alpha^2(1-alpha)^2/(32 pi^2) (c0_probe_4d.py), must give the dimensional-regularization value (pi/4) alpha^2(1-alpha)^2 eps.
usage: python c1_general_d.py
"""
    import mpmath as mp
    def J(d):
        return -2*mp.quad(lambda c: (1-c*c)**((d-3)/2)*(c*c-(1-c*c)/(d-1))*mp.log(c), [0, 1])

    def cT_over_t(d):
        g = mp.gamma(d/2-1)/(4*mp.pi**(d/2))
        K_over_t = -8*mp.gamma(d-2)/(g**2*mp.pi**(d/2)*2**(4-d)*mp.gamma(2-d/2))
        Om = 2*mp.pi**((d-1)/2)/mp.gamma((d-1)/2)
        volH = mp.pi**((d-2)/2)*mp.gamma(1-d/2)
        dV_over_t = -K_over_t*Om*J(d)/(2*mp.pi)**d
        return -mp.sin(mp.pi*(d-2)/2)*2*mp.pi*volH*dV_over_t

    def main():
        print('J(3) =', mp.nstr(J(3), 12), '  expected -2/3')
        print('J(4) =', mp.nstr(J(4), 12), '  -pi/8 =', mp.nstr(-mp.pi/8, 12))
        # d = 3: t = -h/(2 pi)
        print('d = 3:  c_T/h =', mp.nstr(cT_over_t(3)*(-1/(2*mp.pi)), 12), '  64/(3 pi) =', mp.nstr(64/(3*mp.pi), 12))
        # d = 4 - eps: c_T/(eps t) -> limit; with t = 2a = -alpha^2(1-alpha)^2/(16 pi^2)
        for eps in (1e-3, 1e-5):
            r = cT_over_t(4-eps)/eps
            print(f'd = 4 - {eps:g}: c_T/(eps a^2(1-a)^2) =', mp.nstr(r*(-1/(16*mp.pi**2)), 10), '  pi/4 =', mp.nstr(mp.pi/4, 10))

    def cli(argv):
        with fuzzy.logged('analytic/covariant_entropy/c1_general_d.log'):
            main()
    return namespace(locals(), globals())


@section
def b1_blocks():
    """Step 1 of the d = 3.5 slab test: building blocks on H^{n+1}, n = d - 2 (boundary dimension of the AdS slice).
 G(D, chi): propagator of (-lap + D(D-n)) on H^{n+1};  a_l(D1, D2): bubble coefficients, G_D1 G_D2 = sum_l a_l G_{D1+D2+2l};
 G0(D): dimensionally regularized coincident value.  Checks (printed): the paper's n = 1 forms, the flat singularity,
 the product identity at n = 1 and 1.5, and the coincident value as the regular part of G at small chi.
usage: python b1_blocks.py
"""
    import numpy as np
    import mpmath as mp
    from scipy.special import gammaln
    def G(D, chi, n):
        xi = 1/mp.cosh(chi)
        return mp.gamma(D)/(2*mp.pi**(n/2)*mp.gamma(D-n/2+1))*(xi/2)**D*mp.hyp2f1(D/2, (D+1)/2, D-n/2+1, xi**2)

    def a_coef(D1, D2, n, L):
        """a_l for l = 0..L-1 (numpy, log-gamma); (x)_y = Gamma(x+y)/Gamma(x)."""
        l = np.arange(L, dtype=float); D12 = D1+D2
        lp = lambda x, y: gammaln(x+y) - gammaln(x)
        lg = (lp(n/2, l) - gammaln(l+1) - np.log(2*np.pi**(n/2)) + lp(D12+2*l, 1-n/2) + lp(D12+l-n+1, l)
              - lp(D1+l, 1-n/2) - lp(D2+l, 1-n/2) - lp(D12+l-n/2, l))
        return np.exp(lg)

    def G0(D, n):
        return mp.gamma(D)*mp.gamma((1-n)/2)/((4*mp.pi)**((n+1)/2)*mp.gamma(1+D-n))

    def main():
        mp.mp.dps = 30
        # (i) n = 1 reproduces the paper's AdS2 propagator and bubble coefficients
        D, chi = mp.mpf('0.8'), mp.mpf('0.7')
        paper = mp.gamma(D)*(2*mp.cosh(chi))**(-D)/(2*mp.sqrt(mp.pi)*mp.gamma(D+0.5))*mp.hyp2f1(D/2, (D+1)/2, D+0.5, mp.cosh(chi)**-2)
        print('(i)  n=1 propagator vs paper:', mp.nstr(G(D, chi, 1)/paper - 1, 3))
        # (ii) flat singularity G ~ Gamma((n-1)/2)/(4 pi^{(n+1)/2} chi^{n-1})
        n = mp.mpf('1.5'); c = mp.mpf('1e-6')
        print('(ii) n=1.5 singular coefficient ratio:', mp.nstr(G(D, c, n)*c**(n-1)/(mp.gamma((n-1)/2)/(4*mp.pi**((n+1)/2))), 10))
        # (iii) product identity at n = 1 and n = 1.5
        for nn in (1.0, 1.5):
            for (D1, D2) in ((0.75, 1.25), (1.6, 2.3)):
                a = a_coef(D1, D2, nn, 4000)
                for chi in (0.6, 1.5):
                    lhs = G(D1, chi, nn)*G(D2, chi, nn)
                    rhs = mp.fsum(mp.mpf(a[l])*G(D1+D2+2*l, chi, nn) for l in range(200))
                    print(f'(iii) n={nn} D=({D1},{D2}) chi={chi}: product/sum - 1 = {mp.nstr(lhs/rhs-1, 3)}')
        # (iv) coincident value = regular part at small chi (n = 1.5)
        for D in (mp.mpf('0.75'), mp.mpf('2.1')):
            c = mp.mpf('1e-8'); sing = mp.gamma((n-1)/2)/(4*mp.pi**((n+1)/2))*c**(1-n)
            print('(iv) n=1.5 D=', D, ': G - singular =', mp.nstr(G(D, c, n)-sing, 10), '  G0 =', mp.nstr(G0(D, n), 10))

    def cli(argv):
        with fuzzy.logged('analytic/covariant_entropy/b1_blocks.log'):
            main()
    return namespace(locals(), globals())


@section
def b2_gap():
    """Step 2: twisted coincident shift and the spectator coefficient on S^1 x H^{n+1}.
D_tw = sum_{s in Z+alpha} G0(n/2+|s|) - sum_{s in Z} G0(n/2+|s|), with G0(n/2+x) = g R(x), R(x) = Gamma(x+n/2)/Gamma(x+1-n/2),
g = Gamma((1-n)/2)/(4 pi)^{(n+1)/2}.  Since n R(x) = P(x+1) - P(x) with P(x) = Gamma(x+n/2)/Gamma(x-n/2), the convergent sum
sum_{m>=0} [R(m+alpha) + R(m+1-alpha) - R(m) - R(m+1)] telescopes to [P(0) + P(1) - P(alpha) - P(1-alpha)]/n (n < 2).
The spectators' s = 0 mode screens it: c_sp = -2 D_tw / dG0/dD at D = n/2 (common normalization of App. D, where each mode
product enters without its 1/2pi).  Checks: d = 4 (n = 2) gives D_tw = -alpha(1-alpha)/4pi; n -> 1 gives -A/2pi and c_sp = -4A/pi^2
with A = pi(1-2 alpha) tan(pi alpha)/2 (paper, App. D.5); the telescoped value against brute-force partial sums.
usage: python b2_gap.py
"""
    import mpmath as mp
    mp.mp.dps = 30

    def P(x, n): return mp.gamma(x+n/2)/mp.gamma(x-n/2) if x-n/2 not in (0, -1, -2) else mp.mpf(0)
    def g(n): return mp.gamma((1-n)/2)/(4*mp.pi)**((n+1)/2)
    def D_tw(al, n): return g(n)*(P(0, n)+P(1, n)-P(al, n)-P(1-al, n))/n
    def dG0(D, n): return mp.gamma(D)*g(n)/mp.gamma(1+D-n)*(mp.digamma(D)-mp.digamma(1+D-n))
    def c_sp(al, n): return -2*D_tw(al, n)/dG0(n/2, n)
    A = lambda al: mp.pi*(1-2*al)*mp.tan(mp.pi*al)/2

    def main():
        for al in (mp.mpf('0.25'), mp.mpf('0.5') - mp.mpf('1e-12')):
            print(f'alpha={mp.nstr(al,4)}:')
            print('  n=2 (d=4):  D_tw =', mp.nstr(D_tw(al, mp.mpf(2) - mp.mpf('1e-20')), 12), '  expected', mp.nstr(-al*(1-al)/(4*mp.pi), 12))
            e = mp.mpf('1e-15')
            print('  n->1 (d=3): D_tw =', mp.nstr(D_tw(al, 1+e), 10), '  -A/2pi =', mp.nstr(-A(al)/(2*mp.pi), 10),
                  ';  c_sp =', mp.nstr(c_sp(al, 1+e), 10), '  -4A/pi^2 =', mp.nstr(-4*A(al)/mp.pi**2, 10))
            n = mp.mpf('1.5'); R = lambda x: mp.exp(mp.loggamma(x+n/2)-mp.loggamma(x+1-n/2))
            f = lambda m: R(m+al)+R(m+1-al)-R(m)-R(m+1)
            # partial sums S(M) = S_inf + b M^-1/2 + c M^-3/2 (summand ~ m^{n-3}), extrapolated from three M
            Ms, S, acc, m0 = [2000, 8000, 32000], [], mp.mpf(0), 0
            for M in Ms:
                acc += mp.fsum(f(m) for m in range(m0, M)); m0 = M; S.append(acc)
            x = mp.lu_solve(mp.matrix([[1, mp.mpf(M)**-0.5, mp.mpf(M)**-1.5] for M in Ms]), mp.matrix(S))
            print(f'  n=1.5 brute force, extrapolated {mp.nstr(g(n)*x[0], 12)}   telescoped {mp.nstr(D_tw(al, n), 12)}')
            print('  n=1.5: D_tw =', mp.nstr(D_tw(al, n), 12), '  dG0(n/2) =', mp.nstr(dG0(n/2, n), 12), '  c_sp =', mp.nstr(c_sp(al, n), 12))

    def cli(argv):
        with fuzzy.logged('analytic/covariant_entropy/b2_gap.log'):
            main()
    return namespace(locals(), globals())


@section
def b3_pi0():
    """Step 3: the untwisted kernel Pi0_kappa(nu) on S^1 x H^{n+1}.
Mode sum: Pi0_kappa = sum_{s in Z} B(n/2+|s|, n/2+|kappa-s|),  B(D1,D2)(nu) = sum_l a_l/(nu^2 + (D1+D2+2l-n/2)^2).
All its poles sit on one ladder, L = n + kappa + 2j, and each level coefficient c_j is a finite sum over (s, l).
Conjecture: Pi0_kappa = C |Gamma((kappa+n/2+i nu)/2) / Gamma((kappa+2-n/2+i nu)/2)|^2 (n = 1: the paper's form, C = pi/8),
whose residues are c_j = 4C (a+2j)(-1)^j Gamma(a+j) / (j! Gamma(1-n/2-j) Gamma(kappa+1+j)), a = kappa+n/2.
Test: the ratio of mode-sum levels to these residues is one constant C for every j and kappa.
usage: python b3_pi0.py
"""
    import numpy as np
    from scipy.special import gammaln, gamma, rgamma
    _b1_blocks = b1_blocks(); a_coef = _b1_blocks.a_coef
    def levels_untwisted(kappa, n, J):
        """c_j for j = 0..J-1 from pairs (s, kappa-s): |s|+|kappa-s| = kappa + 2p, p = 0 (s in [0,kappa]) or p >= 1 (s = -p, kappa+p)."""
        c = np.zeros(J)
        pairs = [(s, 0) for s in range(0, kappa+1)] + [(s, p) for p in range(1, J) for s in (-p, kappa+p)]
        for s, p in pairs:
            D1, D2 = n/2+abs(s), n/2+abs(kappa-s)
            a = a_coef(D1, D2, n, J-p)
            c[p:] += a
        return c

    def residues(kappa, n, J, C=1.0):
        j = np.arange(J, dtype=float); a = kappa+n/2
        # 1/Gamma(1-n/2-j) via rgamma (finite at poles); sign (-1)^j
        return 4*C*(a+2*j)*(-1)**j*np.exp(gammaln(a+j)-gammaln(j+1)-gammaln(kappa+1+j))*rgamma(1-n/2-j)

    def main():
        for n in (1.0, 1.5):
            print(f'n = {n}:')
            for kappa in (0, 1, 2, 5):
                c = levels_untwisted(kappa, n, 60); r = residues(kappa, n, 60)
                ratio = c/r
                print(f'  kappa={kappa}: C = c_j/res_j at j=0,1,5,20,59: ' + ' '.join(f'{ratio[j]:.12f}' for j in (0, 1, 5, 20, 59))
                      + f'   (pi/8 = {np.pi/8:.12f})')

    def cli(argv):
        with fuzzy.logged('analytic/covariant_entropy/b3_pi0.log'):
            main()
    return namespace(locals(), globals())


@section
def b4_sector():
    """Steps 4-5: the fixed-sector integrals I_kappa = int dnu mu(nu) dPi_kappa(nu)/Pi0_kappa(nu) on S^1 x H^{n+1}, n = d - 2.
dPi_kappa = T_kappa - Pi0_kappa + S_kappa, where
  T_kappa = sum_{s in Z+alpha} B(n/2+|s|, n/2+|s-kappa|): three ladders of levels (inner pairs 0<s<kappa at n+kappa+2j;
            s > kappa at n+kappa+2alpha+2j; s < 0 at n+kappa+2-2alpha+2j), each level a finite sum over (pair, l);
  Pi0_kappa: untwisted ladder n+kappa+2j (b3_pi0.py);
  S_kappa = c_sp d/dD B(D, n/2+kappa) at D = n/2, c_sp from b2_gap.py (screening by the spectators' s = 0 mode).
mu(nu) = |Gamma(i nu + n/2)/Gamma(i nu)|^2 (H^{n+1} Plancherel up to a constant; n = 1: nu tanh(pi nu)).
T and Pi0 carry the same smooth level weight exp(-(L/Lcut)^2), Lcut = Lmax/5, so their tails cancel; the ratio uses the
closed-form Pi0 (exact) in the denominator.  The paper (App. D.5) proves I_kappa = 0 at n = 1; this tests n = 1.5.
usage: python b4_sector.py n alpha kappa[,kappa...] [Lmax=2000] [numax=60]

Each run is logged under its arguments. The research logs concatenated several runs:
b4_validation_d3*.log (n = 1, the proved d = 3 identity) and b5_d3p5.log (n = 1.5, d = 7/2).
"""
    import sys, json, os
    import numpy as np
    from scipy.special import gammaln, digamma, loggamma
    _b1_blocks = b1_blocks(); a_coef = _b1_blocks.a_coef
    _b3_pi0 = b3_pi0(); residues = _b3_pi0.residues
    import mpmath as mp
    _b2_gap = b2_gap(); c_sp_mp = _b2_gap.c_sp
    C_N = {1.0: np.pi/8, 1.5: 0.325425314933}

    def ladders(kappa, n, al, Lmax, twisted):
        """Return arrays (Lpos, coef) of all levels below Lmax: Lpos = level - n/2 (pole at nu^2 = -Lpos^2)."""
        out_L, out_c = [], []
        def add(D1, D2, p):            # pair with D1 + D2 = base + 2p; levels base + 2(p + l)
            lmax = int((Lmax - (D1+D2))//2) + 1
            if lmax <= 0: return
            a = a_coef(D1, D2, n, lmax)
            out_L.append(D1 + D2 + 2*np.arange(lmax) - n/2); out_c.append(a)
        if twisted:
            for m in range(0, kappa):                      # inner: 0 < s = m + al < kappa
                add(n/2+m+al, n/2+kappa-m-al, 0)
            p = 0
            while n+kappa+2*al+2*p < Lmax:                # s = kappa + p + al
                add(n/2+kappa+p+al, n/2+p+al, p); p += 1
            p = 0
            while n+kappa+2-2*al+2*p < Lmax:              # s = -(p + 1) + al < 0
                add(n/2+p+1-al, n/2+kappa+p+1-al, p); p += 1
        else:
            for s in range(0, kappa+1):
                add(n/2+s, n/2+kappa-s, 0)
            p = 1
            while n+kappa+2*p < Lmax:
                add(n/2+p, n/2+kappa+p, p); add(n/2+kappa+p, n/2+p, p); p += 1
        L = np.concatenate(out_L); c = np.concatenate(out_c)
        # merge equal levels (each ladder is an arithmetic sequence), so the nu sum runs over distinct levels only
        key = np.round(L*1e6).astype(np.int64)
        uk, inv = np.unique(key, return_inverse=True)
        cs = np.bincount(inv, weights=c); Ls = np.bincount(inv, weights=L)/np.bincount(inv)
        return Ls, cs

    def bubble_sum(L, c, nu, Lcut=None):
        """Level sum; with Lcut, each level is weighted by exp(-(L/Lcut)^2), the same smooth cutoff for twisted and untwisted
        ladders, so that the half-spacing offset between them leaves no boundary term."""
        wL = np.exp(-(L/Lcut)**2) if Lcut else 1.0
        return ((c*wL)[None, :]/(nu[:, None]**2 + L[None, :]**2)).sum(axis=1)

    def spectator(kappa, n, nu, csp, lmax=20000, h=1e-6):
        D2 = n/2+kappa; l = np.arange(lmax)
        a = a_coef(n/2, D2, n, lmax)
        da = (a_coef(n/2+h, D2, n, lmax) - a_coef(n/2-h, D2, n, lmax))/(2*h)
        Lam = n/2 + kappa + 2*l
        den = nu[:, None]**2 + Lam[None, :]**2
        return csp*(da[None, :]/den - a[None, :]*2*Lam[None, :]/den**2).sum(axis=1)

    def pi0_closed(kappa, n, nu):
        z1 = loggamma((kappa+n/2+1j*nu)/2); z2 = loggamma((kappa+2-n/2+1j*nu)/2)
        return C_N[n]*np.exp(2*(z1.real - z2.real))

    def mu(nu, n):
        return np.exp(2*(loggamma(1j*nu+n/2).real - loggamma(1j*nu).real))

    def sector(n, al, kappa, Lmax, numax, nnu=400):
        # Gauss-Legendre on [0, numax] in panels, plus a tail fit A/nu^2 + B/nu^3 from the last panel
        xg, wg = np.polynomial.legendre.leggauss(16)
        edges = np.concatenate([np.linspace(0, 4, 9), np.geomspace(4, numax, 40)[1:]])
        nu = np.concatenate([(b-a)/2*xg + (a+b)/2 for a, b in zip(edges[:-1], edges[1:])])
        w = np.concatenate([(b-a)/2*wg for a, b in zip(edges[:-1], edges[1:])])
        csp = float(c_sp_mp(mp.mpf(al), mp.mpf(n) if n != 1.0 else mp.mpf(1)+mp.mpf('1e-15')))
        Lt, ct = ladders(kappa, n, al, Lmax, True); Lu, cu = ladders(kappa, n, al, Lmax, False)
        Lcut = Lmax/5                                   # Gaussian weight; levels beyond Lmax are below e^-25
        dpi = bubble_sum(Lt, ct, nu, Lcut) - bubble_sum(Lu, cu, nu, Lcut) + spectator(kappa, n, nu, csp)
        f = mu(nu, n)*dpi/pi0_closed(kappa, n, nu)
        sel = nu > numax/4                              # tail beyond numax: A/nu^2 + B/nu^3 + C/nu^4 fitted on [numax/4, numax]
        A = np.linalg.lstsq(np.c_[nu[sel]**-2, nu[sel]**-3, nu[sel]**-4], f[sel], rcond=None)[0]
        tail = A[0]/numax + A[1]/(2*numax**2) + A[2]/(3*numax**3)
        I = (w*f).sum() + tail
        return dict(n=n, alpha=al, kappa=kappa, Lmax=Lmax, numax=numax, I=float(I), abs_int=float((w*abs(f)).sum()),
                    tail=float(tail), tail_coef=[float(x) for x in A], f_at=[[float(x), float(y)] for x, y in zip(nu[::40], f[::40])])

    def main():
        n = float(sys.argv[1]); al = float(sys.argv[2]); ks = [int(k) for k in sys.argv[3].split(',')]
        Lmax = float(sys.argv[4]) if len(sys.argv) > 4 else 2000; numax = float(sys.argv[5]) if len(sys.argv) > 5 else 60
        for k in ks:
            r = sector(n, al, k, Lmax, numax)
            print(f"n={n} alpha={al} kappa={k} Lmax={Lmax:g} numax={numax:g}: I = {r['I']:+.3e}   int|f| = {r['abs_int']:.4f}"
                  f"   tail {r['tail']:+.2e} (A={r['tail_coef'][0]:+.4f})", flush=True)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('analytic/covariant_entropy/b4_sector_' + '_'.join(argv[1:]).replace(',', '-') + '.log'):
            main()
    return namespace(locals(), globals())


@section
def b6_sensitivity():
    """Sensitivity of the d = 3.5 slab test: rescale the spectator (gap-equation) coefficient by 1 + delta and recompute I_kappa.
A test that cannot fail would show no response; here a 1% error in the screening coefficient should move I_kappa far above
the ~1e-6 residuals of b5_d3p5.log.
usage: python b6_sensitivity.py
"""
    b = globals()['b4_sector']()   # was: import b4_sector as b


    def main():
        orig = b.c_sp_mp
        for delta in (0.0, 0.01, -0.01):
            b.c_sp_mp = lambda al, n, d=delta: orig(al, n)*(1+d)
            for k in (0, 1):
                r = b.sector(1.5, 0.5, k, 16000, 40)
                print(f'd=3.5 alpha=0.5 kappa={k} c_sp x (1{delta:+.2f}): I = {r["I"]:+.3e}  (int|f| = {r["abs_int"]:.4f})', flush=True)

    def cli(argv):
        with fuzzy.logged('analytic/covariant_entropy/b6_sensitivity.log'):
            main()
    return namespace(locals(), globals())


STEPS = {'c1_run': c1_run, 'alpha': alpha, 'largeN_traces': largeN_traces, 'uv_uniformity': uv_uniformity, 'fulltwist': fulltwist, 'largeN_full_twist': largeN_full_twist, 'sector_position': sector_position, 'sector_check2': sector_check2, 'Jbubble': Jbubble, 'Jprecise': Jprecise, 'tadpole_test': tadpole_test, 'tadpole_proof': tadpole_proof, 'PQ_conjecture': PQ_conjecture, 'PQ_antiderivative': PQ_antiderivative, 'P1_exact': P1_exact, 'paper_form_check': paper_form_check, 'sector_closed': sector_closed, 'sector_psi': sector_psi, 'v_closed_form': v_closed_form, 'c0_probe_4d': c0_probe_4d, 'c1_general_d': c1_general_d, 'b1_blocks': b1_blocks, 'b2_gap': b2_gap, 'b3_pi0': b3_pi0, 'b4_sector': b4_sector, 'b6_sensitivity': b6_sensitivity}
REPORT = [('largeN_traces',), ('c1_run', 'sum', 'c1_tab_160_4000_8000.npz'), ('paper_form_check',), ('c1_general_d',), ('b4_sector', '1.5', '0.5', '0,1,2,3', '16000', '40')]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
