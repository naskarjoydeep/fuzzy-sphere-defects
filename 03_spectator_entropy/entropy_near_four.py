"""Near four dimensions: does a spectator still raise the defect entropy, and by how much?

The partial twist (n_t twisted complex planes and N - 2 n_t untwisted spectators) in d = 4 - eps, through O(eps^2).
The steps give the first order (eq:decoupleeps), the spectator s = 0 channel and its Hartree resummation (eq:spgap,
eq:Fsp, eps_c, the imaginary eps^{3/2} term), and the analytic O(eps^2) pieces in MS-bar (coupling counterterm, cactus
diagrams, melon), assembled into the paper's Tables 9 and 10 (Sec. 4.3 and App. D.9, "The entropy near four
dimensions"). Every step is an mpmath/sympy script whose printed output is stored under its research path, .log.

usage: python entropy_near_four.py <step>      (python entropy_near_four.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def eps2_common():
    """Shared d-dimensional ingredients for the R(N, n_t, alpha) steps of ENTROPY_EPS2 (2026-10-26). mpmath, high precision.

Conventions: V = (lam/4!)(phi^2)^2, lam = 16 pi^2 lb (lb = lambda-bar), MS-bar:
    lam_B = M^eps 16 pi^2 (lb + (N+8) lb^2/(3 eps)),   M^eps = (e^gamma/(4 pi))^{eps/2}  (mu R = 1),
    lb* = 3 eps/(N+8) + 9 (3N+14) eps^2/(N+8)^3.
Frame S^1 x H^{d-1}, unit radius, d = 4 - eps; Vol_4 = 2 pi Vol(H^{d-1}), Vol(H^{d-1}) = pi^{(d-2)/2} Gamma(1 - d/2).
dg(e, a): coincident shift per real twisted component (GHJK 2.16 halved), d = 4 - e.
gH(e, D): coincident value of one real H^{d-1} field of boundary dimension D (dim reg); 1/(2 pi) of it per circle mode.
F~ = -sin(pi eps/2) F.
laurent(f, kmin, kmax): coefficients c_k of f(eps) = sum_k c_k eps^k, from mp.taylor of eps^{-kmin} f.
"""
    from mpmath import mp, mpf, gamma, sin, pi, euler, log, exp, zeta, psi, rgamma
    mp.dps = 40


    def dg(e, a):
        d = 4 - e
        return (d - 1)*gamma(d/2 - a)*gamma(d/2 - 1 + a)*sin(pi*a)*gamma(d/2)/(2*pi**(d/2 + 1)*gamma(d)*(2 - d))


    def volH(e):
        d = 4 - e
        return pi**((d - 2)/2)*gamma(1 - d/2)


    def vol4(e):
        return 2*pi*volH(e)


    def Meps(e):
        return exp(e/2*(euler - log(4*pi)))


    def gH(e, D):
        d = 4 - e
        return gamma((3 - d)/2)*gamma(D)*rgamma(D - d + 3)/(4*pi)**((d - 1)/2)


    def lb_star(e, N):
        return 3*e/(N + 8) + 9*(3*N + 14)*e**2/(N + 8)**3


    def laurent(f, kmin, kmax, r=mpf('0.25'), M=32):
        """f(e) has a pole of order -kmin at e = 0; return {k: c_k} for k = kmin..kmax.
        Trapezoidal rule on |e| = r (exponentially accurate, error ~ (r/R)^M, R = nearest other singularity)."""
        from mpmath import mpc, expj
        vals = [f(r*expj(2*pi*j/M)) for j in range(M)]
        out = {k: sum(v*expj(-2*pi*j*k/M) for j, v in enumerate(vals))/(M*r**k) for k in range(kmin, kmax + 1)}
        return {k: c.real for k, c in out.items()}   # f is real on the real axis
    return namespace(locals(), globals())


@section
def eps2_hadamard():
    """d = 4 Hadamard finite parts on S^1 x H^3 (unit radius), used by R(iv). Calibrated in eps2_r4a_calibration.py.

hadamard(f, am3, am1): Fin = lim_{rho0 -> 0} [ int_{rho > rho0} f dmu + am1 log rho0 - am3/(2 rho0^2) ],
dmu = dtheta 4 pi sinh^2(chi) dchi on (-pi, pi) x (0, 60);  rho^2 = theta^2 + chi^2 (geodesic distance).
Returns (Fin, diagnostics); diagnostics = fitted coefficients (b_-1, a1, a3) of the subtracted angular integral
Phi(rho) - am3/rho^3 - am1/rho ~ b_-1/rho + a1 rho + a3 rho^3 near 0 (b_-1 must vanish if am1 is right).
"""
    import numpy as np
    from scipy.integrate import quad
    PI = np.pi
    C4, Om4 = 1/(4*PI**2), 2*PI**2


    def G0(t, c):
        return 1/(8*PI**2*(np.cosh(c) - np.cos(t)))


    def Ga(t, c, a):
        """complex twisted propagator, per-real-component normalization, theta in (-pi, pi)."""
        z1, z2 = np.exp(1j*t - c), np.exp(-1j*t - c)
        return (z1**a/(1 - z1) + z2**(1 - a)/(1 - z2))/(8*PI**2*np.sinh(c))


    def hadamard(f, am3, am1, rmin=0.02):
        w = lambda t, c: 4*PI*np.sinh(c)**2*f(t, c)
        Phi = lambda r: quad(lambda s: r*w(r*np.cos(s), r*np.sin(s)), 0, PI, epsabs=0, epsrel=1e-12, limit=200)[0]
        sub = lambda r: Phi(r) - am3/r**3 - am1/r
        rs = np.array([0.02, 0.03, 0.05, 0.08, 0.12]); vs = np.array([sub(r) for r in rs])
        bm1, a1, a3 = np.linalg.lstsq(np.vstack([1/rs, rs, rs**3]).T, vs, rcond=None)[0]
        a1, a3 = np.linalg.lstsq(np.vstack([rs, rs**3]).T, vs, rcond=None)[0]
        inner = quad(sub, rmin, 1, epsabs=0, epsrel=1e-11, limit=200)[0] + a1*rmin**2/2 + a3*rmin**4/4
        ann = quad(Phi, 1, PI, epsabs=0, epsrel=1e-11, limit=200)[0]
        out = quad(lambda t: quad(lambda c: w(t, c), np.sqrt(max(PI**2 - t**2, 0)), 60.0, epsabs=0, epsrel=1e-11, limit=200)[0],
                   -PI, PI, epsabs=0, epsrel=1e-10, limit=200)[0]
        return inner + ann + out - am3/2, (bm1, a1, a3)


    def dG(t, c, a):
        """G_a - G0 without cancellation: (1/(8 pi^2 sinh chi)) [ -e^{(a-1)w/2} sinh(a w/2)/sinh(w/2) + (same with v) ],
        w = i theta - chi, v = i theta + chi;  -> -a(1-a)/(8 pi^2) at coincidence."""
        w, v = 1j*t - c, 1j*t + c
        br = -np.exp((a - 1)*w/2)*np.sinh(a*w/2)/np.sinh(w/2) + np.exp((a - 1)*v/2)*np.sinh(a*v/2)/np.sinh(v/2)
        return br/(8*PI**2*np.sinh(c))


    def dP(t, c, a):
        """|G_a|^2 - G0^2 = 2 G0 Re dG + |dG|^2."""
        g = dG(t, c, a)
        return 2*G0(t, c)*g.real + abs(g)**2
    return namespace(locals(), globals())


@section
def entropy_eps():
    """Leading epsilon expansion of the defect entropy for a partial twist: n_t twisted real components
(one complex pair, n_t = 2) and N - n_t untwisted spectators, V = (lam/4!)(phi^2)^2, lam* = 48 pi^2 eps/(N+8).

First order: delta log<D> = -(lam/4!) int <(phi^2)^2>_D = -(lam/24) n_t(n_t+2) 2pi Vol(H^{d-1}) dhat^2,
with dhat the coincident shift per twisted real component on S^1 x H^{d-1} (d=4: -a(1-a)/(8 pi^2)).
Spectators do not enter at this order (their free one-point function vanishes).
Two continuations to d=3: (i) keep Vol(H^{d-1}) = pi^{(d-2)/2} Gamma(1-d/2) and set eps=1;
(ii) F~ = -sin(pi p/2) log<D>, p = d-2, expanded to O(eps).
"""
    from mpmath import mp, mpf, pi, gamma, sin, diff
    mp.dps = 20


    def dhat4(a):
        return -a*(1 - a)/(8*pi**2)


    def dlogD(N, nt, a, eps):
        d = 4 - eps
        lam = 48*pi**2*eps/(N + 8)
        vol = pi**((d - 2)/2)*gamma(1 - d/2)
        return -(lam/24)*nt*(nt + 2)*2*pi*vol*dhat4(a)**2


    def logg_natural(N, nt, a):
        return dlogD(N, nt, a, mpf(1))


    def logg_Ftilde(N, nt, a):
        # log g = -F~ at d=3; at O(eps): F~ = -sin(pi p/2) log<D>, take the O(eps) term of sin(pi(2-eps)/2) dlogD
        f = lambda e: sin(pi*(2 - e)/2)*dlogD(N, nt, a, e)
        return diff(f, mpf('1e-30')) * 1  # d/d eps at 0 times eps=1

    def main():
        a = mpf(1)/2
        print('half twist, n_t = 2: interaction part of log g at O(eps), and the O(3)->O(2) change')
        for name, fn in [('Vol(H^{d-1}) kept', logg_natural), ('F-tilde, O(eps)', logg_Ftilde)]:
            g3, g2 = fn(3, 2, a), fn(2, 2, a)
            print(f'  {name:18s}: O(3) {mp.nstr(g3, 6)}   O(2) {mp.nstr(g2, 6)}   Delta = {mp.nstr(g2 - g3, 6)}')
        print('large-N check, N * (interaction log g) at half twist, vs the paper\'s d=3 value 8/(9 pi^2) =', mp.nstr(8/(9*pi**2), 6))
        for name, fn in [('Vol kept', logg_natural), ('F-tilde', logg_Ftilde)]:
            print(f'  {name:9s}: N*log g_int -> {mp.nstr(fn(10**8, 2, a)*10**8, 6)}')
        print('alpha dependence of the O(3)->O(2) change (Vol kept):')
        for a in [mpf('0.1'), mpf('0.25'), mpf('0.4'), mpf('0.5')]:
            print(f'  alpha={float(a)}: {mp.nstr(logg_natural(2, 2, a) - logg_natural(3, 2, a), 6)}')
        # spectator s=0 mode: m^2 r^2 = (lam/6) n_t dhat r^2 = -n_t a(1-a) eps/(N+8)
        print('spectator s=0 potential mu = -n_t a(1-a) eps/(N+8): O(3) half twist', mp.nstr(-2*0.25/11, 6), '-> Dhat = 1 - eps/2 +- i', mp.nstr((2*0.25/11)**0.5, 6), 'sqrt(eps)')

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/open_questions/entropy_eps.log'):
            main()
    return namespace(locals(), globals())


@section
def eps2_s1_anchor():
    """S1 of ENTROPY_EPS2_PLAN.md: reproduce the known first-order result in the plan's conventions.

S = int [ (1/2)(d phi)^2 + (lam4/4)(phi_a phi_a)^2 ],  lam4* = 8 pi^2 eps/(N+8);  n_t twisted planes, twist alpha.
Checks: (i) dg_tw^(4) = -alpha(1-alpha)/(8 pi^2) two ways; (ii) F~_1; (iii) GHJK (5.3) at n_t = N_G, N = 2 N_G;
(iv) eq:decoupleeps and the 9e-4 at O(3)->O(2), alpha = 1/2; (v) tab:entropy last row to its printed digits.

Check (v) reproduces the large-N limit of the fixed-N first-order coefficient, (pi/2) a^2 (1-a)^2. The paper
now compares with the screened large-N coefficient (pi/4) a^2 (1-a)^2 instead (App. D, entropy_eps2_checks).
"""
    import sympy as sp
    from mpmath import mp, mpf, zeta, pi as mpi


    def main():
        eps = sp.symbols('epsilon', positive=True)
        N, nt, a, NG = sp.symbols('N n_t alpha N_G')
        pi = sp.pi

        # (i) per real twisted component on S^1 x H^3: (1/2pi) sum_{s in Z+a} [-(|s|)/(4pi)] minus untwisted (Hurwitz)
        mp.dps = 30
        for av in ['0.1', '0.25', '0.5']:
            x = mpf(av)
            modes = -(zeta(-1, x) + zeta(-1, 1 - x) - 2*zeta(-1))/(4*mpi)/(2*mpi)
            assert abs(modes + x*(1 - x)/(8*mpi**2)) < mpf(10)**-25
        d = sp.symbols('d')
        ghjk216 = (d - 1)*sp.gamma(d/2 - a)*sp.gamma(d/2 - 1 + a)*sp.sin(pi*a)*sp.gamma(d/2)/(pi**(d/2 + 1)*sp.gamma(d)*(2 - d))
        dg = -a*(1 - a)/(8*pi**2)
        assert sp.simplify(sp.gammasimp(ghjk216.subs(d, 4))/2 - dg) == 0
        print('(i)  dg_tw^(4) = -alpha(1-alpha)/(8 pi^2): Hurwitz mode sum and GHJK (2.16)/2 agree')

        # (ii) first order: F_1 = (lam4/4) <(phi^2)^2>_D Vol_4, <(phi^2)^2> = (sum g)^2 + 2 sum g^2, spectators g = 0
        lam4 = 8*pi**2*eps/(N + 8)
        Vol4 = 2*pi*(-2*pi/eps)
        F1 = lam4/4*((2*nt*dg)**2 + 2*(2*nt*dg**2))*Vol4
        Ft1 = sp.factor(sp.simplify(-(pi*eps/2)*F1))
        assert sp.simplify(Ft1 - pi*nt*(nt + 1)*a**2*(1 - a)**2*eps/(4*(N + 8))) == 0
        print('(ii) F~_1 =', Ft1)

        # (iii) GHJK (5.3): eps N_G (N_G+1) Vol(H^3) (1-v)^2 v^2 / (8 pi (N_G+4)), Vol(H^3) -> -2pi/eps
        assert sp.simplify(F1.subs({nt: NG, N: 2*NG}) - eps*NG*(NG + 1)*(-2*pi/eps)*a**2*(1 - a)**2/(8*pi*(NG + 4))) == 0
        print('(iii) GHJK (5.3) reproduced at n_t = N_G, N = 2 N_G (and (5.4): the untwisted O(lambda) term vanishes)')

        # (iv) eq:decoupleeps and O(3) -> O(2) at half twist, eps = 1
        dec = sp.simplify(Ft1.subs(nt, 1).subs(N, N - 1) - Ft1.subs(nt, 1))
        assert sp.simplify(dec - pi*a**2*(1 - a)**2*eps/(2*(N + 7)*(N + 8))) == 0
        o32 = sp.N(dec.subs({N: 3, a: sp.Rational(1, 2), eps: 1}), 6)
        print('(iv) log g_{N-1} - log g_N =', sp.factor(dec), ';  O(3)->O(2), alpha = 1/2:', o32)
        assert abs(o32 - sp.Rational(8925, 10**7)) < 1e-7

        # (v) tab:entropy last row: lim N (N^2 * decoupleeps) at eps = 1 = (pi/2) a^2 (1-a)^2
        printed = {'0.1': '0.012723', '0.2': '0.040212', '0.25': '0.055223', '0.3': '0.069272', '0.4': '0.090478', '0.5': '0.098175'}
        coef = sp.limit(dec*N**2, N, sp.oo).subs(eps, 1)
        for av, s in printed.items():
            val = sp.N(coef.subs(a, sp.Rational(av)), 10)
            assert f'{float(val):.6f}' == s, (av, val)
        print('(v)  tab:entropy last row (pi/2)a^2(1-a)^2 reproduced to 6 digits at all six angles')

    def cli(argv):
        mp.dps = 30
        with fuzzy.logged('analytic/open_questions/eps2_s1_anchor.log'):
            main()
    return namespace(locals(), globals())


@section
def entropy_bf_channel():
    """The BF-critical spectator channel of the partial-twist defect entropy in d = 4 - eps (2026-10-26).

Setting: O(N) model, V = (lam/4!)(phi^2)^2, lam* = 48 pi^2 eps/(N+8); n_t twisted planes (2 n_t real components,
twist alpha) and N_s = N - 2 n_t untwisted spectators.  Frame S^1 x H^3 (unit radius), Vol_4 = 2 pi Vol(H^{3-eps}),
Vol(H^{3-eps}) = -2 pi/eps + O(1);  F~ = sin(pi p/2) log<D> = -(pi eps/2) F + ...

Steps checked here (SymPy):
 1. a real field on H^3 with boundary dimension D has dim-reg coincident value -(D-1)/(4 pi); its S^1 x H^3 mode
    s contributes 1/(2 pi) of that; the twisted sum gives dg = -alpha(1-alpha)/(8 pi^2) per real component.
 2. per real s = 0 mode, F(D) - F(1) = -(D-1)^3 Vol(H^3)/(12 pi)  (from dF/dm^2 = Vol g/2, m^2 = D(D-2)).
 3. Hartree (cactus-resummed) functional for the spectators' s = 0 mode, y = D_0 - 1:
       gap:  y^2 + b y - mu0 = 0,  b = eps (N_s+2)/(N+8),  mu0 = -2 n_t alpha(1-alpha) eps/(N+8)
       F~_sp(y) = -(pi N_s/12) y^3 - (pi N_s b/16) y^2  =  (pi N_s/96) [b^3 + 6 b mu0 -+ D^{3/2}],  D = b^2 + 4 mu0.
 4. expansions: fixed N (eps -> 0) and large N (eps fixed).
"""
    import sympy as sp


    def main():
        eps = sp.symbols('epsilon', positive=True)
        N, nt, a, y, b, mu0, Dl, m2 = sp.symbols('N n_t alpha y b mu0 Delta m2')
        Ns = N - 2*nt
        pi = sp.pi

        # --- 1. coincident values ------------------------------------------------------------------------------
        d = sp.symbols('d')
        gH = sp.gamma((3 - d)/2)*sp.gamma(Dl)/((4*pi)**((d - 1)/2)*sp.gamma(Dl - d + 3))   # real field on H^{d-1}
        g4 = sp.simplify(sp.limit(gH, d, 4))
        assert sp.simplify(g4 + (Dl - 1)/(4*pi)) == 0
        # twisted complex mode sum, D_s = 1 + |s|, s in Z + alpha, Hurwitz-regularized, minus untwisted
        S = lambda q: -sp.zeta(-1, q)  # sum_{n>=0} (n+q) regularized -> zeta(-1,q)
        tw = -(sp.zeta(-1, a) + sp.zeta(-1, 1 - a)) / (4*pi)          # sum_s g4(1+|s|) = -(1/4pi) sum |s|
        un = -(2*sp.zeta(-1)) / (4*pi)
        dg = sp.simplify(sp.expand_func((tw - un)/(2*pi)))
        dg = sp.simplify(sp.expand(sp.expand_func(sp.zeta(-1, a, evaluate=False)).rewrite(sp.bernoulli))) if False else dg
        dg_num = [(sp.N(((tw - un)/(2*pi)).subs(a, v)), sp.N(-v*(1 - v)/(8*pi**2))) for v in [sp.Rational(1, 2), sp.Rational(1, 5)]]
        assert all(abs(x - z) < 1e-14 for x, z in dg_num)
        print('1. g_H3(D) = -(D-1)/(4 pi);  dg_tw = -alpha(1-alpha)/(8 pi^2) checked at alpha = 1/2, 1/5')

        # --- 2. per-mode free energy ---------------------------------------------------------------------------
        VolH = sp.symbols('V_H')
        dF_dD = sp.Rational(1, 2)*VolH*g4*sp.diff(Dl*(Dl - 2), Dl)
        FD = sp.integrate(dF_dD, (Dl, 1, Dl))
        assert sp.simplify(FD + (Dl - 1)**3*VolH/(12*pi)) == 0
        print('2. F(D) - F(1) = -(D-1)^3 Vol(H^3)/(12 pi) per real s = 0 mode')

        # --- 3. Hartree functional ---------------------------------------------------------------------------
        lam = 48*pi**2*eps/(N + 8)
        dgt = -a*(1 - a)/(8*pi**2)
        gsp = -y/(8*pi**2)                          # s = 0 part of the spectator coincident value, mu = y^2
        J = -y**3/(24*pi**2)                        # int (1/2) g d mu, per unit 4-volume (2 pi Vol_H = Vol_4)
        assert sp.simplify(sp.diff(J, y) - sp.Rational(1, 2)*gsp*2*y) == 0
        Phi = Ns*(J - sp.Rational(1, 2)*y**2*gsp) \
            + lam/24*((2*nt*dgt + Ns*gsp)**2 + 2*(2*nt*dgt**2 + Ns*gsp**2)) \
            - lam/24*((2*nt*dgt)**2 + 2*(2*nt*dgt**2))  # remove the y = 0 (first-order) piece
        gap = sp.factor(sp.diff(Phi, y)*16*pi**2/Ns)
        b_def = eps*(Ns + 2)/(N + 8)
        mu0_def = -2*nt*a*(1 - a)*eps/(N + 8)
        assert sp.simplify(gap - (y**2 + b_def*y - mu0_def)) == 0
        print('3. gap equation: y^2 + b y - mu0 = 0 with b = eps(N_s+2)/(N+8), mu0 = -2 n_t a(1-a) eps/(N+8)')
        assert sp.simplify(sp.diff(mu0_def, eps)*(N + 8) - lam/3*nt*dgt*(N + 8)/eps) == 0  # mu0 = (lam/3) n_t dg
        Ft = -(pi*eps/2)*(2*pi*(-2*pi/eps))*Phi        # F~ = -(pi eps/2) Vol_4 Phi, leading continuation
        Ft = sp.simplify(Ft)
        Ft_y = -(pi*Ns/12)*y**3 - (pi*Ns*b_def/16)*y**2
        diff_on_shell = sp.simplify(sp.expand(Ft - Ft_y).subs(mu0_def, mu0))
        # on the gap equation the difference vanishes: reduce modulo the gap polynomial
        r = sp.rem(sp.expand(Ft - Ft_y), sp.expand(y**2 + b_def*y - mu0_def), y)
        assert sp.simplify(r) == 0
        print('   on shell: F~_sp = -(pi N_s/12) y^3 - (pi N_s b/16) y^2')
        Dd = b**2 + 4*mu0
        for sgn in (+1, -1):
            yr = (-b + sgn*sp.sqrt(Dd))/2
            val = sp.simplify(sp.expand(-(pi*Ns/12)*yr**3 - (pi*Ns*b/16)*yr**2))
            closed = pi*Ns/96*(b**3 + 6*b*mu0 - sgn*Dd**sp.Rational(3, 2))
            assert sp.simplify(sp.expand(val - closed)) == 0
        print('   closed form: F~_sp(y_+-) = (pi N_s/96)[b^3 + 6 b mu0 -+ (b^2+4mu0)^{3/2}],  y_+- = (-b +- sqrt(b^2+4mu0))/2')

        # --- 4. expansions ---------------------------------------------------------------------------------------
        t = sp.symbols('t', positive=True)   # eps = t^2 bookkeeping for half-integer powers
        Fpm = lambda sgn: pi*Ns/96*(b_def**3 + 6*b_def*mu0_def - sgn*(b_def**2 + 4*mu0_def)**sp.Rational(3, 2))
        ser = sp.series(Fpm(+1).subs(eps, t**2), t, 0, 6).removeO()
        print('4a. fixed N, eps -> 0 (y_+ branch):')
        for k in range(0, 6):
            c = sp.simplify(ser.coeff(t, k))
            if c != 0:
                print(f'    eps^{sp.Rational(k, 2)}: {sp.factor(c)}')
        c3 = sp.simplify(ser.coeff(t, 3))
        assert sp.simplify(c3**2 + (pi*Ns/12)**2*(2*nt*a*(1 - a)/(N + 8))**3) == 0   # purely imaginary
        print('    => eps^{3/2} term = +-i (pi N_s/12) |mu0|^{3/2}: purely imaginary; Re F~ starts at eps^2')
        x = sp.symbols('x', positive=True)   # x = 1/N
        ln = sp.series(Fpm(+1).subs(N, 1/x), x, 0, 3).removeO()
        print('4b. large N, eps fixed (y_+ branch):')
        for k in range(1, 3):
            print(f'    N^-{k}: {sp.factor(sp.simplify(ln.coeff(x, k)))}')
        L1 = sp.simplify(ln.coeff(x, 1))
        assert sp.simplify(L1 + pi/4*nt**2*a**2*(1 - a)**2*eps) == 0
        F1 = pi*nt*(nt + 1)*a**2*(1 - a)**2*eps/(4*(N + 8))
        print('    first-order term at large N:', sp.factor(sp.limit(F1*N, N, sp.oo)), '/N; sum with spectator channel:',
              sp.factor(sp.limit(F1*N, N, sp.oo) + L1), '/N  (exchange part only: screened)')
        L2 = sp.simplify(ln.coeff(x, 2))
        print('    classical part of N^-2 (eps-independent piece):', sp.factor(L2.subs(eps, 0)),
              ' vs large-N saddle -(pi/3) n_t^3 a^3 (1-a)^3')
        assert sp.simplify(L2.subs(eps, 0) + pi/3*nt**3*a**3*(1 - a)**3) == 0
        epsc = sp.solve(sp.Eq(b_def**2 + 4*mu0_def, 0), eps)
        print('5. merger (real fixed points for eps >= eps_c):', [sp.factor(e) for e in epsc if e != 0])

    def cli(argv):
        with fuzzy.logged('analytic/open_questions/entropy_bf_channel.log'):
            main()
    return namespace(locals(), globals())


@section
def eps2_s4_defect_coupling():
    """S4 of ENTROPY_EPS2_PLAN.md: the spectator s = 0 channel as a defect coupling f int d^2 sigma phi_sp^a phi_sp^a.

d = 4, flat space, defect = plane; bulk <phi phi> = 1/(4 pi^2 x^2); phi_sp -> b0 phi^ with phi^ unit-normalized, Delta^ = 1.
 (a) trivial-defect bulk-to-defect coefficient of the s = 0 mode: b0^2 = 1/(4 pi^2).
 (b) Wick data of O = phi^a phi^a (N_s components): C_OO = 2 N_s, C_OOO = 8 N_s, <phi^a phi^a O> = 2.
 (c) first order in f: gamma(phi^) = 4 pi f  [log coefficient of int d^2s /(|x-s|^2 |s|^2) is 4 pi/x^2];
     beta_f = 4 pi f^2 [(S_1/2) C_OOO/C_OO]; so u = 4 pi f = Delta^ - 1 and beta_u = u^2: the f^2 coefficient
     matches the radial (Riccati) flow of the AdS_3 mode, w = chi/z, w'' = (1+m^2) w, u = w'/w, du/dL = mu - u^2.
 (d) source: a bulk mass mu/(2 r^2) phi^2 gives int d^2x_perp mu b0^2/(2 r^2) = pi mu b0^2 log -> 4 pi^2 b0^2 mu = mu.
 (e) b-term: the coupling induces <phi_a^2> = -f int d^2 s 2 b0^2/(r^2+s^2)^2 = -u/(8 pi^2 r^2), i.e. the S2/S3
     spectator coincident value; fed into the mass, (lam/6)(N_s+2) <phi_a^2>, it gives mu = mu0 - b u.
     So beta_u = mu0 - b u - u^2, the S3 gap equation, with u_+ IR-stable (beta' = -sqrt(D)).
 (f) Cardy: for dg/dl = y g - pi B g^2 (unit-normalized g), Delta(defect b) = -y^3/B^2, and F~ = pi b/6
     (log<D> = (b/3) log R <-> 1/eps).  This must reproduce F~(y_+) - F~(y_-) = -(pi N_s/48) D^{3/2} of S3,
     including the b-term, and the gradient potential (pi N_s/8) int (u^2 + b u - mu0) du gives the closed form.
 (g) numerics: the flow from the BF point u = 0 reaches u_+ when D > 0, and walks (L ~ pi/sqrt|D|) when D < 0.
"""
    import itertools
    import sympy as sp
    from mpmath import mp, mpf, quad, log, pi as mpi, odefun, sqrt as msqrt


    def main():
        pi = sp.pi
        # (a) s = 0 part of 1/(4 pi^2 (r^2 + r'^2 - 2 r r' cos t + y^2)) at r = r' = 0 is 1/(4 pi^2 y^2)  -> b0^2 = 1/(4 pi^2)
        b0sq = sp.Rational(1, 4)/pi**2

        # (b) Wick contractions for a generalized free field with unit propagators


        def wick(fields):
            """fields: list of (point, component); sum over perfect matchings of delta(component) for distinct points."""
            if not fields:
                return 1
            (p0, c0), rest = fields[0], fields[1:]
            tot = 0
            for i, (p, c) in enumerate(rest):
                if p != p0 and c == c0:
                    tot += wick(rest[:i] + rest[i + 1:])
            return tot


        for Ns in [1, 2, 3]:
            O = lambda pt: [[(pt, a), (pt, a)] for a in range(Ns)]
            COO = sum(wick(x + y) for x in O(1) for y in O(2))
            COOO = sum(wick(x + y + z) for x in O(1) for y in O(2) for z in O(3))
            CffO = sum(wick([(1, 0), (2, 0)] + z) for z in O(3))
            assert (COO, COOO, CffO) == (2*Ns, 8*Ns, 2)
        print('(b) C_OO = 2 N_s, C_OOO = 8 N_s, <phi^a phi^a O> = 2  (Wick, N_s = 1, 2, 3)')

        # (c) log coefficient of I(a) = int_{|s|>a, |s-x|>a} d^2 s /(|x-s|^2 |s|^2), x = 1: I(a1) - I(a2) = 4 pi log(a2/a1)
        mp.dps = 15


        def I(a):
            f = lambda rr, th: rr/((rr**2 - 2*rr*mp.cos(th) + 1)*rr**2)
            # region |s| > a, excluding the disk |s - x| < a: integrate in polar coords about 0, subtract the small disk numerically
            tot = quad(lambda th: quad(lambda rr: f(rr, th) if (rr**2 - 2*rr*mp.cos(th) + 1) > a**2 else 0,
                                       [a, 1 - a, 1, 1 + a, 3, mp.inf]), [-mp.pi, -a, 0, a, mp.pi])
            return tot


        slope = (I(mpf('0.02')) - I(mpf('0.04')))/log(2)
        assert abs(slope - 4*mpi) < 0.05, slope
        print(f'(c) log coefficient {mp.nstr(slope, 5)} vs 4 pi = {mp.nstr(4*mpi, 5)}; <phi^ phi^> (1 - 2 f (4 pi) log x) gives gamma(phi^) = 4 pi f')
        S1 = 2*pi
        assert S1/2*sp.Rational(8, 2) == 4*pi                       # beta_f = (S_1/2)(C_OOO/C_OO) f^2 = 4 pi f^2
        f, u, L, m2 = sp.symbols('f u L m2')
        assert sp.expand((4*pi*f**2*4*pi).subs(f, u/(4*pi))) == u**2  # u = 4 pi f: beta_u = 4 pi beta_f = u^2
        z = sp.symbols('z', positive=True)
        w = sp.Function('w')
        chi = z*w(sp.log(z))
        ode = sp.simplify((z**2*sp.diff(chi, z, 2) - z*sp.diff(chi, z) - m2*chi)/z)  # k = 0 mode on H^3 (Poincare)
        Ls = sp.symbols('Ls')
        ode_L = sp.simplify(ode.subs(sp.log(z), Ls).doit())
        print('    radial mode equation in L = log z:', sp.simplify(ode_L), ' -> w\'\' = (1+m^2) w, Riccati du/dL = mu - u^2')

        # (d) source coefficient
        mu = sp.symbols('mu')
        r = sp.symbols('r', positive=True)
        src = sp.integrate(2*pi*r*mu*b0sq/(2*r**2)*r, (r, 1, sp.E))/1   # coefficient of log: int 2 pi r dr mu b0^2/(2 r^2)
        assert sp.simplify(sp.integrate(2*pi*r*mu*b0sq/(2*r**2), (r, 1, sp.E)) - pi*mu*b0sq) == 0
        assert sp.simplify(4*pi*pi*mu*b0sq - mu) == 0
        print('(d) source: pi mu b0^2 per log, so beta_u gains +mu (matches the Riccati source)')

        # (e) induced one-point function
        s = sp.symbols('s', positive=True)
        one_pt = -f*sp.integrate(2*pi*s*2*b0sq/(r**2 + s**2)**2, (s, 0, sp.oo))
        assert sp.simplify(one_pt - (-(4*pi*f)/(8*pi**2*r**2))) == 0
        print('(e) <phi_a^2> = -u/(8 pi^2 r^2) with u = 4 pi f: the S3 coincident value; mu = mu0 - b u')

        # (f) Cardy's formula with the b-term, against the S3 closed form
        mu0, b, Ns_ = sp.symbols('mu0 b N_s', real=True)
        D = b**2 + 4*mu0
        up, um = (-b + sp.sqrt(D))/2, (-b - sp.sqrt(D))/2
        beta = mu0 - b*u - u**2
        assert sp.simplify(sp.diff(beta, u).subs(u, up) + sp.sqrt(D)) == 0      # u_+ IR stable
        # expand about u_-: u = u_- + v, g = sqrt(2 N_s) v/(4 pi) (unit-normalized O): dg/dL = y g - pi B g^2
        y_c = sp.simplify(sp.diff(beta, u).subs(u, um))                         # = +sqrt(D)
        B_c = 4/sp.sqrt(2*Ns_)
        v = sp.symbols('v')
        gsym = sp.symbols('g')
        beta_v = sp.expand(beta.subs(u, um + v))
        beta_g = sp.expand(sp.sqrt(2*Ns_)/(4*pi)*beta_v.subs(v, 4*pi*gsym/sp.sqrt(2*Ns_)))
        assert sp.simplify(beta_g - (y_c*gsym - pi*B_c*gsym**2)) == 0
        dF_cardy = pi/6*(-y_c**3/B_c**2)
        closed = lambda sg: pi*Ns_/96*(b**3 + 6*b*mu0 - sg*D**sp.Rational(3, 2))
        assert sp.simplify(dF_cardy - (closed(1) - closed(-1))) == 0
        pot = pi*Ns_/8*sp.integrate(u**2 + b*u - mu0, (u, 0, u))
        for sg, uu in [(1, up), (-1, um)]:
            assert sp.simplify(sp.expand(pot.subs(u, uu) - closed(sg))) == 0
        print('(f) Cardy (B^2 = 8/N_s, y = sqrt D) gives F~(y_+) - F~(y_-) = -(pi N_s/48) D^{3/2} = S3; the gradient potential',
              '(pi N_s/8) int(u^2+bu-mu0) reproduces the S3 closed form at both fixed points')

        # (g) the flow from the BF point, O(3) half twist (N = 3, n_t = 1)
        for e in [mpf(3), mpf(1)]:
            bb = 3*e/11
            m0 = -e/22
            Dv = bb**2 + 4*m0
            sol = odefun(lambda LL, uu: m0 - bb*uu - uu**2, 0, mpf(0))
            if Dv > 0:
                u_end = sol(mpf(60))
                print(f'(g) eps = {mp.nstr(e, 2)}: D = {mp.nstr(Dv, 4)} > 0, u(L=60) = {mp.nstr(u_end, 8)} vs u_+ = '
                      f'{mp.nstr((-bb + msqrt(Dv))/2, 8)}')
                assert abs(u_end - (-bb + msqrt(Dv))/2) < 1e-6
            else:
                # time to run away to u = -1 from u = 0 ~ (2/sqrt|D|)(atan terms); compare with the walking scale pi/sqrt|D|
                LL = mpf(0); uu = mpf(0); h = mpf('0.001')
                while uu > -1:
                    uu += h*(m0 - bb*uu - uu**2); LL += h
                print(f'(g) eps = {mp.nstr(e, 2)}: D = {mp.nstr(Dv, 4)} < 0, no fixed point; u reaches -1 at L = {mp.nstr(LL, 4)}'
                      f' (walking scale pi/sqrt|D| = {mp.nstr(mpi/msqrt(-Dv), 4)})')

    def cli(argv):
        mp.dps = 15
        with fuzzy.logged('analytic/open_questions/eps2_s4_defect_coupling.log'):
            main()
    return namespace(locals(), globals())


@section
def eps2_s5_d3_epsc():
    """S5 of ENTROPY_EPS2_PLAN.md (as re-scoped by the parent session):
 (a) d = 3 inputs: spectator s = 0 response g_sp = -y/8 (y = Dhat - 1/2) from psi'(1/2) = pi^2/2, and
     dg_tw = -A_alpha/(4 pi^2) from GHJK (2.16)/2 at d = 3.
 (b) the same channel at large N in d = 3: the sigma-singlet constraint 2 n_t dg_tw + N_s g_sp = 0 gives
     y = -4 n_t A/(pi^2 N_s) (eq:sqrta at leading order); the literal Hartree channel at lam -> infinity gives
     y = -4 n_t A/(pi^2 (N_s+2)), which at n_t = 1 is eq:sqrta with N exactly (Dhat_sp = 0.365 at N = 3).
     Both are the small-|y| root on the singular side, i.e. the y_+ branch of S3 (at large N in 4-eps y_+ -> mu0/b ~ 1/N,
     while y_- -> -b ~ eps stays finite).
 (c) eps_c(alpha, N) = 8 n_t alpha(1-alpha)(N+8)/(N_s+2)^2 for O(3), O(4) at n_t = 1, and the twist below which the
     leading-order channel is real at eps = 1.
"""
    import sympy as sp
    from mpmath import mp, mpf, psi, pi as mpi, tan, sqrt, findroot


    def main():
        mp.dps = 20
        y, a, d = sp.symbols('y alpha d')
        pi = sp.pi
        # (a)
        assert sp.simplify(sp.polygamma(1, sp.Rational(1, 2)) - pi**2/2) == 0
        odd = sp.series(-(sp.polygamma(0, sp.Rational(1, 2) + y) - sp.polygamma(0, sp.Rational(1, 2) - y))/2/(2*pi), y, 0, 2).removeO()
        assert sp.simplify(odd/(2*pi) + y/8) == 0          # per real component on S^1 x H^2: -y/8
        A = pi/2*(1 - 2*a)*sp.tan(pi*a)
        ghjk = (d - 1)*sp.gamma(d/2 - a)*sp.gamma(d/2 - 1 + a)*sp.sin(pi*a)*sp.gamma(d/2)/(pi**(d/2 + 1)*sp.gamma(d)*(2 - d))
        for av in [sp.Rational(1, 10), sp.Rational(1, 4), sp.Rational(2, 5)]:
            assert abs(sp.N(ghjk.subs({d: 3, a: av})/2 + A.subs(a, av)/(4*pi**2), 30)) < 1e-25
        print('(a) d = 3: g_sp(s=0) = -y/8 per real component;  dg_tw = -A_alpha/(4 pi^2) (GHJK 2.16/2)')

        # (b)
        Aal = lambda x: mpf(1) if x == mpf('0.5') else mpi/2*(1 - 2*x)*tan(mpi*x)
        for Nv in [3, 4, 10]:
            x = mpf('0.5'); Ns = Nv - 2
            y_constraint = -4*Aal(x)/(mpi**2*Ns)            # sigma-singlet constraint, exact in N_s
            y_hartree = -4*Aal(x)/(mpi**2*(Ns + 2))         # Hartree at infinite coupling (exchange '+2' kept)
            y_paper = -4*Aal(x)/(mpi**2*Nv)                 # eq:sqrta as printed (leading order in 1/N)
            print(f'(b) N = {Nv}, alpha = 1/2: Dhat_sp = 1/2 + y:  constraint(N_s) {mp.nstr(0.5 + y_constraint, 4)},'
                  f'  Hartree(N_s+2) {mp.nstr(0.5 + y_hartree, 4)},  eq:sqrta {mp.nstr(0.5 + y_paper, 4)}')
        assert abs(0.5 - 4/(mpi**2*3) - mpf('0.3649')) < 1e-4
        # identification with y_+: in 4 - eps at large N, y_+ = (-b + sqrt(b^2 + 4 mu0))/2 -> mu0/b = -2 n_t a(1-a)/(N_s+2)
        eps, N, nt = sp.symbols('epsilon N n_t', positive=True)
        Ns = N - 2*nt
        b = eps*(Ns + 2)/(N + 8); mu0 = -2*nt*a*(1 - a)*eps/(N + 8)
        yp = (-b + sp.sqrt(b**2 + 4*mu0))/2
        ym = (-b - sp.sqrt(b**2 + 4*mu0))/2
        assert sp.simplify(sp.limit((yp*N).subs({a: sp.Rational(1, 2), nt: 1}), N, sp.oo) + sp.Rational(1, 2)) == 0
        assert sp.simplify(sp.limit(ym.subs({a: sp.Rational(1, 2), nt: 1}), N, sp.oo) + eps) == 0
        print('    4 - eps, large N: y_+ N -> -2 n_t a(1-a) [the analogue of -4A/pi^2], y_- -> -eps: the paper uses y_+')

        # (c)
        epsc = lambda Nv, x, ntv=1: 8*ntv*x*(1 - x)*(Nv + 8)/(Nv - 2*ntv + 2)**2
        print('(c) eps_c (real fixed points for eps >= eps_c), n_t = 1:')
        print('    alpha   :  0.1     0.25    0.3     0.4     0.5')
        for Nv in [3, 4]:
            row = '  '.join(f'{float(epsc(Nv, mpf(x))):.3f}' for x in ['0.1', '0.25', '0.3', '0.4', '0.5'])
            amax = findroot(lambda x: epsc(Nv, x) - 1, mpf('0.1'))
            print(f'    O({Nv})    :  {row}    real at eps = 1 only for alpha < {mp.nstr(amax, 4)} (or > {mp.nstr(1 - amax, 4)})')

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/open_questions/eps2_s5_d3_epsc.log'):
            main()
    return namespace(locals(), globals())


@section
def eps2_r1_counterterm():
    """R(i): the coupling-counterterm insertion into the first-order diagram, d = 4 - eps, MS-bar (mu R = 1).

F_ct = M^eps 16 pi^2 dlb (1/24) <(phi^2)^2>_D Vol_4,  dlb = (N+8) lb^2/(3 eps),  <(phi^2)^2>_D = 4 n_t(n_t+1) dg(d)^2.
Claim (units of lb^2):  F_ct = -(N+8) n_t(n_t+1) a^2(1-a)^2/(18 eps^2) [1 + K(a) eps + O(eps^2)],
K(a) = 7/2 - gamma_E - log 2 - psi(1+a) - psi(2-a).   Calibration: a -> 0 gives zero.
"""
    from mpmath import mp, mpf, euler, log, psi, nstr
    _eps2_common = eps2_common(); dg, vol4, Meps, pi, laurent = _eps2_common.dg, _eps2_common.vol4, _eps2_common.Meps, _eps2_common.pi, _eps2_common.laurent
    mp.dps = 40


    def Fct_over_lb2(e, a, N, nt):
        return Meps(e)*16*pi**2*(N + 8)/(3*e)/24*4*nt*(nt + 1)*dg(e, a)**2*vol4(e)


    K = lambda a: mpf(7)/2 - euler - log(2) - psi(0, 1 + a) - psi(0, 2 - a)

    def main():
        for a in [mpf('0.1'), mpf('0.25'), mpf('0.5')]:
            for N, nt in [(3, 1), (2, 1), (6, 2)]:
                c = laurent(lambda e: Fct_over_lb2(e, a, N, nt), -2, 0)
                lead = -(N + 8)*nt*(nt + 1)*a**2*(1 - a)**2/18
                assert abs(c[-2]/lead - 1) < mpf(10)**-20 and abs(c[-1]/c[-2] - K(a)) < mpf(10)**-20
            print(f'R(i) alpha = {nstr(a, 3)}: c_-2 = {nstr(c[-2], 12)} (N=6,n_t=2), K = c_-1/c_-2 = {nstr(K(a), 12)}, c_0/c_-2 = {nstr(c[0]/c[-2], 12)}')
        c = laurent(lambda e: Fct_over_lb2(e, mpf(10)**-8, 3, 1), -2, -1)
        assert abs(c[-2]) < mpf(10)**-15 and abs(c[-1]) < mpf(10)**-15
        print('R(i) passed: c_-2 = -(N+8) n_t(n_t+1) a^2(1-a)^2/18, K(a) closed form (20 digits); calibration a -> 0: 0')

    def cli(argv):
        mp.dps = 40
        with fuzzy.logged('analytic/open_questions/eps2_r1_counterterm.log'):
            main()
    return namespace(locals(), globals())


@section
def eps2_r2_cactus_twisted():
    """R(ii): the cactus with twisted loops, d = 4 - eps, MS-bar (mu R = 1).

F = -(1/4) sum_a (m_a^2)^2 Tr[G_a G_a];  twisted plane: m_t^2 = (lam_B/3)(n_t+1) dg(d),  Tr[G G] = -Vol_4 dg_t/dm^2 per real
component, so F_cac,t = (1/2) n_t (m_t^2)^2 Vol_4 dg_t/dm^2, with
  dg_t/dm^2 = (1/2pi) P(d) S(eps),  P = Gamma((3-d)/2)/(4pi)^{(d-1)/2},
  S(eps) = sum_{s in Z+a} R'(s)/(2s),  R(s) = Gamma(s + 1 - eps/2)/Gamma(s + eps/2)   [D = d/2 - 1 + |s|].
Large s: R'(s)/(2s) = (1/2)[(1-eps) s^{-1-eps} - (1+eps) c2 s^{-3-eps} + O(s^{-5-eps})], c2 = eps(1-eps)(2-eps)/24
(only even inverse powers since a + b = 1 in Gamma(s+a)/Gamma(s+b)); the two leading terms are summed as Hurwitz zetas.
Claims: S = 1/eps - 1 - (psi(a) + psi(1-a))/2 + O(eps);
  F_cac,t = n_t (n_t+1)^2 a^2(1-a)^2/(9 eps^2) [1 + K_t eps],
  K_t = 7/2 - gamma_E - 2 log 2 - (3/2)(psi(a) + psi(1-a)) - 1/a - 1/(1-a).   Calibration: a -> 0 gives zero.
"""
    from mpmath import mp, mpf, gamma, rgamma, psi, zeta, diff, nsum, inf, euler, log, nstr
    _eps2_common = eps2_common(); dg, vol4, Meps, pi, laurent = _eps2_common.dg, _eps2_common.vol4, _eps2_common.Meps, _eps2_common.pi, _eps2_common.laurent
    mp.dps = 20


    def S(e, a, nmax=300):
        """Hurwitz-continued mode sum; remainder after two asymptotic terms decays like s^-5, summed to nmax."""
        c2 = e*(1 - e)*(2 - e)/24
        def q(s):
            return gamma(s + 1 - e/2)*rgamma(s + e/2)*(psi(0, s + 1 - e/2) - psi(0, s + e/2))/(2*s)
        asym = lambda s: ((1 - e)*s**(-1 - e) - (1 + e)*c2*s**(-3 - e))/2
        rem = mp.fsum(q(n + a) - asym(n + a) + q(n + 1 - a) - asym(n + 1 - a) for n in range(nmax))
        Z = lambda x: zeta(x, a) + zeta(x, 1 - a)
        return (1 - e)*Z(1 + e)/2 - (1 + e)*c2*Z(3 + e)/2 + rem


    def P(e):
        d = 4 - e
        return gamma((3 - d)/2)/(4*pi)**((d - 1)/2)


    def Fcac_over_lb2(e, a, nt):
        m2_over_lb = Meps(e)*16*pi**2/3*(nt + 1)*dg(e, a)
        return mpf(1)/2*nt*m2_over_lb**2*vol4(e)*P(e)*S(e, a)/(2*pi)


    Kt = lambda a: mpf(7)/2 - euler - 2*log(2) - mpf(3)/2*(psi(0, a) + psi(0, 1 - a)) - 1/a - 1/(1 - a)

    def main():
        for a in [mpf('0.25'), mpf('0.5')]:
            cS = laurent(lambda e: S(e, a), -1, 0)
            assert abs(cS[-1] - 1) < mpf(10)**-12 and abs(cS[0] - (-1 - (psi(0, a) + psi(0, 1 - a))/2)) < mpf(10)**-12
            c = laurent(lambda e: Fcac_over_lb2(e, a, 1), -2, -1)
            lead = 1*(2)**2*a**2*(1 - a)**2/9
            assert abs(c[-2]/lead - 1) < mpf(10)**-12 and abs(c[-1]/c[-2] - Kt(a)) < mpf(10)**-10
            print(f'R(ii) alpha = {nstr(a, 3)}: S = 1/eps + {nstr(cS[0], 10)};  c_-2 = {nstr(c[-2], 10)} (n_t=1),'
                  f'  K_t = {nstr(c[-1]/c[-2], 12)}')
        small = mpf(10)**-6
        c = laurent(lambda e: Fcac_over_lb2(e, small, 1), -2, -1)
        assert abs(c[-2]) < mpf(10)**-9 and abs(c[-1]) < mpf(10)**-5
        print('R(ii) passed: c_-2 = n_t(n_t+1)^2 a^2(1-a)^2/9, K_t closed form; calibration a -> 0 gives', nstr(c[-1], 3))

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/open_questions/eps2_r2_cactus_twisted.log'):
            main()
    return namespace(locals(), globals())


@section
def eps2_r3_cactus_spectator():
    """R(iii): the cactus with s != 0 spectator loops (the s = 0 mode is the resummed channel of S3/S4), d = 4 - eps, MS-bar.

F_cac,sp = (1/4) N_s mu0^2 Vol_4 dg_sp/dm^2|_{s != 0},  mu0 = (lam_B/3) n_t dg(d),  dg_sp/dm^2 = (1/2pi) P(d) S_sp,
S_sp = sum_{s in Z, s != 0} R'(|s|)/(2|s|)  (same summand and asymptotics as R(ii)).
Claims: S_sp = 1/eps - 1 + gamma_E + O(eps);
  F_cac,sp = N_s n_t^2 a^2(1-a)^2/(18 eps^2) [1 + K_sp eps],  K_sp = 7/2 - 2 log 2 - psi(1+a) - psi(2-a).
Calibration: a -> 0 gives zero.  Pole bookkeeping (units n_t a^2(1-a)^2 lb^2/(18 eps^2)): ct -(N+8)(n_t+1), twisted cactus
+2(n_t+1)^2, spectator cactus +N_s n_t; so the melon must supply +[(N+2) + 4(n_t+1)] (checked in R(iv)).
"""
    from mpmath import mp, mpf, gamma, rgamma, psi, zeta, euler, log, nstr
    _eps2_common = eps2_common(); dg, vol4, Meps, pi, laurent = _eps2_common.dg, _eps2_common.vol4, _eps2_common.Meps, _eps2_common.pi, _eps2_common.laurent
    mp.dps = 20


    def S_sp(e, nmax=300):
        c2 = e*(1 - e)*(2 - e)/24
        q = lambda s: gamma(s + 1 - e/2)*rgamma(s + e/2)*(psi(0, s + 1 - e/2) - psi(0, s + e/2))/(2*s)
        asym = lambda s: ((1 - e)*s**(-1 - e) - (1 + e)*c2*s**(-3 - e))/2
        rem = mp.fsum(q(n) - asym(n) for n in range(1, nmax))
        return 2*((1 - e)*zeta(1 + e)/2 - (1 + e)*c2*zeta(3 + e)/2 + rem)


    def P(e):
        d = 4 - e
        return gamma((3 - d)/2)/(4*pi)**((d - 1)/2)


    def Fsp_over_lb2(e, a, N, nt):
        mu0_over_lb = Meps(e)*16*pi**2/3*nt*dg(e, a)
        return mpf(1)/4*(N - 2*nt)*mu0_over_lb**2*vol4(e)*P(e)*S_sp(e)/(2*pi)

    def main():
        cS = laurent(S_sp, -1, 0)
        assert abs(cS[-1] - 1) < mpf(10)**-12 and abs(cS[0] - (euler - 1)) < mpf(10)**-12
        Ksp = lambda a: mpf(7)/2 - 2*log(2) - psi(0, 1 + a) - psi(0, 2 - a)
        for a in [mpf('0.25'), mpf('0.5')]:
            c = laurent(lambda e: Fsp_over_lb2(e, a, 3, 1), -2, -1)
            lead = 1*1*a**2*(1 - a)**2/18
            assert abs(c[-2]/lead - 1) < mpf(10)**-12 and abs(c[-1]/c[-2] - Ksp(a)) < mpf(10)**-10
            print(f'R(iii) alpha = {nstr(a, 3)}: c_-2 = {nstr(c[-2], 10)} (N=3, n_t=1), K_sp = {nstr(c[-1]/c[-2], 12)}')
        c = laurent(lambda e: Fsp_over_lb2(e, mpf(10)**-6, 3, 1), -2, -1)
        assert abs(c[-2]) < mpf(10)**-12 and abs(c[-1]) < mpf(10)**-12
        print(f'R(iii) passed: S_sp = 1/eps + gamma - 1; c_-2 = N_s n_t^2 a^2(1-a)^2/18; K_sp closed form; calibration a -> 0: {nstr(c[-1], 2)}')

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/open_questions/eps2_r3_cactus_spectator.log'):
            main()
    return namespace(locals(), globals())


@section
def eps2_r4a_calibration():
    """R(iv-a): calibrate the 'd = 4 Hadamard finite part + local d-dim terms' evaluation of dim-reg integrals on S^1 x H^{d-1}.

Test 1 (vertex type): int |G_a|^2 = -(1/2pi) P(d) S(eps) exactly (R(ii)); method: Fin_B + Omega_d C_d^2 / eps (Q0 = 1).
Test 2 (sunset type, phi = 1): int G0^3 = C_d^3 pi^{(d-2)/2} Gamma(d-2)/Gamma(3(d-2)/2) * pi Gamma(1+p)Gamma(1+q)Gamma(-1-p-q)/
        (Gamma(-p)Gamma(-q)Gamma(2+p+q)),  p = (d-6)/4, q = -(d-2)   [flat-space map, y_par integral then 2d formula];
        method: Fin_3 + T(d)/(2 eps), T(d) = Omega_d C_d^3 (d-2)(d+2)/(24 d).
d = 4 numerics: measure dtheta 4 pi sinh^2 chi dchi on (-pi, pi) x (0, inf); geodesic distance rho^2 = theta^2 + chi^2.
"""
    import numpy as np
    from scipy.integrate import quad
    from mpmath import mp, mpf, gamma as mg, pi as mpi, nstr
    _eps2_common = eps2_common(); laurent = _eps2_common.laurent

    _eps2_hadamard = eps2_hadamard(); PI, C4, Om4, G0, Ga, hadamard = _eps2_hadamard.PI, _eps2_hadamard.C4, _eps2_hadamard.Om4, _eps2_hadamard.G0, _eps2_hadamard.Ga, _eps2_hadamard.hadamard
    mp.dps = 20
    Cd = lambda e: mg((4 - e)/2 - 1)/(4*mpi**((4 - e)/2))
    Omd = lambda e: 2*mpi**((4 - e)/2)/mg((4 - e)/2)

    def main():
        # Test 1

        _r2 = eps2_r2_cactus_twisted(); ns = {'P': _r2.P, 'S': _r2.S}   # the research script exec'd these definitions of R(ii)
        for a in [0.5, 0.25]:
            FinB, _ = hadamard(lambda t, c: abs(Ga(t, c, a))**2, 0.0, 1/(8*PI**2))
            exact = laurent(lambda e: -ns['P'](e)*ns['S'](e, mpf(a))/(2*mpi), -1, 0)
            loc = laurent(lambda e: Omd(e)*Cd(e)**2/e, -1, 0)
            assert abs(exact[-1] - loc[-1]) < 1e-15
            print(f'Test 1 (alpha={a}): exact finite {nstr(exact[0], 12)}  method {FinB + float(loc[0]):.12f}  diff {float(exact[0]) - FinB - float(loc[0]):.2e}')
        # Test 2
        def I3(e):
            d = 4 - e; p = (d - 6)/4; q = -(d - 2)
            return Cd(e)**3*mpi**((d - 2)/2)*mg(d - 2)/mg(3*(d - 2)/2)*mpi*mg(1 + p)*mg(1 + q)*mg(-1 - p - q)/(mg(-p)*mg(-q)*mg(2 + p + q))
        T = lambda e: Omd(e)*Cd(e)**3*(2 - e)*(6 - e)/(24*(4 - e))
        Fin3, fit = hadamard(lambda t, c: G0(t, c)**3, Om4*C4**3, float(T(mpf(0))))
        ex3 = laurent(I3, -1, 0)
        loc3 = laurent(lambda e: T(e)/(2*e), -1, 0)
        print(f'Test 2: pole exact {nstr(ex3[-1], 12)} local {nstr(loc3[-1], 12)};  finite exact {nstr(ex3[0], 12)}  method {Fin3 + float(loc3[0]):.12f}'
              f'  diff {float(ex3[0]) - Fin3 - float(loc3[0]):.2e}   (small-rho fit a1, a3 = {fit[0]:.3e}, {fit[1]:.3e})')

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/open_questions/eps2_r4a_calibration.log'):
            main()
    return namespace(locals(), globals())


@section
def eps2_r4b_melon_integrals():
    """R(iv-b): the two melon integrals in dimensional regularization, at alpha = 1/2, 1/4 and a small-alpha calibration.

J3 = int G0^2 dP  (two twisted + two spectator lines; also 2 J3 inside M_TT),  J2 = int dP^2,  dP = |G_a|^2 - G0^2.
Local structure at coincidence (phi0 = dg = -a(1-a)/(8 pi^2) at d = 4, phi = Re dG, box phi = -(d-2)^2 phi0/4):
  J3: power a_-3 = 2 Om4 C4^3 phi0; log a_-1 = Om4 C4^2 phi0^2 (the 2 phi part has none at d = 4: EOM);
      DR = Fin3 + Om_d C_d^3 phi0(d) 2(d-2)/(24 d) + Om_d C_d^2 phi0(d)^2/eps
  J2: log a_-1 = 4 Om4 C4^2 phi0^2;  DR = Fin2 + 4 Om_d C_d^2 phi0(d)^2/eps
Calibration: alpha -> 0 (dP -> 0) gives zero.  Diagnostics: the fitted residual 1/rho coefficient must vanish.
"""
    import numpy as np
    from mpmath import mp, mpf, gamma as mg, pi as mpi, nstr
    _eps2_hadamard = eps2_hadamard(); PI, C4, Om4, G0, Ga, dP, hadamard = _eps2_hadamard.PI, _eps2_hadamard.C4, _eps2_hadamard.Om4, _eps2_hadamard.G0, _eps2_hadamard.Ga, _eps2_hadamard.dP, _eps2_hadamard.hadamard
    _eps2_common = eps2_common(); dg, laurent = _eps2_common.dg, _eps2_common.laurent
    mp.dps = 20
    Cd = lambda e: mg((4 - e)/2 - 1)/(4*mpi**((4 - e)/2))
    Omd = lambda e: 2*mpi**((4 - e)/2)/mg((4 - e)/2)

    def main():
        res = {}
        for a in [0.5, 0.25, 0.02]:
            ph0 = -a*(1 - a)/(8*PI**2)
            dPa = lambda t, c: dP(t, c, a)
            F3, d3 = hadamard(lambda t, c: G0(t, c)**2*dPa(t, c), 2*Om4*C4**3*ph0, Om4*C4**2*ph0**2)
            F2, d2 = hadamard(lambda t, c: dPa(t, c)**2, 0.0, 4*Om4*C4**2*ph0**2)
            am = mpf(a)
            loc3 = laurent(lambda e: Omd(e)*Cd(e)**3*dg(e, am)*2*(2 - e)/(24*(4 - e)) + Omd(e)*Cd(e)**2*dg(e, am)**2/e, -1, 0)
            loc2 = laurent(lambda e: 4*Omd(e)*Cd(e)**2*dg(e, am)**2/e, -1, 0)
            J3 = (float(loc3[-1]), F3 + float(loc3[0])); J2 = (float(loc2[-1]), F2 + float(loc2[0]))
            res[a] = (J3, J2)
            print(f'alpha={a}: J3 = {J3[0]:.10e}/eps + {J3[1]:.10e}   [resid 1/rho {d3[0]:.1e} vs a_-1 {Om4*C4**2*ph0**2:.1e}]')
            print(f'          J2 = {J2[0]:.10e}/eps + {J2[1]:.10e}   [resid 1/rho {d2[0]:.1e}]')
            print(f'          (Fin3 = {F3:.10e}, E_sunset = {float(Omd(0)*Cd(0)**3)*ph0*2*2/(24*4):.4e}, Fin2 = {F2:.10e})')
        small = res[0.02]
        print('calibration: J3, J2 finite parts at alpha = 0.02 scaled by 1/alpha^2:', f'{small[0][1]/0.02**2:.3e}', f'{small[1][1]/0.02**2:.3e}',
              '(both -> 0 as alpha -> 0)')
        store.put('analytic/open_questions/eps2_r4b_melon_integrals.npy', np.array([[a, *res[a][0], *res[a][1]] for a in res]).tolist())

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/open_questions/eps2_r4b_melon_integrals.log'):
            main()
    return namespace(locals(), globals())


@section
def eps2_r4c_melon_assembly():
    """R(iv-c): assemble the melon, F_melon = -(lam_B^2/36) n_t Vol_4 [ (N+2) J3 + (n_t+1) J2 ], in units of lb^2, and check
its pole against the bookkeeping of R(i)-(iii): c_-2(melon) = +[(N+2) + 4(n_t+1)] n_t a^2(1-a)^2/18.
lam_B^2 = M^{2eps} (16 pi^2)^2 lb^2; J = j_-1/eps + j_0 (R(iv-b)); Vol_4(eps) and M^eps from eps2_common.
"""
    import numpy as np
    from mpmath import mp, mpf, pi, nstr
    _eps2_common = eps2_common(); vol4, Meps, laurent = _eps2_common.vol4, _eps2_common.Meps, _eps2_common.laurent
    mp.dps = 20

    def main():
        data = {round(float(r[0]), 4): r[1:] for r in np.array(store.get('analytic/open_questions/eps2_r4b_melon_integrals.npy'))}
        pref = laurent(lambda e: -Meps(e)**2*(16*pi**2)**2/36*vol4(e), -1, 0)     # = p_-1/eps + p_0
        out = {}
        for a, (j3m, j30, j2m, j20) in data.items():
            for N, nt in [(3, 1), (2, 1), (4, 1)]:
                Jm = (N + 2)*j3m + (nt + 1)*j2m; J0 = (N + 2)*j30 + (nt + 1)*j20
                cm2 = nt*pref[-1]*Jm; cm1 = nt*(pref[-1]*J0 + pref[0]*Jm)
                expect = ((N + 2) + 4*(nt + 1))*nt*a**2*(1 - a)**2/18
                assert abs(cm2/expect - 1) < 1e-9, (a, N, cm2, expect)
                out[(a, N, nt)] = (float(cm2), float(cm1))
                print(f'alpha={a} N={N} n_t={nt}: c_-2 = {float(cm2):.10e} (expected {expect:.10e}),  c_-1 = {float(cm1):.10e}')
        store.put('analytic/open_questions/eps2_r4c_melon_assembly.npy', np.array([[k[0], k[1], k[2], *v] for k, v in out.items()]).tolist())
        print('R(iv-c) passed: melon pole = +[(N+2) + 4(n_t+1)] n_t a^2(1-a)^2/18, so the total pole of R(i)-(iv) cancels')

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/open_questions/eps2_r4c_melon_assembly.log'):
            main()
    return namespace(locals(), globals())


@section
def eps2_r5_assembly():
    """R(v): the O(eps^2) defect entropy of a partial twist in MS-bar, and the O(3) -> O(2) change.

F = F_1 + F_ct + F_cac,t + F_cac,sp + F_melon (+ the resummed s = 0 spectator channel, added to F~ directly):
  F_1 = lb (f_-1/eps + f_0 + ...),  two loops = lb^2 (C_-2/eps^2 + C_-1/eps + ...), C_-2 = 0;
  lb* = 3 eps/(N+8) + l2 eps^2, l2 = 9(3N+14)/(N+8)^3;  F~ = -sin(pi eps/2) F.
  F~ = F~_free + eps * F~1 + eps^2 * F~2 + O(eps^{5/2}),
  F~1 = -(pi/2) 3 f_-1/(N+8),
  F~2 = -(pi/2)[3 f_0/(N+8) + l2 f_-1 + 9 C_-1/(N+8)^2] - (pi/8) n_t N_s (N_s+2) a(1-a)/(N+8)^2  (S3 real part).
mu R = e^t: K, f_0 -> +t; K_t, K_sp, melon -> +2t per c_-2 (one or two powers of M^eps). F~2 must not depend on t.
The free part is N-independent at fixed n_t and cancels in Delta = F~(N-1) - F~(N).
"""
    import numpy as np
    from mpmath import mp, mpf, pi, euler, log, psi, nstr
    _eps2_common = eps2_common(); vol4, Meps, laurent = _eps2_common.vol4, _eps2_common.Meps, _eps2_common.laurent
    mp.dps = 20
    J = {round(float(r[0]), 4): [mpf(x) for x in r[1:]] for r in np.array(store.get('analytic/open_questions/eps2_r4b_melon_integrals.npy'))}
    K = lambda a: mpf(7)/2 - euler - log(2) - psi(0, 1 + a) - psi(0, 2 - a)
    Kt = lambda a: mpf(7)/2 - euler - 2*log(2) - mpf(3)/2*(psi(0, a) + psi(0, 1 - a)) - 1/a - 1/(1 - a)
    Ksp = lambda a: mpf(7)/2 - 2*log(2) - psi(0, 1 + a) - psi(0, 2 - a)
    pref = laurent(lambda e: -Meps(e)**2*(16*pi**2)**2/36*vol4(e), -1, 0)


    def pieces(a, N, nt, t=0):
        a = mpf(a); Ns = N - 2*nt; A2 = a**2*(1 - a)**2
        fm1 = -nt*(nt + 1)*A2/6; f0 = fm1*(K(a) + t)
        c_ct = -(N + 8)*nt*(nt + 1)*A2/18; c_t = nt*(nt + 1)**2*A2/9; c_sp = Ns*nt**2*A2/18
        j3m, j30, j2m, j20 = J[round(float(a), 4)]
        Jm = (N + 2)*j3m + (nt + 1)*j2m; J0 = (N + 2)*j30 + (nt + 1)*j20
        c_me = nt*pref[-1]*Jm; c_me1 = nt*(pref[-1]*J0 + pref[0]*Jm) + 2*t*c_me
        Cm2 = c_ct + c_t + c_sp + c_me
        Cm1 = {'ct': c_ct*(K(a) + t), 'cactus_t': c_t*(Kt(a) + 2*t), 'cactus_sp': c_sp*(Ksp(a) + 2*t), 'melon': c_me1}
        l2 = mpf(9)*(3*N + 14)/(N + 8)**3
        F1 = -pi/2*3*fm1/(N + 8)
        F2 = {'first-order (K)': -pi/2*3*f0/(N + 8), 'lambda* at eps^2': -pi/2*l2*fm1}
        for k, v in Cm1.items():
            F2[k] = -pi/2*9*v/(N + 8)**2
        F2['s=0 spectator (S3)'] = -pi/8*nt*Ns*(Ns + 2)*a*(1 - a)/(N + 8)**2
        return F1, F2, Cm2

    def main():
        for a in ['0.5', '0.25']:
            print(f'alpha = {a}, n_t = 1:')
            tot = {}
            for N in [2, 3, 4]:
                F1, F2, Cm2 = pieces(a, N, 1)
                _, F2t, _ = pieces(a, N, 1, t=mpf('0.7'))
                s0, s1 = sum(F2.values()), sum(F2t.values())
                assert abs(Cm2) < 1e-12 and abs(s0 - s1) < 1e-12, (Cm2, s0, s1)
                tot[N] = (F1, s0, F2)
                print(f'  N={N}: F~ - F~_free = {nstr(F1, 8)} eps + {nstr(s0, 8)} eps^2   [poles cancel: C_-2 = {nstr(Cm2, 2)};'
                      f' mu-independent: {nstr(s1 - s0, 2)}]')
            for (Nh, Nl) in [(3, 2), (4, 3)]:
                d1 = tot[Nl][0] - tot[Nh][0]; d2 = tot[Nl][1] - tot[Nh][1]
                print(f'  Delta O({Nh})->O({Nl}) = {nstr(d1, 6)} eps + {nstr(d2, 6)} eps^2;  at eps = 1: {nstr(d1 + d2, 6)}'
                      f'  (ratio eps^2/eps term {nstr(d2/d1, 4)})')
                for k in tot[Nh][2]:
                    print(f'      {k:22s}: {nstr(tot[Nl][2][k] - tot[Nh][2][k], 6)}')

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/open_questions/eps2_r5_assembly.log'):
            main()
    return namespace(locals(), globals())


@section
def entropy_tables():
    """Every number of the paper's entropy tables near four dimensions (2026-10-26):
 Table tab:entropy (a), rows 'near d = 4, O(eps)' and 'difference'; Table tab:entropy (b); Table tab:entropyeps2.
Inputs: eq:loggfree (quadrature), eq:cresult (closed form), the O(eps^2) pieces of eps2_r5_assembly.py (imported),
and the resummed spectator channel eq:Fsp, F_sp = (pi N_sp/96)[w^3 + 6 w v - (w^2 + 4 v)^{3/2}] (real part, upper root),
w = eps (N_sp + 2)/(N + 8), v = -2 n_t alpha(1-alpha) eps/(N + 8), eps_c = 8 n_t alpha(1-alpha)(N+8)/(N_sp+2)^2.
One twisted complex component (n_t = 1); Delta = log g_{N-1} - log g_N at eps = 1; large N is eq:decouple, c/(N(N-1)).
usage: python entropy_tables.py   (run from this folder; needs eps2_r4b_melon_integrals.npy)
"""
    import contextlib, io
    import numpy as np
    from scipy.integrate import quad
    r5 = eps2_r5_assembly()   # the research script imported it with its printing silenced

    def loggfree(a):
        f = lambda v: v*np.tanh(np.pi*v)*np.log((1 - 2*np.exp(-2*np.pi*v)*np.cos(2*np.pi*a) + np.exp(-4*np.pi*v))/(-np.expm1(-2*np.pi*v))**2)
        return quad(f, 0, np.inf, limit=400, epsabs=1e-14)[0]
    cres = lambda a: 16/(9*np.pi)*a*(1 - a)*(1 - 2*a)*np.tan(np.pi*a) if a != 0.5 else 8/(9*np.pi**2)
    def Fsp_re(a, N, eps=1.0, nt=1):
        Ns = N - 2*nt; w = eps*(Ns + 2)/(N + 8); v = -2*nt*a*(1 - a)*eps/(N + 8); D = w*w + 4*v
        return np.pi*Ns/96*(w**3 + 6*w*v - (D**1.5 if D >= 0 else 0.0))
    epsc = lambda a, N, nt=1: 8*nt*a*(1 - a)*(N + 8)/(N - 2*nt + 2)**2

    def main():
        print('tab:entropy (a):  alpha | log g_free | eq:cresult | near d=4 O(eps) (pi/4)a^2(1-a)^2 | difference')
        for a in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5):
            n4 = np.pi/4*a*a*(1 - a)**2
            print(f'  {a:4}  {loggfree(a):.6f}  {cres(a):.6f}  {n4:.6f}  {100*(n4/cres(a) - 1):+.0f}%')
        print('tab:entropy (b) and tab:entropyeps2 (units 1e-3 where marked):')
        for (Nh, a) in ((3, 0.25), (3, 0.5), (4, 0.25), (4, 0.5)):
            Nl = Nh - 1
            F1h, F2h, _ = r5.pieces(str(a), Nh, 1); F1l, F2l, _ = r5.pieces(str(a), Nl, 1)
            d1 = float(F1l - F1h); d2 = float(sum(F2l.values()) - sum(F2h.values()))
            s0 = float(F2l['s=0 spectator (S3)'] - F2h['s=0 spectator (S3)']); an = d2 - s0
            res = d1 + an + (Fsp_re(a, Nl) - Fsp_re(a, Nh))
            big = cres(a)/(Nh*(Nh - 1))
            print(f'  O({Nh})->O({Nl}) a={a}: O(eps) {d1:.2e} ({100*(d1/big - 1):+.0f}%)  through O(eps^2) {d1 + d2:.2e} ({100*((d1 + d2)/big - 1):+.0f}%)'
                  f'  large N {big:.2e} | eps2: analytic {1e3*an:.2f}  s=0 {1e3*s0:.2f}  resummed sum {1e3*res:.2f}  eps_c(O({Nh})) {epsc(a, Nh):.2f}')

    def cli(argv):
        with fuzzy.logged('analytic/open_questions/entropy_tables.log'):
            main()
    return namespace(locals(), globals())


@section
def entropy_eps2_checks():
    """Checks and numbers for ENTROPY_EPS2.md (2026-10-26).  Conventions as in entropy_bf_channel.py.

 A. O(eps) free energy against GHJK (5.3)-(5.4) (full twist: n_t = N_G complex fields, N = 2 N_G).
 B. O(eps) conformal weight against GHJK (5.8), and the spectator term of a partial twist.
 C. Large N: the O(1/N) fluctuation trace at leading order in eps is the exchange (Fock) term only.
 D. The first-order diagram continued to O(eps^2) in MS-bar (mu R = 1): the continuation constants.
 E. Numbers: log g_{N-1} - log g_N and the O(3) -> O(2) half twist, with the spectator channel.

Part E labels the large-N limit of the fixed-N coefficient, (pi/2) a^2 (1-a)^2, as "[paper]": that was the draft
when the check was written. The paper now uses the screened (pi/4) a^2 (1-a)^2, which this check calls "correct".
"""
    import sympy as sp
    from mpmath import mp, mpf, pi as mpi, sqrt as msqrt, tan as mtan, mpc


    def main():
        eps = sp.symbols('epsilon', positive=True)
        N, nt, a, v, beta, d = sp.symbols('N n_t alpha v beta d')
        pi = sp.pi
        lam = 48*pi**2*eps/(N + 8)
        VolH = -2*pi/eps
        dg = -a*(1 - a)/(8*pi**2)

        # A -----------------------------------------------------------------------------------------------------
        F1 = lam/24*4*nt*(nt + 1)*dg**2*2*pi*VolH
        NG = sp.symbols('N_G')
        ghjk53 = eps*NG*(NG + 1)*VolH*(1 - a)**2*a**2/(8*pi*(NG + 4))   # GHJK (5.3) interaction term, Vol(H^3) -> -2pi/eps
        assert sp.simplify(F1.subs({nt: NG, N: 2*NG}) - ghjk53) == 0
        Ft1 = sp.simplify(-(pi*eps/2)*F1)
        print('A. F~_1 =', sp.factor(Ft1), ' = GHJK (5.3) at n_t = N_G, N = 2 N_G')

        # B -----------------------------------------------------------------------------------------------------
        # coincident values on S^1_beta x H^3 per real component (GHJK 5.7 halved): twisted, and untwisted spectator
        gt = (6*a**2 - 6*a + 1)/(12*beta**2) - 1/(48*pi**2)
        gs = 1/(12*beta**2) - 1/(48*pi**2)
        assert sp.simplify(gt.subs(beta, 2*pi) - dg) == 0 and gs.subs(beta, 2*pi) == 0
        Ns = N - 2*nt
        Sg = 2*nt*gt + Ns*gs
        X = Sg**2 + 2*(2*nt*gt**2 + Ns*gs**2)
        F1b = lam/24*X*beta                      # per Vol(H^3)
        h1 = sp.simplify(-(2*pi/3)*sp.diff(F1b, beta).subs(beta, 2*pi))   # h = -(2pi/((d-1)Vol_H)) dF/dbeta at d = 4
        ghjk58 = NG*(NG + 1)*(1 - a)*a*(9*(1 - a)*a - 2)*eps/(72*pi*(NG + 4))
        assert sp.simplify(h1.subs({nt: NG, N: 2*NG}) - ghjk58) == 0
        h1_closed = nt*a*(1 - a)*eps*((nt + 1)*(9*a*(1 - a) - 2) - Ns)/(36*pi*(N + 8))
        assert sp.simplify(h1 - h1_closed) == 0
        # normalization of h = (2pi/3)<T_thth> at d = 4 from the free field: sum_{Z+a}|s|^3 - sum_Z |s|^3 = -a^2(1-a)^2/2
        Tfree = sp.gamma(-sp.Rational(1, 2))/(pi*(4*pi)**sp.Rational(3, 2))*(-a**2*(1 - a)**2/2)
        assert sp.simplify(2*pi/3*Tfree - a**2*(1 - a)**2/(12*pi)) == 0
        print('B. h_1 =', sp.factor(h1_closed), ' (GHJK 5.8 at N_s = 0; spectator term -n_t N_s a(1-a) eps/(36 pi (N+8)))')

        # C -----------------------------------------------------------------------------------------------------
        # sigma propagator at large N, leading order in eps: (N/2 G0^2)^{-1} -> (2/N) * 8 pi^2 eps delta  (bubble pole 1/(8pi^2 eps))
        bubble_pole = 1/(8*pi**2*eps)              # int d^dy G0(x,y)^2, G0 = 1/(4 pi^2 x^2)
        Dsig = 1/(N/2*bubble_pole)
        # F gets +(1/2) Tr[D dSigma], dSigma(x,x) = (1/2) sum_a g_a^2 (dim reg: spectator term G0(x,x) dG_sp = 0)
        Ffluct = sp.Rational(1, 2)*Dsig*sp.Rational(1, 2)*(2*nt*dg**2)*2*pi*VolH
        Fexch = lam/24*2*(2*nt*dg**2)*2*pi*VolH
        assert sp.simplify(sp.limit((Ffluct - Fexch)*N, N, sp.oo)) == 0
        cl = sp.limit(-(pi*eps/2)*Ffluct*N, N, sp.oo)
        print('C. large-N 4-eps coefficient of F~ at O(1/N):', sp.factor(cl), ' (exchange only; strict-eps limit gives',
              sp.factor(sp.limit(Ft1*N, N, sp.oo)), ')')

        # D -----------------------------------------------------------------------------------------------------
        dgd = (d - 1)*sp.gamma(d/2 - a)*sp.gamma(d/2 - 1 + a)*sp.sin(pi*a)*sp.gamma(d/2)/(2*pi**(d/2 + 1)*sp.gamma(d)*(2 - d))
        assert sp.simplify(sp.gammasimp(dgd.subs(d, 4)) - dg) == 0
        k1 = sp.simplify(-sp.diff(sp.log(dgd), d).subs(d, 4))     # dg(4-eps) = dg (1 + k1 eps + ...)
        volH = pi**((d - 2)/2)*sp.gamma(1 - d/2)
        vol1 = sp.simplify(sp.series((volH/(-2*pi/eps)).subs(d, 4 - eps), eps, 0, 2).removeO().coeff(eps, 1))
        lam2 = 9*(3*N + 14)/(N + 8)**3     # lambda-bar* = 3eps/(N+8) + lam2 eps^2, lambda = 16 pi^2 lambda-bar (MS-bar)
        msbar = (sp.EulerGamma - sp.log(4*pi))/2   # (e^gamma/4pi)^{eps/2} in lambda_B = mu^eps (e^gamma/4pi)^{eps/2} 16pi^2 lam
        K = sp.simplify(2*k1 + vol1 + msbar)
        print('D. F~_1 = F~_1^(eps) [1 + eps (K(alpha) + 3(3N+14)/(N+8)^2)],  K = 2 k1 + vol + MSbar:')
        print('   vol term =', sp.simplify(vol1), ';  K(1/2) =', sp.N(K.subs(a, sp.Rational(1, 2)), 8),
              ';  K(1/4) =', sp.N(K.subs(a, sp.Rational(1, 4)), 8))

        # E -----------------------------------------------------------------------------------------------------
        mp.dps = 20


        def pieces(Nv, ntv, av, e=1):
            Nv, av, e = mpf(Nv), mpf(av), mpf(e)
            Nsv = Nv - 2*ntv
            F1v = mpi*ntv*(ntv + 1)*av**2*(1 - av)**2*e/(4*(Nv + 8))
            bv = e*(Nsv + 2)/(Nv + 8)
            m0 = -2*ntv*av*(1 - av)*e/(Nv + 8)
            D = mpc(bv**2 + 4*m0)
            Fsp = mpi*Nsv/96*(bv**3 + 6*bv*m0 - D**1.5) if Nsv > 0 else mpc(0)
            strict2 = -mpi/8*ntv*Nsv*(Nsv + 2)*av*(1 - av)*e**2/(Nv + 8)**2
            epsc = 8*ntv*av*(1 - av)*(Nv + 8)/(Nsv + 2)**2 if Nsv > 0 else None
            return F1v, Fsp, strict2, epsc


        print('E. eps = 1, n_t = 1.  Delta = log g_{N-1} - log g_N.  Columns: O(eps); strict eps^2 spectator piece;',
              'resummed spectator Re (Im of F~_N); total O(eps) + resummed Re; eps_c of the merger at N')
        for Nv in [3, 4, 5, 6, 10]:
            for av in ['0.5'] if Nv != 3 else ['0.1', '0.25', '0.4', '0.5']:
                A1, B1, s1, ec = pieces(Nv, 1, av)
                A0, B0, s0, _ = pieces(Nv - 1, 1, av)
                dO = A0 - A1; ds = s0 - s1; dr = (B0 - B1).real
                print(f'   N={Nv} a={av}: {mp.nstr(dO, 4)}  {mp.nstr(ds, 4)}  {mp.nstr(dr, 4)} (Im {mp.nstr(B1.imag, 3)})'
                      f'  {mp.nstr(dO + dr, 4)}  eps_c={mp.nstr(ec, 4)}')
        cpaper = lambda x: 8/(9*mpi**2) if x == mpf('0.5') else 16/(9*mpi)*x*(1 - x)*(1 - 2*x)*mtan(mpi*x)
        print('   large N, eps = 1: (pi/4)a^2(1-a)^2 [correct limit] vs (pi/2)a^2(1-a)^2 [paper] vs c(a) [3d]:')
        for av in ['0.1', '0.2', '0.25', '0.3', '0.4', '0.5']:
            x = mpf(av)
            print(f'     a={av}: {mp.nstr(mpi/4*x**2*(1-x)**2, 5)}  {mp.nstr(mpi/2*x**2*(1-x)**2, 5)}  {mp.nstr(cpaper(x), 5)}')

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('analytic/open_questions/entropy_eps2_checks.log'):
            main()
    return namespace(locals(), globals())


@section
def melon_structure():
    """Structure of the O(lambda^2) defect free energy for a partial twist (2026-10-26), for ENTROPY_EPS2.md section 6.

1. Melon tensor: -(1/48) V_abcd V_efgh G_ae G_bf G_cg G_dh = -(lam^2/144)[(Tr G G^T)^2 + 2 Tr (G G^T)^2],
   checked with random non-symmetric propagator matrices; a twisted plane has the block [[R, I], [-I, R]], so
   G G^T = |G_alpha|^2 1 and the defect part is
   F_melon = -(lam^2/36) n_t int int [ (N+2) G0^2 dP + (n_t+1) dP^2 ],  dP = |G_alpha|^2 - G0^2.
2. UV poles (units lam^2 n_t dg^2 / (36 * 8 pi^2 eps), bubble pole int G0^2 = 1/(8 pi^2 eps)):
   melon vertex parts -(N+2) - 4(n_t+1); cactus -2(n_t+1)^2 (twisted) - N_s n_t (spectators);
   counterterm (N+8)(n_t+1) from delta-lam = (N+8) lam^2/(48 pi^2 eps).  They cancel identically.
"""
    import numpy as np
    import sympy as sp


    def main():
        rng = np.random.default_rng(1)
        n = 5
        d = np.eye(n)
        V = (np.einsum('ab,cd->abcd', d, d) + np.einsum('ac,bd->abcd', d, d) + np.einsum('ad,bc->abcd', d, d))/3  # lam = 1
        for _ in range(3):
            G = rng.normal(size=(n, n))
            lhs = np.einsum('abcd,efgh,ae,bf,cg,dh->', V, V, G, G, G, G)
            GG = G @ G.T
            rhs = (3*np.trace(GG)**2 + 6*np.trace(GG @ GG))/9
            assert abs(lhs - rhs) < 1e-9*abs(rhs)
        R, I = rng.normal(), rng.normal()
        B = np.array([[R, I], [-I, R]])
        assert np.allclose(B @ B.T, (R**2 + I**2)*np.eye(2))
        print('1. melon tensor identity and twisted-plane block checked')

        N, nt = sp.symbols('N n_t')
        Ns = N - 2*nt
        poles = -(N + 2) - 4*(nt + 1) - 2*(nt + 1)**2 - Ns*nt + (N + 8)*(nt + 1)
        assert sp.expand(poles) == 0
        print('2. UV poles of melon + cactus + coupling counterterm cancel for all N, n_t')

    def cli(argv):
        with fuzzy.logged('analytic/open_questions/melon_structure.log'):
            main()
    return namespace(locals(), globals())


STEPS = {'entropy_eps': entropy_eps, 'eps2_s1_anchor': eps2_s1_anchor, 'entropy_bf_channel': entropy_bf_channel, 'eps2_s4_defect_coupling': eps2_s4_defect_coupling, 'eps2_s5_d3_epsc': eps2_s5_d3_epsc, 'eps2_r1_counterterm': eps2_r1_counterterm, 'eps2_r2_cactus_twisted': eps2_r2_cactus_twisted, 'eps2_r3_cactus_spectator': eps2_r3_cactus_spectator, 'eps2_r4a_calibration': eps2_r4a_calibration, 'eps2_r4b_melon_integrals': eps2_r4b_melon_integrals, 'eps2_r4c_melon_assembly': eps2_r4c_melon_assembly, 'eps2_r5_assembly': eps2_r5_assembly, 'entropy_tables': entropy_tables, 'entropy_eps2_checks': entropy_eps2_checks, 'melon_structure': melon_structure}
REPORT = [('eps2_s1_anchor',), ('entropy_eps2_checks',), ('eps2_r5_assembly',), ('entropy_tables',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
