"""What do the epsilon expansion and large N predict for the junction of pinned lines?

The shape factor r = Delta_j(pi)/2 C_tilt at one loop, 1 + (6 zeta(3)/pi^2) eps (eq:roneloop), and at two loops
(eq:rtwoloop; Table 4): the two-vertex tree, the one-loop determinant around the pinned profile and the classical cubic
term, assembled in minimal subtraction. Independently of every graph, the junction at large N is a classical saddle on
S^1 x H^2 (r_inf = 1.551, Delta_{+-} = 0.276 N), and its continuation to d = 4 - eps reproduces the two-loop coefficient
at N = infinity (paper, Sec. 5.1 and App. E, "The shape factor at two loops" and "The junction at large N").

The d -> 2 sigma-model wall (eq:walld2) and the Kosterlitz-Thouless anchor of Table 7 are closed forms in the text.

usage: python junction_theory.py <step> [args]      (python junction_theory.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace, figpath          # noqa: E402,F401


@section
def junction_oneloop():
    """One-loop pinned-line junction (Cuomo, Komargodski, Mezei, Raviv-Moshe, arXiv:2406.10186, as quoted in App. E):
Delta_j(eta) = (M+8)(1-cos eta)/(4 pi^2) + eps (1-cos eta)[pi^2(3M^2+29M+106) + 6(M+8)^2 (1-cos eta) zeta(3)]/(8 pi^4 (M+8)).
C_t is the coefficient of 1 - cos(eta); the wall is Delta_j(pi); r = Delta_j(pi)/(2 C_t).

arXiv:2406.10186 is Cuomo, He and Komargodski (the paper's Ref. Cuomo:2024psk); M is the number of components N.
"""
    from mpmath import mp, zeta, pi, mpf
    mp.dps = 15

    def main():
        for M in (2, 3, 4):
            c0 = (M + 8) / (4 * pi**2)
            c1 = (3 * M**2 + 29 * M + 106) / (8 * pi**2 * (M + 8))
            q1 = 6 * (M + 8)**2 * zeta(3) / (8 * pi**4 * (M + 8))      # coefficient of eps (1-cos)^2
            Ct = lambda e: c0 + c1 * e
            wall = lambda e: 2 * Ct(e) + 4 * q1 * e
            print(f'M={M}: C_t = {float(c0):.4f} + {float(c1):.4f} eps -> {float(Ct(1)):.3f} (eps=1);  '
                  f'Delta_+- = {float(2*c0):.3f} + {float(2*c1 + 4*q1):.3f} eps -> {float(wall(1)):.3f};  '
                  f'r = 1 + {float(4*q1/(2*c0)):.4f} eps (check 6 zeta3/pi^2 = {float(6*zeta(3)/pi**2):.4f})')

    def cli(argv):
        mp.dps = 15
        with fuzzy.logged('analytic/junction_oneloop.log'):
            main()
    return namespace(locals(), globals())


@section
def ddim():
    """d-dimensional (d = 4 - eps) pieces of the O(eps^2) pinned-line junction (2026-10-24).

Conventions (Cuomo-Komargodski-Mezei 2112.10634): S = int 1/2 (dphi)^2 + lam0/4! (phi^2)^2 + h0 int n.phi, minimal subtraction
lam0 = M^eps (lam + dlam/eps), h0 = M^{eps/2} (h + dh/eps), dh = lam h^3/(12 (4pi)^2), dlam = (N+8) lam^2/(3 (4pi)^2).
Junction at the origin, half-lines along the tau axis; theta = polar angle from the tau > 0 half-line.
Half-line potentials (h stripped): a = C_d rho^{3-d} f_a(th) for tau < 0 (singular at th = pi), b = C_d rho^{3-d} f_b(th) for tau > 0,
    f_a = sin^{3-d}(th) al(th), f_b = f_a(pi - th), al(th) = int_0^th sin^{d-4},  C_d = Gamma(d/2-1)/(4 pi^{d/2}).
t = 1 - cos(eta).  Junction dimension Delta = -(lam d_lam + h/2 d_h) Res, Res = simple-pole residue of log<D> (bare, in renormalized
couplings), with infrared-consistent counterterm graphs (checked: reproduces the CHK O(eps) (1-c) coefficient, B0 = pi(1 - log pi)).
Graphs (coefficients of t^2, written as A(eps) (ML)^{n eps}/(n eps) per coupling monomial):
  (b) lam h^4:   A_b = -(1/6) C_d^4 Om_{d-2} int sin^{d-2} f_P^2                                   (primitive)
  (c) lam^2 h^6: A_c = (1/72) C_d^6 Om_{d-2} [4 Q(fP fS, fP fS) + 8 Q(fa fb^2, fS^2 fa)]           (near-line pole)
      Q(L, k) = int sin^{d-2} L psi_k,  (-Lap_{S^{d-1}} - m^2) psi_k = k,  m^2 = al_c (al_c + d - 2), al_c = 11 - 3d.
"""
    from mpmath import mp, mpf, gamma, pi, sin, cos, betainc, hyp2f1, quad, diff, zeta, log, sqrt
    mp.dps = 25


    def C(d):
        return gamma(d / 2 - 1) / (4 * pi ** (d / 2))


    def Om(n):      # area of the unit S^n
        return 2 * pi ** ((n + 1) / mpf(2)) / gamma((n + 1) / mpf(2))


    def alpha_pi(d):
        return sqrt(pi) * gamma((d - 3) / 2) / gamma((d - 2) / 2)


    def al(th, d):
        th = mpf(th)
        if th <= pi / 2:
            return betainc((d - 3) / 2, mpf(1) / 2, 0, sin(th) ** 2) / 2
        return alpha_pi(d) - al(pi - th, d)


    def fa(th, d):
        return sin(th) ** (3 - d) * al(th, d)


    def fb(th, d):
        return fa(pi - th, d)


    def A_b(eps):
        d = 4 - mpf(eps)
        I = quad(lambda t: sin(t) ** (d - 2) * (fa(t, d) * fb(t, d)) ** 2, [0, pi / 2, pi])
        return -C(d) ** 4 * Om(d - 2) * I / 6


    # ---- S^{d-1} Green's function for (-Lap - m^2) on zonal functions -------------------------------------------------
    def u1(th, d):
        a = 11 - 3 * d
        return hyp2f1(-a, a + d - 2, (d - 1) / 2, sin(th / 2) ** 2)


    def u2(th, d):
        return u1(pi - th, d)


    def wronsk(d):
        """calW = sin^{d-2} (u1' u2 - u1 u2'), constant; G(th,th') = u1(th<) u2(th>)/calW for -(p u')' - m^2 p u = p k."""
        t0 = pi / 2
        du1 = diff(lambda t: u1(t, d), t0)
        du2 = diff(lambda t: u2(t, d), t0)
        return sin(t0) ** (d - 2) * (du1 * u2(t0, d) - u1(t0, d) * du2)


    def hyp_1mz(a, b, c, w):
        """2F1(a,b;c;1-w) for small w via the 1-z connection formula (w given exactly, so no cancellation in 1-w)."""
        t1 = gamma(c) * gamma(c - a - b) / (gamma(c - a) * gamma(c - b)) * hyp2f1(a, b, a + b - c + 1, w)
        t2 = w ** (c - a - b) * gamma(c) * gamma(a + b - c) / (gamma(a) * gamma(b)) * hyp2f1(c - a, c - b, c - a - b + 1, w)
        return t1 + t2


    def u1_at(th, d, near_pi=False):
        """u1(th); with near_pi=True, th is the distance w = pi - theta and the value u1(pi - w) is returned accurately."""
        a = 11 - 3 * d
        A, B, Cc = -a, a + d - 2, (d - 1) / 2
        if not near_pi:
            return hyp2f1(A, B, Cc, sin(th / 2) ** 2)
        return hyp_1mz(A, B, Cc, sin(th / 2) ** 2)

    def main():
        Ab0 = A_b(0)
        print('A_b(0) =', Ab0, ' vs -zeta(3)/(64 pi^6) =', -zeta(3) / (64 * pi ** 6))
        dAb = diff(A_b, 0)
        print("A_b'(0) =", dAb, '  ratio to A_b(0):', dAb / Ab0)

    def cli(argv):
        mp.dps = 25
        with fuzzy.logged('junction_eps2/ddim.log'):
            main()
    return namespace(locals(), globals())


@section
def classical_c0():
    """Finite part c0 of the classical two-loop (lam^2 h^6) t^2 coefficient A_c(eps) = c_{-1}/eps + c0 (see ddim.py).
"""
    from mpmath import mp, mpf, pi, sin, quad, diff, zeta, log
    D = globals()['ddim']()   # was: import ddim as D

    mp.dps = 22


    def Pc(eps):
        d = 4 - mpf(eps)
        return D.C(d) ** 6 * D.Om(d - 2) / 72


    def pairing4(Lfun, kfun):
        """Q(L,k) at d = 4: int int (s L)(th) g_D(th,th') (s k)(th'), g_D = min (pi - max)/pi."""
        def inner(x):
            return ((pi - x) / pi) * quad(lambda y: y * sin(y) * kfun(y), [0, x]) + (x / pi) * quad(lambda y: (pi - y) * sin(y) * kfun(y), [x, pi])
        return quad(lambda x: sin(x) * Lfun(x) * inner(x), [0, pi / 2, pi])


    def fa4(t): return t / sin(t)
    def fb4(t): return (pi - t) / sin(t)
    def fS4(t): return pi / sin(t)
    def fP4(t): return fa4(t) * fb4(t)


    def J(eps):
        """int_0^pi sin^{d-2} fa fb^2 u1 (pi - th)^{2 eps}, folded onto (0, pi/2) with exact symmetric forms (no sin(pi - tiny))."""
        d = 4 - mpf(eps); e = mpf(eps)
        def g(t):   # integrand at th = t (t <= pi/2) plus at th = pi - t
            s = sin(t)
            lo = s ** (d - 2) * D.fa(t, d) * D.fb(t, d) ** 2 * D.u1_at(t, d) * (pi - t) ** (2 * e)
            hi = s ** (d - 2) * D.fb(t, d) * D.fa(t, d) ** 2 * D.u1_at(t, d, near_pi=True) * t ** (2 * e)
            return lo + hi
        return quad(g, [0, pi / 4, pi / 2])


    def F0W(eps):
        d = 4 - mpf(eps)
        return D.alpha_pi(d) ** 3 * J(eps) / D.wronsk(d)

    def main():
        # Q1 (no subdivergence) at d = 4
        Q1 = pairing4(lambda t: fP4(t) * fS4(t), lambda t: fP4(t) * fS4(t))
        print('Q1(0) =', Q1)
        # Q2 regular part at d = 4: (1/pi) int s^2 L [u2 I1 + u1 R2]
        L = lambda t: fa4(t) * fb4(t) ** 2
        I1 = lambda t: pi ** 2 * quad(lambda y: y ** 2 / sin(y) ** 2, [0, t])
        R2 = lambda t: quad(lambda w: pi ** 2 * (pi - w) * w / sin(w) ** 2 - pi ** 3 / w, [0, pi - t])   # w = pi - y
        u1 = lambda t: t / sin(t)
        u2 = lambda t: (pi - t) / sin(t)
        reg = quad(lambda t: sin(t) ** 2 * L(t) * (u2(t) * I1(t) + u1(t) * R2(t)), [0, pi / 2, pi]) / pi
        print('Q2 regular part (d=4) =', reg)
        # pole piece (F0 J / (2 eps W)): residue and O(eps^0) part
        q2m1 = F0W(0) / 2
        print('q2_{-1} =', q2m1, ' vs 3 pi^3 zeta(3) =', 3 * pi ** 3 * zeta(3))
        dF0W = diff(F0W, 0)
        Q2fin = reg + dF0W / 2
        print('d/deps(F0 J/W) at 0 =', dF0W, '   Q2fin =', Q2fin)
        P0, dP = Pc(0), diff(Pc, 0)
        cm1 = 8 * P0 * q2m1
        c0 = P0 * (4 * Q1 + 8 * Q2fin) + 8 * dP * q2m1
        print('c_{-1} =', cm1, ' predicted -A_b(0)/(3(4pi)^2) =', zeta(3) / (3072 * pi ** 8))
        print('c0 =', c0)

    def cli(argv):
        mp.dps = 22
        with fuzzy.logged('junction_eps2/classical_c0.log'):
            main()
    return namespace(locals(), globals())


@section
def classical_cubic():
    """O(eps^2) junction of the pinned O(N) line, classical sector (2026-10-24).

d = 4: S = int 1/2 (dphi)^2 + (lam/4!)(phi^2)^2 + h int dtau n(tau).phi, n = e_1 (tau < 0), e(eta) (tau > 0).
Free classical field Phi0 = h (A e_1 + B e_eta), A = (pi - th)/(4 pi^2 rho sin th), B = th/(4 pi^2 rho sin th) (junction at rho = 0,
th = polar angle from the tau > 0 half-line).  Junction energy E(eta) = coefficient of log(L/a) in S_cl, with t = 1 - cos(eta).
  S1 = (lam/4!) int (Phi0^2)^2:  (1-c)^2 term -> lam h^4 zeta(3)/(64 pi^6)  [CHK check: 6(N+8) zeta(3) eps/(8 pi^4) at the fixed point]
  S2 = -(lam^2/72) int int (Phi0^2 Phi0)_a G (Phi0^2 Phi0)_a: radial scaling + zonal S^3 harmonics reduce G* to the Dirichlet
       Green's function g(th,th') = min(th,th')(pi - max)/pi of -d^2/dth^2 acting on sin(th) * source.
  The (1-c)^3 term arises only here at O(eps^2):  E2 = (N+8) <ua/s^2, ub/s^2> eps^2 / (4 pi^7),
       u = th (pi - th), a = pi - th, b = th, <X,Y> = int int X(th) g(th,th') Y(th').
"""
    import numpy as np
    from mpmath import mp, quad, sin, pi, zeta, mpf
    mp.dps = 30


    def gD(x, y):
        return min(x, y) * (pi - max(x, y)) / pi


    def X(th):    # u a / s^2
        return th * (pi - th) ** 2 / sin(th) ** 2


    def Y(th):    # u b / s^2
        return th ** 2 * (pi - th) / sin(th) ** 2


    def pairing(f, g):
        # <f, g> = int_0^pi f(x) [ int_0^pi g_D(x,y) g(y) dy ] dx, with the inner integral split at y = x
        def inner(x):
            return ((pi - x) / pi) * quad(lambda y: y * g(y), [0, x]) + (x / pi) * quad(lambda y: (pi - y) * g(y), [x, pi])
        return quad(lambda x: f(x) * inner(x), [0, pi / 2, pi])

    def main():
        # check of the method: S1 (1-c)^2 angular integral int psi^2 (pi-psi)^2 / sin^2 = 6 pi zeta(3)
        I1 = quad(lambda t: t ** 2 * (pi - t) ** 2 / sin(t) ** 2, [0, pi])
        print('int u^2/s^2 =', I1, ' 6 pi zeta(3) =', 6 * pi * zeta(3))
        P = pairing(X, Y)
        print('<ua/s^2, ub/s^2> =', P)
        print('E2 / ((N+8) eps^2) = <>/(4 pi^7) =', P / (4 * pi ** 7))
        print('contribution to r at O(eps^2): 4 E2 / C0 = 4 <> / pi^5 =', 4 * P / pi ** 5)

    def cli(argv):
        mp.dps = 30
        with fuzzy.logged('junction_eps2/classical_cubic.log'):
            main()
    return namespace(locals(), globals())


@section
def loop_d0():
    """Finite part d0 of the one-loop determinant graph (lam^2 h^4) t^2 coefficient, A_d(eps) = d_{-1}/eps + d0.
W_d^{t^2} = (lam0^2 h0^4/36) t^2 int int G^2 [(N+6) P P' + 2 a^2 b'^2],  G^2 = C_d^2 |x-y|^{4-2d}.
Funk-Hecke on homogeneous functions (degree -p, p = 2(d-3)):  kappa_l = K(eps) rho_l(eps),
  K = pi^{d/2} Gamma(eps/2)/Gamma(2-eps),  rho_l = G((l+6-d)/2) G((l+3d-10)/2) / (G((l+2d-6)/2) G((l+10-2d)/2)).
R(f,g) = Om_{d-2} sum_l kappa_l f_l g_l N_l = K [Lam - eps Sig_psi] + O(eps^2) with Lam = Om int sin^{d-2} f g (exact in d) and
  Sig_psi = Om_2 sum psi((l+2)/2) f_l g_l (pi/2) at d = 4 (Chebyshev-U coefficients).
A_d = (C_d^6/36)[(N+6) R(fP,fP) + 2 R(fa^2, fb^2)];  R(fa^2,fb^2) - R(fP,fP) = DeltaR (position space, d = 4, finite).
"""
    from mpmath import mp, mpf, pi, sin, cos, quad, diff, zeta, psi, gamma, log, euler, nsum, inf
    D = globals()['ddim']()   # was: import ddim as D

    mp.dps = 20


    def rho_l(l, eps):
        d = 4 - mpf(eps)
        return gamma((l + 6 - d) / 2) * gamma((l + 3 * d - 10) / 2) / (gamma((l + 2 * d - 6) / 2) * gamma((l + 10 - 2 * d) / 2))


    def K(eps):
        d = 4 - mpf(eps)
        return pi ** (d / 2) * gamma(mpf(eps) / 2) / gamma(2 - mpf(eps))


    def b_l(l):
        """Chebyshev-U coefficient of f_P at d = 4:  sin(th) f_P = th (pi - th)/sin(th) = sum_l b_l sin((l+1) th)."""
        if l % 2:
            return mpf(0)
        return 2 / pi * quad(lambda t: t * (pi - t) / sin(t) * sin((l + 1) * t), [0, pi / 2, pi])


    def Lam(eps):     # Om_{d-2} int sin^{d-2} fP^2 = -6 A_b / C_d^4
        d = 4 - mpf(eps)
        return -6 * D.A_b(eps) / D.C(d) ** 4


    def I_rho(a_, b_):
        """int_0^inf 2 rho drho / ((rho^2 - 2 rho cos a + 1)(rho^2 - 2 rho cos b + 1)), a, b in (0, pi) after reflection."""
        def F(x):
            if x == 0:
                return mp.inf
            s = sin(x)
            return mpf(1) if s == 0 else (pi - x) / s
        if a_ == 0:
            return mpf(0)      # coincident point: the integrand bracket vanishes there
        if abs(a_ - b_) < mpf(10) ** (-8) * (a_ + b_):          # degenerate cos a = cos b: limit -F'(a)/sin(a)
            return -diff(F, a_) / sin(a_)
        return (F(a_) - F(b_)) / (2 * sin((a_ + b_) / 2) * sin((b_ - a_) / 2))   # cos a - cos b without cancellation


    def refl(x):
        x = abs(x)
        return 2 * pi - x if x > pi else x


    def DeltaR():
        fa = lambda t: t / sin(t); fb = lambda t: (pi - t) / sin(t)
        def inner(tx):
            g = lambda ty: sin(ty) ** 2 * (fa(tx) ** 2 * fb(ty) ** 2 - fa(tx) * fb(tx) * fa(ty) * fb(ty)) * I_rho(refl(tx - ty), refl(tx + ty))
            return quad(g, [0, tx, pi])
        return 4 * pi * 2 * pi * quad(lambda tx: sin(tx) ** 2 * inner(tx), [0, pi / 2, pi])

    def main():
        print('check rho_l expansion, l=4: rho(1e-6) =', rho_l(4, mpf('1e-6')), ' 1 - eps psi(3) =', 1 - mpf('1e-6') * psi(0, 3))
        K1 = (K(mpf('1e-8')) * mpf('1e-8') / (2 * pi ** 2) - 1) / mpf('1e-8')
        print('K1 numeric =', K1, ' analytic 1 - 3gamma/2 - log(pi)/2 =', 1 - 1.5 * euler - log(pi) / 2)
        # Sigma_psi^{PP}: b_l ~ 4/(l+1); sum over even l with tail estimate
        import sys
        if len(sys.argv) > 1:
            SigPP = mpf(sys.argv[1])
        else:
            Lmax = 400
            S = mpf(0)
            for l in range(0, Lmax + 1, 2):
                S += psi(0, mpf(l + 2) / 2) * b_l(l) ** 2
            tail = nsum(lambda j: psi(0, mpf(2 * j + 2) / 2) * (mpf(4) / (2 * j + 1)) ** 2, [Lmax // 2 + 1, inf])
            SigPP = 4 * pi * (pi / 2) * (S + tail)
        print('Sig_psi^PP =', SigPP, '  (b_400 vs 4/401:', b_l(400), mpf(4) / 401, ')')
        dR = DeltaR()
        print('DeltaR =', dR)
        L0, dL = Lam(0), diff(Lam, 0)
        C4 = D.C(mpf(4)); dC6 = diff(lambda e: D.C(4 - mpf(e)) ** 6, 0)
        print('Lam(0) =', L0, ' = 4pi*6pi*zeta3 =', 24 * pi ** 2 * zeta(3), "  Lam'(0) =", dL)
        pref = 2 * pi ** 2 / 36
        dm1 = lambda N: pref * C4 ** 6 * (N + 8) * L0
        d0 = lambda N: pref * ((N + 8) * (dC6 * L0 + C4 ** 6 * (dL + K1 * L0)) - C4 ** 6 * ((N + 8) * SigPP - dR / pi ** 2))
        for N in (1, 2, 3, 4):
            print(f'N={N}: d_-1 = {dm1(N)}  (pred {(N + 8) * zeta(3) / (3072 * pi ** 8)})   d0 = {d0(N)}')

    def cli(argv):
        sys.argv = list(argv)
        mp.dps = 20
        with fuzzy.logged('junction_eps2/loop_d0{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def assemble():
    """O(eps^2) shape factor r and wall Delta_+- of the pinned O(N) line (2026-10-24).
Delta_j(eta) = C t + D t^2 + E t^3, t = 1 - cos(eta).  Residue method, CKM minimal subtraction:
  D = -A_b(0) lam h^4 - lam^2 h^6 [c0 + 5 A_b'/(9 (4pi)^2)] - lam^2 h^4 [d0(N) + 4 (N+8) A_b'/(9 (4pi)^2)]
  E = (N+8) <ua/s^2, ub/s^2> eps^2/(4 pi^7)    (classical_cubic.py)
Fixed point: lam*/(4pi)^2 = 3eps/(N+8) + 9(3N+14)eps^2/(N+8)^3,  h*^2 = N+8 + eps (4N^2+45N+170)/(2N+16)   [CKM (3.6), (3.18)]
C = (N+8)/(4pi^2) + eps (3N^2+29N+106)/(8 pi^2 (N+8))   [CHK; reproduced here, B0 = pi(1 - log pi)].
"""
    from mpmath import mp, mpf, pi, zeta, diff
    D = globals()['ddim']()   # was: import ddim as D
    Cc = globals()['classical_c0']()   # was: import classical_c0 as Cc
    Ld = globals()['loop_d0']()   # was: import loop_d0 as Ld

    mp.dps = 20
    SigPP = mpf('-43.220631393624571759')
    DeltaR = mpf('6325.6873440470512868')
    P_cubic = mpf('103.199187953125183289367348604')


    def pieces():
        Ab0 = D.A_b(0); dAb = diff(D.A_b, 0)
        # c0 (classical)
        Q1 = Cc.pairing4(lambda t: Cc.fP4(t) * Cc.fS4(t), lambda t: Cc.fP4(t) * Cc.fS4(t))
        from mpmath import quad, sin
        Lf = lambda t: Cc.fa4(t) * Cc.fb4(t) ** 2
        I1 = lambda t: pi ** 2 * quad(lambda y: y ** 2 / sin(y) ** 2, [0, t])
        R2 = lambda t: quad(lambda w: pi ** 2 * (pi - w) * w / sin(w) ** 2 - pi ** 3 / w, [0, pi - t])
        reg = quad(lambda t: sin(t) ** 2 * Lf(t) * ((pi - t) / sin(t) * I1(t) + t / sin(t) * R2(t)), [0, pi / 2, pi]) / pi
        q2m1 = Cc.F0W(0) / 2; dF0W = diff(Cc.F0W, 0)
        P0, dP = Cc.Pc(0), diff(Cc.Pc, 0)
        c0 = P0 * (4 * Q1 + 8 * (reg + dF0W / 2)) + 8 * dP * q2m1
        # d0 (loop) pieces
        K1 = 1 - mpf(3) / 2 * mp.euler - mp.log(pi) / 2
        L0, dL = Ld.Lam(0), diff(Ld.Lam, 0)
        C4 = D.C(mpf(4)); dC6 = diff(lambda e: D.C(4 - mpf(e)) ** 6, 0)
        def d0(N):
            return 2 * pi ** 2 / 36 * ((N + 8) * (dC6 * L0 + C4 ** 6 * (dL + K1 * L0)) - C4 ** 6 * ((N + 8) * SigPP - DeltaR / pi ** 2))
        return Ab0, dAb, c0, d0


    def r_series(N, Ab0, dAb, c0, d0):
        N = mpf(N); g = 16 * pi ** 2
        lam1, lam2 = g * 3 / (N + 8), g * 9 * (3 * N + 14) / (N + 8) ** 3
        h0sq, h1 = N + 8, (4 * N ** 2 + 45 * N + 170) / (2 * N + 16)
        C0, C1 = (N + 8) / (4 * pi ** 2), (3 * N ** 2 + 29 * N + 106) / (8 * pi ** 2 * (N + 8))
        D1 = -Ab0 * lam1 * h0sq ** 2
        D2 = (-Ab0 * (lam2 * h0sq ** 2 + 2 * lam1 * h0sq * h1)
              - lam1 ** 2 * h0sq ** 3 * (c0 + 5 * dAb / (9 * g))
              - lam1 ** 2 * h0sq ** 2 * (d0(N) + 4 * (N + 8) * dAb / (9 * g)))
        E2 = (N + 8) * P_cubic / (4 * pi ** 7)
        r1 = 2 * D1 / C0
        r2 = 2 * D2 / C0 - 2 * D1 * C1 / C0 ** 2 + 4 * E2 / C0
        # wall Delta_+- = 2C + 4D + 8E
        w0, w1 = 2 * C0, 2 * C1 + 4 * D1
        w2 = 4 * D2 + 8 * E2           # + 2 C2 (two-loop stiffness, not computed)
        return r1, r2, D1, D2, E2, w0, w1, w2

    def main():
        Ab0, dAb, c0, d0 = pieces()
        print('A_b(0) =', Ab0, " A_b' =", dAb, ' c0 =', c0)
        print('6 zeta3/pi^2 =', 6 * zeta(3) / pi ** 2)
        for N in (1, 2, 3, 4, 10, 100, 10 ** 6):
            r1, r2, D1, D2, E2, w0, w1, w2 = r_series(N, Ab0, dAb, c0, d0)
            r1, r2, D2, E2, w0, w1, w2 = [float(x) for x in (r1, r2, D2, E2, w0, w1, w2)]
            print(f'N={N}: r = 1 + {r1:.6f} eps + {r2:+.6f} eps^2   [2D2/C0 {2*D2/(w0/2):+.5f}, 4E2/C0 {4*E2/(w0/2):+.5f}]'
                  f'   Delta_+- = {w0:.5f} + {w1:.5f} eps + ({w2:+.5f} + 2 C2) eps^2')

    def cli(argv):
        mp.dps = 20
        with fuzzy.logged('junction_eps2/assemble.log'):
            main()
    return namespace(locals(), globals())


@section
def junction_largeM():
    """Internal-angle junction of the pinned O(M) line at large M (2026-10-24).

S^1 x H^2 frame, H^2 = {(t, phi)}, ds^2 = (dt^2 + dphi^2)/sin^2 phi, with the junction at the boundary point phi = 0 = pi
"joined" (t = log of the distance to the junction).  Straight pinned line: sigma = 1/4, phi_a = sqrt(M) v0 e_1 (constant),
transverse modes Dh_s = 1/2 + sqrt(s^2 + 1/4) (tilt at 1, s = +-1 at the golden ratio), and the gap equation fixes
    v0^2 = (1/4 pi^2) sum_s [psi(1/2 + sqrt(s^2 + 1/4)) - psi(1/2 + |s|)].
Junction: pins e_1 on phi = pi and e(eta) on phi = 0.  phi_a/sqrt(M) = A(phi) e_1 + B(phi) e(eta), with
    -A'' + (sigma - 1/4) A / sin^2 phi = 0,  A(0) = 0, A(pi) = v0;   B likewise with B(0) = v0, B(pi) = 0,
and the gap equation  dG_sigma(phi) := G_sigma(x,x) - G_{1/4}(x,x) = v0^2 - V(phi)^2,  V^2 = A^2 + B^2 + 2 cos(eta) A B,
    G(x,x) = (1/2 pi) sum_s int dw/(2 pi) g_{s,w}(phi,phi),  [-d^2 + w^2 + (s^2 - 1/4 + sigma)/sin^2 phi] g = delta.
Hellmann-Feynman: dF/deta = -2 pi v0 sin(eta) B'(pi), and with sigma = 1/4 (A, B linear) F = 2 v0^2 (1 - cos eta), so
    r_inf = Delta(pi)/(2 C_t) = (1/2) int_0^pi sin(eta) b(eta) deta,   b = -pi B'(pi)/v0  (= 1 without relaxation).
"""
    import numpy as np
    from scipy.special import digamma
    pi = np.pi


    def v0sq(S=200000):
        s = np.arange(1, S + 1, dtype=float)
        terms = digamma(0.5 + np.sqrt(s ** 2 + 0.25)) - digamma(0.5 + s)
        tail = 1 / (8 * S)                                  # terms ~ 1/(8 s^2)
        return (2 * np.log(2) + 2 * (terms.sum() + tail)) / (4 * pi ** 2)


    class Grid:
        def __init__(self, N):
            self.N = N
            self.h = pi / (N + 1)
            self.phi = self.h * np.arange(1, N + 1)
            self.s2 = np.sin(self.phi) ** 2


    def tridiag_diag_inv(d, off):
        """Diagonal of the inverse of the symmetric tridiagonal matrix (diag d, off-diagonal off), batched over leading axes."""
        N = d.shape[-1]
        # forward/backward continuants (Usmani)
        th = np.empty_like(d); ph = np.empty_like(d)
        th[..., 0] = d[..., 0]
        th1 = d[..., 0] * d[..., 1] - off ** 2
        # use ratios for stability: L-U factorization from both ends
        l = np.empty_like(d); u = np.empty_like(d)
        l[..., 0] = d[..., 0]
        for i in range(1, N):
            l[..., i] = d[..., i] - off ** 2 / l[..., i - 1]
        u[..., N - 1] = d[..., N - 1]
        for i in range(N - 2, -1, -1):
            u[..., i] = d[..., i] - off ** 2 / u[..., i + 1]
        return 1 / (l + u - d)


    def omega_nodes(n=48, c=2.0):
        x, w = np.polynomial.legendre.leggauss(n)
        x = 0.5 * (x + 1); w = 0.5 * w
        om = c * x / (1 - x)
        return om, w * c / (1 - x) ** 2


    def dG(grid, dsig, S=40, nw=48, c=2.0):
        """G_sigma - G_{1/4} at coincident points on the grid, sigma = 1/4 + dsig (FD, |s| <= S, w-quadrature) + local tail."""
        h, s2 = grid.h, grid.s2
        om, wq = omega_nodes(nw, c)
        out = np.zeros(grid.N)
        off = -1 / h ** 2
        for s in range(0, S + 1):
            mult = 1 if s == 0 else 2
            base = 2 / h ** 2 + (s ** 2) / s2                   # sigma = 1/4: (s^2 - 1/4 + 1/4)/sin^2
            d0 = base[None, :] + om[:, None] ** 2
            d1 = d0 + dsig[None, :] / s2[None, :]
            diff = tridiag_diag_inv(d1, off) - tridiag_diag_inv(d0, off)
            out += mult * (wq[:, None] * diff).sum(0) / h / pi / (2 * pi)      # (L_h^{-1})_ii / h = g(phi_i, phi_i)
        # local large-momentum tail: dg ~ -dsig/(4 sin^2 k^3), summed over s > S (all w): (1/2pi) 2 sum_{s>S} int dw/pi ...
        k_tail = 0.0
        sS = np.arange(S + 1, S + 4000)[:, None]
        # int_0^inf dw/pi (w^2 + a^2)^(-3/2) = 1/(pi a^2), a = s/sin phi
        k_tail = (2 * (1 / (pi * (sS ** 2 / s2[None, :]))).sum(0)) / (2 * pi)
        out += -dsig / (4 * s2) * k_tail
        return out


    def lap(grid):
        N, h = grid.N, grid.h
        return (np.diag(np.full(N, 2 / h ** 2)) + np.diag(np.full(N - 1, -1 / h ** 2), 1) + np.diag(np.full(N - 1, -1 / h ** 2), -1))


    def solve_V(grid, dsig, v0):
        """A, B on the grid (Dirichlet data A(0)=0, A(pi)=v0; B(0)=v0, B(pi)=0) for -V'' + dsig V/sin^2 = 0."""
        L = lap(grid) + np.diag(dsig / grid.s2)
        rA = np.zeros(grid.N); rA[-1] = v0 / grid.h ** 2
        rB = np.zeros(grid.N); rB[0] = v0 / grid.h ** 2
        return np.linalg.solve(L, rA), np.linalg.solve(L, rB), L


    def jac_G(grid, S=40, nw=48, c=2.0):
        """d dG_i / d sigma_j at sigma = 1/4 (FD Green's functions, |s| <= S) plus the local tail on the diagonal."""
        h, s2, N = grid.h, grid.s2, grid.N
        om, wq = omega_nodes(nw, c)
        J = np.zeros((N, N))
        L0 = lap(grid)
        for s in range(0, S + 1):
            mult = 1 if s == 0 else 2
            for w, q in zip(om, wq):
                Ginv = np.linalg.inv(L0 + np.diag(s ** 2 / s2 + w ** 2))
                J -= mult * q / pi / (2 * pi) * (Ginv ** 2) / h / s2[None, :]
        sS = np.arange(S + 1, S + 4000)[:, None]
        k_tail = (2 * (1 / (pi * (sS ** 2 / s2[None, :]))).sum(0)) / (2 * pi)
        J -= np.diag(k_tail / (4 * s2))
        return J


    def solve_junction(grid, eta, v0, JG, S=40, nw=48, tol=1e-11, maxit=60, verbose=False):
        dsig = np.zeros(grid.N)
        for it in range(maxit):
            A, B, L = solve_V(grid, dsig, v0)
            V2 = A ** 2 + B ** 2 + 2 * np.cos(eta) * A * B
            R = dG(grid, dsig, S, nw) - v0 ** 2 + V2
            if verbose:
                print(f'   it {it}: max|R| {np.max(abs(R)):.2e}  min dsig {dsig.min():+.4f}', flush=True)
            if np.max(abs(R)) < tol:
                break
            Linv = np.linalg.inv(L)
            dA = -Linv * (A / grid.s2)[None, :]
            dB = -Linv * (B / grid.s2)[None, :]
            JV = 2 * (A + np.cos(eta) * B)[:, None] * dA + 2 * (B + np.cos(eta) * A)[:, None] * dB
            dsig = dsig - np.linalg.solve(JG + JV, R)
        Bp = (-4 * B[-1] + B[-2]) / (2 * grid.h)           # B(pi) = 0, second-order one-sided slope
        return dict(dsig=dsig, A=A, B=B, b=-pi * Bp / v0, it=it, res=float(np.max(abs(R))))


    def r_inf(N=200, S=40, nw=48, neta=10, verbose=True):
        grid = Grid(N)
        v0 = np.sqrt(v0sq())
        JG = jac_G(grid, S, nw)
        x, w = np.polynomial.legendre.leggauss(neta)
        etas = pi / 2 * (x + 1); we = pi / 2 * w
        bs = []
        for eta in etas:
            r = solve_junction(grid, eta, v0, JG, S, nw)
            bs.append(r['b'])
            if verbose:
                print(f'  eta/pi={eta / pi:.4f}: b = {r["b"]:.6f}  min dsig {r["dsig"].min():+.5f}  iters {r["it"]}  res {r["res"]:.1e}', flush=True)
        bs = np.array(bs)
        rr = 0.5 * np.sum(we * np.sin(etas) * bs)
        # free-shape check: with b = 1 the quadrature gives exactly 1
        return rr, etas, bs


    def jac_G_at(grid, dsig, S=40, nw=48, c=2.0):
        """d dG_i / d sigma_j at sigma = 1/4 + dsig (full Newton)."""
        h, s2, N = grid.h, grid.s2, grid.N
        om, wq = omega_nodes(nw, c)
        J = np.zeros((N, N))
        L0 = lap(grid) + np.diag(dsig / s2)
        for s in range(0, S + 1):
            mult = 1 if s == 0 else 2
            for w, q in zip(om, wq):
                Ginv = np.linalg.inv(L0 + np.diag(s ** 2 / s2 + w ** 2))
                J -= mult * q / pi / (2 * pi) * (Ginv ** 2) / h / s2[None, :]
        sS = np.arange(S + 1, S + 4000)[:, None]
        k_tail = (2 * (1 / (pi * (sS ** 2 / s2[None, :]))).sum(0)) / (2 * pi)
        J -= np.diag(k_tail / (4 * s2))
        return J


    def residual(grid, dsig, eta, v0, S, nw):
        A, B, L = solve_V(grid, dsig, v0)
        V2 = A ** 2 + B ** 2 + 2 * np.cos(eta) * A * B
        return dG(grid, dsig, S, nw) - v0 ** 2 + V2, A, B, L


    def solve_junction_newton(grid, eta, v0, S=40, nw=48, tol=1e-11, maxit=40, dsig0=None, verbose=False):
        dsig = np.zeros(grid.N) if dsig0 is None else dsig0.copy()
        R, A, B, L = residual(grid, dsig, eta, v0, S, nw)
        for it in range(maxit):
            nr = np.max(abs(R))
            if verbose:
                print(f'   it {it}: max|R| {nr:.2e}  min sigma {0.25 + dsig.min():+.5f}', flush=True)
            if nr < tol:
                break
            JG = jac_G_at(grid, dsig, S, nw)
            Linv = np.linalg.inv(L)
            dA = -Linv * (A / grid.s2)[None, :]
            dB = -Linv * (B / grid.s2)[None, :]
            JV = 2 * (A + np.cos(eta) * B)[:, None] * dA + 2 * (B + np.cos(eta) * A)[:, None] * dB
            step = -np.linalg.solve(JG + JV, R)
            lam = 1.0
            while lam > 1e-4:
                trial = dsig + lam * step
                Rt, At, Bt, Lt = residual(grid, trial, eta, v0, S, nw)
                if np.all(np.isfinite(Rt)) and np.max(abs(Rt)) < (1 - 0.3 * lam) * nr:
                    break
                lam /= 2
            dsig, R, A, B, L = trial, Rt, At, Bt, Lt
        Bp = (-4 * B[-1] + B[-2]) / (2 * grid.h)
        return dict(dsig=dsig, A=A, B=B, b=-pi * Bp / v0, it=it, res=float(np.max(abs(R))))

    def main():
        print(f'v0^2 = {v0sq():.10f}')

    def cli(argv):
        with fuzzy.logged('largeN/junction_largeM.log'):
            main()
    return namespace(locals(), globals())


@section
def junction_largeM_run():
    """r_inf from the self-consistent large-M junction (junction_largeM.py): Gauss-Legendre in eta with continuation.
"""
    import sys
    import json
    import time
    import numpy as np
    _junction_largeM = globals()['junction_largeM'](); Grid, v0sq, solve_junction_newton = _junction_largeM.Grid, _junction_largeM.v0sq, _junction_largeM.solve_junction_newton   # was: from junction_largeM import Grid, v0sq, solve_junction_newton

    pi = np.pi


    def run(N, S, nw, neta=12):
        grid = Grid(N); v0 = np.sqrt(v0sq())
        x, w = np.polynomial.legendre.leggauss(neta)
        etas = pi / 2 * (x + 1); we = pi / 2 * w
        bs, smin, prev = [], [], None
        t0 = time.time()
        for eta in etas:
            r = solve_junction_newton(grid, eta, v0, S, nw, dsig0=prev)
            prev = r['dsig']
            assert r['res'] < 1e-9, (eta, r['res'])
            bs.append(r['b']); smin.append(0.25 + r['dsig'].min())
        rinf = 0.5 * np.sum(we * np.sin(etas) * np.array(bs))
        rp = solve_junction_newton(grid, pi, v0, S, nw, dsig0=prev)
        out = dict(N=N, S=S, nw=nw, neta=neta, r_inf=float(rinf), b_pi=float(rp['b']), sigma_min_pi=float(0.25 + rp['dsig'].min()),
                   etas=etas.tolist(), b=[float(b) for b in bs], seconds=time.time() - t0)
        print(f'N={N} S={S} nw={nw} neta={neta}: r_inf = {rinf:.6f}   b(pi) = {rp["b"]:.5f}   sigma_min(pi) = {out["sigma_min_pi"]:+.5f}  [{out["seconds"]:.0f}s]', flush=True)
        return out

    def main():
        res = [run(*cfg) for cfg in [(120, 30, 40, 12), (160, 40, 48, 12), (120, 30, 40, 16), (240, 50, 56, 12)]]
        store.put('largeN/junction_largeM_results.json', res)

    def cli(argv):
        with fuzzy.logged('largeN/junction_largeM_run.log'):
            main()
    return namespace(locals(), globals())


@section
def junction_largeM_d():
    """Large-N junction of the pinned O(N) line in general dimension d (2026-10-24): an independent check of the O(eps^2) shape factor.

Frame S^{d-2} x H^2 (H^2 angle phi), straight line: sigma = s0 = (4-d)(d-2)/4 (Cuomo-Komargodski-Mezei), constant phi = sqrt(N) v0 e_1.
Transverse mode l on S^{d-2}: weight w_l = deg(l)/Vol(S^{d-2}), deg(l) = (2l+d-3) Gamma(l+d-3)/(Gamma(d-2) l!), and potential
l(l+d-3)/sin^2 phi (conformal coupling (d-2)(d-4)/4 cancels s0, so the classical field is linear at sigma = s0, as in d = 3).
Line dimensions Dh_l = 1/2 + sqrt((l+(d-3)/2)^2 + s0) (tilt at 1).  Gap: v0^2 = (1/2pi) sum_l w_l [psi(Dh_l) - psi(l + (d-2)/2)].
Junction exactly as in junction_largeM.py; r_inf(d) = (1/2) int sin(eta) b(eta) d eta, b = -pi B'(pi)/v0.
"""
    import numpy as np
    from scipy.special import digamma, gammaln
    _junction_largeM = globals()['junction_largeM'](); Grid, tridiag_diag_inv, omega_nodes, lap, solve_V = _junction_largeM.Grid, _junction_largeM.tridiag_diag_inv, _junction_largeM.omega_nodes, _junction_largeM.lap, _junction_largeM.solve_V   # was: from junction_largeM import Grid, tridiag_diag_inv, omega_nodes, lap, solve_V

    pi = np.pi


    def weights(d, L):
        l = np.arange(L + 1, dtype=float)
        vol = 2 * pi ** ((d - 1) / 2) / np.exp(gammaln((d - 1) / 2))
        if abs(d - 3) < 1e-12:
            deg = np.where(l == 0, 1.0, 2.0)
        else:
            deg = (2 * l + d - 3) * np.exp(gammaln(l + d - 3) - gammaln(d - 2) - gammaln(l + 1))
        return deg / vol


    def s0(d):
        return (4 - d) * (d - 2) / 4


    def v0sq(d, L=4000000):
        l = np.arange(L + 1, dtype=float)
        w = weights(d, L)
        terms = digamma(0.5 + np.sqrt((l + (d - 3) / 2) ** 2 + s0(d))) - digamma(l + (d - 2) / 2)
        # tail: terms ~ s0/(2 (l+(d-3)/2)^2) * ... ~ s0/(2 l^2); w_l ~ 2 l^{d-3}/(Gamma(d-2) vol)
        vol = 2 * pi ** ((d - 1) / 2) / np.exp(gammaln((d - 1) / 2))
        A = 2 / (np.exp(gammaln(d - 2)) * vol) * s0(d) / 2
        tail = A * L ** (d - 4) / (4 - d) if d < 4 else 0.0
        return (np.sum(w * terms) + tail) / (2 * pi)


    def dG_d(grid, dsig, d, L=60, nw=48, c=2.0):
        h, s2 = grid.h, grid.s2
        om, wq = omega_nodes(nw, c)
        w = weights(d, L)
        out = np.zeros(grid.N)
        off = -1 / h ** 2
        for l in range(L + 1):
            base = 2 / h ** 2 + l * (l + d - 3) / s2
            d0 = base[None, :] + om[:, None] ** 2
            d1 = d0 + dsig[None, :] / s2[None, :]
            diff = tridiag_diag_inv(d1, off) - tridiag_diag_inv(d0, off)
            out += w[l] * (wq[:, None] * diff).sum(0) / h / pi
        # local tail l > L: dg ~ -dsig/(4 sin^2 k^3), int dw/pi (w^2 + a^2)^{-3/2} = 1/(pi a^2), a^2 = l(l+d-3)/sin^2
        lt = np.arange(L + 1, L + 200001, dtype=float)
        wt = weights(d, L + 200000)[L + 1:]
        S = np.sum(wt / (lt * (lt + d - 3)))
        vol = 2 * pi ** ((d - 1) / 2) / np.exp(gammaln((d - 1) / 2))
        Lf = L + 200000
        S += 2 / (np.exp(gammaln(d - 2)) * vol) * Lf ** (d - 4) / (4 - d) if d < 4 else 0.0
        out += -dsig / (4 * pi) * S
        return out


    def jac_G_d(grid, dsig, d, L=60, nw=48, c=2.0):
        h, s2, N = grid.h, grid.s2, grid.N
        om, wq = omega_nodes(nw, c)
        w = weights(d, L)
        J = np.zeros((N, N))
        L0 = lap(grid) + np.diag(dsig / s2)
        for l in range(L + 1):
            for ww, q in zip(om, wq):
                Ginv = np.linalg.inv(L0 + np.diag(l * (l + d - 3) / s2 + ww ** 2))
                J -= w[l] * q / pi * (Ginv ** 2) / h / s2[None, :]
        lt = np.arange(L + 1, L + 200001, dtype=float)
        wt = weights(d, L + 200000)[L + 1:]
        S = np.sum(wt / (lt * (lt + d - 3)))
        vol = 2 * pi ** ((d - 1) / 2) / np.exp(gammaln((d - 1) / 2))
        Lf = L + 200000
        S += 2 / (np.exp(gammaln(d - 2)) * vol) * Lf ** (d - 4) / (4 - d) if d < 4 else 0.0
        J -= np.diag(np.full(N, S / (4 * pi)))
        return J


    def solve(grid, eta, v0, d, L, nw, dsig0=None, tol=1e-12, maxit=40):
        dsig = np.zeros(grid.N) if dsig0 is None else dsig0.copy()
        def res(ds):
            A, B, Lm = solve_V(grid, ds, v0)
            return dG_d(grid, ds, d, L, nw) - v0 ** 2 + A ** 2 + B ** 2 + 2 * np.cos(eta) * A * B, A, B, Lm
        R, A, B, Lm = res(dsig)
        for it in range(maxit):
            nr = np.max(abs(R))
            if nr < tol:
                break
            JG = jac_G_d(grid, dsig, d, L, nw)
            Li = np.linalg.inv(Lm)
            dA = -Li * (A / grid.s2)[None, :]; dB = -Li * (B / grid.s2)[None, :]
            JV = 2 * (A + np.cos(eta) * B)[:, None] * dA + 2 * (B + np.cos(eta) * A)[:, None] * dB
            step = -np.linalg.solve(JG + JV, R)
            lam = 1.0
            while lam > 1e-4:
                tr = dsig + lam * step
                Rt, At, Bt, Lt = res(tr)
                if np.all(np.isfinite(Rt)) and np.max(abs(Rt)) < (1 - 0.3 * lam) * nr:
                    break
                lam /= 2
            dsig, R, A, B, Lm = tr, Rt, At, Bt, Lt
        Bp = (-4 * B[-1] + B[-2]) / (2 * grid.h)
        return dict(dsig=dsig, b=-pi * Bp / v0, res=float(np.max(abs(R))), it=it)


    def r_inf(d, N=120, L=60, nw=40, neta=12, full=False):
        grid = Grid(N); v0 = np.sqrt(v0sq(d))
        x, wgt = np.polynomial.legendre.leggauss(neta)
        etas = pi / 2 * (x + 1); we = pi / 2 * wgt
        bs, prev = [], None
        for eta in etas:
            r = solve(grid, eta, v0, d, L, nw, dsig0=prev)
            assert r['res'] < 1e-9, (d, eta, r['res'])
            prev = r['dsig']; bs.append(r['b'])
        rr = 0.5 * np.sum(we * np.sin(etas) * np.array(bs))
        return (rr, v0 ** 2, etas, np.array(bs)) if full else (rr, v0 ** 2)

    def main():
        import sys
        for d in [float(x) for x in sys.argv[1:]] or [3.0]:
            r, v2 = r_inf(d)
            eps = 4 - d
            print(f'd={d}: v0^2 = {v2:.6f}  r_inf = {r:.6f}   eps-series 1 + 0.730763 eps - 0.679500 eps^2 = {1 + 0.730763 * eps - 0.6795 * eps ** 2:.6f}', flush=True)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('largeN/junction_largeM_d{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def junction_largeM_d_scan():
    """Converged large-N r_inf(d): FD grids N = 240, 360, 480 at mode cutoff L = 12 (local WKB tail above), extrapolated in h^2.
(The first scan used N = 120, L = 60: modes with l h > 1 are unresolved on the grid and shift r by ~3e-4 at d = 3.9.)
"""
    import sys, json, time
    import numpy as np
    _junction_largeM_d = globals()['junction_largeM_d'](); r_inf = _junction_largeM_d.r_inf   # was: from junction_largeM_d import r_inf

    out = {}

    def main():
        for d in [float(x) for x in sys.argv[1:]]:
            rs, bb = {}, {}
            for N in (240, 360, 480):
                t = time.time(); r, v2, etas, bs = r_inf(d, N=N, L=12, full=True); rs[N] = r; bb[str(N)] = bs.tolist()
                print(f'd={d} N={N}: r = {r:.8f}  v0^2 = {v2:.8f}  [{time.time() - t:.0f}s]', flush=True)
            h2 = np.array([(np.pi / (N + 1)) ** 2 for N in rs]); r = np.array(list(rs.values()))
            c = np.polyfit(h2, r, 1); r0 = c[1]
            # quadratic-in-h^2 fit as an error estimate
            c2 = np.polyfit(h2, r, 2)
            eps = 4 - d
            print(f'd={d}: r_inf -> {r0:.8f} (h^4 fit {c2[2]:.8f})   (r-1)/eps = {(r0 - 1) / eps:.6f}', flush=True)
            out[d] = dict(N={str(k): v for k, v in rs.items()}, r0=r0, r0_quad=c2[2], v0sq=v2, etas=etas.tolist(), b=bb)
            store.put('largeN/junction_largeM_d_scan.json', out)

    def cli(argv):
        sys.argv = list(argv)
        with fuzzy.logged('largeN/junction_largeM_d_scan{args}.log'.replace('{args}', '_'.join(argv[1:]))):
            main()
    return namespace(locals(), globals())


@section
def junction_largeM_d_fit():
    """Fit the converged large-N junction profile b(eta) = 1 + beta1 t + beta2 t^2 + ... (t = 1 - cos eta) at each d and compare with the
eps expansion at N -> infinity:  Delta = C t + D t^2 + E t^3  =>  b = 1 + (2D/C) t + (3E/C) t^2,
    beta1 = 0.730763 eps - 2.028422 eps^2,   beta2 = (3/4)(4 E2/C0) eps^2 = 1.011692 eps^2,   r = 1 + beta1 + (4/3) beta2 + 2 beta3 + ...
"""
    import json
    import numpy as np


    def main():
        dat = store.get('largeN/junction_largeM_d_scan.json')
        rows = []
        for d, v in sorted(dat.items(), key=lambda kv: -float(kv[0])):
            d = float(d); eps = 4 - d
            Ns = sorted(int(n) for n in v['b'])
            # the grid error is ~h^(1 + O(eps)) (dsig vanishes at the line with a non-integer power), so extrapolate in h + h^2, not h^2
            h = np.array([np.pi / (N + 1) for N in Ns])
            Mh = np.vstack([np.ones(len(h)), h, h ** 2]).T
            B = np.array([v['b'][str(N)] for N in Ns])                 # (nN, neta)
            b0 = np.linalg.solve(Mh, B)[0]
            r0 = np.linalg.solve(Mh, np.array([v['N'][str(N)] for N in Ns]))[0]
            t = 1 - np.cos(np.array(v['etas']))
            c = np.polynomial.polynomial.polyfit(t, b0 - 1, [1, 2, 3, 4, 5])
            rows.append((eps, r0, c[1], c[2], c[3]))
            print(f'eps={eps:.3f}: r={r0:.7f}  (r-1)/eps={(r0 - 1) / eps:.5f}  beta1/eps={c[1] / eps:.5f}  beta2/eps^2={c[2] / eps ** 2:.4f}  beta3/eps^3={c[3] / eps ** 3:.3f}')
        rows = np.array(rows)
        if len(rows) >= 4:
            small = rows[rows[:, 0] <= 0.21]
            e = small[:, 0]
            for name, col, p in [('(r-1)/eps', 1, None), ('beta1/eps', 2, None)]:
                y = (small[:, col] - 1) / e if col == 1 else small[:, col] / e
                f = np.polyfit(e, y, min(3, len(e) - 1))
                print(f'{name}: fit a0 = {f[-1]:.5f}, a1 = {f[-2]:.4f}' + (f', a2 = {f[-3]:.3f}' if len(f) > 2 else ''))
                z = (y - 0.730763) / e
                f = np.polyfit(e, z, 2)
                print(f'{name}: with a0 = 0.730763 fixed, a1 = {f[-1]:.4f}, a2 = {f[-2]:.3f}')
            y = small[:, 3] / e ** 2
            f = np.polyfit(e, y, min(2, len(e) - 1))
            print(f'beta2/eps^2: fit -> {f[-1]:.4f} (+ {f[-2]:.3f} eps)')

    def cli(argv):
        with fuzzy.logged('largeN/junction_largeM_d_fit.log'):
            main()
    return namespace(locals(), globals())


@section
def ct_largeN_eps():
    """Exact large-N tilt normalization of the pinned O(N) line in d = 4 - eps (2026-10-25).

Frame S^{d-2} x H^2, straight line: sigma = s0 = (4-d)(d-2)/4, phi = sqrt(N) v0 e_1 constant, and the classical junction
energy is N Vol(S^{d-2}) v0^2 (1 - cos eta)/pi, so
    C_t/N = Vol(S^{d-2}) v0^2 / pi,   v0^2 = (1/2pi) sum_l w_l [psi(1/2 + sqrt(nu_l^2 + s0)) - psi(1/2 + nu_l)],
    w_l = deg_l / Vol(S^{d-2}),  nu_l = l + (d-3)/2,  deg_l = (2l+d-3) Gamma(l+d-3)/(Gamma(d-2) l!).
Hence C_t/N = (1/2pi^2) sum_l deg_l [psi(1/2+sqrt(nu_l^2+s0)) - psi(1/2+nu_l)]   (d = 3: 2 v0^2 = 0.0888264).
The sum converges like l^{d-5}; mpmath's Euler-Maclaurin summation handles the slow tail.
Checks: C_t/N -> 1/(4 pi^2) at d = 4 and slope 3/(8 pi^2) (large-N limit of the one-loop C_t).
usage: python ct_largeN_eps.py
"""
    from mpmath import mp, mpf, psi, sqrt, gamma, nsum, inf, pi, rgamma, loggamma, exp
    mp.dps = 25


    def Ct_over_N(d, L=200000):
        """Explicit sum to L (float64) plus the tail: summand -> s0 nu^{d-5}/Gamma(d-2) (1 + O(nu^-2)), summed exactly as a
        Hurwitz zeta; the neglected tail correction is O(L^{d-7}) ~ 1e-11."""
        import numpy as np
        from scipy.special import digamma, gammaln
        from mpmath import zeta as hzeta
        dd = float(d); s0 = (4 - dd) * (dd - 2) / 4
        l = np.arange(L + 1, dtype=float); nu = l + (dd - 3) / 2
        deg = np.ones_like(l)
        if abs(dd - 3) > 1e-12:
            deg[1:] = (2 * l[1:] + dd - 3) * np.exp(gammaln(l[1:] + dd - 3) - gammaln(dd - 2) - gammaln(l[1:] + 1))
        else:
            deg[1:] = 2.0
        body = np.sum(deg * (digamma(0.5 + np.sqrt(nu ** 2 + s0)) - digamma(0.5 + nu)))
        tail = s0 / np.exp(gammaln(dd - 2)) * float(hzeta(5 - mpf(d), L + 1 + (mpf(d) - 3) / 2)) if dd < 4 else 0.0
        return (body + tail) / (2 * np.pi ** 2)


    def Ct_over_N_mp(d):
        d = mpf(d)
        s0 = (4 - d) * (d - 2) / 4
        def term(l):
            nu = l + (d - 3) / 2
            deg = (2 * l + d - 3) * exp(loggamma(l + d - 3) - loggamma(d - 2) - loggamma(l + 1)) if l > 0 else mpf(1)
            return deg * (psi(0, mpf(1) / 2 + sqrt(nu ** 2 + s0)) - psi(0, mpf(1) / 2 + nu))
        return nsum(term, [0, inf], method='euler-maclaurin') / (2 * pi ** 2)

    def main():
        import numpy as np
        print('d = 3 check: C_t/N =', Ct_over_N(3.0), ' (2 v0^2 = 0.0888264)')
        eps = [mpf(x) for x in ('0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.08', '0.1', '0.12', '0.15')]
        vals = [Ct_over_N(4 - e) for e in eps]
        for e, v in zip(eps, vals):
            print(f'eps={float(e):.3f}: C_t/N = {float(v):.12f}')
        E = np.array([float(e) for e in eps]); V = np.array([float(v) for v in vals])
        for deg in (3, 4, 5):
            c = np.polyfit(E, V, deg)[::-1]
            print(f'fit deg {deg}: C0 = {c[0]:.9f} (1/4pi^2 = {1/(4*np.pi**2):.9f}),  C1 = {c[1]:.8f} (3/8pi^2 = {3/(8*np.pi**2):.8f}),  C2 = {c[2]:.6f},  C3 = {c[3]:.4f}')
        C0, C1 = 1 / (4 * np.pi ** 2), 3 / (8 * np.pi ** 2)
        y = (V - C0 - C1 * E) / E ** 2
        for deg in (1, 2, 3):
            c = np.polyfit(E, y, deg)[::-1]
            print(f'with C0, C1 exact, deg {deg}: C2 = {c[0]:.7f}, C3 = {c[1]:.5f}')
        print('d = 3 (sum + tail):', Ct_over_N(3.0))

    def cli(argv):
        mp.dps = 25
        with fuzzy.logged('largeN/ct_largeN_eps.log'):
            main()
    return namespace(locals(), globals())


STEPS = {'junction_oneloop': junction_oneloop, 'ddim': ddim, 'classical_c0': classical_c0, 'classical_cubic': classical_cubic, 'loop_d0': loop_d0, 'assemble': assemble, 'junction_largeM': junction_largeM, 'junction_largeM_run': junction_largeM_run, 'junction_largeM_d': junction_largeM_d, 'junction_largeM_d_scan': junction_largeM_d_scan, 'junction_largeM_d_fit': junction_largeM_d_fit, 'ct_largeN_eps': ct_largeN_eps}
REPORT = [('junction_oneloop',), ('assemble',), ('junction_largeM_d_fit',), ('ct_largeN_eps',)]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
