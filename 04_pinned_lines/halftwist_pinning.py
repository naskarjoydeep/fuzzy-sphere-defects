"""Does the pinned partial twist survive at half twist at large N, and how do the lowest twisted modes shift at O(1/N)?

At N = infinity the pinned partial twist loses its condensate at alpha = 1/2: the spectator mode s = 0 reaches the
Breitenlohner-Freedman value. The order-1/N shift of the pinning mode s = 1/2 on the unpinned partial twist decides
whether the pin stays relevant. Only the exchange of the auxiliary field sigma contributes at this order, so the
rainbow self-energy Sigma = D_sigma G_alpha, projected onto the modes e^{i s theta} K_Delta of S^1 x H^2, gives the
shift (paper, App. E.7, eq:pinlaw and eq:halftwistpin). We evaluate it in two schemes with independent codes:
  analytic continuation in the exponent of D_sigma, with the local terms subtracted in a chordal ball
  (hs1_ingredients ... hs6_alpha);
  a chordal cutoff rho > eps with a Hadamard finite-part fit (cutoff_eigen, cutoff_alpha).
On untwisted lines both give every mode with s != 0 the shift Delta_phi - 1/2 = 4/(3 pi^2 N), which fixes the
normalization and the sign. At half twist the pinning mode has Delta_1/2 = 3/2 - Delta_phi, so the pin stays relevant.
For a general twist the two lowest modes, |s| = alpha and 1 - alpha, shift by (4|s| - 3)(Delta_phi - 1/2).

Notation: the scripts write N delta Delta, or Sigma-hat per C_sigma = 16/(pi^2 N), and gamma_phi = Delta_phi - 1/2.
The docstrings keep their research notes: S1-S6 are the steps of the plan for this calculation (the "brief", not
included), the CCS route is the cutoff scheme, and the report points were the checkpoints at which the two schemes were
compared.

usage: python halftwist_pinning.py <step> [args]      (python halftwist_pinning.py help lists the steps)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuzzy                                             # noqa: E402
from fuzzy import store, section, namespace          # noqa: E402,F401


@section
def hs1_ingredients():
    """S1 of fun/briefs_2026-10-27/halftwist_delta_gamma.md: check the ingredients.

1. G_tw antiperiodic in theta.
2. G_tw - G_0 -> -1/(4 pi^2) at coincidence.
3. theta-modes of G_tw (s in Z+1/2) and G_0 (s in Z) equal Q_{|s|-1/2}(cosh chi)/(4 pi^2).
4. C_sigma = 16/(pi^2 N) from D_sigma(p) = -16|p|/N, and gamma_phi = +4/(3 pi^2 N)
   from Sigma = D_sigma G_0, G^-1 = G_0^-1 - Sigma.
"""
    import json
    import mpmath as mp
    import sympy as sp

    mp.mp.dps = 30


    def dt(th, ch):
        return mp.sqrt(2 * (mp.cosh(ch) - mp.cos(th)))


    def G0(th, ch):
        return 1 / (4 * mp.pi * dt(th, ch))


    def Gtw(th, ch):
        Z = mp.cos(th / 2) / mp.cosh(ch / 2)
        return G0(th, ch) * (2 / mp.pi) * mp.asin(Z)


    def Q(nu, ch):
        return mp.legenq(nu, 0, mp.cosh(ch), type=3).real

    def main():
        out = {}

        # 1. antiperiodicity
        pts = [(0.3, 0.2), (1.1, 0.7), (2.5, 1.3), (-0.8, 0.05), (3.0, 2.0)]
        e1 = max(abs(Gtw(t + 2 * mp.pi, c) + Gtw(t, c)) / abs(Gtw(t, c)) for t, c in pts)
        out['antiperiodic_max_rel_err'] = float(e1)

        # 2. coincidence limit of dG = G_tw - G_0, approached along several directions.
        # 1 - Z ~ rho^2/8 cancels catastrophically, so work at 80 digits; the remainder is O(rho^2) (about rho^2/24).
        e2 = 0
        with mp.workdps(80):
            for psi in [0.0, 0.4, 0.9, 1.3, mp.pi / 2]:
                errs = []
                for r in [mp.mpf('1e-8'), mp.mpf('1e-10')]:
                    th, ch = r * mp.cos(psi), r * mp.sin(psi)
                    dG = Gtw(th, ch) - G0(th, ch)
                    errs.append(abs(dG * 4 * mp.pi**2 + 1))
                e2 = max(e2, errs[1])
                assert errs[1] < 1e-15 and 5e3 < errs[0] / errs[1] < 2e4, errs  # quadratic in rho
        out['coincidence_dG_rel_err_at_rho_1e-10'] = float(e2)

        # 3. theta-modes at 10 points
        e3 = 0
        rows = []
        for s, ch in [(0.5, 0.3), (0.5, 1.2), (1.5, 0.4), (1.5, 2.0), (2.5, 0.8),
                      (0, 0.3), (0, 1.5), (1, 0.6), (2, 1.0), (3, 0.5)]:
            s = mp.mpf(s)
            ch = mp.mpf(ch)
            f = Gtw if s % 1 else G0
            mode = mp.quad(lambda t: mp.cos(s * t) * f(t, ch), [-mp.pi, 0, mp.pi]) / (2 * mp.pi)
            ref = Q(abs(s) - mp.mpf(1) / 2, ch) / (4 * mp.pi**2)
            e3 = max(e3, abs(mode / ref - 1))
            rows.append([float(s), float(ch), float(mode), float(ref)])
        out['mode_max_rel_err'] = float(e3)
        out['mode_rows'] = rows

        # 4. C_sigma and gamma_phi (sympy, d = 3)
        x, p, a, dl, N = sp.symbols('x p a delta N', positive=True)


        def ft(two_a, d=3):
            """int d^d y e^{iky} |y|^{-two_a} = pi^{d/2} 2^{d-two_a} Gamma(d/2-two_a/2)/Gamma(two_a/2) |k|^{two_a-d}."""
            A = sp.Rational(two_a) if not isinstance(two_a, sp.Expr) else two_a
            return sp.pi**sp.Rational(d, 2) * 2**(d - A) * sp.gamma(sp.Rational(d, 2) - A / 2) / sp.gamma(A / 2)


        # D_sigma(x) = int d^3p/(2pi)^3 e^{ipx} (-16|p|/N): |p|^{+1} is two_a = -1
        Csig = sp.simplify(-16 / N * ft(-1) / (2 * sp.pi)**3)
        out['C_sigma'] = str(Csig)
        assert sp.simplify(Csig - 16 / (sp.pi**2 * N)) == 0

        # Sigma(x) = C_sigma |x|^-4 * 1/(4 pi |x|) = C_sigma/(4 pi) |x|^{-5+2 delta} (dim-reg in the exponent)
        Sig_p = Csig / (4 * sp.pi) * ft(5 - 2 * dl) * p**(2 - 2 * dl)
        ser = sp.series(Sig_p / p**2, dl, 0, 1).removeO()
        logcoef = sp.simplify(sp.expand(ser).coeff(sp.log(p)))
        # G^-1 = p^2 - Sigma = p^2 (1 - logcoef log p) = p^{2 - 2 gamma}  =>  gamma = logcoef/2
        gam = sp.simplify(logcoef / 2)
        out['gamma_phi_times_N'] = str(sp.simplify(gam * N))
        assert sp.simplify(gam - 4 / (3 * sp.pi**2 * N)) == 0

        ok = e1 < 1e-20 and e2 < 1e-10 and e3 < 1e-12
        out['gate'] = 'PASS' if ok else 'FAIL'
        print(json.dumps(out, indent=1))
        store.put('analytic/halftwist_gamma/hs1_ingredients.json', out)   # was: json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1)

    def cli(argv):
        mp.mp.dps = 30
        main()
    return namespace(locals(), globals())


@section
def hs2_local():
    """S2: local expansion of the eigenvalue integrand at coincidence (sympy).

Everything is per C_sigma.  The 3d integrand of Sigma-hat, including the measure, is
    2 pi sinh(chi) dchi dtheta * dt^{-4} G(theta,chi) cos(s theta) P_{-Delta}(cosh chi).
Two coordinate systems:
  geodesic: (theta, chi) = rho (cos psi, sin psi);
  chordal:  u = 2 sin(theta/2), v = 2 sinh(chi/2), (u, v) = R (cos phi, sin phi), so dt = R exactly
            and 2 pi sinh(chi) dchi dtheta = 2 pi v du dv / sqrt(1 - u^2/4).
Checks:
  1. the log-order (lambda-pole) coefficient of the bulk piece, after the angular integral,
     is proportional to Delta(Delta-1) - s^2 + 1/4 in both systems;
  2. the twisted-only piece dt^{-4} dG has a pure power divergence and no log term
     (its expansion contains only even powers above the leading one).
"""
    import json
    import sympy as sp

    def main():
        s, D = sp.symbols('s Delta')
        r, ang = sp.symbols('r ang', positive=True)
        onshell = D * (D - 1) - s**2 + sp.Rational(1, 4)
        out = {}

        # P_{-Delta}(z) = 2F1(Delta, 1-Delta; 1; (1-z)/2) to second order in (z-1)
        def P_series(zm1):
            w = -zm1 / 2
            return 1 + D * (1 - D) * w + D * (D + 1) * (1 - D) * (2 - D) / 4 * w**2

        # ---- geodesic polar coordinates ----
        th, ch = r * sp.cos(ang), r * sp.sin(ang)
        dt2 = 2 * (sp.cosh(ch) - sp.cos(th))
        meas = 2 * sp.pi * sp.sinh(ch) * r          # dtheta dchi = rho drho dpsi
        bulk = meas * dt2**sp.Rational(-5, 2) / (4 * sp.pi) * sp.cos(s * th) * P_series(sp.cosh(ch) - 1)
        ser = sp.series(bulk, r, 0, 0).removeO()     # terms r^{-3} .. r^{-1}
        c_m1 = sp.simplify(sp.integrate(sp.expand(ser).coeff(r, -1), (ang, 0, sp.pi)))
        c_m2 = sp.simplify(sp.integrate(sp.expand(ser).coeff(r, -2), (ang, 0, sp.pi)))
        out['geodesic_log_coeff'] = str(sp.factor(c_m1))
        out['geodesic_log_coeff_over_onshell'] = str(sp.simplify(c_m1 / onshell))
        out['geodesic_rho^-2_coeff'] = str(c_m2)
        assert sp.simplify(c_m1 / onshell).free_symbols == set()
        assert c_m2 == 0

        # ---- chordal coordinates ----
        u, v = r * sp.cos(ang), r * sp.sin(ang)
        theta_u = 2 * sp.asin(u / 2)
        f = sp.cos(s * theta_u) * P_series(v**2 / 2) / sp.sqrt(1 - u**2 / 4)
        meas_c = 2 * sp.pi * v * r                   # du dv = R dR dphi
        bulk_c = meas_c * r**-5 / (4 * sp.pi) * f
        ser_c = sp.expand(sp.series(bulk_c, r, 0, 0).removeO())
        k_m1 = sp.simplify(sp.integrate(ser_c.coeff(r, -1), (ang, 0, sp.pi)))
        out['chordal_log_coeff'] = str(sp.factor(k_m1))
        out['chordal_log_coeff_over_onshell'] = str(sp.simplify(k_m1 / onshell))
        assert sp.simplify(k_m1 / onshell).free_symbols == set()
        assert sp.simplify(k_m1 - c_m1) == 0         # the residue is the same in both systems
        assert ser_c.coeff(r, -2) == 0

        # local coefficients used by the S3 subtraction: f = 1 + a u^2 + b v^2 + ...
        U, V = sp.symbols('U V')
        fu = sp.cos(s * 2 * sp.asin(U / 2)) * P_series(V**2 / 2) / sp.sqrt(1 - U**2 / 4)
        a = sp.simplify(sp.diff(fu, U, 2).subs({U: 0, V: 0}) / 2)
        b = sp.simplify(sp.diff(fu, V, 2).subs({U: 0, V: 0}) / 2)
        out['a'], out['b'] = str(a), str(b)
        assert sp.simplify(a + 2 * b - onshell / 2) == 0

        # ---- twisted-only piece, chordal: dG = -(2/pi) asin(R/(2c)) / (4 pi R), c = cosh(chi/2) = sqrt(1+v^2/4)
        c = sp.sqrt(1 + v**2 / 4)
        dG = -(2 / sp.pi) * sp.asin(r / (2 * c)) / (4 * sp.pi * r)
        tw_c = sp.expand(sp.series(meas_c * r**-4 * dG * f, r, 0, 2).removeO())
        lead = sp.simplify(sp.integrate(tw_c.coeff(r, -2), (ang, 0, sp.pi)))
        odd = [sp.simplify(tw_c.coeff(r, k)) for k in (-1, 1)]
        out['twisted_leading_R^-2_coeff'] = str(lead)
        out['twisted_odd_coeffs'] = [str(x) for x in odd]
        assert sp.simplify(lead + 1 / sp.pi) == 0 and all(x == 0 for x in odd)

        out['gate'] = 'PASS'
        print(json.dumps(out, indent=1))
        store.put('analytic/halftwist_gamma/hs2_local.json', out)   # was: json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1)

    def cli(argv):
        main()
    return namespace(locals(), globals())


@section
def hs3_eigen():
    """S3: regularized eigenvalue Sigma-hat^FP(s, Delta) in the chordal scheme, per C_sigma.

Sigma-hat(s,Delta) = int_{-pi}^{pi} dtheta cos(s theta) int_0^oo 2 pi sinh(chi) dchi
                     dt^{-4+2 lam} G(theta,chi) P_{-Delta}(cosh chi),   FP at lam -> 0.
Chordal ball R = dt < eps, with u = 2 sin(theta/2) = R cos(phi), v = 2 sinh(chi/2) = R sin(phi):
  Sigma^FP = int_{R>eps} K  +  int_{R<eps} (K - T)  +  FP int_{R<eps} T dt^{2 lam},
  T = (sin phi / 2) R^-3 (1 + a u^2 + b v^2)  [bulk],  - sin(phi) R^-2 / (2 pi)  [twisted only],
  a = 1/8 - s^2/2, b = Delta(Delta-1)/4 (hs2_local.json), so
  FP int T = -1/(2 eps^2) + (a + 2 b)/3 log(eps)  [+ 1/(pi eps) for the twisted kernel].
kind = 'bulk': trivial defect, G = G0 = 1/(4 pi dt).
kind = 'tw':   alpha = 1/2 twisted pair, G = G0 (2/pi) asin(cos(theta/2)/cosh(chi/2)).
closed_bulk(s, Delta): the trivial eigenvalue in closed form (z integral, then the 2d Riesz potential
of r^k e^{i s theta}), continued in a = 5/2 - lam; valid for integer s.
"""
    import json
    import sys
    import mpmath as mp

    mp.mp.dps = 25


    def P(D, ch):
        return mp.hyp2f1(D, 1 - D, 1, -mp.sinh(ch / 2)**2)


    def G_outer(kind, th, ch):
        dt = mp.sqrt(2 * (mp.cosh(ch) - mp.cos(th)))
        g0 = 1 / (4 * mp.pi * dt)
        if kind == 'bulk':
            return dt, g0
        return dt, g0 * (2 / mp.pi) * mp.asin(mp.cos(th / 2) / mp.cosh(ch / 2))


    def outer(kind, s, D, eps, chmax=30):
        che = 2 * mp.asinh(eps / 2)

        def th_int(ch, th0):
            f = lambda th: (lambda dg: dg[0]**-4 * dg[1])(G_outer(kind, th, ch)) * mp.cos(s * th)
            return mp.quad(f, [th0, mp.pi / 2, mp.pi] if th0 < mp.pi / 2 else [th0, mp.pi])

        def chi_int(ch, near):
            if near:
                th0 = 2 * mp.asin(mp.sqrt(max(eps**2 / 4 - mp.sinh(ch / 2)**2, 0)))
            else:
                th0 = mp.mpf(0)
            return 2 * mp.pi * mp.sinh(ch) * P(D, ch) * th_int(ch, th0)

        I1 = mp.quad(lambda ch: chi_int(ch, True), [0, che])
        pts = [che] + [x for x in [0.5, 1, 2, 4, 8, 15] if x > che] + [chmax]
        I2 = mp.quad(lambda ch: chi_int(ch, False), pts)
        return 2 * (I1 + I2)                     # theta < 0 half


    def inner(kind, s, D, eps):
        a = mp.mpf(1) / 8 - s**2 / 2
        b = D * (D - 1) / 4

        def f(R, phi):
            # the bulk subtraction cancels R^-3 against R^-3 (1 + O(R^2)): add digits as R -> 0
            extra = int(-3 * mp.log10(R)) + 5 if R < 1 else 5
            with mp.workdps(mp.mp.dps + extra):
                return +f_hi(mp.mpf(R), mp.mpf(phi))

        def f_hi(R, phi):
            u, v = R * mp.cos(phi), R * mp.sin(phi)
            th, ch = 2 * mp.asin(u / 2), 2 * mp.asinh(v / 2)
            c = mp.sqrt(1 + v**2 / 4)
            w = mp.cos(s * th) * P(D, ch) / mp.sqrt(1 - u**2 / 4)
            meas = 2 * mp.pi * R**2 * mp.sin(phi)
            val = meas * R**-4 * (1 / (4 * mp.pi * R)) * w - mp.sin(phi) / 2 * R**-3 * (1 + a * u**2 + b * v**2)
            if kind == 'tw':
                dG = -(2 / mp.pi) * mp.asin(R / (2 * c)) / (4 * mp.pi * R)
                val += meas * R**-4 * dG * w + mp.sin(phi) * R**-2 / (2 * mp.pi)
            return val

        return mp.quad(f, [0, eps], [0, mp.pi / 2, mp.pi])


    def ball(kind, s, D, eps):
        a = mp.mpf(1) / 8 - s**2 / 2
        b = D * (D - 1) / 4
        B = -1 / (2 * eps**2) + (a + 2 * b) / 3 * mp.log(eps)
        if kind == 'tw':
            B += 1 / (mp.pi * eps)
        return B


    def sigma_hat(kind, s, D, eps=mp.mpf('0.1')):
        s, D, eps = mp.mpf(s), mp.mpf(D), mp.mpf(eps)
        parts = (outer(kind, s, D, eps), inner(kind, s, D, eps), ball(kind, s, D, eps))
        return sum(parts), parts


    def closed_bulk(s, D, h=mp.mpf('1e-12')):
        l = abs(mp.mpf(s))
        D = mp.mpf(D)

        def E(a):
            return (mp.pi**1.5 * mp.gamma(mp.mpf(3) / 2 - a) / mp.gamma(a)
                    * mp.gamma((l + a - D) / 2) * mp.gamma((l + a + D - 1) / 2)
                    / (mp.gamma((l - a - D + 3) / 2) * mp.gamma((l - a + D + 2) / 2)))

        with mp.workdps(50):
            a0 = mp.mpf(5) / 2
            val = (E(a0 - h) + E(a0 + h)) / 2 / (4 * mp.pi)   # constant Laurent term in lam
        return val


    def main():
        out = {'bulk_vs_closed': [], 'eps_scan': []}
        # (i) trivial kernel vs the closed form, on and off shell
        for s, D in [(1, 1.5), (2, 2.5), (1, 1.3), (0, 0.8), (2, 2.2)]:
            num, parts = sigma_hat('bulk', s, D)
            cf = closed_bulk(s, D)
            out['bulk_vs_closed'].append([s, D, mp.nstr(num, 15), mp.nstr(cf, 15), float(abs(num - cf))])
            print('bulk', s, D, mp.nstr(num, 15), mp.nstr(cf, 15), flush=True)
        # (ii) ball-size independence, on shell: trivial n=1 and twisted s=3/2 (not the target mode)
        for kind, s, D in [('bulk', 1, 1.5), ('tw', 1.5, 2)]:
            vals = [sigma_hat(kind, s, D, e)[0] for e in ['0.05', '0.1', '0.2']]
            spread = max(vals) - min(vals)
            out['eps_scan'].append([kind, s, D, [mp.nstr(x, 15) for x in vals], float(spread)])
            print(kind, s, D, [mp.nstr(x, 15) for x in vals], flush=True)
        ok = all(r[4] < 1e-9 for r in out['bulk_vs_closed']) and all(r[4] < 1e-9 for r in out['eps_scan'])
        out['gate'] = 'PASS' if ok else 'FAIL'
        print(out['gate'])
        store.put('analytic/halftwist_gamma/hs3_eigen.json', out)   # was: json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1)

    def cli(argv):
        mp.mp.dps = 25
        main()
    return namespace(locals(), globals())


@section
def hs4_calib():
    """S4: calibration on the trivial defect, Sigma-hat_0^FP(n, 1/2 + n) for n = 0, 1, 2 (per C_sigma).

The brief expected 0 (or a scheme constant a + b n^2). The closed form of hs3_eigen.closed_bulk gives
-n/6 on shell, i.e. Sigma-hat_0 = -C_sigma n / 6 = -2 gamma_phi n, so delta Delta = -Sigma-hat/(2n) = gamma_phi
for every n >= 1: the chordal part alone already gives each trivial mode the full bulk gamma_phi.
This script checks the numerics against that, and tests the a + b n^2 form (fit on n = 0, 1, test on n = 2).
"""
    import json
    import mpmath as mp
    h = globals()['hs3_eigen']()   # was: import hs3_eigen as h

    def main():
        rows = []
        for n in [0, 1, 2]:
            D = mp.mpf(n) + mp.mpf(1) / 2
            num = {e: h.sigma_hat('bulk', n, D, e)[0] for e in ['0.1', '0.2']}
            cf = h.closed_bulk(n, D)
            rows.append(dict(n=n, Delta=float(D), num_eps01=mp.nstr(num['0.1'], 16), num_eps02=mp.nstr(num['0.2'], 16),
                             closed=mp.nstr(cf, 16), minus_n_over_6=mp.nstr(-mp.mpf(n) / 6, 16),
                             err_vs_closed=float(max(abs(x - cf) for x in num.values())),
                             dDelta_times_N=(mp.nstr(-16 / mp.pi**2 * num['0.1'] / (2 * n), 12) if n else 'BF: first order degenerates')))
            print(rows[-1], flush=True)

        v = [mp.mpf(r['num_eps01']) for r in rows]
        a_fit, b_fit = v[0], v[1] - v[0]
        pred2 = a_fit + 4 * b_fit
        out = dict(rows=rows,
                   gamma_phi_times_N=mp.nstr(4 / (3 * mp.pi**2), 12),
                   brief_expectation_zero=all(abs(x) < 1e-6 for x in v),
                   a_plus_bn2=dict(a=mp.nstr(a_fit, 12), b=mp.nstr(b_fit, 12), predicted_n2=mp.nstr(pred2, 12),
                                   actual_n2=mp.nstr(v[2], 12), fits=bool(abs(pred2 - v[2]) < 1e-6 * abs(v[2]))),
                   linear_in_n=dict(slope=mp.nstr(v[1] - v[0], 12),
                                    fits=bool(abs(v[2] - 2 * v[1]) < 1e-9 and abs(v[0]) < 1e-9)))
        out['verdict'] = ('FAILS the brief: Sigma-hat_0(n, 1/2+n) = -n C_sigma/6 = -2 gamma_phi n (not 0, not a + b n^2); '
                          'the chordal part gives every trivial mode delta Delta = +gamma_phi')
        print(json.dumps({k: out[k] for k in ['brief_expectation_zero', 'a_plus_bn2', 'linear_in_n']}, indent=1))
        store.put('analytic/halftwist_gamma/hs4_calib.json', out)   # was: json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1)

    def cli(argv):
        mp.mp.dps = 25   # set by hs3_eigen on import in the research run
        main()
    return namespace(locals(), globals())


@section
def hs5_twisted():
    """S5: the twisted mode s = 1/2, Delta = 1 at alpha = 1/2.

Agreed with CCS after report point 1: gamma_{1/2} = -N Sigma-hat^FP_tw(1/2, 1)/(2 Delta - 1), taken in full
(the chordal part already carries gamma_phi, hs4_calib.json), and delta gamma = gamma_{1/2} - 4/(3 pi^2).
N C_sigma = 16/pi^2.
Checks: ball size eps = 0.05, 0.1, 0.2; chi tail chmax = 30 vs 40; and a regulator check of the
twisted-only part (kernel dt^-4 dG on theta in (-pi, pi)): continuation vs hard chordal cutoff R > eps
with the Hadamard finite part, lim [I(eps) + 1/(pi eps)], extrapolated in eps.
"""
    import json
    import mpmath as mp
    h = globals()['hs3_eigen']()   # was: import hs3_eigen as h

    def main():
        s, D = mp.mpf(1) / 2, mp.mpf(1)
        NC = 16 / mp.pi**2
        out = {}

        vals = {}
        for e in ['0.05', '0.1', '0.2']:
            vals[e] = h.sigma_hat('tw', s, D, e)[0]
            print('eps', e, mp.nstr(vals[e], 16), flush=True)
        tail40 = h.outer('tw', s, D, mp.mpf('0.1'), chmax=40) + h.inner('tw', s, D, mp.mpf('0.1')) + h.ball('tw', s, D, mp.mpf('0.1'))
        sig = vals['0.1']
        gam = -NC * sig / (2 * D - 1)
        out['Sigma_tw_per_Csigma'] = {k: mp.nstr(v, 16) for k, v in vals.items()}
        out['Sigma_tw_chmax40'] = mp.nstr(tail40, 16)
        out['eps_spread'] = float(max(vals.values()) - min(vals.values()))
        out['chi_tail_shift'] = float(abs(tail40 - sig))
        out['gamma_half_times_N'] = mp.nstr(gam, 14)
        out['delta_gamma_times_N'] = mp.nstr(gam - 4 / (3 * mp.pi**2), 14)
        out['sign'] = 'negative: pinned line survives' if gam < 0 else 'positive: pinned line does not survive'
        print(json.dumps(out, indent=1), flush=True)

        # regulator check on the twisted-only part
        e0 = mp.mpf('0.1')
        cont = (h.outer('tw', s, D, e0) - h.outer('bulk', s, D, e0)
                + h.inner('tw', s, D, e0) - h.inner('bulk', s, D, e0) + 1 / (mp.pi * e0))
        had = []
        for e in ['0.01', '0.02', '0.04', '0.08']:
            e = mp.mpf(e)
            I = h.outer('tw', s, D, e) - h.outer('bulk', s, D, e)
            had.append((e, I + 1 / (mp.pi * e)))
            print('hadamard eps', mp.nstr(e, 3), mp.nstr(had[-1][1], 16), flush=True)
        # I + 1/(pi eps) = I0 + c1 eps + c2 eps^2 + c3 eps^3: solve on the four points
        A = mp.matrix([[1, e, e**2, e**3] for e, _ in had])
        y = mp.matrix([v for _, v in had])
        I0 = mp.lu_solve(A, y)[0]
        out['twisted_only_continuation'] = mp.nstr(cont, 14)
        out['twisted_only_hadamard_cutoff'] = mp.nstr(I0, 14)
        out['twisted_only_hadamard_raw'] = [[mp.nstr(e, 3), mp.nstr(v, 14)] for e, v in had]
        out['regulator_diff'] = float(abs(cont - I0))
        out['gate'] = 'PASS' if out['eps_spread'] < 1e-9 and out['chi_tail_shift'] < 1e-9 and out['regulator_diff'] < 1e-7 else 'CHECK'
        print(json.dumps({k: out[k] for k in ['twisted_only_continuation', 'twisted_only_hadamard_cutoff', 'regulator_diff', 'gate']}, indent=1))
        store.put('analytic/halftwist_gamma/hs5_twisted.json', out)   # was: json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1)

    def cli(argv):
        mp.mp.dps = 25   # set by hs3_eigen on import in the research run
        main()
    return namespace(locals(), globals())


@section
def hs6_alpha():
    """S6: general twist alpha. Modes s = alpha (Delta = 1/2 + alpha) and s = alpha - 1 (Delta = 3/2 - alpha);
their dimensions sum to 2 at N = oo.

Sigma-hat = bulk part (hs3_eigen 'bulk', mpmath, on theta in (-pi, pi)) + twisted-only part (here, numpy).
Twisted-only kernel: dt^-4 dG_alpha projected with e^{-is theta}, i.e. Re dG cos(s theta) + Im dG sin(s theta).
dG_alpha = (1/4pi^2) int_chi^oo dt dS(theta,t) / sqrt(2(cosh t - cosh chi)), with
  dS = S_alpha - S_0 = -expm1(alpha w)/expm1(w) + expm1(alpha w')/expm1(w'),  w = i theta - t, w' = i theta + t
(the poles at w, w' = 0 cancel exactly).  dS is odd in t, so dS/sinh t is a function of t^2; with
cosh t = cosh chi cosh eta the integrand dS sqrt(cosh chi) cosh(eta/2)/sinh t is analytic in eta.
Partition of unity psi(R) (1 for R < Ra, erfc step, 0 for R > Rb) in the chordal radius R = dt:
  polar part  int_0^Rb dR int_0^pi dphi psi 2 pi sin(phi) R^-2 {W - dG(0)},
  ball term   4 pi dG(0) [-1/Ra + int_Ra^Rb psi R^-2 dR]   (continuation, no pole),
  rest        2 int_0^pi dtheta int_0^chimax 2 pi sinh(chi) dchi (1 - psi) dt^-4 W'.
Gates: dG at alpha = 1/2 against the closed form; theta-modes of G_alpha against Q/(4pi^2) at alpha = 0.3;
twisted-only part at alpha = 1/2 against hs3/hs5 (s = 1/2, 3/2, 5/2); node doubling.
"""
    import json
    import sys
    import numpy as np
    from numpy.polynomial.legendre import leggauss
    from scipy.special import hyp2f1, erfc
    import mpmath as mp
    h = globals()['hs3_eigen']()   # was: import hs3_eigen as h


    def gl(edges, n):
        x, w = leggauss(n)
        X, W = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            X.append((b - a) / 2 * x + (a + b) / 2)
            W.append((b - a) / 2 * w)
        return np.concatenate(X), np.concatenate(W)


    ETA, WETA = gl([0, 1, 3, 7, 14, 25, 40, 60], 32)


    def dG(alpha, th, ch, chunk=20000):
        th, ch = np.broadcast_arrays(np.asarray(th, float), np.asarray(ch, float))
        shape, th, ch = th.shape, th.ravel(), ch.ravel()
        out = np.empty(th.size, complex)
        for i in range(0, th.size, chunk):      # keep the (points x eta) arrays small
            out[i:i + chunk] = _dG(alpha, th[i:i + chunk], ch[i:i + chunk])
        return out.reshape(shape)


    def _dG(alpha, th, ch, eta=ETA, weta=WETA):
        th = th[:, None]
        ch = ch[:, None]
        sh = np.sqrt(np.sinh(ch)**2 + np.cosh(ch)**2 * np.sinh(eta)**2)    # sinh t
        t = np.arcsinh(sh)
        w, wp = 1j * th - t, 1j * th + t
        dS = -np.expm1(alpha * w) / np.expm1(w) + np.expm1(alpha * wp) / np.expm1(wp)
        f = dS * np.sqrt(np.cosh(ch)) * np.cosh(eta / 2) / sh
        return (f * weta).sum(-1) / (4 * np.pi**2)


    def P(D, ch):
        return hyp2f1(D, 1 - D, 1, -np.sinh(ch / 2)**2)


    def psi(R, Ra, Rb):
        Rm, k = (Ra + Rb) / 2, 6.5 / ((Rb - Ra) / 2)
        out = 0.5 * erfc(k * (R - Rm))
        return np.where(R < Ra, 1.0, np.where(R > Rb, 0.0, out))


    def twisted_only(alpha, s, D, n=60, Ra=0.5, Rb=1.5, chimax=22.0):
        g0 = dG(alpha, 0.0, 0.0).real
        # polar part
        R, wR = gl([0, Ra, Rb], n)
        ph, wph = gl([0, np.pi / 2, np.pi], n // 2)
        RR, PP = np.meshgrid(R, ph, indexing='ij')
        u, v = RR * np.cos(PP), RR * np.sin(PP)
        th, ch = 2 * np.arcsin(u / 2), 2 * np.arcsinh(v / 2)
        g = dG(alpha, th, ch)
        W = (g.real * np.cos(s * th) + g.imag * np.sin(s * th)) * P(D, ch) / np.sqrt(1 - u**2 / 4)
        integ = psi(RR, Ra, Rb) * 2 * np.pi * np.sin(PP) * RR**-2 * (W - g0)
        polar = np.einsum('i,j,ij->', wR, wph, integ)
        Rr, wr = gl([Ra, Rb], 4 * n)
        ball = 4 * np.pi * g0 * (-1 / Ra + (wr * psi(Rr, Ra, Rb) / Rr**2).sum())
        # rest, theta in (0, pi) doubled
        tq, wt = gl([0, 0.5, 1.2, 2.2, np.pi], n)
        cq, wc = gl([0, 0.5, 1.2, 2.5, 5, 9, 14, chimax], n)
        TT, CC = np.meshgrid(tq, cq, indexing='ij')
        dt2 = 2 * (np.cosh(CC) - np.cos(TT))
        Rg = np.sqrt(dt2)
        g = dG(alpha, TT, CC)
        W = (g.real * np.cos(s * TT) + g.imag * np.sin(s * TT)) * P(D, CC)
        integ = 2 * 2 * np.pi * np.sinh(CC) * (1 - psi(Rg, Ra, Rb)) * dt2**-2 * W
        rest = np.einsum('i,j,ij->', wt, wc, integ)
        return polar + ball + rest, dict(polar=polar, ball=ball, rest=rest, dG0=g0)


    def gates():
        out = {}
        # (1) dG at alpha = 1/2 vs closed form, including points near coincidence
        rng = np.random.default_rng(1)
        th = np.concatenate([rng.uniform(-3, 3, 200), rng.uniform(-1e-3, 1e-3, 50), [0.0]])
        ch = np.concatenate([rng.uniform(0, 8, 200), rng.uniform(0, 1e-3, 50), [0.0]])
        num = dG(0.5, th, ch)
        R = np.sqrt(4 * np.sin(th / 2)**2 + 4 * np.sinh(ch / 2)**2)
        c = np.cosh(ch / 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            ref = np.where(R > 0, -(2 / np.pi) * np.arcsin(R / (2 * c)) / (4 * np.pi * np.where(R > 0, R, 1)), -1 / (4 * np.pi**2))
        out['dG_half_vs_closed_maxabs'] = float(np.max(np.abs(num - ref)))
        # (2) theta-modes of G_alpha = G_0 + dG at alpha = 0.3
        th, wth = gl([-np.pi, -1, 0, 1, np.pi], 60)
        errs = []
        for chi in [0.4, 1.0, 2.5]:
            G = 1 / (4 * np.pi * np.sqrt(2 * (np.cosh(chi) - np.cos(th)))) + dG(0.3, th, np.full_like(th, chi))
            for sm in [0.3, -0.7, 1.3, -1.7]:
                mode = (wth * np.exp(-1j * sm * th) * G).sum() / (2 * np.pi)
                ref = float(mp.legenq(abs(sm) - 0.5, 0, mp.cosh(chi), type=3).real) / (4 * np.pi**2)
                errs.append(abs(mode - ref) / abs(ref))
        out['G_alpha03_mode_maxrel'] = float(max(errs))
        # (3) twisted-only part at alpha = 1/2 against the mpmath route (full 'tw' minus 'bulk').
        # Only Delta < 2 is usable: at half-integer s the bulk-only piece on (-pi, pi) decays pointwise as
        # e^{(Delta - 5/2) chi}, so at Delta = 2, 3 the split itself converges slowly or diverges (first run).
        rows = []
        for sv, Dv in [(0.5, 1.0), (1.5, 1.2), (0.5, 0.8)]:
            ref = float(h.sigma_hat('tw', sv, Dv)[0] - h.sigma_hat('bulk', sv, Dv)[0])
            a, _ = twisted_only(0.5, sv, Dv)
            b, _ = twisted_only(0.5, sv, Dv, n=90)
            rows.append([sv, Dv, a, b, ref, abs(a - ref), abs(a - b)])
        out['twisted_only_half'] = rows
        # (4) charge conjugation: alpha -> 1 - alpha with s -> -s, which tests the imaginary part of dG
        c1, _ = twisted_only(0.3, 0.3, 0.8)
        c2, _ = twisted_only(0.7, -0.3, 0.8)
        out['conjugation_diff'] = abs(c1 - c2)
        out['gate'] = ('PASS' if out['dG_half_vs_closed_maxabs'] < 1e-13 and out['G_alpha03_mode_maxrel'] < 1e-11
                       and all(r[5] < 1e-9 and r[6] < 1e-10 for r in rows) and out['conjugation_diff'] < 1e-11 else 'FAIL')
        return out


    def run(alphas):
        NC = 16 / np.pi**2
        rows = []
        for al in alphas:
            pair = []
            for sv, Dv in [(al, 0.5 + al), (al - 1, 1.5 - al)]:
                tw, parts = twisted_only(al, sv, Dv)
                tw2, _ = twisted_only(al, sv, Dv, n=90)
                tw3, _ = twisted_only(al, sv, Dv, Ra=0.4, Rb=1.2)
                bulk = float(h.sigma_hat('bulk', sv, Dv)[0])
                sig = bulk + tw
                gam = -NC * sig / (2 * Dv - 1)
                pair.append(gam)
                rows.append(dict(alpha=al, s=sv, Delta=Dv, bulk=bulk, twisted_only=tw, sigma_per_Csigma=sig,
                                 Ngamma=gam, Ngamma_over_Ngammaphi=gam / (4 / (3 * np.pi**2)),
                                 node_doubling=abs(tw - tw2), partition_change=abs(tw - tw3), dG0=float(parts['dG0']),
                                 law=(4 * al - 3) if sv > 0 else (1 - 4 * al)))
                print(rows[-1], flush=True)
            rows.append(dict(alpha=al, Ngamma_sum=pair[0] + pair[1], over_gammaphi=(pair[0] + pair[1]) / (4 / (3 * np.pi**2))))
            print(rows[-1], flush=True)
        return rows


    def main():
        g = gates()
        print(json.dumps(g, indent=1), flush=True)
        res = {'gates': g}
        if g['gate'] == 'PASS':
            res['results'] = run([0.3, 0.4, 0.1, 0.2])  # 0.1, 0.2: tests the linear law found at 0.3, 0.4
        store.put('analytic/halftwist_gamma/hs6_alpha.json', res)   # was: json.dump(res, open(__file__.replace('.py', '.json'), 'w'), indent=1)

    def cli(argv):
        mp.mp.dps = 25   # set by hs3_eigen on import in the research run
        main()
    return namespace(locals(), globals())


@section
def cutoff_eigen():
    """O(1/N) rainbow eigenvalue on S^1 x H^2 with a chordal cutoff and Hadamard finite part (CCS route for delta_gamma).

Sigma(theta, chi) = D(rho) G(theta, chi),  D = C_sigma / rho^4,  rho^2 = 2(cosh chi - cos theta),  C_sigma = 16/pi^2 (units 1/N),
G = 1/(4 pi rho) times F(Z) with F = 1 (trivial defect) or (2/pi) arcsin Z, Z = cos(theta/2)/cosh(chi/2) (half twist).
Eigenvalue on the on-shell mode (s, Delta): I = int_{-pi}^{pi} dtheta cos(s theta) int_0^inf 2 pi sinh chi dchi P_{Delta-1}(cosh chi) Sigma,
cut at rho > eps; the finite part I0 is read off a fit I(eps) = a/eps^2 + b/eps + c log eps + I0 + d eps + e eps^2.
delta Delta = -I0/(2 Delta - 1), on top of the uniform gamma_phi from the Weyl factor.
usage: python cutoff_eigen.py triv|half s Delta
"""
    import sys
    import numpy as np
    import mpmath as mp
    from scipy.integrate import quad
    from scipy.interpolate import CubicSpline

    CS = 16 / np.pi ** 2


    def legendre_spline(nu, chimax=40.0, n=4000):
        if abs(nu) < 1e-14:
            return lambda c: np.ones_like(np.asarray(c, float))
        x = np.concatenate([np.linspace(0, 2, n // 2), np.linspace(2, chimax, n // 2)[1:]])
        y = np.array([float(mp.legenp(nu, 0, mp.cosh(c), type=3).real) * float(mp.exp(-max(nu, 0) * c)) for c in x])
        sp = CubicSpline(x, y)
        return lambda c: sp(c) * np.exp(max(nu, 0) * np.asarray(c, float))


    def integrand_theta(theta, chi, s, kind):
        rho2 = 2 * (np.cosh(chi) - np.cos(theta))
        rho = np.sqrt(rho2)
        G = 1 / (4 * np.pi * rho)
        if kind == 'half':
            Z = np.cos(theta / 2) / np.cosh(chi / 2)
            G = G * (2 / np.pi) * np.arcsin(np.clip(Z, -1, 1))
        return np.cos(s * theta) * CS / rho2 ** 2 * G


    def I_eps(eps, s, Delta, kind, P):
        def outer(chi):
            c = np.cosh(chi) - eps ** 2 / 2
            if c >= 1:
                lo = 0.0
            else:
                lo = np.arccos(c)
            val, _ = quad(integrand_theta, lo, np.pi, args=(chi, s, kind), epsabs=1e-14, epsrel=1e-12, limit=400)
            return 2 * val * 2 * np.pi * np.sinh(chi) * P(chi)
        ce = np.arccosh(1 + eps ** 2 / 2)
        pts = [0.0, ce, 2 * ce, 0.5, 2.0, 8.0, 30.0]
        tot = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            v, _ = quad(outer, a, b, epsabs=1e-13, epsrel=1e-11, limit=400)
            tot += v
        return tot


    def main():
        kind, s, Delta = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
        P = legendre_spline(Delta - 1)
        eps = np.array([0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.2])
        I = np.array([I_eps(e, s, Delta, kind, P) for e in eps])
        for e, v in zip(eps, I):
            print(f'  eps {e:.3f}: I = {v:.10f}', flush=True)
        for label, cols in (('with log', lambda e: [e ** -2, e ** -1, np.log(e), np.ones_like(e), e, e ** 2]),
                            ('no log', lambda e: [e ** -2, e ** -1, np.ones_like(e), e, e ** 2])):
            A = np.vstack(cols(eps)).T
            coef = np.linalg.lstsq(A, I, rcond=None)[0]
            i0 = coef[3] if label == 'with log' else coef[2]
            extra = f'  log coeff {coef[2]:+.3e}' if label == 'with log' else ''
            print(f'{kind} s={s} Delta={Delta} [{label}]: a {coef[0]:.6f} b {coef[1]:.6f}{extra}  I0 {i0:+.6e}  -> delta Delta = {-i0 / (2 * Delta - 1):+.6e}')

    def cli(argv):
        mp.mp.dps = 15   # mpmath's default, as in the research run
        with fuzzy.logged('analytic/halftwist_largeN/cutoff_eigen_' + '_'.join(argv[1:4]) + '.log'):
            main()
    return namespace(locals(), globals())


@section
def cutoff_alpha():
    """cutoff_eigen.py at general twist alpha: the O(1/N) rainbow eigenvalue of the modes s in Z + alpha on S^1 x H^2.

G_alpha = G_0 + dG, with G_0 = 1/(4 pi rho) and the regular part
  dG(theta, chi) = (1/4 pi^2) int_chi^inf dt dS(theta, t) / sqrt(2(cosh t - cosh chi)),
  dS = sum_{s in Z+alpha} e^{i s theta - |s| t} - sum_{n in Z} (same) = -expm1(a w)/expm1(w) + expm1(-a w')/expm1(-w'),
  w = i theta - t, w' = -i theta - t  (no pole at w = 0; the coincidence singularity sits entirely in G_0).
With cosh t = cosh chi cosh eta the t-measure is sqrt(cosh chi) cosh(eta/2) d eta / sinh t, smooth at eta = 0.
dG(-theta) = conj dG(theta), so the projected kernel Re[e^{-i s theta} G_alpha] is even and is integrated over [0, pi] twice.
dG is tabulated once (x e^{chi/2}) and splined; the eigenvalue then follows cutoff_eigen.py: chordal cutoff rho > eps,
fit I(eps) = a/eps^2 + b/eps (+ c log eps) + I0 + d eps + e eps^2, delta Delta = -I0/(2 Delta - 1) per C_sigma CS = 16/pi^2.
usage: python cutoff_alpha.py gate alpha | python cutoff_alpha.py run alpha s Delta
"""
    import sys
    import time
    import numpy as np
    import mpmath as mp
    from scipy.integrate import quad
    from scipy.interpolate import RectBivariateSpline
    _cutoff_eigen = globals()['cutoff_eigen'](); CS, legendre_spline = _cutoff_eigen.CS, _cutoff_eigen.legendre_spline   # was: from cutoff_eigen import CS, legendre_spline


    def cexpm1(z):
        x, y = z.real, z.imag
        return np.expm1(x) * np.cos(y) - 2 * np.sin(y / 2) ** 2 + 1j * np.exp(x) * np.sin(y)


    def eta_rule(n=48):
        xg, wg = np.polynomial.legendre.leggauss(n)
        cuts = [0, 0.5, 2, 6, 14, 30, 50, 80]
        X, W = [], []
        for a, b in zip(cuts[:-1], cuts[1:]):
            X.append((b - a) / 2 * xg + (a + b) / 2); W.append((b - a) / 2 * wg)
        return np.concatenate(X), np.concatenate(W)


    def dG_direct(theta, chi, alpha, rule):
        """dG at broadcastable theta, chi (arrays), by the eta quadrature."""
        eta, w = rule
        th, ch = np.asarray(theta, float)[..., None], np.asarray(chi, float)[..., None]
        cm1 = 2 * np.sinh(ch / 2) ** 2 * np.cosh(eta) + 2 * np.sinh(eta / 2) ** 2     # cosh t - 1, no cancellation
        t = 2 * np.arcsinh(np.sqrt(cm1 / 2))
        sh = np.sqrt(cm1 * (cm1 + 2))
        W1, W2 = 1j * th - t, 1j * th + t                                            # w and -w'
        dS = -cexpm1(alpha * W1) / cexpm1(W1) + cexpm1(alpha * W2) / cexpm1(W2)
        f = np.sqrt(np.cosh(ch)) * np.cosh(eta / 2) / sh * dS
        return (f * w).sum(-1) / (4 * np.pi ** 2)


    def table(alpha, rule):
        th = np.linspace(-0.5, np.pi, 351)
        chi = np.concatenate([np.linspace(0, 2, 201), np.linspace(2, 32, 301)[1:]])
        Z = np.empty((len(th), len(chi)), complex)
        for j, c in enumerate(chi):
            Z[:, j] = dG_direct(th, np.full_like(th, c), alpha, rule) * np.exp(c / 2)
        re = RectBivariateSpline(th, chi, Z.real, kx=3, ky=3, s=0)
        im = RectBivariateSpline(th, chi, Z.imag, kx=3, ky=3, s=0)
        return lambda t, c: (re.ev(t, c) + 1j * im.ev(t, c)) * np.exp(-c / 2)


    def gate(alpha):
        rule = eta_rule()
        # (1) alpha = 1/2 against the closed form G_0 [(2/pi) arcsin Z - 1]
        th = np.array([0.0, 0.3, 1.0, 2.0, 3.0, 0.01]); chi = np.array([0.0, 0.2, 0.5, 1.5, 4.0, 0.01])
        rho = np.sqrt(2 * (np.cosh(chi) - np.cos(th)))
        with np.errstate(divide='ignore', invalid='ignore'):
            exact = (2 / np.pi * np.arcsin(np.cos(th / 2) / np.cosh(chi / 2)) - 1) / (4 * np.pi * rho)
        exact[0] = -1 / (4 * np.pi ** 2)                      # coincidence limit, A_{1/2} = 1
        got = dG_direct(th, chi, 0.5, rule)
        print('gate 1 (alpha=1/2 vs arcsin): max |diff| %.1e, max |Im| %.1e' % (np.max(abs(got.real - exact)), np.max(abs(got.imag))))
        rule2 = eta_rule(96)
        print('gate 1b (eta rule doubled, alpha=%g): max |diff| %.1e' % (alpha, np.max(abs(dG_direct(th, chi, alpha, rule) - dG_direct(th, chi, alpha, rule2)))))
        # (2) theta-modes of G_alpha on (-pi, pi) against Q_{|s|-1/2}(cosh chi)/(4 pi^2)
        xg, wg = np.polynomial.legendre.leggauss(400)
        for c in (0.4, 1.0):
            T = np.pi * xg
            G = 1 / (4 * np.pi * np.sqrt(2 * (np.cosh(c) - np.cos(T)))) + dG_direct(T, np.full_like(T, c), alpha, rule)
            for s in (alpha, alpha - 1, alpha + 1, alpha - 2):
                m = (np.pi * wg * np.exp(-1j * s * T) * G).sum() / (2 * np.pi)
                q = float(mp.legenq(abs(s) - 0.5, 0, mp.cosh(c), type=3).real) / (4 * np.pi ** 2)
                print(f'gate 2 chi={c} s={s:+.2f}: mode {m.real:.12f} {m.imag:+.1e}   Q/4pi^2 {q:.12f}   diff {abs(m - q):.1e}')
        # (3) spline against direct, off grid
        t0 = time.time(); F = table(alpha, rule); print(f'table built [{time.time() - t0:.0f}s]')
        rng = np.random.default_rng(1)
        tt, cc = rng.uniform(0, np.pi, 200), np.concatenate([rng.uniform(0, 0.3, 100), rng.uniform(0.3, 10, 100)])
        print('gate 3 (spline vs direct, 200 off-grid points): max |diff| %.1e' % np.max(abs(F(tt, cc) - dG_direct(tt, cc, alpha, rule))))


    def I_eps(eps, s, P, F):
        def inner(theta, chi):
            rho2 = 2 * (np.cosh(chi) - np.cos(theta))
            G0 = 1 / (4 * np.pi * np.sqrt(rho2))
            return CS / rho2 ** 2 * (np.cos(s * theta) * G0 + (np.exp(-1j * s * theta) * F(theta, chi)).real)

        def outer(chi):
            c = np.cosh(chi) - eps ** 2 / 2
            lo = 0.0 if c >= 1 else np.arccos(c)
            val, _ = quad(inner, lo, np.pi, args=(chi,), epsabs=1e-14, epsrel=1e-12, limit=400)
            return 2 * val * 2 * np.pi * np.sinh(chi) * P(chi)
        ce = np.arccosh(1 + eps ** 2 / 2)
        pts = [0.0, ce, 2 * ce, 0.5, 2.0, 8.0, 30.0]
        return sum(quad(outer, a, b, epsabs=1e-13, epsrel=1e-11, limit=400)[0] for a, b in zip(pts[:-1], pts[1:]))


    def run(alpha, s, Delta, eps=(0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.2)):
        t0 = time.time()
        rule = eta_rule(); F = table(alpha, rule); P = legendre_spline(Delta - 1)
        print(f'alpha={alpha} s={s} Delta={Delta}: table [{time.time() - t0:.0f}s]', flush=True)
        eps = np.array(eps); I = []
        for e in eps:
            t1 = time.time(); I.append(I_eps(e, s, P, F)); print(f'  eps {e:.3f}: I = {I[-1]:.10f}  [{time.time() - t1:.0f}s]', flush=True)
        I = np.array(I)
        out = []
        for label, cols in (('with log', lambda e: [e ** -2, e ** -1, np.log(e), np.ones_like(e), e, e ** 2]),
                            ('no log', lambda e: [e ** -2, e ** -1, np.ones_like(e), e, e ** 2]),
                            ('no log, cubic', lambda e: [e ** -2, e ** -1, np.ones_like(e), e, e ** 2, e ** 3])):
            for sub in (slice(None), slice(1, None), slice(None, -1)):
                A = np.vstack(cols(eps[sub])).T
                coef = np.linalg.lstsq(A, I[sub], rcond=None)[0]
                i0 = coef[3] if label == 'with log' else coef[2]
                out.append(-i0 / (2 * Delta - 1))
                extra = f' log {coef[2]:+.1e}' if label == 'with log' else ''
                print(f'  [{label:13s} eps {eps[sub][0]:.2f}-{eps[sub][-1]:.2f}] I0 {i0:+.8f}{extra} -> N delta Delta = {-i0 / (2 * Delta - 1):+.6f}')
        g = 4 / (3 * np.pi ** 2)
        out = np.array(out)
        print(f'alpha={alpha} s={s} Delta={Delta}: N delta Delta = {np.mean(out):+.6f} (spread {np.ptp(out):.1e}) = {np.mean(out) / g:+.5f} gamma_phi  [{time.time() - t0:.0f}s]')


    def main():
        if sys.argv[1] == 'gate':
            gate(float(sys.argv[2]))
        else:
            run(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))

    def cli(argv):
        mp.mp.dps = 15   # mpmath's default, as in the research run
        with fuzzy.logged('analytic/halftwist_largeN/cutoff_alpha_' + '_'.join(argv[1:5]) + '.log'):
            main()
    return namespace(locals(), globals())


STEPS = {'hs1_ingredients': hs1_ingredients, 'hs2_local': hs2_local, 'hs3_eigen': hs3_eigen, 'hs4_calib': hs4_calib, 'hs5_twisted': hs5_twisted, 'hs6_alpha': hs6_alpha, 'cutoff_eigen': cutoff_eigen, 'cutoff_alpha': cutoff_alpha}
REPORT = [('hs1_ingredients',), ('hs2_local',), ('hs4_calib',), ('hs5_twisted',), ('hs6_alpha',),
          ('cutoff_eigen', 'triv', '1', '1.5'), ('cutoff_eigen', 'triv', '2', '2.5'),
          ('cutoff_eigen', 'half', '0.5', '1'), ('cutoff_eigen', 'half', '1.5', '2'), ('cutoff_eigen', 'half', '2.5', '3')]

if __name__ == '__main__':
    fuzzy.cli(__file__, STEPS, REPORT)
