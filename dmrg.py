"""DMRG for the same Hamiltonians (TeNPy 0.10): one fermion mode per site, ordered by m, with U(1) and Z2 charges,
so each run gives the lowest state of one sector (dmrg_twist.py), and a resumable parallel job runner
(dmrg_jobs.py). Job lists are stored in the question files, under keys twist/jobs_*.json.

usage: python dmrg.py <jobs key, e.g. twist/jobs_o2wf.json> <nworkers> <threads_per_worker>
"""
import os
import sys
import tempfile
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import fuzzy                                             # noqa: E402
from fuzzy import store                                   # noqa: E402

# ----------------------------------------------------------------------------------------------------
# from twist/dmrg_twist.py
"""DMRG (TeNPy 0.10) for the fuzzy-sphere monodromy defects.

One fermion mode per chain site, sites ordered by orbital angular momentum m (so momentum-conserving
interactions are as local as they can be). Conserved U(1) charges per mode are passed in, and TeNPy
blocks the MPS by them, so each run targets one (N_e, L_z, flavour-charge) sector and DMRG returns the
lowest state there. The Hamiltonian terms come unchanged from the ED builders (twist.py / o2twist.py),
so the ED spectra validate this code directly.
"""
import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.algorithms import dmrg


def fermion_site(charges, names, mods=None):
    """Single fermion mode; |1> carries `charges` under the charges `names` (mod 1 = U(1), 2 = Z2)."""
    chinfo = npc.ChargeInfo(mods or [1] * len(names), names)
    leg = npc.LegCharge.from_qflat(chinfo, [[0] * len(names), list(charges)])
    C = np.array([[0., 1.], [0., 0.]])
    Cd = C.T.copy()
    Nn = np.diag([0., 1.])
    JW = np.diag([1., -1.])
    site = Site(leg, ['empty', 'full'], C=C, Cd=Cd, N=Nn, JW=JW)
    site.need_JW_string |= {'C', 'Cd'}
    return site


def build_model(mode_m, mode_charges, charge_names, terms, onsite, mods=None):
    """mode_m[k]: angular momentum of ED mode k; terms: (k1,k2,k3,k4,v) for c+_k1 c+_k2 c_k3 c_k4;
    onsite[k]: coefficient of n_k.  Returns (model, order) with order[pos] = ED mode index."""
    order = sorted(range(len(mode_m)), key=lambda k: (mode_m[k], k))
    pos = {k: i for i, k in enumerate(order)}
    sites = [fermion_site(mode_charges[k], charge_names, mods) for k in order]
    from tenpy.models.lattice import TrivialLattice
    lat = TrivialLattice(sites, bc_MPS='finite')
    model = CouplingModel(lat)
    li = lambda p: list(lat.mps2lat_idx(p))
    for (k1, k2, k3, k4, v) in terms:           # add_local_term handles JW strings and reordering
        model.add_local_term(v, [('Cd', li(pos[k1])), ('Cd', li(pos[k2])),
                                 ('C', li(pos[k3])), ('C', li(pos[k4]))])
    for k, v in enumerate(onsite):
        if v != 0:
            model.add_onsite_term(v, pos[k], 'N')
    H = model.calc_H_MPO()
    return MPOModel(lat, H), order


def product_state(order, occupied_modes):
    return ['full' if k in occupied_modes else 'empty' for k in order]


def run_dmrg(model, order, occupied_modes, chi_max=400, sweeps=12, svd_min=1e-10, verbose=0,
             orthogonal_to=None):
    """Lowest state in the sector of `occupied_modes`; with orthogonal_to=[psi0, ...] the lowest state
    orthogonal to those (excited states within the same sector)."""
    psi = MPS.from_product_state(model.lat.mps_sites(), product_state(order, set(occupied_modes)),
                                 bc='finite')
    params = {'mixer': True, 'max_E_err': 1e-9, 'max_S_err': 1e-6, 'min_sweeps': 4,
              'max_sweeps': sweeps, 'trunc_params': {'chi_max': chi_max, 'svd_min': svd_min},
              'chi_list': {0: min(64, chi_max), 2: min(160, chi_max), 4: chi_max},
              'combine': True}
    eng = dmrg.TwoSiteDMRGEngine(psi, model, params, orthogonal_to=orthogonal_to or [])
    E, psi = eng.run()
    return E, psi, eng


def compress_mpo(H, rel_tol=1e-12):
    """SVD compression of a finite MPO, dropping singular values < rel_tol * max on each bond (numerically
    lossless at rel_tol = 1e-12, but a truncation, not an exact algebraic compression).

    Left-to-right QR puts the MPO in left-canonical form (Frobenius norm), then a right-to-left SVD
    truncates.  The finite-state-machine identity indices are lost on interior bonds (IdL/IdR = None
    there); TeNPy's DensityMatrixMixer handles that case.
    """
    from tenpy.networks.mpo import MPO
    Ws = [W.copy() for W in H._W]
    L = len(Ws)
    # project the boundary bonds onto their boundary states (TeNPy keeps dim 2 there: IdL and IdR)
    m0 = np.zeros(Ws[0].get_leg('wL').ind_len, bool); m0[H.IdL[0]] = True
    Ws[0].iproject(m0, 'wL')
    mL = np.zeros(Ws[-1].get_leg('wR').ind_len, bool); mL[H.IdR[L]] = True
    Ws[-1].iproject(mL, 'wR')
    for i in range(L - 1):
        W = Ws[i].combine_legs([['wL', 'p', 'p*'], ['wR']], qconj=[+1, -1])
        Q, R = npc.qr(W, inner_labels=['wR', 'wL'])
        Ws[i] = Q.split_legs([0]).itranspose(['wL', 'wR', 'p', 'p*'])
        R = R.split_legs([1])
        Ws[i + 1] = npc.tensordot(R, Ws[i + 1], axes=('wR', 'wL')).itranspose(['wL', 'wR', 'p', 'p*'])
    for i in range(L - 1, 0, -1):
        W = Ws[i].combine_legs([['wL'], ['wR', 'p', 'p*']], qconj=[+1, -1])
        U, S, VH = npc.svd(W, inner_labels=['wR', 'wL'])
        keep = S > rel_tol * S.max()
        U.iproject(keep, 'wR')
        VH.iproject(keep, 'wL')
        S = S[keep]
        Ws[i] = VH.split_legs([1]).itranspose(['wL', 'wR', 'p', 'p*'])
        US = U.scale_axis(S, 'wR').split_legs([0])
        Ws[i - 1] = npc.tensordot(Ws[i - 1], US, axes=('wR', 'wL')).itranspose(['wL', 'wR', 'p', 'p*'])
    return MPO(H.sites, Ws, bc='finite', IdL=[0] + [None] * L, IdR=[None] * L + [0])


# ----------------------------------------------------------------------------------------------------
# from twist/test_dmrg_ising.py
"""Validate TeNPy DMRG against ED for the Ising twist defect (and the untwisted bulk)."""
import sys
import time
import numpy as np
from fuzzy import couplings_for, orbital_ms, interaction_terms, TwistBasis, build_H_fast
import scipy.sparse.linalg as sla

def ising_setup(N, twisted, h=3.16):
    G0, G1, _ = couplings_for(N, 4.75, 1.0)
    mp, mm = orbital_ms(N, False), orbital_ms(N, twisted)
    mode_m = list(mp) + list(mm)
    charges = [(1, int(round(2 * m)), 0) for m in mp] + [(1, int(round(2 * m)), 1) for m in mm]
    terms = interaction_terms(N, twisted, G0, G1)
    onsite = [-h] * len(mp) + [+h] * len(mm)
    return mode_m, charges, terms, onsite, len(mp)

def occupied_for(N, twisted, twoLz, z2):
    """Product state in the target sector: fill all + orbitals, then move electrons to - orbitals."""
    s = (N - 1) / 2
    mp, mm = orbital_ms(N, False), orbital_ms(N, twisted)
    occ = set(range(len(mp)))
    if z2 == -1:
        # move one electron from + orbital a to - orbital b with 2(m_b - m_a) = twoLz
        for a, ma in enumerate(mp):
            for b, mb in enumerate(mm):
                if int(round(2 * (mb - ma))) == twoLz:
                    return (occ - {a}) | {len(mp) + b}
    else:
        if twoLz == 0:
            return occ
        # hop one + electron to ... need a free + orbital: use two - electrons instead
        for a1, ma1 in enumerate(mp):
            for a2, ma2 in enumerate(mp):
                if a2 <= a1: continue
                for b1, mb1 in enumerate(mm):
                    for b2, mb2 in enumerate(mm):
                        if b2 <= b1: continue
                        if int(round(2 * (mb1 + mb2 - ma1 - ma2))) == twoLz:
                            return (occ - {a1, a2}) | {len(mp) + b1, len(mp) + b2}
    raise ValueError



# ----------------------------------------------------------------------------------------------------
# from twist/dmrg_jobs.py
"""Parallel DMRG jobs for the fuzzy-sphere monodromy defects (Ising twist and O(2) at angle alpha).

A job = lowest state in one symmetry sector. Results go to dmrg_results/<name>.json.
usage:  python dmrg_jobs.py <jobfile.json> <nworkers> <threads_per_worker>
"""
import os
import sys
import json
import time
import subprocess
import numpy as np

HERE = 'twist'
OUT = os.path.join(HERE, 'dmrg_results')
LOCKS = tempfile.gettempdir()


def job_name(j):
    if j['kind'] in ('ising', 'o2'):
        return f"{j['kind']}_N{j['N']}_a{j['alpha']}_s{j['sector'][0]}_{j['sector'][1]}_chi{j['chi']}"
    q, K, par = j['sector']                       # O(N) models of onmodel.py: [pair charges, K, parities]
    flat = '_'.join(str(x) for x in list(q) + [K] + list(par))
    return f"{j['kind']}_N{j['N']}_a{j['alpha']}_s{flat}_chi{j['chi']}"


# ------------------------------------------------------------------------------------------------
def ising_problem(N, alpha, sector):
    from fuzzy import couplings_for, orbital_ms, interaction_terms
    twisted = alpha != 0
    h = 3.16
    G0, G1, _ = couplings_for(N, 4.75, 1.0)
    mp, mm = orbital_ms(N, False), orbital_ms(N, twisted)
    mode_m = list(mp) + list(mm)
    charges = [(1, int(round(2 * m)), 0) for m in mp] + [(1, int(round(2 * m)), 1) for m in mm]
    terms = interaction_terms(N, twisted, G0, G1)
    onsite = [-h] * len(mp) + [+h] * len(mm)
    pass
    occ = occupied_for(N, twisted, sector[0], sector[1])
    return mode_m, charges, ['N', '2Lz', 'Z2'], [1, 1, 2], terms, onsite, occ


def o2_problem(N, alpha, sector, D):
    from fuzzy import couplings_for
    orbitals = fuzzy.o2twist().orbitals
    build_terms = fuzzy.o2twist().build_terms
    G0, G1, _ = couplings_for(N, 4.0, 1.0)
    orb = orbitals(N, alpha)
    tw = 1 if alpha != 0 else 0
    off = [0, len(orb[0]), len(orb[0]) + len(orb[1])]
    mode_m, charges = [], []
    Qf = [+1, 0, -1]
    for f in range(3):
        for j, m in enumerate(orb[f]):
            mode_m.append(m)
            charges.append((1, j + (tw if f == 2 else 0), Qf[f]))
    two, U = build_terms(N, alpha, G0, G1)
    onsite = []
    for f in range(3):
        for j in range(len(orb[f])):
            onsite.append(U[f][j] + (D if f != 1 else 0.0))
    # product state: all electrons in flavour 0 (Q=0, K=0); then promote to reach (Q, K)
    Q, K = sector
    s = (N - 1) / 2
    base = [off[1] + j for j in range(N)]                 # flavour-0 modes, all filled
    Kc = lambda occ: sum(charges[k][1] for k in occ) - int(round(s * N))
    Qc = lambda occ: sum(charges[k][2] for k in occ)
    occ0 = set(base)
    cands = [occ0]
    # single promotions 0 -> +1 or 0 -> -1, and pair promotions
    f_modes = {f: [off[f] + j for j in range(len(orb[f]))] for f in range(3)}
    import itertools
    for a in base:
        for f in (0, 2):
            for b in f_modes[f]:
                cands.append((occ0 - {a}) | {b})
    for a1, a2 in itertools.combinations(base, 2):
        for b1 in f_modes[0]:
            for b2 in f_modes[2]:
                cands.append((occ0 - {a1, a2}) | {b1, b2})
        for b1, b2 in itertools.combinations(f_modes[0], 2):
            cands.append((occ0 - {a1, a2}) | {b1, b2})
    occ = next(c for c in cands if Qc(c) == Q and Kc(c) == K)
    return mode_m, charges, ['N', 'K', 'Q'], [1, 1, 1], two, onsite, occ


def on_problem(name, N, alpha, sector, h):
    """O(N) models of onmodel.py (configs in run_on_defect.CONFIGS); sector = [pair charges, K, parities]."""
    import itertools
    M = fuzzy.onmodel()
    CONFIGS = fuzzy.run_on_defect().CONFIGS
    cfg = CONFIGS[name]
    fl, orbs, koff, two, mu = M.build_terms(cfg['model'], cfg['Nv'], N, alpha, cfg['U'], h,
                                            pairs=cfg['pairs'], twist=cfg['twist'])
    off = np.concatenate([[0], np.cumsum([len(o) for o in orbs])]).astype(int)
    npair, par_fl = len(fl.pairs), list(fl.unpaired)
    mode_m, charges, fmode = [], [], []
    for f in range(fl.F):
        for jj, m in enumerate(orbs[f]):
            mode_m.append(m); fmode.append((f, jj))
            charges.append(tuple([1, jj + koff[f]] + [int(fl.qpair[f, p]) for p in range(npair)]
                                 + [1 if f == g else 0 for g in par_fl]))
    names = ['N', 'K'] + [f'Q{p}' for p in range(npair)] + [f'P{i}' for i in range(len(par_fl))]
    mods = [1, 1] + [1] * npair + [2] * len(par_fl)
    q_t, K_t, par_t = sector
    par_t = [0 if x > 0 else 1 for x in par_t]
    s = (N - 1) / 2
    def tag(occ):
        c = np.array([charges[k] for k in occ]).sum(0)
        return (list(c[2:2 + npair]), int(c[1] - round(s * N)), [int(x) % 2 for x in c[2 + npair:]])
    base = set(range(N))                           # psi_0 filled: the paramagnet, K = 0
    others = [k for k in range(len(mode_m)) if k >= N]
    for npro in range(0, 4):
        for rem in itertools.combinations(sorted(base), npro):
            for add in itertools.combinations(others, npro):
                occ = (base - set(rem)) | set(add)
                qq, KK, pp = tag(occ)
                if qq == list(q_t) and KK == K_t and pp == par_t:
                    return mode_m, charges, names, mods, two, list(mu), occ
    raise RuntimeError('no product state in sector')


# ------------------------------------------------------------------------------------------------
def run_job(j):
    from tenpy.models.model import MPOModel
    pass
    t0 = time.time()
    if j['kind'] == 'ising':
        prob = ising_problem(j['N'], j['alpha'], tuple(j['sector']))
    elif j['kind'] == 'o2':
        prob = o2_problem(j['N'], j['alpha'], tuple(j['sector']), j['D'])
    else:
        prob = on_problem(j['kind'], j['N'], j['alpha'], j['sector'], j['h'])
    mode_m, charges, names, mods, terms, onsite, occ = prob
    shift = 0.0
    if j.get('nstates', 1) > 1:
        # excited states use a projector, whose null space sits at energy 0; shift every level (N_e is
        # fixed in a sector) far below 0 so the projected-out direction is never the lowest
        shift = 2.0
        onsite = [v - shift for v in onsite]
    model, order = build_model(mode_m, charges, names, terms, onsite, mods=mods)
    Hc = compress_mpo(model.H_MPO)
    mc = MPOModel(model.lat, Hc)
    tb = time.time() - t0
    E, psi, eng = run_dmrg(mc, order, occ, chi_max=j['chi'], sweeps=j.get('sweeps', 14))
    st = eng.sweep_stats
    E = E + shift * len(occ)
    res = dict(j, E=float(E), chi_reached=int(max(psi.chi)), mpo_chi=int(max(Hc.chi)),
               trunc=float(st['max_trunc_err'][-1]), E_sweeps=[float(x) for x in st['E']],
               t_build=tb, t_total=time.time() - t0)
    if j['kind'] == 'o2':
        # Hellmann-Feynman input for tuning D: dE/dD = <N_D>, N_D = occupation of the charged flavours
        nexp = psi.expectation_value('N')
        res['nD'] = float(sum(nexp[p] for p in range(len(order)) if charges[order[p]][2] != 0))
    if j.get('nstates', 1) > 1:                   # excited states in the same sector
        states, Es, trs = [psi], [float(E)], [float(st['max_trunc_err'][-1])]
        for _ in range(j['nstates'] - 1):
            E2, psi2, eng2 = run_dmrg(mc, order, occ, chi_max=j['chi'], sweeps=j.get('sweeps', 14),
                                      orthogonal_to=list(states))
            states.append(psi2); Es.append(float(E2 + shift * len(occ)))
            trs.append(float(eng2.sweep_stats['max_trunc_err'][-1]))
        res.update(E_list=Es, trunc_list=trs, t_total=time.time() - t0)
    store.put(os.path.join(OUT, job_name(j) + '.json'), res)
    return res


if __name__ == '__main__':
    if sys.argv[1] == '--one':                      # worker mode: run a single job given as JSON
        j = json.loads(sys.argv[2])
        # skip if another runner finished or is running this job (lock file holds the worker's pid)
        lock = os.path.join(LOCKS, job_name(j) + '.running')
        if store.exists(os.path.join(OUT, job_name(j) + '.json')):
            print(f"SKIP {job_name(j)} (done)", flush=True); sys.exit(0)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            txt = open(lock).read().strip()
            try:
                os.kill(int(txt), 0)
                print(f"SKIP {job_name(j)} (running)", flush=True); sys.exit(0)
            except (ValueError, ProcessLookupError):       # stale lock from a killed worker
                if not txt:
                    sys.exit(0)
                os.remove(lock)
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        try:
            r = run_job(j)
        finally:
            os.remove(lock)
        print(f"DONE {job_name(j)} E={r['E']:.9f} trunc={r['trunc']:.1e} chi={r['chi_reached']} "
              f"mpo={r['mpo_chi']} t={r['t_total']:.0f}s", flush=True)
        sys.exit(0)
    jobs = store.get(sys.argv[1])
    store.target = store._where[store.key(sys.argv[1])]
    nw, nt = int(sys.argv[2]), int(sys.argv[3])
    env = dict(os.environ, FUZZY_TARGET=store.target, OMP_NUM_THREADS=str(nt), OPENBLAS_NUM_THREADS=str(nt),
               MKL_NUM_THREADS=str(nt), VECLIB_MAXIMUM_THREADS=str(nt))
    todo = [j for j in jobs if not store.exists(os.path.join(OUT, job_name(j) + '.json'))]
    running = []
    py = sys.executable
    while todo or running:
        while todo and len(running) < nw:
            j = todo.pop(0)
            p = subprocess.Popen([py, __file__, '--one', json.dumps(j)], env=env, cwd=ROOT)
            running.append((p, j))
        time.sleep(5)
        for p, j in list(running):
            if p.poll() is not None:
                running.remove((p, j))
                if p.returncode != 0:
                    print(f"FAILED {job_name(j)} rc={p.returncode}", flush=True)
    print("ALL JOBS FINISHED", flush=True)
