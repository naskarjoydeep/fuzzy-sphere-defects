# fuzzy-sphere-defects

Supporting code and data for the paper *Monodromy Defects and Bulk Renormalization Group Flows on the Fuzzy Sphere*
by Kimyeong Lee and Joydeep Naskar, arXiv:2611.XXXXX.

We put monodromy (twist) line defects of the 3d Ising, O(2), O(3) and O(4) critical points on the fuzzy sphere by
giving the twisted fermion flavours a fractional Aharonov–Bohm flux through the poles. The lowest Landau level then
keeps only the orbitals that are regular at both poles, and every matrix element is a closed-form Beta-function
integral. We diagonalize the resulting Hamiltonians exactly (ED) or with DMRG.

## Citation

If you use this code or data, please cite:

```bibtex
@article{Lee:2026xxx,
    author = "Lee, Kimyeong and Naskar, Joydeep",
    title = "{Monodromy Defects and Bulk Renormalization Group Flows on the Fuzzy Sphere}",
    eprint = "XXXX.XXXXX",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    year = "2026"
}
```

## One file per physics question

| file | question |
|---|---|
| `fuzzy.py` | the engine: LLL orbitals and Fock bases, the Ising, O(N)-vector and spin-one Hamiltonians with fractional flux, bulk tuning, the regulator's local potential, the anisotropy-flow models |
| `dmrg.py` | the same Hamiltonians for TeNPy, and a resumable parallel job runner |
| **`01_known_results/`** | **monodromy defects at the bulk fixed points**. These overlap with Qian (arXiv:2610.27926) and Tang–Zhu (arXiv:2610.31269), which appeared while this work was in progress |
| `ising_twist.py` | What is the Z2 twist line of the 3d Ising CFT? |
| `o2_twist.py` | What is the O(2) monodromy line at a general angle, and does it depend on the regulator? |
| `o3_o4_twists.py` | How do the O(3) and O(4) monodromy lines compare with O(2)? |
| `free_control.py` | Does the method reproduce the exact free monodromy defect? |
| **`02_rg_flow/`** | **monodromy defects along bulk RG flows (the new results)** |
| `trajectory.py` | Where is the critical surface along the anisotropy flows O(3) → O(2), O(3) → Ising and O(4) → O(3)? |
| `easy_plane.py` | Is the monodromy transported onto the O(2) half-twist, couplings included? |
| `easy_axis.py` | Is the monodromy erased, leaving a trivial line? |
| `cpt.py` | Does conformal perturbation theory account for the response of the bulk to the anisotropy? |
| **`03_spectator_entropy/`** | **spectators and the defect entropy** |
| `defect_entropy.py` | Does a spectator raise the defect entropy at large N, and why is the answer fixed by the conformal weight? |
| `entropy_near_four.py` | Near four dimensions, does a spectator still raise the defect entropy, and by how much? |
| `spectator_response.py` | How do the untwisted spectators respond to a partial twist, and is the twisted line stable? |
| `cap_entanglement.py` | Can cap entanglement on the fuzzy sphere measure the defect entropy? |
| **`04_pinned_lines/`** | **pinning, junctions and line stability** |
| `junctions.py` | How large is the domain-wall dimension of a pinned O(N) line, measured on the lattice? |
| `ward_stiffness.py` | What is the stiffness of a pinned line, and why does the small-angle curvature converge so slowly? |
| `wall_correction.py` | Which correction law controls the extrapolation of the wall? |
| `junction_theory.py` | What do the epsilon expansion and large N predict for the junction of pinned lines? |
| `monodromy_pinning.py` | Where does pinning a charged mode of the O(2) monodromy lead, and how does the twist change its walls? |
| `halftwist_pinning.py` | Does the pinned partial twist survive at half twist at large N, and how do the lowest twisted modes shift at order 1/N? |
| `pinned_flow.py` | Does pinning commute with the bulk flows, and which pin survives which branch? |
| `pinned_entropy.py` | Can an ordered line of pinned monodromy lines disorder into the unpinned line? Walls against entropy at large N |
| **`05_five_dimensions/`** | **a construction in five dimensions** |
| `fuzzy_s4.py` | Can a monodromy defect be built on the fuzzy four-sphere, and how close is it to the free five-dimensional defect? |

Every question file has a data file of the same name (`ising_twist.json`, …). It holds every result the paper uses
for that question, so nothing has to be recomputed to check a number. The exception is `05_five_dimensions`, whose data
file holds only the model couplings its steps start from; the five-dimensional results are not included.

## Usage

```bash
pip install -r requirements.txt
python fuzzy.py test                         # self-tests of the engine
python 02_rg_flow/easy_plane.py help         # the steps of one question
python 02_rg_flow/easy_plane.py report       # the paper's numbers, from the stored data
python 02_rg_flow/easy_plane.py dflow 6 0.5  # recompute one step (here the defect along the flow at N_m = 6)
```

We ran everything with Python 3.11, numpy 1.25, scipy 1.16, numba 0.58, sympy 1.14, mpmath 1.3 and TeNPy 0.10.
TeNPy is needed only by `dmrg.py` and the DMRG steps.

## How the files are organized

- **Sections.** Each question file is a set of *sections*, one per original research script. The code inside a
  section is that script, unchanged except for how it reads and writes results. A section runs on first use and
  keeps its names to itself, so different sections can reuse names. Run a section with
  `python <question>.py <section> [args]`, where the arguments are those the original script took; its docstring
  gives them.
- **Stored results.** Results are stored under the path they had in the research tree (`twist/...` for the
  fixed-point defects, `flow/...` for the flows), which is also how the docstrings refer to them. Results computed
  by a step are written into the data file of the question being run, so delete a key there to force a
  recomputation.
- **Notation.** As in the paper, N counts vector components and N_m counts orbitals on S^2. Step arguments keep
  the research scripts' names, where N is the number of orbitals.
- **Units.** Dimensions are in bulk units `gamma_B = E(dphi) - E(phi)` of the untwisted model at the same size, or
  in displacement units, where the lowest neutral spin-one defect level is set to 2. Each section says which.
- **DMRG jobs.** Job lists are stored under `twist/jobs_*.json`, for example
  `python dmrg.py twist/jobs_o2wf.json 2 4`.
- **Five dimensions.** The research scripts of `05_five_dimensions` used paths relative to their folder `s4d/`, so
  `fuzzy_s4.py` sets the store's base to `s4d/`. Their large intermediate files (twisted bands `band_I*.npz`, 12–56 MB;
  interaction terms `terms_*.pkl`, up to 13 MB; compressed MPOs) are not stored: `save_band`, `build_I` and `build_I3`
  rebuild them into `05_five_dimensions/s4d_files/`, which git ignores, and the other steps read them there.
  `fuzzy_s4.json` holds only the couplings of the models (`s4d/i3A.json`, `s4d/i3B.json`, `s4d/i4A.json`,
  `s4d/designB1c_I2*.json`), and the steps add their results to it.
- **Data sizes.** Most data files are small. The largest, `defect_entropy.json` (14 MB), holds the large-N fluctuation
  tables.
- **Code written for this package.** Everything is research code merged verbatim, except three short blocks that only
  read stored results and print the paper's numbers: `cap_entanglement.py cap_alpha_scan` (the inline code that fitted
  the stored twist scan was not kept), `fuzzy_s4.py s4_summary` (which needs the five-dimensional results, not
  included here) and `free_control.py cd_series`. Each says so in its docstring.

## Results and where they come from

### 01 — fixed-point defects (paper, Appendix A)

| result | step | stored as |
|---|---|---|
| Ising twist spectrum, ED N_m ≤ 14 and DMRG to N_m = 20, in bulk and displacement units | `ising_twist.py run_defect`, `dmrg.py`, `ising_twist.py dmrg_tables` | `twist/run_N*.log`, `twist/dmrg_results/ising_*` |
| Ising even scalar over odd mode, 2.46 at N_m = 14 | `ising_twist.py ising_levels 14 10` | `twist/zgh_results/ising_levels_N14.json` |
| Ising beta and defect three-point coefficients, N_m = 8–14 | `ising_twist.py ising_amplitudes N` | `.../ising_amplitudes/N*.json` |
| O(2) charged mode and alpha_c in two regulators | `o2_twist.py run_on_defect o2wf N alphas`, `o2_twist.py run_o2defect`, `o2_twist.py on_tables` | `twist/on_results/o2wf_*`, `twist/dmrg_results/o2*` |
| Two-mode energies against one loop at any alpha, with the free control at the same angles (Fig. 8) | `o2_twist.py run_on_defect o2gauss N 0.35,0.45` (N = 6–10), `o2_twist.py contact_alpha` | `twist/on_results/o2gauss_N*`, `.../contact_alpha.json` |
| Pole estimates of defect three-point coefficients | `o2_twist.py pole_ope o2wf N alpha [lam]` | `.../pole_ope/*` |
| O(3) one-plane, centre and reflection twists; O(4) two-plane twist | `o3_o4_twists.py run_on_defect o3wf N alphas`, `o3_o4_twists.py o3_twists centre N` | `twist/on_results/o3wf_*`, `.../o3_twists/*` |
| Free-scalar control: beta_s^2, correlator, response to the local potential | `free_control.py defect_corr run N [lam]`, `free_control.py core_scaling run o2gauss Ns` | `twist/defect_corr_*`, `twist/core_scaling_*` |
| Free displacement normalization C_D from the projected rotation, ED to N_m = 10 and DMRG at N_m = 12, 14 (App. C.2) | `free_control.py free_cd`, `free_control.py disp_tilt run o2gauss_lam0 N`, `free_control.py dmrg_ward o2gauss N chi --lam 0`, `free_control.py cd_series` | `twist/disp_results/tilt_o2gauss*`, `twist/dmrg_ward_results/*`, `twist/free_cd.json` |

Representative values:

- **Ising Z2 twist.**
  - The lowest odd mode is 0.9227 (N_m = 10) and 0.9184 (N_m = 20) in displacement units, against Monte Carlo
    0.9187(6).
  - Its bulk-to-defect coefficient is 0.862, flat over N_m = 8–14.
- **O(2) half-twist.**
  - The lowest charged mode is 0.9065 (vector, N_m = 14) and 0.8798 (spin-one, N_m = 14).
  - It becomes relevant at alpha_c = 0.3936 (vector) and 0.3949 (spin-one), both at N_m = 14.
- **Free scalar, N_m = 10, local potential removed.** beta_1/2^2 = 0.710 and beta_3/2^2 = 0.431, against the exact
  2/pi and 4/(3 pi).
- **Free displacement.** C_D rises from 0.0059 (N_m = 6) to 0.0276 (N_m = 14), 12% to 54% of the exact 1/(2 pi^2); an
  N_m^-1 fit of the last three sizes gives 1.02 times the exact value.

### 02 — RG flows (paper, Section 3, Appendices B and C)

| result | step | stored as |
|---|---|---|
| critical trajectory, both branches, N_m = 5–9 | `trajectory.py partB_flow N g1,g2,... const` | `flow/partB_const_N*.json` |
| O(4) → O(3) trajectory | `trajectory.py flow_run o4a A N...`, then `... B N g1,g2,...` | `flow/gf_o4a_*` |
| trajectory controls (App. C.1) | `trajectory.py traj_validate A\|B\|C N`, `trajectory.py traj_analyze`, `trajectory.py retune_test N` | `flow/traj_*`, `flow/retune_*` |
| defect levels along the flow, endpoint against O(2) (Fig. 2(a), Table 1) | `easy_plane.py dflow N 0.5`, `easy_plane.py dflow_analyze`; the N_m = 9 scan ran as `dflow 9 0.5 lite`, and `easy_plane.py dflow_fill_B0 9 0.5 -0.09` adds its charge-two level | `flow/dflow_a0.500_N*`, `flow/dflow_fill_B0_a0.500_N9.json`, `flow/dflow_report.txt` |
| beta along the flow, with extrapolations (Fig. 2(b), Table 5) | `easy_plane.py dflow_c ope N`, `easy_plane.py dflow_c_endpoint run\|report` | `flow/dflow_c_*` |
| universal crossover in Y = \|g\| N_m^{y_t/2} (Fig. 9(a,b)) | `easy_plane.py crossover_collapse` | `flow/crossover_collapse.json` |
| O(4) → O(3) endpoint against the O(3) one-plane half-twist | `easy_plane.py dflow_c o4 N` | `flow/dflow_c_o4_N*` |
| frozen-limit shifts and the local-potential subtraction (Figs. 2(c), 9(c)) | `easy_axis.py dflow_frozen N`, `easy_axis.py core_residue frozen N lams`, `easy_axis.py core_residue_analyze` | `flow/core_frozen_*`, `flow/core_residue_report.txt` |
| pinned-spectator domain wall | `easy_axis.py scan`, `easy_axis.py core_residue pin N lams`, `easy_axis.py core_pin_full Ns` | `twist/spectator_pinning/results/*`, `flow/core_pin_*` |
| overlap g-function of the residual line, and its controls | `easy_axis.py gfunc_core N lams`, `easy_axis.py gfunc_core pin N`, `easy_axis.py gfunc_ising Ns` | `flow/gfunc_core_*`, `twist/gfunc_ising_results.json` |
| CPT slopes, curvature and predictions (Table 7) | `cpt.py partA_slopes N`, `cpt.py cpt_lattice N`, `cpt.py cpt_scheme` | `flow/partA_*`, `flow/cpt_*` |
| amplitudes of the nearly null descendant, to N_m = 16 | `cpt.py imp_thermal N`, `cpt.py imp_admixture`, `cpt.py dmrg_box ...`, `cpt.py box_converge analyze` | `flow/imp_*`, `flow/dmrg_box*`, `flow/box_converge_summary.json` |
| beta_1/2 of the half-twist through O(eps^2) (App. A, eq:betaeps2) | `easy_plane.py beta_eps`, `beta_check3d`, `sunset_tensor`, `beta_eps2`, `beta_eps2_modes` | `analytic/*.log`, `analytic/open_questions/*.log` |

Representative values:

- **Easy-plane endpoint at N_m = 8, g = -0.09.** The charge-one, charge-two, displacement-candidate and neutral gaps
  are (0.9402, 2.0135, 2.0478, 1.8963). The independent O(2) model gives (0.9406, 2.0176, 2.0522, 1.8984). The
  charge-two gaps differ from O(2) by -0.262, -0.229, -0.202 and -0.180% at N_m = 6–9.
- **beta.** At the endpoint it is 0.8514, 0.8536, 0.8551, 0.8562 over N_m = 6–9, against 0.8503, 0.8527, 0.8543,
  0.8555 in the O(2) model. Both extrapolate to 0.865–0.875.
- **Easy axis at N_m = 9.** The residual line has 1 - g = 4.2 × 10^-5. Pinning phi_3 in the same Hamiltonian gives
  g = 0.598.
- **beta through O(eps^2).** At eps = 1, expanded strictly, beta_1/2 = 0.8480 (O(2)) and 0.8453 (O(3)): the ultraviolet
  O(3) coupling lies 0.0027 below the O(2) one.

### 03 — spectators and the defect entropy (paper, Section 4, Appendix D)

| result | step | stored as |
|---|---|---|
| large-N fluctuation trace by sector, its cutoff sums and c = (64/3 pi) h_free (Table 9(a), Fig. 3(b)) | `defect_entropy.py c1 ...`, `c1_run sum ...`, `alpha a L`, `largeN_traces` | `largeN/c1_*.npz`, `largeN/*.log` |
| the fixed-sector identity, the full twist of Giombi et al., regulator and d = 7/2 checks | `defect_entropy.py sector_* tadpole_* PQ_* fulltwist c1_general_d b1_blocks ... b6_sensitivity` | `largeN/*.log`, `analytic/covariant_entropy/*.log` |
| the partial twist near d = 4 through O(eps^2), the spectator channel, eps_c (Tables 9(b), 10) | `entropy_near_four.py entropy_eps`, `entropy_bf_channel`, `eps2_*`, `entropy_tables` | `analytic/open_questions/*.log` |
| neutral line spectrum of a partial twist (Table 8) | `spectator_response.py singlet_channel`, `singlet_check` | `largeN/singlet_channel.log`, `twist/singlet_check.log` |
| spectator shifts of the O(3) defects, one twisted real component against a twisted plane (Table 2, Fig. 3(a)) | `spectator_response.py spectator_shift_ratio` | `twist/spectator_shift_ratio.*` |
| cap entanglement of the free scalar and O(2) model, B_cap | `cap_entanglement.py cap_ee ...`, `cap_alpha_scan`, `cap_two_shapes` | `twist/cap_ee_*`, `twist/cap_two_shapes_*` |

Representative values:

- **Large N, alpha = 1/2.** The direct trace and the closed form both give 0.090063, against log g_free = 0.066717.
  Near d = 4 at O(eps) the same quantity is (pi/4) alpha^2 (1 - alpha)^2, 0.049 at alpha = 1/2.
- **Partial twists.** For O(3) → O(2) at alpha = 1/2 the spectator raises the entropy by 8.9 × 10^-4 at O(eps) and
  4.5 × 10^-3 through O(eps^2), against 1.5 × 10^-2 at large N.
- **Cap entanglement.** The twist dependence has the free shape, but its normalization is B_cap = 1.59, 1.55, 1.59,
  1.65 at N_m = 6–9.
- **Spectator shifts.** Delta_phi - Dh_sp, with the local potential subtracted and extrapolated linearly in 1/N_m over
  N_m = 6–9, is 0.049 for the reflection (one twisted real component) and 0.110 for the twisted plane, a ratio of 0.45
  (0.46–0.47 at each N_m), against 1/2 at large N.

### 04 — pinned lines (paper, Section 5, Appendix E)

| result | step | stored as |
|---|---|---|
| ED and DMRG angle scans of the junction, O(2) vector and spin-one, O(3), O(4); the audit (Tables 3, 11–14, 16, Figs. 5, 10, 11) | `junctions.py theta_on`, `theta_dey`, `dmrg_theta`, `dmrg_dey`, `dmrg_wall`, `o4_package`, `junction_analysis` | `twist/theta_junction_results/*`, `flow/dmrg_pin_out/*`, `fun/paper_2026_10_24_physics/figure_sources/JUNCTION_ROUTE_AUDIT.json` |
| Ward stiffness C^HQ by ED and DMRG; curvature bias and the tilt tower | `ward_stiffness.py tilt_test`, `dmrg_hq`, `ct_ward`, `dmrg_ward_ct`, `junction_pt2`, `junction_spectral`, `curvature_bias` | `twist/tilt_test_results/*`, `twist/ct_ward_results/*`, `twist/theta_junction_results/*`, `twist/curvature_bias.json` |
| the slow neutral operator and the wall correction law | `wall_correction.py neutral_parity`, `junction_firstorder*`, `slow_response`, `dey_series` | `twist/theta_junction_results/*`, `twist/dey_series.log` |
| shape factor at one and two loops, the large-N junction saddle (Table 16, eq:rtwoloop, eq:junctionr2assembly) | `junction_theory.py junction_oneloop`, `assemble`, `junction_largeM*`, `ct_largeN_eps` | `junction_eps2/*`, `largeN/*` |
| pinned monodromy line: the 3:1 test, the endpoint, its junctions, large N (Tables 4, 17–21) | `monodromy_pinning.py pin_flow`, `kr_*`, `even_*`, `disp_even`, `largeN_straight`, `junction_twisted_largeN` | `twist/kr_results/*`, `twist/even_results/*`, `twist/pin_even_*`, `twist/disp_results/even_*`, `analytic/monodromy_pinning/*` |
| half twist: the ingredients, the local expansion and the ball-size checks of the continuation scheme | `halftwist_pinning.py hs1_ingredients`, `hs2_local`, `hs3_eigen` | `analytic/halftwist_gamma/hs1_*`, `hs2_*`, `hs3_*` |
| half twist: untwisted calibration, every mode with s ≠ 0 shifted by Delta_phi - 1/2 | continuation `halftwist_pinning.py hs4_calib`; cutoff `cutoff_eigen triv 1 1.5`, `cutoff_eigen triv 2 2.5` | `analytic/halftwist_gamma/hs4_calib.json`, `analytic/halftwist_largeN/cutoff_eigen_triv_*` |
| half twist: the pinning mode, Delta_1/2 = 3/2 - Delta_phi (eq:halftwistpin), and the modes s = 3/2, 5/2 | continuation `halftwist_pinning.py hs5_twisted`; cutoff `cutoff_eigen half 0.5 1`, `half 1.5 2`, `half 2.5 3` | `analytic/halftwist_gamma/hs5_twisted.json`, `analytic/halftwist_largeN/cutoff_eigen_half_*` |
| general twist: the two lowest twisted modes, (4\|s\| - 3)(Delta_phi - 1/2) (eq:pinlaw) | continuation `halftwist_pinning.py hs6_alpha`; cutoff `cutoff_alpha gate 0.3`, `cutoff_alpha run 0.5 0.5 1.0` (prints -1.00017), `run 0.3 0.3 0.8` (-1.80045), `run 0.3 -0.7 1.2` (-0.19997), in units of Delta_phi - 1/2 | `analytic/halftwist_gamma/hs6_alpha.json`, `analytic/halftwist_largeN/cutoff_alpha_*` |
| the pinned O(3) half-twist along the easy-plane flow, against the pinned O(2) monodromy line (Fig. 4(c,d)) | `pinned_flow.py pinned_flow gate N`, `run N 0,-0.0025,-0.005,-0.01,-0.02,-0.045,-0.09` (N = 6, 7, 8), `robust N` | `flow/pinned_flow_N*.json`, `flow/pinned_flow_robust_*`, with `twist/pin_even_o2wf_N*_a0.5.json` |
| pins along the two branches: the phi_3 pin ends on the pinned Ising line on the easy axis, the charged pin there and the phi_3 pin on the easy plane decouple (App. E.9, Table 24, Figs. 1, 4(a,b)) | `pinned_flow.py pin_branches gate N` (N = 5, 6, 7), `neutral N frozen,0.09,0.03,0.01,0,-0.01,-0.03,-0.09` (N = 6, 7, 8), `neutral_lm 9 frozen,0.09,0.045,0.01,0,-0.01,-0.045,-0.09`, `charged N frozen,0.09,0.03,0.01` (N = 6, 7, 8), `finite N` (N = 6, 7); `pinned_flow.py ising_pinned_ref 6,7,8,9,10,11,12` | `flow/pin_branches_*`, `flow/ising_pinned_ref.json` |
| entropic and wall thresholds of an ordered line at large N, at leading order and with the entropy of order N^0 (Tables 22, 23, Fig. 6) | `pinned_entropy.py entropy_window`, `pinned_det trace 1600 1600 160 160`, `twisted_det trace alpha 1000 1000 120 120` (alpha = 0, 0.1, …, 0.5), `twisted_summary` | `analytic/pinned_entropy/entropy_window.json`, `largeN/pinned_det_*`, `largeN/twisted_det_*` |

`halftwist_pinning.py` asks whether the pinned partial twist survives at half twist at large N (paper, App. E.7). At
N = infinity its condensate vanishes there, because the spectator mode s = 0 reaches the Breitenlohner–Freedman value,
so the order-1/N shift of the pinning mode s = 1/2 on the unpinned partial twist decides: the pin stays relevant if this
mode moves below Delta = 1. The shift comes from the exchange of the auxiliary field alone, and the file computes it in
two schemes with independent codes, analytic continuation in the exponent of the sigma propagator (`hs1_ingredients` …
`hs6_alpha`) and a chordal cutoff with a finite-part fit (`cutoff_eigen`, `cutoff_alpha`). Both are calibrated on
untwisted lines, where every mode with s ≠ 0 must shift by Delta_phi - 1/2 = 4/(3 pi^2 N). The number of components is
N throughout; no orbitals enter. `python 04_pinned_lines/halftwist_pinning.py report` reruns `hs1_ingredients`,
`hs2_local`, `hs4_calib`, `hs5_twisted`, `hs6_alpha` and the five `cutoff_eigen` runs, and regenerates their stored
results unchanged, in about 12 minutes (719 s measured, peak memory 1.1 GB), most of it in `hs5_twisted` and `hs6_alpha`. The `cutoff_alpha` steps are not in the report because their logs record run
times; rerun them one at a time as listed above (about 20 s each) and compare with the values given there.

`pinned_flow.py` follows the pinned O(3) half-twist along the easy-plane flow and divides each quantity by the pinned
O(2) monodromy line at the same N_m. Its steps `pin_branches` and `ising_pinned_ref` follow the same charged pin and a
pin on the untwisted component phi_3 along both branches, with the pinned line of the Ising model as an independent
realization of the easy-axis endpoint; 'frozen' in a g list is the frozen easy-axis limit g = 3. `report` runs two
gates: the O(2) model sent through the pinned-flow code reproduces the pinned O(2) line to all printed digits, and the
phi_3 pin at the O(3) point reproduces the spectator-pinning energies of `02_rg_flow/easy_axis.py` to 1e-15.
`pinned_entropy.py` sets the entropy released by pinning, at leading order
and at order N^0, against the leading wall threshold. Its `report` reruns `entropy_window` (about 5 minutes) and
`twisted_summary`, which reads the stored regulated traces; the traces themselves take 10 minutes (`pinned_det`) to an
hour (`twisted_det`, per twist), and their (lam, mode) tables are rebuilt into `04_pinned_lines/pinned_entropy_files/`,
which git ignores.

Representative values:

- **Shape factor.** One loop gives 1.731 at eps = 1; the [1/1] Padé of the two-loop series gives 1.336 (O(2)), and the
  lattice tilt-excess fits give 1.38–1.41.
- **Walls.** The measured O(2) vector wall extrapolates to 0.85–0.91, below one; O(3) gives 0.93–1.04 and O(4)
  1.01–1.06.
- **Pinned monodromy line.** At alpha = 0.3 the wall is 0.70–0.71, and the ratios to alpha = 0 follow the large-N
  saddle (wall ratio 0.87–0.89 against 0.831).
- **Half twist at large N.** Shifts are N(Delta - Delta|_{N=inf}) in units of Delta_phi - 1/2 = 4/(3 pi^2 N) = 0.135095/N.
  - Untwisted calibration, s = 1, 2: +1 exactly in the continuation scheme (Sigma-hat = -n/6 per C_sigma), and
    0.999999–1.000004 in the cutoff scheme.
  - Pinning mode at alpha = 1/2: -1 in the continuation scheme (Sigma-hat = 1/12 per C_sigma to 16 digits, unchanged
    under the ball size and the chi cutoff), and -1.00006 (fit without a log term) or -1.00046 (with one) in the cutoff
    scheme. So Delta_1/2 = 3/2 - Delta_phi and the pin stays relevant, which gives varsigma_* = Delta_phi - 1/2 and
    phi_*^2 = 1/(4 pi sqrt(3N)) in eq:halftwistpin.
  - Higher modes at alpha = 1/2: s = 3/2 gives 0 in the continuation scheme (`hs3_eigen`) and 0.0002–0.0018 in the
    cutoff scheme; s = 5/2 gives 0.250–0.253 in the cutoff scheme. Neither follows 4|s| - 3.
  - General twist, |s| = alpha and 1 - alpha: the continuation scheme reproduces 4|s| - 3 to 2.2 × 10^-11 at
    alpha = 0.1, 0.2, 0.3, 0.4 (`hs6_alpha`) and to 2 × 10^-14 at alpha = 1/2 (`hs5_twisted`), and each pair sums to -2.
    The cutoff scheme gives -1.80045 and -0.19997 at alpha = 0.3, against -1.8 and -0.2, and -1.00017 at alpha = 1/2,
    within its fit uncertainty of about 10^-3.
- **Pinned flow.** At the easy-plane endpoint g = -0.09 the pinned O(3) line matches the pinned O(2) monodromy line at
  N_m = 6–8: C^HQ to 0.03% and the wall to 0.5%. At g = 0 the spectator raises the wall by 2.8, 4.1 and 5.2%. Keeping the
  local potential or reweighting the second pin bilinear moves the endpoint ratios by at most 0.1%.
- **Pins along the two branches.** In the frozen easy-axis limit the phi_3-pinned half-twist has wall 0.750–0.755 and
  lowest level 1.534–1.543 at N_m = 6–9, against 0.774–0.783 and 1.572–1.587 for the pinned line of the Ising model; the
  ratio of the two, which does not depend on the unit, is 0.489 against 0.493 at N_m = 8. The twist changes this wall by
  0.0016–0.0033 at g = 0 and by nothing in the frozen limit. On the easy plane at g = -0.09 the phi_3-pinned wall falls as
  0.139, 0.095, 0.067, 0.048 at N_m = 6–9; on the easy axis the charged pin's wall falls to 0.043 in the frozen limit at
  N_m = 8, and its Ward matrix element vanishes.
- **Entropy against walls.** The entropy released by pinning at order N^0 is +0.0294, +0.0255, +0.0131, -0.0092,
  -0.0399 and -0.0549 at alpha = 0, 0.1, …, 0.5 (log g_pin = -0.153673 N - 0.0294 at alpha = 0). It moves the entropic
  threshold from 4.51 to 4.32 at alpha = 0 and from 14.59 to 15.75 at alpha = 1/2, above the leading wall threshold
  (3.63 to 7.62) at every twist.

### 05 — five dimensions (paper, Section 6, Appendix F)

| result | step | written as |
|---|---|---|
| twisted bands at I = 2, 3, 4 | `fuzzy_s4.py twisted_band I K`, `save_band I 4 nsave` | `s4d/band_I*.log` |
| interaction terms of models A and B, the I = 2 coupling map | `fuzzy_s4.py build_I ...`, `build_I3 i3A.json`, `design` | `terms_*.pkl` in `s4d_files/`, from the couplings `s4d/i3A.json`, `s4d/i3B.json`, `s4d/i4A.json`, `s4d/designB1c_I2*.json` |
| bulk scans and defect levels by DMRG (Table 25) | `fuzzy_s4.py bulk_scan ...`, `defect_dmrg ...`, `fig_curves`, `lpar_measure ...`, `par_spin2 dmrg ...` | `s4d/bulkscan/*`, `s4d/defdmrg/*`, `s4d/note/*_points.json`, `s4d/lpar/*` |
| exact correlator, overlaps, equal-time correlator, tilt Ward identity (App. F.4) | `fuzzy_s4.py corr5d exact\|run\|dmrg ...`, `core5d ...`, `ward5d ...`, `ward5d_dmrg ...` | `s4d/lpar/*_corr5d.json`, `*_core5d.json`, `*_ward5d.json` |
| the parallel-generator test: the first moment of the energy density along the defect, I = 2 by ED and DMRG, I = 3 by DMRG (App. F.5, Table 26) | `fuzzy_s4.py lambda5 gate terms_designB1c_I2.pkl designB1c_I2.json`, `lambda5 ed terms_designB1c_I2.pkl terms_designB1c_I2_bulk.pkl designB1c_I2.json 8.3`, `lambda5 dmrg terms_designB1c_I2.pkl designB1c_I2.json 8.3 200 10 7`, `lambda5 dmrg terms_i3A.pkl i3A.json 4.5 400 10 10 1` | `s4d/lpar/*_lambda5.json` |
| exact density weights f_n of the projected Landau level (App. F.3); band multiplets and the rotation algebra (App. F.2); the vacuum-improved moment (Table 26); ED against DMRG at I = 2 and chi = 400 against 800 at I = 3 | `fuzzy_s4.py audit_s4d` | `s4d/checks_2026_11_07/checks.json` |

The five-dimensional results are not included in this repository. The steps write them under the names in the last
column, and `fuzzy_s4.py report` needs them. Of `audit_s4d`, `harmonic_checks` needs nothing and `improvement_test`
needs only the I = 2 terms and the result of `lambda5 ed`; `fig_curves` also reads the I = 2 field scan
`s4d/curve_I2_designB1c_I2.json`, which is not included either. The values below are those of the paper.

Representative values:

- **Table 25.** With the displacement candidate at four, I = 4 model A at h = 6 gives psi_1/2 = 2.018, psi_3/2 = 3.005,
  the s_par = 1 descendant 3.374 and the s_par = 2 candidate 4.999 (L_par^2 = 6.045), against the free 2, 3, 3, 4.
- **Normalizations.** At I = 3, b_3/2^2/b_1/2^2 = 0.78 against the exact 4/3. The tilt carries 3.8%, 19% and 11% of its
  exact weight on the displacement family at I = 2, 3, 4.
- **Density weights.** f_n(I) = I! (I+3)! / ((I-n)! (I+n+3)!), so f_1 = I/(I+4): 1/3, 3/7 and 1/2 at I = 2, 3, 4.
- **Parallel generator (Table 26).** The largest share of the off-diagonal weight of the moment acting on psi_1/2 is 0.486
  at 2.715 at I = 2 (0.579 at 2.673 with the vacuum-improved moment), against the descendant expected at 2.983; at
  I = 3 it is 0.215 at 3.306, within nine resolved states that carry 0.448 of the weight.
