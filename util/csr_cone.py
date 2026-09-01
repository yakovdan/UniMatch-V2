"""Exact cone projection for GGR mode ``cone`` (the CSR-class successor).

The operation
-------------
Given $k \\le 20$ foreground class-prototype gradients $G_c \\in \\mathbb{R}^D$
(rows of ``protos``; background and unseen classes are the *caller's* job to
exclude) and a vector $g \\in \\mathbb{R}^D$ (the head slice of the unsupervised
gradient), replace $g$ by its Euclidean projection onto the polyhedral cone

    $K = \\{\\, x : \\langle x, G_c \\rangle \\ge 0 \\ \\ \\forall c \\,\\}$,

i.e. the smallest edit after which the unsupervised gradient opposes no
foreground class prototype.  This is what the clipping CSR-class *meant* to do;
clipping coordinates in a Gram-Schmidt basis does not enforce these constraints
(and destabilised every seed), the projection enforces them exactly.

Why the dual, and why it equals Dykstra's algorithm
---------------------------------------------------
Primal:  $\\min_x \\tfrac{1}{2}\\|x - g\\|^2$  s.t.  $A x \\ge 0$, with the rows
of $A$ the unit-normalised prototypes.  Stationarity of the Lagrangian gives
$x = g + A^\\top \\lambda$ with $\\lambda \\ge 0$, and substituting back yields
the dual, a tiny non-negative quadratic program in $\\lambda$ alone:

    $\\min_{\\lambda \\ge 0} \\ \\tfrac{1}{2}\\lambda^\\top M \\lambda
     + q^\\top \\lambda$,  $M = A A^\\top$ (the $k \\times k$ Gram matrix),
    $q = A g$ (the raw alignments).

Cyclic coordinate descent on this NNQP has a closed-form step,

    $s_c = q_c + (M \\lambda)_c$  (the current alignment
    $\\langle A_c, x_{\\mathrm{cur}} \\rangle$),
    $\\lambda_c \\leftarrow \\max(0, \\ \\lambda_c - s_c / M_{cc})$,

and is *exactly* Dykstra's cyclic-projection algorithm specialised to
half-spaces (Han 1988; Gaffke & Mathar 1989): the per-constraint correction
vector Dykstra remembers is $\\lambda_c A_c$.  Convergence needs only
convexity, not a full-rank $M$ -- duplicated or near-parallel prototypes make
$\\lambda$ non-unique but leave the projection $P(g)$ unique.

KKT conditions (what "solved" means)
------------------------------------
    stationarity            $x = g + A^\\top \\lambda$   (holds by construction)
    dual feasibility        $\\lambda \\ge 0$            (holds by construction)
    primal feasibility      $s = q + M\\lambda \\ge 0$   -> residual rho_feas
    complementary slackness $\\lambda_c s_c = 0$         -> residual rho_comp

Mid-iteration the primal point is *infeasible*, so there is no certified
duality gap to read off; the two residuals above are the honest monitors and
both live in $k$-dimensional arithmetic.

Why unit-normalising the prototypes is safe
-------------------------------------------
A half-space $\\{x : \\langle x, G_c \\rangle \\ge 0\\}$ is invariant under
positive rescaling of its normal, so normalising changes neither $K$ nor
$P(g)$.  It *does* fix the Gram conditioning: raw prototype norms span ~8-39,
so raw $M_{cc}$ would span three orders of magnitude; in the unit convention
$M_{cc} = 1$ exactly.  The normalisation is done in $k$-space
($M = M_{\\mathrm{raw}} / (n n^\\top)$, $q = q_{\\mathrm{raw}} / n$) so no
$(k, D)$ temporary is ever allocated.

Cost and precision
------------------
Only two $D$-sized operations exist: the reductions building $M$ and $q$
(GPU, input dtype), and the final matvec $g + \\sum_c \\lambda_c G_c$.  The
$k \\times k$ solve itself runs in float64 on the CPU -- microseconds, and
solver error lands orders of magnitude below the gradient noise floor.

References: R. L. Dykstra 1983; Boyle & Dykstra 1986; Han 1988;
Gaffke & Mathar 1989; Deutsch & Hundal 1994 (linear rate for polyhedra).
Project context: seed_attribution_experiment/GGR-class.md section 4.1.
"""

import sys
from typing import Dict, List, Optional, Tuple

import torch


def cone_rectify(protos: torch.Tensor, g: torch.Tensor, *, tol: float = 1e-8,
                 max_sweeps: int = 500, min_proto_norm: float = 1e-12
                 ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Project ``g`` onto the cone that opposes no prototype; the one public entry point.

    Args:
        protos: ``(k, D)`` prototype rows $G_c$ (foreground only, unseen classes
            already dropped by the caller). Never modified. ``k = 0`` is legal.
        g: ``(D,)`` vector to rectify, same device as ``protos``. Never modified.
        tol: stop when both KKT residuals (see ``_kkt_residuals``) are <= tol.
        max_sweeps: hard cap on coordinate-descent sweeps; on hitting it the best
            $\\lambda$ so far is still applied (``diag['converged'] = False``) --
            a training run is never crashed by the solver.
        min_proto_norm: rows with $\\|G_c\\| <$ this are dropped (belt-and-braces;
            the trainer's ``seen`` mask already excludes them).

    Returns:
        ``(g_rect, diag)``.
        ``g_rect``: the projection $P(g)$, with ``g``'s dtype and device. On the
        no-op paths -- $k = 0$ after filtering, $\\min_c q_c \\ge 0$ (``g``
        already in the cone, which covers $\\|g\\| = 0$), or $\\lambda = 0$ --
        the function returns **``g`` itself** (the same object, zero-copy) and
        ``diag['fired'] == 0.0``; otherwise a fresh tensor.
        ``diag``: plain floats/lists only (no tensors, no ``grad/`` prefixes;
        the caller owns wandb naming):

        - ``fired``      1.0 iff ``g_rect is not g``
        - ``k``          rows kept after the norm filter
        - ``active``     number of strictly positive multipliers
        - ``sweeps``     CD sweeps executed
        - ``converged``  bool, both residuals <= tol
        - ``rho_feas`` / ``rho_comp``  final KKT residuals
        - ``delta_norm`` $\\|g_{rect} - g\\| = \\sqrt{\\lambda^\\top M \\lambda}$
          (computed in $k$-space; exact because $M$ is in the unit convention)
        - ``edit_frac``  ``delta_norm / max(||g||, eps)``
        - ``q`` / ``s`` / ``lam``  per-row lists in the *caller's* row order
          (before-alignments, after-alignments, multipliers; a filtered row
          reports $q = s = \\lambda = 0.0$ so per-class mapping stays aligned)

    The edit is always a non-negative combination of the prototypes
    ($\\lambda \\ge 0$ every sweep by construction), and
    $\\|g_{rect} - g\\| \\le \\|g\\|$ (projection onto a cone through the origin).
    """
    # 1. degenerate input: k == 0 -> return g itself, all-zero diagnostics.
    # 2. _gram_and_alignments: the D-sized reductions; get (M, q, inv_norms,
    #    g_norm_sq, keep_mask) with M, q already fp64/CPU in the unit convention.
    # 3. early exit: min(q) >= 0 -> already feasible, return g itself
    #    (diag reports q and s = q, lam = 0).
    # 4. _solve_nnqp_cd on the kept rows.
    # 5. lambda == 0 (possible despite step 3 only via numerics) -> no-op path.
    # 6. _apply_correction with lam_scaled = (lam * inv_norms) scattered back to
    #    the caller's row order, cast to protos.dtype on protos.device.
    # 7. _diagnostics; return (g_rect, diag).
    with torch.no_grad():
        k0 = protos.shape[0]
        e = torch.zeros(0, dtype=torch.float64)
        if k0 == 0:
            return g, _diagnostics(False, torch.zeros(0, dtype=torch.bool), e, e, e, 0, True, 0, 0, 0, 0)

        M, q, inv_norms, g_norm_sq, keep_mask = _gram_and_alignments(protos, g, min_proto_norm)

        if not keep_mask.any():
            return g, _diagnostics(False, keep_mask, e, e, e, 0, True, 0, 0, 0, g_norm_sq)

        if torch.min(q) >=0: # already feasible
            z = torch.zeros_like(q)
            return g, _diagnostics(False, keep_mask, z, q, q,0, True, 0.0, 0.0, 0.0, g_norm_sq)

        lam, s, sweeps, rho_feas, rho_comp, converged =_solve_nnqp_cd(M, q, g_norm_sq, tol, max_sweeps)

        if torch.allclose(lam, torch.zeros_like(lam)): # numerical safety
            return g, _diagnostics(False, keep_mask, lam, q, s, sweeps, converged, rho_feas, rho_comp, 0.0, g_norm_sq)

        lam_full = torch.zeros(k0, dtype=torch.float64)
        lam_full[keep_mask] = lam * inv_norms
        lam_full = lam_full.to(device=protos.device, dtype=protos.dtype)
        g_rectified = _apply_correction(protos, g, lam_full)

        delta_norm = float((lam @ (M @ lam)).clamp(min=0).sqrt())

        return g_rectified, _diagnostics(True, keep_mask, lam, q, s, sweeps, converged, rho_feas, rho_comp,
                                    delta_norm, g_norm_sq)




def _gram_and_alignments(protos: torch.Tensor, g: torch.Tensor,
                         min_proto_norm: float
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """The only $D$-sized reductions; everything downstream is $k$-dimensional.

    Computes on ``g``'s device in the input dtype:
        ``M_raw = protos @ protos.T``   (one (k, D) x (D, k) matmul)
        ``q_raw = protos @ g``          (one (k, D) x (D,) matvec)
        ``g_norm_sq = g . g``
    then filters rows with $\\|G_c\\| <$ ``min_proto_norm`` and rescales in
    $k$-space to the unit-normal convention:

        $M = M_{raw} / (n n^\\top)$,  $q = q_{raw} / n$,  $n_c = \\|G_c\\|$
        (equivalent to normalising the rows first -- the cone is unchanged --
        but without materialising a second (k, D) tensor).

    Returns:
        ``(M, q, inv_norms, g_norm_sq, keep_mask)`` where ``M`` ``(k', k')`` and
        ``q`` ``(k',)`` are **float64 CPU** (the solve precision), ``inv_norms``
        ``(k',)`` maps unit-convention multipliers back onto the raw rows
        ($\\lambda_{scaled} = \\lambda / n$), ``g_norm_sq`` is a float, and
        ``keep_mask`` ``(k,)`` bool marks the surviving rows in the caller's
        original order.
    """
    # - row norms from M_raw.diagonal().sqrt() (no extra D-reduction needed)
    # - keep_mask = norms > min_proto_norm; index M_raw/q_raw down to kept rows
    # - divide by outer(n, n) / n, then .double().cpu()
    with torch.no_grad():
        m_raw = protos @ protos.T
        q_raw = protos @ g
        g_norm_sq = float(torch.dot(g, g))

        m64 = m_raw.double().cpu()
        q64 = q_raw.double().cpu()
        norms = m64.diagonal().clamp(min=0).sqrt()

        keep_mask = norms > min_proto_norm
        norms = norms[keep_mask]
        q = q64[keep_mask] / norms
        m = m64[keep_mask][:, keep_mask] / torch.outer(norms, norms)
        m = 0.5 * (m + m.T)
        m.fill_diagonal_(1.0)

        inv_norms = norms.reciprocal()
        return m, q, inv_norms, g_norm_sq, keep_mask




def _solve_nnqp_cd(M: torch.Tensor, q: torch.Tensor, g_norm_sq: float,
                   tol: float, max_sweeps: int,
                   lam_init: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, int, float, float, bool]:
    """Cyclic coordinate descent on the dual NNQP (== Dykstra for half-spaces).

    Solves $\\min_{\\lambda \\ge 0} \\tfrac{1}{2}\\lambda^\\top M \\lambda
    + q^\\top \\lambda$ over float64 CPU tensors.  One sweep visits every
    coordinate once in fixed order:

        $s_c = q_c + (M \\lambda)_c$          # current alignment of constraint c
        $\\lambda_c \\leftarrow \\max(0, \\lambda_c - s_c / M_{cc})$

    The update order does not affect the limit (the projection is unique);
    Deutsch & Hundal (1994) give an asymptotically linear rate for polyhedra.
    Each sweep is followed by a Lawson-Hanson face step: solve the equality
    system on the active face $\\{c : \\lambda_c > 0\\}$ (SVD-based lstsq,
    minimum-norm) and take the standard non-negativity-preserving step.  Pure
    CD needs ~cond(M) sweeps on near-parallel prototypes; the face step makes
    those converge in a few.
    ``M[c, c] <= eps`` coordinates are skipped -- cannot occur for kept rows
    ($M_{cc} = 1$ exactly in the unit convention) but the guard documents the
    invariant and costs nothing.

    Args:
        M, q: unit-convention Gram and alignments, float64 CPU, shape (k, k)/(k,).
        g_norm_sq: $\\|g\\|^2$, needed by the residuals' denominators.
        tol: convergence threshold on both residuals from ``_kkt_residuals``.
        max_sweeps: hard iteration cap; never raises on hitting it.
        lam_init: optional warm start (internal/testing only -- the public API is
            stateless by design); clamped to $\\ge 0$. ``None`` -> zeros.

    Returns:
        ``(lam, s, sweeps, rho_feas, rho_comp, converged)`` -- the multipliers,
        the final alignments $s = q + M\\lambda$, sweeps executed, the final
        residuals, and whether both dropped below ``tol``.
    """


    kp = q.shape[0]
    if lam_init is not None:
        lam = lam_init.detach().to(torch.float64).clamp(min=0.0).clone()
        s = q + M @ lam
    else:
        lam = torch.zeros(kp, dtype=torch.float64)
        s = q.clone()  # s = q + M @ 0
    # numpy views: zero-copy aliases of the same CPU float64 buffers.  The
    # inner loop is pure scalar work; ~10k torch dispatches at the sweep cap
    # would cost 50-200 ms where numpy scalars cost microseconds.
    Mn, ln, sn = M.numpy(), lam.numpy(), s.numpy()
    sweeps, rho_feas, rho_comp, converged = 0, float('inf'), float('inf'), False
    for sweep in range(1, max_sweeps + 1):
        # --- one cyclic CD sweep (the Dykstra pass; monotone in the dual) ---
        for c in range(kp):
            Mcc = Mn[c, c]
            if Mcc <= 1e-12:  # cannot happen for kept rows (M_cc = 1);
                continue  # the guard documents the invariant
            new = ln[c] - sn[c] / Mcc  # unconstrained coordinate minimiser,
            if new < 0.0:  # clipped to the non-negative orthant
                new = 0.0
            d = new - ln[c]
            if d != 0.0:  # settled coordinates cost nothing
                ln[c] = new
                sn += d * Mn[:, c]  # s = q + M @ lam, rank-1 refresh
        # --- Lawson-Hanson face acceleration: pure CD crawls when rows are
        # near-parallel (Gram cond ~1e6 would need ~1e6 sweeps), so once per
        # sweep solve the equality system on the current active face and take
        # the standard non-negativity-preserving step.  driver='gelsd' (SVD)
        # is essential: it returns a minimum-norm solution on singular faces
        # (duplicate rows), where the default CPU driver gelsy returns garbage
        # (residual O(1) vs 1e-15).  Both step types are monotone in the dual
        # objective, and the residual check below remains the only judge. ---
        F = (lam > 0.0).nonzero(as_tuple=True)[0]
        if F.numel() > 0:
            z = torch.linalg.lstsq(M[F][:, F], -q[F], driver='gelsd').solution
            if bool((z >= 0.0).all()):
                lam[F] = z  # face optimum already non-negative: jump
            else:
                lam_F = lam[F]
                neg = z < 0.0  # step until the first coordinate hits 0
                alpha = float((lam_F[neg] / (lam_F[neg] - z[neg])).min())
                lam[F] = (lam_F + alpha * (z - lam_F)).clamp(min=0.0)
        sweeps = sweep
        # once per sweep (not per coordinate): fresh residuals, and adopt the
        # recomputed s so the incremental updates cannot drift
        rho_feas, rho_comp, s_fresh = _kkt_residuals(M, q, lam, g_norm_sq)
        sn[:] = s_fresh.numpy()
        if rho_feas <= tol and rho_comp <= tol:
            converged = True
            break
    return lam, s, sweeps, rho_feas, rho_comp, converged


def _kkt_residuals(M: torch.Tensor, q: torch.Tensor, lam: torch.Tensor,
                   g_norm_sq: float) -> Tuple[float, float, torch.Tensor]:
    """The two KKT conditions not enforced by construction, as scale-free residuals.

    With $s = q + M\\lambda$ (the alignments of the *current* primal iterate
    $x = g + A^\\top \\lambda$) and the iterate's energy expanded in $k$-space,

        $\\|x\\|^2 = \\|g\\|^2 + 2 \\lambda^\\top q + \\lambda^\\top M \\lambda$,

    the residuals are

        rho_feas = $\\max_c \\max(0, -s_c) / \\max(\\|x\\|, 10^{-30})$
                   -- the worst still-violated *cosine* against a prototype;
        rho_comp = $\\sum_c \\lambda_c \\max(s_c, 0) / \\max(\\|g\\|^2, 10^{-30})$
                   -- multipliers still pushing on satisfied constraints.

    ``max(s_c, 0)`` (not the signed $\\lambda^\\top s$) so a negative term cannot
    cancel a positive one and pass the tolerance spuriously; the negative side is
    rho_feas's job.  rho_comp is normalised by the *input* energy $\\|g\\|^2$,
    not $\\|x\\|^2$: in the apex regime $P(g) \\to 0$ while $\\lambda$ scales
    with $\\|g\\|$, so $\\lambda^\\top s \\sim \\varepsilon \\|g\\|^2$ at an
    essentially perfect solve and an $\\|x\\|^2$ denominator would make the
    tolerance unreachable.  The clamps and rho_feas's $\\|x\\|$ denominator are
    load-bearing in that same regime ($\\|x\\| \\to 0$).
    These are KKT residuals, not a duality gap -- the iterate is infeasible
    mid-run, so no certified gap exists.

    Returns:
        ``(rho_feas, rho_comp, s)``.
    """


    s = q + M @ lam  # fresh O(k^2) recomputation, no drift
    if s.numel() == 0:
        return 0.0, 0.0, s
    # ||x||^2 expanded entirely in k-space; mathematically >= 0, but in the
    # apex regime it is a difference of large numbers -- clamp before sqrt
    x_norm_sq = max(g_norm_sq + 2.0 * float(lam @ q) + float(lam @ (M @ lam)), 0.0)
    rho_feas = float((-s).clamp(min=0.0).max()) / max(x_norm_sq ** 0.5, 1e-30)
    # normalised by the INPUT energy ||g||^2: lambda scales with ||g||, so in
    # the apex regime (||x|| -> 0) an ||x||^2 denominator would be unreachable
    rho_comp = float((lam * s.clamp(min=0.0)).sum()) / max(g_norm_sq, 1e-30)
    return rho_feas, rho_comp, s


def _apply_correction(protos: torch.Tensor, g: torch.Tensor,
                      lam_scaled: torch.Tensor) -> torch.Tensor:
    """The one $D$-sized write: ``g + lam_scaled @ protos``, as a fresh tensor.

    ``lam_scaled`` is the unit-convention $\\lambda$ divided by the row norms
    ($\\lambda / n$), indexed in the *raw* row order of ``protos`` (filtered rows
    carry 0), cast to ``protos.dtype`` on ``protos.device`` by the caller.
    Pure function: ``g`` is not modified; in-place semantics (the trainer's
    ``g_u_scope.copy_(...)``) are the caller's decision.
    """
    with torch.no_grad():
        return g + lam_scaled @ protos


def _diagnostics(fired: bool, keep_mask: torch.Tensor, lam: torch.Tensor,
                 q: torch.Tensor, s: torch.Tensor, sweeps: int, converged: bool,
                 rho_feas: float, rho_comp: float, delta_norm: float,
                 g_norm_sq: float) -> Dict[str, object]:
    """Assemble the plain-Python diagnostics dict described in ``cone_rectify``.

    Scatters the solver's kept-row vectors back to the caller's original row
    order using ``keep_mask`` (filtered rows report $q = s = \\lambda = 0.0$:
    their unit-convention alignment was never computed, and the raw alignment
    of a ~zero prototype is ~0, so 0.0 is the honest sentinel), converts
    everything to floats/lists -- no tensors escape, so the dict is directly
    wandb-loggable -- and leaves key naming (``grad/...`` prefixes, class names)
    entirely to the caller.

    ``lam``, ``q``, ``s`` arrive in kept-row order with ``keep_mask.sum()``
    entries; ``keep_mask`` has one entry per row the caller passed, so the
    returned lists always have the caller's length and per-class mapping in the
    trainer never shifts, whatever was filtered.
    """
    k0 = keep_mask.shape[0]
    # scatter kept-row values into the caller's row order; filtered rows stay 0
    lam_full = torch.zeros(k0, dtype=torch.float64)
    q_full = torch.zeros(k0, dtype=torch.float64)
    s_full = torch.zeros(k0, dtype=torch.float64)
    lam_full[keep_mask] = lam
    q_full[keep_mask] = q
    s_full[keep_mask] = s
    return {
        'fired': 1.0 if fired else 0.0,
        'k': int(keep_mask.sum()),               # rows kept after the norm filter
        'active': int((lam > 0).sum()),          # constraints with lambda_c > 0
        'sweeps': int(sweeps),
        'converged': bool(converged),
        'rho_feas': float(rho_feas),
        'rho_comp': float(rho_comp),
        'delta_norm': float(delta_norm),         # ||g_rect - g||, from k-space
        'edit_frac': float(delta_norm) / max(g_norm_sq ** 0.5, 1e-30),
        'q': q_full.tolist(),                    # alignments before the edit
        's': s_full.tolist(),                    # alignments after the edit
        'lam': lam_full.tolist(),                # multipliers, caller's row order
    }


# ---------------------------------------------------------------------------
# self-tests (run: `python util/csr_cone.py` from the repo root; exit 0 == pass)
# ---------------------------------------------------------------------------


def _random_problem(D: int, k: int, case: str, dtype: torch.dtype,
                    gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one ``(protos, g)`` test instance.

    ``case`` selects the regime:
        'generic'       -- i.i.d. gaussian rows and g;
        'near_parallel' -- rows = shared direction + small noise
                           (Gram condition number ~1e6);
        'duplicate'     -- two identical rows (singular M: lambda non-unique,
                           the projection still unique);
        'zero_row'      -- one all-zero row (exercises min_proto_norm filtering);
        'feasible'      -- g = non-negative combination of rows + noise
                           orthogonalised against all rows (P(g) must be g itself);
        'apex'          -- g ~ -sum_c G_c scaled (deep in the polar cone: all
                           constraints active, P(g) ~ 0).
    Row norms are drawn from ~U[8, 39] to mimic the measured prototype spread.
    """
    rows = torch.randn(k, D, generator=gen, dtype=torch.float64)
    if case == 'near_parallel':
        # rows = one shared direction + 1e-3 relative noise -> Gram cond ~1e6
        shared = torch.randn(D, generator=gen, dtype=torch.float64)
        rows = shared.unsqueeze(0) + 1e-3 * rows
    # rescale every row to a norm in the measured prototype range U[8, 39]
    target = 8.0 + 31.0 * torch.rand(k, generator=gen, dtype=torch.float64)
    rows = rows * (target / rows.norm(dim=1).clamp_min(1e-30)).unsqueeze(1)
    if case == 'duplicate' and k >= 2:
        rows[1] = rows[0]          # singular Gram; k = 1 falls back to 'generic'
    if case == 'zero_row':
        rows[0] = 0.0              # must be dropped by the min_proto_norm filter

    if case == 'feasible':
        # g = A^T M^{-1} y with y >= 0.1 guarantees A g = M M^{-1} y = y >= 0.1
        # (unit rows A, invertible M -- the base rows are generic here).  The
        # noise must live in the orthogonal complement of the row span, removed
        # via the Gram -- sequential Gram-Schmidt against non-orthogonal rows
        # would NOT orthogonalise against the set.
        A = rows / rows.norm(dim=1, keepdim=True)
        Mk = A @ A.T
        y = torch.rand(k, generator=gen, dtype=torch.float64) + 0.1
        g = A.T @ torch.linalg.solve(Mk, y)
        w = torch.randn(D, generator=gen, dtype=torch.float64)
        g = g + w - A.T @ torch.linalg.solve(Mk, A @ w)
        if float((A @ g).min()) < 0:   # belt and braces: fall back to the pure
            g = A.T @ torch.linalg.solve(Mk, y)  # combination, feasible exactly
    elif case == 'apex':
        # deep in the polar cone: -positive combination of the rows, plus a
        # whisker of noise so the projection is near (not exactly) zero
        coeffs = torch.rand(k, generator=gen, dtype=torch.float64) + 0.5
        g = -(coeffs @ rows) + 1e-3 * torch.randn(D, generator=gen, dtype=torch.float64)
    else:
        g = torch.randn(D, generator=gen, dtype=torch.float64)
    return rows.to(dtype), g.to(dtype)


def _ref_project_k2(protos: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Closed-form projection for k = 2, the brute-force reference for test (c).

    Enumerates the candidate set of the exact projection onto the intersection
    of two homogeneous half-spaces:
        - interior: g itself, if feasible;
        - each single face: $g - \\min(0, \\langle a_i, g \\rangle) a_i$
          (unit $a_i$), kept if it satisfies the *other* constraint;
        - the edge $\\{x : \\langle a_1, x \\rangle = \\langle a_2, x \\rangle
          = 0\\}$: remove the span component of g via a 2x2 solve (the "apex"
          of the wedge in the span, keeping the orthogonal complement).
    Returns the feasible candidate nearest to g.  Everything in float64.
    """
    eps = 1e-12
    A = (protos / protos.norm(dim=1, keepdim=True)).double()
    g = g.double()
    f = A @ g
    cands = []
    if float(f.min()) >= -eps:                      # interior: g already feasible
        cands.append(g)
    for i in (0, 1):                                # single-face candidates
        xi = g - min(0.0, float(f[i])) * A[i]
        if float(A[1 - i] @ xi) >= -eps:            # must satisfy the other one
            cands.append(xi)
    # edge candidate: remove the span component.  pinv (not solve) so parallel
    # rows -- the 'duplicate' case, span a line -- go through the same formula;
    # both alignments are exactly 0 afterwards, so it is always feasible.
    Mk = A @ A.T
    edge = g - A.T @ (torch.linalg.pinv(Mk) @ f)
    cands.append(edge)
    return min(cands, key=lambda c: float((g - c).norm()))


def _ref_project_pg(protos: torch.Tensor, g: torch.Tensor,
                    iters: int = 50000) -> torch.Tensor:
    """Slow-but-sure reference: projected gradient on the dual NNQP.

    $\\lambda \\leftarrow \\max(0, \\lambda - (M\\lambda + q) / \\|M\\|_2)$ for
    ``iters`` steps, float64.  Used by test (f) only when scipy is unavailable;
    with scipy present the reference is ``scipy.optimize.nnls(A.T, -g)`` via the
    identity $\\tfrac{1}{2}\\lambda^\\top M\\lambda + q^\\top \\lambda =
    \\tfrac{1}{2}\\|A^\\top \\lambda + g\\|^2 - \\tfrac{1}{2}\\|g\\|^2$.
    """
    A = (protos / protos.norm(dim=1, keepdim=True)).double()
    g = g.double()
    Mk = A @ A.T
    q = A @ g
    L = float(torch.linalg.eigvalsh(Mk)[-1].clamp(min=1e-12))  # Lipschitz constant
    lam = torch.zeros_like(q)
    for _ in range(iters):
        lam = (lam - (Mk @ lam + q) / L).clamp(min=0.0)
    return g + A.T @ lam


def _check(name: str, ok: bool, detail: str = "") -> bool:
    """Print ``PASS name`` / ``FAIL name  <detail>`` and return ``ok``."""
    print(('PASS ' + name) if ok else ('FAIL ' + name + '  ' + detail))
    return ok


def _run_self_tests() -> int:
    """Randomised correctness suite; returns 0 iff every check passes.

    Fixed seed; D in {64, 300}, k in {1, 2, 5, 20}; ~20 trials per test; float64
    inputs for tight tolerances plus a few fp32 trials at loose tolerance.
    Compares always in x-space, never lambda-space (lambda is non-unique when M
    is singular; the projection is unique because K is convex).

        (a) KKT ~ 0: recompute the alignments of the returned g_rect against the
            unit rows directly in fp64; assert feasibility >= -10*tol*||x||,
            lam >= 0, complementarity <= 10*tol*||g||^2, and that diag's
            residuals agree.
        (b) idempotence: ||P(P(g)) - P(g)|| <= 1e-6 ||g|| in float64, and the
            second call is a no-op or a negligible (<= 1e-6 ||g||) touch-up.
        (c) k = 2 closed form: cone_rectify vs _ref_project_k2 to 1e-8.
        (d) no-op: a constructed-feasible g returns the *identical object*
            (``is g``) with fired == 0.
        (e) nonexpansiveness: ||P(x) - P(y)|| <= ||x - y|| (1 + 1e-9) on random
            pairs (projections onto convex sets are 1-Lipschitz).
        (f) external reference: scipy.optimize.nnls if importable (guarded by
            try/except), else _ref_project_pg; compare x to ~1e-6.
        (g) GPU parity: fp32 CUDA in -> fp32 CUDA out, agreeing with the CPU
            fp64 answer to ~1e-4 relative; no-op returns the same object.
            Skipped (with a note) when CUDA is unavailable.
    """
    gen = torch.Generator().manual_seed(0)
    tol = 1e-8
    fails = 0
    cases = ('generic', 'near_parallel', 'duplicate', 'zero_row', 'feasible', 'apex')
    dims = (64, 300)
    ks = (1, 2, 5, 20)

    def unit_kept(protos):
        # the reference view of the constraint set: unit rows, zero rows dropped
        n = protos.double().norm(dim=1)
        keep = n > 1e-12
        return protos.double()[keep] / n[keep].unsqueeze(1), keep

    def run(name, fn):
        nonlocal fails
        try:
            ok, detail = fn()
        except NotImplementedError:
            ok, detail = False, 'NotImplementedError (solver not implemented yet)'
        except Exception as exc:               # a crash is a failure, not an abort
            ok, detail = False, repr(exc)[:160]
        if not _check(name, ok, detail):
            fails += 1

    def test_a():
        # every (D, k, case) once in fp64 at 10*tol; two fp32 trials at 1e-5
        # relative (the correction is cast to fp32, so ~1e-7*||x|| violations
        # are expected and the fp64 bound would fail spuriously)
        specs = [(D, k, c, torch.float64, 10 * tol) for D in dims for k in ks for c in cases]
        specs += [(64, 5, 'generic', torch.float32, 1e-5), (300, 20, 'generic', torch.float32, 1e-5)]
        for D, k, case, dt, rtol in specs:
            protos, g = _random_problem(D, k, case, dt, gen)
            gr, diag = cone_rectify(protos, g, tol=tol)
            A, keep = unit_kept(protos)
            tag = '%s D%d k%d %s' % (case, D, k, str(dt).replace('torch.', ''))
            if A.shape[0] == 0:
                if gr is not g:
                    return False, tag + ': all rows filtered but g was edited'
                continue
            x = gr.double()
            s = A @ x                                   # ground truth, from x itself
            xn = float(x.norm())
            lam_full = torch.tensor(diag['lam'], dtype=torch.float64)
            if float(lam_full.min()) < 0:
                return False, tag + ': negative lambda %.3e' % lam_full.min()
            if float(s.min()) < -rtol * max(xn, 1e-30):
                return False, tag + ': feasibility %.3e at ||x||=%.3e' % (s.min(), xn)
            comp = float((lam_full[keep] * s.clamp(min=0)).sum())
            if comp > rtol * max(float(g.double().norm()) ** 2, 1e-30):
                return False, tag + ': complementarity %.3e' % comp
            if diag['converged'] and (diag['rho_feas'] > tol or diag['rho_comp'] > tol):
                return False, tag + ': converged=True but residuals above tol'
        return True, ''

    def test_b():
        for D, k, case in ((64, 2, 'generic'), (300, 5, 'generic'), (64, 5, 'apex'),
                           (300, 20, 'near_parallel'), (300, 20, 'generic')):
            protos, g = _random_problem(D, k, case, torch.float64, gen)
            g1, _ = cone_rectify(protos, g, tol=tol)
            g2, d2 = cone_rectify(protos, g1, tol=tol)
            if float((g2.double() - g1.double()).norm()) > 1e-6 * max(float(g.norm()), 1e-30):
                return False, '%s D%d k%d: P(P(g)) != P(g)' % (case, D, k)
            # the first solve leaves violations up to tol*||x||, so a second
            # call may legitimately fire a negligible touch-up -- reject only a
            # non-negligible one
            if d2['fired'] != 0.0 and d2['delta_norm'] > 1e-6 * max(float(g.norm()), 1e-30):
                return False, '%s D%d k%d: second call made a non-negligible edit' % (case, D, k)
        return True, ''

    def test_c():
        for D in dims:
            for case in ('generic', 'near_parallel', 'duplicate', 'feasible', 'apex'):
                for _ in range(4):     # 'zero_row' excluded: the reference has no filter
                    protos, g = _random_problem(D, 2, case, torch.float64, gen)
                    gr, _ = cone_rectify(protos, g, tol=tol)
                    err = float((gr.double() - _ref_project_k2(protos, g)).norm())
                    if err > 1e-8 * max(1.0, float(g.norm())):
                        return False, '%s D%d: |x - x_ref| = %.3e' % (case, D, err)
        return True, ''

    def test_d():
        for D in dims:
            for k in ks:
                protos, g = _random_problem(D, k, 'feasible', torch.float64, gen)
                gr, diag = cone_rectify(protos, g, tol=tol)
                if gr is not g or diag['fired'] != 0.0:
                    return False, 'D%d k%d: expected the identical object back' % (D, k)
        return True, ''

    def test_e():
        # projections onto convex sets are 1-Lipschitz; the +1e-7 absolute slack
        # covers two tol-accurate solves
        for D, k in ((64, 2), (64, 5), (300, 20)):
            for _ in range(5):
                protos, x = _random_problem(D, k, 'generic', torch.float64, gen)
                y = x + torch.randn(D, generator=gen, dtype=torch.float64)
                px, _ = cone_rectify(protos, x, tol=tol)
                py, _ = cone_rectify(protos, y, tol=tol)
                lhs = float((px.double() - py.double()).norm())
                rhs = float((x - y).norm()) * (1 + 1e-9) + 1e-7
                if lhs > rhs:
                    return False, 'D%d k%d: %.6e > %.6e' % (D, k, lhs, rhs)
        return True, ''

    def test_f():
        try:
            from scipy.optimize import nnls
        except Exception:
            nnls = None
        for D, k in ((64, 2), (64, 5), (300, 20)):
            for case in ('generic', 'near_parallel', 'duplicate', 'apex'):
                protos, g = _random_problem(D, k, case, torch.float64, gen)
                gr, _ = cone_rectify(protos, g, tol=tol)
                A, _keep = unit_kept(protos)
                if nnls is not None:
                    # dual == nnls(A^T, -g); compare x, never lambda (non-unique)
                    lam_ref, _ = nnls(A.T.numpy(), (-g).numpy())
                    ref = g + A.T @ torch.from_numpy(lam_ref)
                else:
                    ref = _ref_project_pg(protos, g)
                err = float((gr.double() - ref).norm())
                if err > 1e-6 * max(1.0, float(g.norm())):
                    return False, '%s D%d k%d: |x - x_ref| = %.3e (%s)' % (
                        case, D, k, err, 'nnls' if nnls is not None else 'pg')
        return True, ''

    def test_g():
        if not torch.cuda.is_available():
            print('SKIP g_gpu_parity  (no CUDA)')
            return True, ''
        p64, g64 = _random_problem(300, 5, 'generic', torch.float64, gen)
        p32, x32 = p64.float().cuda(), g64.float().cuda()
        gr32, _ = cone_rectify(p32, x32, tol=tol)
        if gr32.dtype != torch.float32 or gr32.device.type != 'cuda':
            return False, 'output dtype/device does not match the input'
        ref, _ = cone_rectify(p32.double().cpu(), x32.double().cpu(), tol=tol)
        err = float((gr32.double().cpu() - ref).norm())
        if err > 1e-4 * max(1.0, float(ref.norm())):
            return False, 'fp32 CUDA vs fp64 CPU differ by %.3e' % err
        pf, gf = _random_problem(300, 5, 'feasible', torch.float64, gen)
        pf32, gf32 = pf.float().cuda(), gf.float().cuda()
        grf, df = cone_rectify(pf32, gf32, tol=tol)
        if grf is not gf32 or df['fired'] != 0.0:
            return False, 'feasible fp32 CUDA input did not no-op to the same object'
        return True, ''

    run('a_kkt', test_a)
    run('b_idempotent', test_b)
    run('c_k2_closed_form', test_c)
    run('d_noop_identity', test_d)
    run('e_nonexpansive', test_e)
    run('f_external_reference', test_f)
    run('g_gpu_parity', test_g)
    print('%d test(s) FAILED' % fails if fails else 'all tests passed')
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_run_self_tests())
