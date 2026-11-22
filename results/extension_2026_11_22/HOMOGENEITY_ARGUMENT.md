# Why the frozen-model scaling intervention is informative

Let L be the linear low-fidelity advection solver. Let V be the family of
unit-maximum-amplitude single waves, with arbitrary phase. Suppose an emulator
f fits L exactly on that entire family (an idealization of the finite training
set). Define, for a nonzero state u,

    a = max(abs(u))
    f_hom(u) = a * f(u/a).

For any positive scale a and unit wave v in V,

    f_hom(a*v) = a*f(v) = a*L(v) = L(a*v).

Both advection teachers preserve the single-mode wave family, changing its
phase and amplitude. Therefore the scaled emulator and teacher agree at every
future step, by induction. Their error against any common reference is the
same, so their superiority ratio is exactly 1 whenever the denominator is
nonzero.

Real models only approximately fit the teacher and see finitely many phases,
so the observed ratio need not be exactly 1. The experiment measures that
deviation instead of assuming it away. Nevertheless, forcing this known scaling
law removes the otherwise unconstrained amplitude direction while retaining
the learned weights and initial prediction. A large original advantage that
disappears under this intervention supports the amplitude-feedback explanation.

This argument is deliberately restricted to linear single-mode advection. It
does not establish anything about nonlinear Burgers dynamics, and it does not
invalidate the paper's separate linear-operator/state-space superiority results.
