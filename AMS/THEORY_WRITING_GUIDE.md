# Theory and paper-writing guide for AMS

## Recommended narrative

Hybrid attacks create two coupled failures. First, DoS and FDI make the observation/action stream incomplete or physically inconsistent. Second, if the reward is computed from attacked quantities, the learning target can be misaligned with the real platoon objective. The reward-teacher layer fixes the direction of the objective using privileged clean physical state and executed input. However, even with a better reward direction, the actor may make unstable local changes when the attack makes the local Q surface sharp, noisy, or poorly identified. AMS addresses this second failure by smoothing Q values over feasible nearby actions on the Ackermann manifold.

Use the following division of responsibility throughout the manuscript:

    Meta reward shaping: objective alignment with clean physical safety.
    AMS: local action-geometry regularization and conservative credit assignment.
    HAPPO/CTDE: multi-agent factorized policy optimization.
    Ackermann projection: physically realizable actuation.

## Notation

Let i in {1,...,N}, N=5, index vehicles. Let s_t be the privileged physical state, o_t^a the possibly attacked observation, a_t^pi the actor proposal, and a_t^e the executed action after projection, shield, and attack effects. Each per-vehicle action is two-dimensional:

    a_i = [v_i, delta_i]

where delta_i is the center-steering command. Let M_A(s) be the feasible Ackermann action manifold at state s, and let P_A be the feasibility projection. Let Q_phi(s,a) be the centralized action value and Q_bar_phi its Polyak target.

## Ideal teacher objective and implemented approximation

The ideal bilevel statement can be written as:

    theta_plus(phi) = U(theta_minus, phi; attacked trajectories)
    phi_star = argmax_phi J_phy(theta_plus(phi))

where J_phy is evaluated from clean physical state and executed action. A first-order truncated meta-gradient is:

    grad_phi J_phy = (d theta_plus / d phi)^T grad_theta_plus J_phy.

In the implementation, the teacher stores a window of physical scores and shaping outputs, uses a moving-average baseline, and optimizes a clipped/proxy alignment term based on normalized physical-score improvement and teacher-output consistency. Therefore write:

    implemented teacher = stable first-order/proxy-aligned approximation

and reserve “meta-gradient” for the ideal objective or for the method name inherited from the reference line of work. Do not imply that the code backpropagates through every simulator transition and optimizer step.

## AMS local action geometry

For a center action a_e, sample K orthogonal directions d_k and a small radius epsilon. The feasible neighbors are:

    a_k_plus  = P_A(a_e + epsilon d_k)
    a_k_minus = P_A(a_e - epsilon d_k).

The conservative local value is:

    Q_smooth(s,a_e) = min over twin critics, then aggregate over the symmetric feasible neighborhood.

The code returns the neighborhood mean mu_Q and variance sigma_Q squared. A useful abstract form is:

    Q_AMS(s,a) = E_k[ min(Q_1(s,a_k), Q_2(s,a_k)) ]
                 - lambda_risk * Std_k[ min(Q_1(s,a_k), Q_2(s,a_k)) ].

The local advantage correction is anchored to the executed action:

    A_AMS = Q_min(s,a_e) - Q_AMS(s, a_pi).

The per-agent correction is obtained by the local credit reduction used by the runner. The final HAPPO update is conceptually:

    A_i^final = Normalize(A_i^HAPPO) + beta_t Normalize(A_i^AMS),

where beta_t = 0 during warm-up and increases linearly to the configured weight during the ramp. The current default target is 0.10 after 50 warm-up updates plus 100 ramp updates.

## Ackermann feasibility argument

The normalized policy command is not four independent wheel angles. The projection maps the two-dimensional command into front left/right steering and rear wheel velocities. A curvature bound and steering limits define the feasible set. Because all AMS neighbors pass through this projection, the local critic probes only physically realizable commands. This supports the paper claim “manifold-aware” rather than merely “action-noise smoothing.”

The rear-wheel steering constraint should be stated explicitly: rear joints have velocity targets only, and their steering targets are identically absent/zero.

## Conditional stability proposition

Present stability as a proposition, not as an unconditional theorem. A suitable statement is:

**Proposition (local ultimate boundedness under bounded hybrid attack).** Assume (i) the nominal platoon tracking error dynamics are locally dissipative under the projected policy, (ii) DoS/FDI, hold, and execution disturbances are bounded, (iii) the teacher residual and AMS critic residual are bounded, and (iv) the action projection and shield are non-expansive in the local neighborhood. Then a quadratic Lyapunov function of normalized spacing, velocity, and acceleration errors has a one-step drift bounded by a negative local term plus a finite disturbance/residual term. Consequently, all errors that remain in the local region enter an ultimate bound whose radius increases monotonically with the attack, hold, teacher, and critic residual bounds. If that bound lies inside the prescribed lane-change, spacing, and road margins, the whole-platoon completion gate remains feasible.

Do not claim that this proposition proves arbitrary hard-b success. It explains why lower local action variance and better physical reward alignment should enlarge the empirical safe region.

## Suggested theorem proof skeleton

1. Define normalized platoon error z_t from adjacent spacing, velocity, acceleration, lateral, heading, and lane-change progress errors.
2. Use local dissipativity to obtain V(z_{t+1}) - V(z_t) <= -c||z_t||^2 plus cross terms.
3. Bound hybrid attack/hold/execution terms by a disturbance constant d_attack.
4. Bound teacher residual and AMS residual by d_teacher and d_AMS.
5. Use non-expansive projection and shield to avoid increasing the local action perturbation bound.
6. Complete the square to obtain a negative quadratic term plus D = d_attack + d_teacher + d_AMS.
7. State the resulting ultimate radius and the margin condition for the joint completion gate.

## How to phrase novelty

The novelty is not simply “we add another critic.” It is the coupling of three constraints that are usually separated:

- an outer physical objective aligned with clean state and executed action;
- an inner local action-value geometry defined on the feasible Ackermann manifold; and
- a whole-platoon completion gate that evaluates five vehicles on one shared spatial path.

The orthogonal neighborhood, executed-action anchor, twin-critic minimum, and warm-up/ramp are implementation mechanisms supporting this coupling.

## Ablation interpretation

The clean four-cell orthogonal experiment should separate teacher-only, AMS-only, and combined effects. Because the early AMS-only pilot had a share-actor confound and failed light, report it as a diagnostic failure and explain that the corrected shared-actor multiseed run is the causal revalidation. A combination beating both isolated components is evidence for complementarity, not proof of a universal interaction theorem.

