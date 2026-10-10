# Persistent latent activity: a hypothesis to review

This note was prepared before revealing the completed midpoint family and before any quote-renewal market fit. It does not alter the reflected-mixture prototype or register another experiment. Finish that prototype's E2E and computational preflight first.

## Primary-source check

[Hu's March 2026 neural-HMM preprint](https://arxiv.org/html/2603.20456v1) reports improvements for a 500-ms volatility-threshold target, unlike our five-second spread/tick target. Its section 5 names venues but does not identify assessment dates or symbols sufficiently for our replication. Equations 4–7 include the current observation in the context used to condition its own emission in equation 12; this needs clarification before interpreting the objective as a normalized generative likelihood. Equation 13 shows only context-dependent affine transforms, whereas the experiment describes masked coupling networks. Those descriptions need reconciliation. Equation 15 also displays a state-conditioned expression while the prose refers to marginal likelihood. These are reasons to require a precise implementation, not evidence that its reported results are false. We do not adopt its headline gains.

[Li's Cambridge thesis abstract](https://www.repository.cam.ac.uk/items/1c7b048d-f82b-44f2-be25-5ad0d8b5bc41) describes latent regime and state-space approaches to order-book data. This motivates investigating persistence; the abstract does not verify our task or the construction below.

## Our proposed distinction

The current two-component renewal mixture draws a new duration/mark component after each block from weights determined by the observed spread/imbalance state. This allows duration/mark dependence within one transition. It does not preserve an additional latent activity regime across blocks. Recent complete durations and marks could contain information about such persistence. That is a hypothesis requiring a matched reset-component control, not a demonstrated information gain.

Let z_n be the observed state at block n and r_n a two-valued latent component governing the next transition. For a complete observed gap and joint mark (g_n,m_n), define

    e_n(r) = p[z_n,r] * (1-p[z_n,r])^(g_n-1) * F[z_n,r](m_n).

The mark supplies z_(n+1). After the transition, draw r_(n+1) from a shared two-by-two matrix A[r_n,:]. An initial row distribution pi starts each independent source interval. For K complete transitions followed by right censoring at E after block t_K, the likelihood is

    pi * [diag(e_0) A] * ... * [diag(e_(K-1)) A] * c,
    c_r = (1-p[z_K,r])^(E-t_K-1).

The final multiplication by A introduces the component for the censored next interval; it is not an extra observed event. K=0 reduces to pi*c. No transition joins source windows.

The causal filter at an observed block propagates as

    alpha_(n+1) = normalize((alpha_n * e_n) A).

At a decision of age a>=1 after the last strictly earlier block, the current component posterior is proportional to alpha_n(r)*(1-p[z_n,r])^(a-1). Draw the initial residual as Geom(p[z_n,r])-1 and integrate future marked transitions using A. Retain the same first/last quote selection, exact target and explicit reflected-price law. Forecast paths can still be cached by initial observed state/component; past history changes their mixing weights.

Offline parameter fitting may use a forward/backward algorithm and the censored terminal factor. Online forecasting must use filtering and elapsed survival only. A smoothed state posterior that includes any later block would leak future information. Component identities must remain aligned across observed states: independent per-state relabeling is invalid when A connects components globally.

## Before any implementation

Review the likelihood and sufficient statistics independently. Specify sparse-state pooling, unseen-mark support during filtering, initialization, all priors, convergence and resource bounds. A future E2E must enumerate tiny latent paths independently, reproduce the matrix likelihood and posteriors, include a censored-only interval, prove exact filtering-prefix invariance, and compare the persistent and reset laws on the same synthetic tapes. The initial mixture's own E2E does not verify this extension. No latent-state labels, market interpretation or accuracy gain should be inferred merely from finding two fitted components.

Independent review identified unresolved design choices. A shared A with identical rows resets to one global component distribution, so it does not contain the existing state-dependent weights w_z as its reset special case. Either compare against a matched global-reset HMM, or specify destination-state-dependent transitions A[z_(n+1),r_n,r_(n+1)] whose identical rows can reproduce w_[z_(n+1)]. This choice must precede implementation and performance attribution.

The empirical exact-mark distribution also assigns zero mass to new marks outside training support. Unlike the current mixture, the proposed filter evaluates those past marks explicitly and could encounter zero likelihood for every component. It needs a coherent observation/coarsening law with a corresponding future sampler. An arbitrary unknown-mark score, discarding the mark when it is inconvenient, or coarsening away the observed next state would not implement the written likelihood. At a recorded bid exactly at the reflection floor, an observation likelihood also requires summing compatible reflected latent marks; the interior-quote training argument does not cover that case. These issues remain open and this extension is not ready to implement.

Keep alpha_n at the last completed block unchanged by queries. Form each decision's age posterior afresh; otherwise repeated queries or a later completed gap double-count survival. In EM, the terminal censored component contributes duration failures but no mark or outgoing transition. The preceding complete transition still contributes an A count because it introduces that censored component. A censored-only interval contributes an initial-distribution count and survival likelihood.
