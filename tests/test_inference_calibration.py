import math
import random

from lob_forge.alpha_factory import (
    one_sided_hac_p_value_mean_le_zero,
    one_sided_separated_batch_t_p_value_mean_le_zero,
)


def test_guarded_inference_null_size_on_declared_gaussian_stress_domain() -> None:
    # Seeded distributional regression, not a universal coverage assertion.
    # The old HAC-only path rejects roughly 25% at n=20, phi=.8.
    rng = random.Random(20260907)
    for n, phi in [(20, 0.0), (20, 0.8), (60, 0.8), (20, 0.95)]:
        rejected = 0
        replications = 2000
        for _ in range(replications):
            state = rng.gauss(0, 1)
            values = []
            for _ in range(n):
                state = phi * state + math.sqrt(1 - phi * phi) * rng.gauss(0, 1)
                values.append(state)
            p_value = max(one_sided_hac_p_value_mean_le_zero(values),
                          one_sided_separated_batch_t_p_value_mean_le_zero(values))
            rejected += p_value < 0.05
        assert rejected / replications < 0.065
