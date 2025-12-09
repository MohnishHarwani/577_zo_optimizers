import jax, jax.numpy as jnp
from celo.factory import get_optimizer
from celo.utils import load_state
from compress_celo import prune_by_magnitude

lopt2 = get_optimizer("celo")
tmpl2 = lopt2.init(jax.random.PRNGKey(0))
theta2 = load_state("./theta_phase2.state", tmpl2)

for k in theta2:
    arrs = jax.tree.leaves(theta2[k])
    total = sum(a.size for a in arrs)
    mean_abs = sum(jnp.abs(a).sum() for a in arrs) / total
    print(k, "total params:", total, "mean |w|:", float(mean_abs))

