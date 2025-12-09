import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import numpy as np

from celo.factory import get_optimizer
from celo.utils import load_state

PHASE2_CKPT = "./theta_phase2.state"  # adjust to your path

def tree_paths(tree):
    """Yield (path_tuple, leaf) for any nested dict/tuple/list/frozendict PyTree."""
    def _walk(node, prefix):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from _walk(v, prefix + (str(k),))
        elif hasattr(node, "items") and not isinstance(node, (str, bytes)):
            # e.g. Flax FrozenDict
            for k, v in node.items():
                yield from _walk(v, prefix + (str(k),))
        elif isinstance(node, (tuple, list)):
            for i, v in enumerate(node):
                yield from _walk(v, prefix + (str(i),))
        else:
            # leaf
            yield prefix, node
    return list(_walk(tree, ()))

def main():
    lopt = get_optimizer("celo")
    theta_tmpl = lopt.init(jax.random.PRNGKey(0))
    theta = load_state(PHASE2_CKPT, theta_tmpl)

    # 1) print structure
    for path, leaf in tree_paths(theta):
        if hasattr(leaf, "shape"):
            print("/".join(path), leaf.shape, leaf.dtype)

    # 2) save to npz
    arrays = {}
    for path, leaf in tree_paths(theta):
        if hasattr(leaf, "shape"):
            key = "/".join(path)
            arrays[key] = np.asarray(leaf)
    np.savez("celo_theta_phase2.npz", **arrays)
    print("Saved celo_theta_phase2.npz with", len(arrays), "arrays")

if __name__ == "__main__":
    main()

