rom functools import partial
from inspect import isgeneratorfunction

def node(fn=None, **kwargs): # Arguments defined in NodeInfo init
    """
    Decorator for nodes. All functions using this decorator will be automatically read by the node manager.
    Parameters
    ----------
        path : str
            The path that the node will appear in.
        name : str
            The display name of the node.
        display : bool
            Whether to display the output in the node.
    """
    if fn is None:
        return partial(node, **kwargs)

    fn.generator = isgeneratorfunction(fn)

    from sdbx.nodes.info import NodeInfo  # Avoid circular import
    fn.info = NodeInfo(fn, **kwargs)

    # from sdbx.nodes.tuner import NodeTuner
    # fn.tuner = NodeTuner(fn)

    return fn
