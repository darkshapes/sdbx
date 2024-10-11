from sdbx.config import config
############## DEBUG TOOLS
# pass using self.debugger(locals())


def s(variables):
    var_list = [
    "__doc__",
    "__name__",
    "__package__",
    "__loader__",
    "__spec__",
    "__annotations__",
    "__builtins__",
    "__file__",
    "__cached__",
    "config",
    "indexer",
    "json",
    "os",
    "defaultdict",
    "IndexManager",
    "logger",
    "psutil",
    "var_list",
    "i"
    "diffusers"
    "transformers"
    "torch"
    "vae"
    "AutoencoderKL"
    "generation"
    "tensor"
    ]
    for each in variables:
        if each not in var_list:
            print(f"{each} = {variables[each]}")