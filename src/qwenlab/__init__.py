"""qwenlab: modular from-scratch Qwen pipeline. Importing this package registers all built-in components."""
__version__ = "0.3.0"

from . import models, tasks, training, data  # noqa: F401  (each import registers its components)
