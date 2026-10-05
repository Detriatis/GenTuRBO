"""toygen/core/registry.py
Simple, decorator-based component registry with optional lazy imports
and a convenience builder. Designed for plug-and-play experiments where
components (models, processes, schedules, datasets, integrators, etc.)
are requested by name from config

Usage
-----

>>> from toygen.core.registry import register, build, get, available
>>> @register('model', 'mlp')
... class MLP(torch.nn.module):
...     def __init__(self, in_dim=2, hidden=64, out_dim=2): ...
...
>>> m = build('model', 'mlp', in_dim=2, hidden=32, out_dim=2)

Lazy registration (import only on first use):
>>> register_lazy('process', 'diffusion_vp', 'toygen.process.diffusion_vp:VPProcess')
>>> proc = build('process', 'diffusion_vp', beta_schedule='cosine')

Aliases:
>>> alias('model', 'tiny_mlp', 'mlp')
>>> get('model', 'tiny_mlp') is get('model', 'mlp')


Notes
-----
- This registery is intentionally lightweight. It stores *callables* or *classes*
- 'build(kind, name, **cfg)' will:
    1) Resolve aliases and perform a lazy import if needed.
    2) If the target is a class or callable, call it with **cfg
    3) If the target has an attribute 'build' call 'target.build(**cfg)' instead (optional pattern)
- Helpful errors with "did you mean" suggestions.
"""

from __future__ import annotations

from collections import defaultdict
import difflib
import importlib
import inspect
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

# # # # # # # # # #
# Internal State  #
# # # # # # # # # #

_REG: Dict[str, Dict[str, Any]] = defaultdict(dict) # canonical name -> obj or lazy
_ALIASES: Dict[str, Dict[str, str]] = defaultdict(dict) # aliases -> canonical

# # # # # # # # # #
# Helpers         #
# # # # # # # # # #

class Lazy:
    """Holds an import path to import on first access.

    Accepts either "pkg.mod:attr" or "pkg.mod.attr"
    """

    def __init__(self, import_path: str):
        self.import_path = import_path
        self._obj: Optional[Any] = None

    def _load(self) -> Any:
        path = self.import_path
        mod_path, sep, attr = path.partition(":")
        if not sep:
            parts = path.split(".")
            if len(parts) < 2:
                raise ImportError(f"Invalid import path for Lazy: {path}")
            mod_path = ".".join(parts[:-1])
            attr = parts[-1]
        module = importlib.import_module(mod_path)
        obj = getattr(module, attr)
        return obj

    @property
    def obj(self) -> Any:
        if self._obj is None:
            self._obj = self._load()
        return self._obj

    def __repr__(self) -> str:
        status = "loaded" if self._obj is not None else "lazy"
        return f"Lazy({self.import_path!r}, {status})"

def _canonical(kind: str, name: str) -> str:
    # Resolve alias -> canonical
    if name in _REG[kind]:
        return name
    if name in _ALIASES[kind]:
        return _ALIASES[kind][name]
    # Not found; prepare helpeful error
    choices = sorted(list(_REG[kind].keys()) + list(_ALIASES[kind].keys()))
    hint = difflib.get_close_matches(name, choices, n=3)
    msg = f"No entry '{name}' registered under kind '{kind}'. Available: {', '.join(choices) or '(none)'}"
    if hint:
        msg += f". Did you mean: {', '.join(hint)}?"
    raise KeyError(msg)

# # # # # # # # # #
# Public API      #
# # # # # # # # # #
def register(kind: str, name: Optional[str] = None, *, aliases: Optional[Iterable[str]] = None):
    """Decorator to register a class or callable under a kind and name.

    Example:
        @register('model', 'mlp', aliases=['tiny_mlp'])
        class MLP(...):
            ...
    """
    def deco(obj: Any):
        nm = name or getattr(obj, "__name__", None)
        if nm is None:
            raise ValueError("Registry: name must be provided or object must have __name__")
        if nm in _REG[kind]:
            raise KeyError(f"Duplicate registration for kind='{kind}', name='{nm}'")
        _REG[kind][nm] = obj
        if aliases:
            for a in aliases:
                if a in _REG[kind] or a in _ALIASES[kind]:
                    raise KeyError(f"Alias '{a}' already used under kind '{kind}'")
                _ALIASES[kind][a] = nm
        return obj

    return deco

def register_lazy(kind: str, name: str, import_path: str, *, aliases: Optional[Iterable[str]] = None) -> None:
    """Register a lazy target by import path ("pkg.mod:attr" or "pkg.mod.attr")."""
    if name in _REG[kind]:
        raise KeyError(f"Duplicate registration for kind='{kind}', name='{name}'")
    _REG[kind][name] = Lazy(import_path)
    if aliases:
        for a in aliases:
            if a in _REG[kind] or a in _ALIASES[kind]:
                raise KeyError(f"Alias '{a}' already under kind '{kind}'")
            _ALIASES[kind][a] = name

def alias(kind: str, alias_name: str, target_name: str) -> None:
    """create an alias for an existing entry."""
    if target_name not in _REG[kind]:
        raise KeyError(f"Cannot alias unknown entry '{target_name}' (kind='{kind}')")
    if alias_name in _REG[kind] or alias_name in _ALIASES[kind]:
        raise KeyError(f"Alias '{alias_name}' already exists under kind '{kind}'")
    _ALIASES[kind][alias_name] = target_name

def available(kind: str) -> List[str]:
    """List canonical names registered under a kind (excludes aliases)."""
    return sorted(_REG[kind].keys())

def get(kind: str, name: str) -> Any:
    """Fetch the raw registered object (resolving aliases and lazy imports)."""
    nm = _canonical(kind, name)
    obj = _REG[kind][nm]
    if isinstance(obj, Lazy):
        obj = obj.obj
        _REG[kind][nm] = obj # cache loaded object
    return obj

def build(kind: str, name: str, /, **cfg: Any) -> Any:
    """Instantiate or call a registered entry with the provided config.

    Rules:
        - If target has a '.build(**cfg)' attribute, call that.
        - Else if target is callable (class or function), call it as 'target(**cfg)'.
        - Else error
    """
    if kind == "dataset" and name not in _REG[kind]:
        # Attempt lazy import by convention data.<name>
        module_name = f"data.{name}"
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Only an actually absent convention module is optional.  Missing
            # dependencies or import bugs inside that module must be visible.
            if exc.name not in {module_name, name}:
                raise
    target = get(kind, name)

    if hasattr(target, "build") and callable(getattr(target, "build")):
        return target.build(**cfg)

    if callable(target):
        try:
            return target(**cfg) # type: ignore[misc]
        except TypeError as e:
            sig = None
            try:
                sig = str(inspect.signature(target))
            except Exception:
                pass
            raise TypeError(f"Failed to build '{name}' of kind '{kind}' with cfg={cfg}. Signature {sig}") from e

    raise TypeError(f"Registered object for kind='{kind}', name='{name} is not callable and has no .build(...)")

def reset(kind: Optional[str] = None) -> None:
    """Clear the registery (for tests). If kind is None, clears all kinds."""
    if kind is None:
        _REG.clear(); _ALIASES.clear()
    else:
        _REG[kind].clear(); _ALIASES[kind].clear()

__all__ = [
    'register', 'register_lazy', 'alias', 'available', 'get', 'build', 'reset'
    ]

# Trigger default registrations for built-in component packages.
for _pkg in ("models", "processes"):
    try:
        __import__(_pkg)
    except ModuleNotFoundError as exc:
        if exc.name != _pkg:
            raise

#TODO write small test suite to check proper loading
