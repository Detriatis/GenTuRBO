#!/usr/bin/env python3
"""Prepare the locked local environments and optionally import frozen artifacts."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]



def external_sources(root: Path = ROOT) -> dict[str, dict]:
    manifest = json.loads((root / "external/manifest.json").read_text())
    sources = {item["name"]: item for item in manifest["sources"]}
    if manifest.get("schema_version") != 1 or len(sources) != len(manifest["sources"]):
        raise ValueError("Invalid or duplicate external source declaration")
    return sources


def poli_baselines_requirement(root: Path = ROOT) -> str:
    source = external_sources(root)["poli-baselines"]
    repository = source["repository"].removesuffix(".git")
    revision = source["revision"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("poli-baselines requires an immutable source commit")
    return f"poli-baselines[lambo2] @ git+{repository}.git@{revision}"


def validate_dependency_sources(root: Path = ROOT) -> dict:
    # Imported only after selecting Python >=3.11; system Python 3.9 may
    # invoke this bootstrap before uv has prepared the requested interpreter.
    import tomllib
    sources = external_sources(root)
    main = tomllib.loads((root / "pyproject.toml").read_text())
    main_lock = tomllib.loads((root / "uv.lock").read_text())
    lambo = tomllib.loads((root / "environments/lambo2/pyproject.toml").read_text())
    lambo_lock = tomllib.loads((root / "environments/lambo2/uv.lock").read_text())
    def package(lock, name):
        found = [item for item in lock["package"] if item["name"] == name]
        if len(found) != 1:
            raise ValueError(f"Expected one locked external distribution: {name}")
        return found[0]
    def repository(url):
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path.removesuffix(".git"), "", ""))
    for name, distribution, project, lock in (("evodiff", "evodiff", main, main_lock),
                                               ("poli", "poli-core", lambo, lambo_lock)):
        declared = sources[name]
        selected = project["tool"]["uv"]["sources"][distribution]
        resolved = package(lock, distribution)["source"].get("git", "")
        parsed = urlsplit(resolved)
        if (repository(selected["git"]) != declared["repository"].removesuffix(".git")
                or selected.get("rev") != declared["revision"]
                or repository(resolved) != declared["repository"].removesuffix(".git")
                or parse_qs(parsed.query).get("rev") != [declared["revision"]]
                or parsed.fragment != declared["revision"]):
            raise ValueError(f"External source declaration drift for {name}: check manifest, project and lock")
    esm = sources["esm"]
    install = esm["installation"]
    expected = f'{install["distribution"]}=={install["version"]}'
    locked = package(main_lock, install["distribution"])
    if (esm["revision"] != expected
            or expected not in main["project"]["optional-dependencies"]["protein"]
            or locked["version"] != install["version"]
            or locked["source"] != {"registry": install["registry"]}
            or locked["sdist"]["url"] != install["sdist_url"]
            or locked["sdist"]["hash"] != install["sdist_hash"]):
        raise ValueError("External source declaration drift for esm: check manifest, project and lock")
    return {"ok": True, "sources": ["evodiff", "poli", "esm"],
            "poli_baselines_requirement": poli_baselines_requirement(root)}


def run(command: list[str], *, environment: Path | None = None) -> None:
    env = dict(os.environ)
    if environment is not None:
        env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def python_at(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "cuda128"), default="cpu")
    parser.add_argument("--python", choices=("3.11", "3.12"), default="3.12")
    parser.add_argument("--protein", action="store_true", help="Install pinned EvoDiff and ESM adapters")
    parser.add_argument("--lambo2", action="store_true", help="Also prepare the isolated Python 3.10 LaMBO-2 runtime")
    parser.add_argument("--dev", action="store_true", help="Include the compact validation suite dependencies")
    parser.add_argument("--artifacts", type=Path, help="Import the companion GenTuRBO-artifacts.tar.gz")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--prefetch-models", action="store_true", help="Download EvoDiff 38M and ESM2 650M weights (requires --protein)")
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_dependency_sources(), indent=2))
        return 0
    uv = shutil.which("uv")
    if uv is None:
        parser.error("Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
    if args.profile == "cuda128" and platform.system() != "Linux":
        parser.error("The CUDA 12.8 profile requires Linux; use cpu for CPU or Apple MPS")
    if args.prefetch_models and not args.protein:
        parser.error("--prefetch-models requires --protein")
    if sys.version_info >= (3, 11):
        validate_dependency_sources()
    else:
        run([uv, "run", "--no-project", "--python", args.python, "--",
             "python", str(Path(__file__).resolve()), "--validate-only"])
    command = [uv, "sync", "--locked", "--project", str(ROOT), "--python", args.python,
               "--extra", args.profile, "--extra", "search", "--extra", "analysis"]
    command += ["--group", "dev"] if args.dev else ["--no-dev"]
    if args.protein:
        command += ["--extra", "protein"]
    run(command, environment=ROOT / ".venv")
    runtime = str(python_at(ROOT / ".venv"))
    run([runtime, "-c", "import genturbo, torch, botorch, gpytorch; print('GenTuRBO environment ready; torch', torch.__version__)"])
    if args.protein:
        run([runtime, "-c", "import evodiff.pretrained, esm; print('Protein adapters import successfully')"])
    if args.lambo2:
        lambo_env = ROOT / ".venv-lambo2"
        run([uv, "sync", "--locked", "--no-dev", "--project", str(ROOT / "environments/lambo2"), "--python", "3.10"], environment=lambo_env)
        lambo_python = str(python_at(lambo_env))
        # The released adapter advertises an unpinned poli URL. Keep its exact
        # adapter commit but install its dependency closure from our lock above.
        run([uv, "pip", "install", "--python", lambo_python, "--no-deps", poli_baselines_requirement()])
        run([lambo_python, "-c", "from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2; print('LaMBO-2 runtime ready')"])
    if args.prefetch_models:
        run([runtime, "-m", "genturbo.model_weights", "--download"])
    if args.artifacts:
        run([runtime, "-m", "genturbo.assets", "import", str(args.artifacts.resolve()), "--root", str(args.artifact_root.resolve())])
    print("Use .venv/bin/genturbo --help (Windows: .venv/Scripts/genturbo.exe).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
