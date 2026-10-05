"""The release's frozen-input boundary must preserve exact, trusted payloads."""
import hashlib
import io
import json
import tarfile
from pathlib import PurePosixPath

import pytest

from genturbo.assets import MANIFEST_NAME, import_bundle, load_manifest, verify_assets


def bundle_fixture(tmp_path, payload=b"frozen scientific input", member=None):
    name = "benchmarks/example/input.dat"
    manifest = {"schema_version": 1, "files": [{"path": name, "group": "example",
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
    declared = tmp_path / "manifest.json"
    declared.write_text(json.dumps(manifest))
    bundle = tmp_path / "inputs.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path, data in [(MANIFEST_NAME, json.dumps(manifest).encode()), (name, member or payload)]:
            info = tarfile.TarInfo(path)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return bundle, declared, name


def test_import_is_verified_repeatable_and_refuses_changed_inputs(tmp_path):
    bundle, manifest, name = bundle_fixture(tmp_path)
    root = tmp_path / "artifacts"
    assert import_bundle(bundle, root, manifest=manifest)["imported"] == 1
    assert verify_assets(root, manifest=manifest)["ok"]
    assert import_bundle(bundle, root, manifest=manifest)["imported"] == 0
    (root / name).write_bytes(b"different scientific input")
    assert verify_assets(root, manifest=manifest)["mismatched"] == [name]
    with pytest.raises(FileExistsError):
        import_bundle(bundle, root, manifest=manifest)
    assert (root / name).read_bytes() == b"different scientific input"


def test_corrupt_bundle_does_not_import_any_payload(tmp_path):
    bundle, manifest, name = bundle_fixture(tmp_path, member=b"modified payload")
    root = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="mismatch"):
        import_bundle(bundle, root, manifest=manifest)
    assert not (root / name).exists()


def test_packaged_manifest_has_only_relative_scientific_payloads():
    manifest = load_manifest()
    assert {entry["group"] for entry in manifest["files"]} == {"ehrlich", "protein", "sokoban", "reference"}
    for entry in manifest["files"]:
        path = PurePosixPath(entry["path"])
        assert not path.is_absolute() and ".." not in path.parts
        assert path.suffix not in {".pdf", ".png", ".svg", ".tex", ".ipynb"}
        assert len(entry["sha256"]) == 64


@pytest.mark.parametrize("algorithm", ["sha256", "md5", "s3-multipart-md5"])
def test_model_cache_checks_provider_hashes_before_loading(tmp_path, algorithm):
    from genturbo.model_weights import verify_weight
    payload = b"fixed checkpoint bytes"
    path = tmp_path / "weights.pt"
    path.write_bytes(payload)
    chunks = [payload[i:i + 8] for i in range(0, len(payload), 8)]
    if algorithm == "s3-multipart-md5":
        checksum = hashlib.md5(b"".join(hashlib.md5(x).digest() for x in chunks)).hexdigest() + f"-{len(chunks)}"
    else:
        checksum = hashlib.new(algorithm, payload).hexdigest()
    spec = {"bytes": len(payload), "multipart_chunk_bytes": 8,
            "checksum": {"algorithm": algorithm, "value": checksum}}
    spec["sha256"] = hashlib.sha256(payload).hexdigest()
    assert verify_weight(path, spec)["sha256"] == spec["sha256"]
    spec["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 checksum mismatch"):
        verify_weight(path, spec)
    spec["sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_bytes(b"X" + payload[1:])
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_weight(path, spec)


@pytest.mark.parametrize("kind, expected", [("evodiff", ("evodiff",)), ("esm2", ("esm2", "esm2-regression"))])
def test_real_protein_loaders_verify_only_their_used_weights(monkeypatch, kind, expected):
    import sys
    from types import SimpleNamespace
    from genturbo import model_weights
    called = []
    def unavailable(*, download, names):
        called.append((download, tuple(names)))
        raise FileNotFoundError("run setup with --protein --prefetch-models")
    def must_not_load(*args, **kwargs):
        raise AssertionError("Unverified weights reached an upstream loader")
    monkeypatch.setattr(model_weights, "prepare_weights", unavailable)
    if kind == "evodiff":
        from models.evodiff import EvoDiffD3PMDenoiser
        backend = SimpleNamespace(D3PM_UNIFORM_38M=must_not_load)
        monkeypatch.setitem(sys.modules, "evodiff", SimpleNamespace(pretrained=backend))
        monkeypatch.setitem(sys.modules, "evodiff.pretrained", backend)
        load = EvoDiffD3PMDenoiser()._load_backend
    else:
        from adapters.protein.esm import ESM2Featurizer
        monkeypatch.setitem(sys.modules, "esm", SimpleNamespace(pretrained=SimpleNamespace(esm2_t33_650M_UR50D=must_not_load)))
        load = ESM2Featurizer(model_name="esm2_t33_650M_UR50D")._load
    with pytest.raises(FileNotFoundError, match="prefetch-models"):
        load()
    assert called == [(False, expected)]


def test_setup_rejects_external_source_version_and_url_drift(tmp_path):
    import importlib.util
    from pathlib import Path
    import shutil
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("release_setup", root / "scripts/setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    for name in ("external/manifest.json", "pyproject.toml", "uv.lock", "environments/lambo2/pyproject.toml", "environments/lambo2/uv.lock"):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    assert setup.validate_dependency_sources(tmp_path)["ok"]
    changes = [
        ("pyproject.toml", "https://github.com/microsoft/evodiff.git", "https://example.invalid/evodiff.git"),
        ("environments/lambo2/uv.lock", "bc38921a51cbb28b835e34a785e830f3ef3b9eb7", "0" * 40),
        ("uv.lock", "https://pypi.org/simple", "https://example.invalid/simple"),
    ]
    for relative, old, new in changes:
        path = tmp_path / relative
        original = path.read_text()
        assert old in original
        path.write_text(original.replace(old, new))
        with pytest.raises(ValueError, match="declaration drift"):
            setup.validate_dependency_sources(tmp_path)
        path.write_text(original)
    manifest_path = tmp_path / "external/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    adapter = next(item for item in manifest["sources"] if item["name"] == "poli-baselines")
    adapter["revision"] = "1" * 40
    manifest_path.write_text(json.dumps(manifest))
    assert setup.poli_baselines_requirement(tmp_path).endswith("@" + "1" * 40)
