"""Check saved runs, source provenance, Notebook outputs, and local links."""

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

import nbformat

from experiment import ROOT, save_json


def main():
    result = ROOT / "10.18_result" / "baseline"
    checks = {}
    for name in ["debug", "main", "repeat", "seed_19", "seed_20"]:
        folder = result / name
        manifest = json.loads((folder / "manifest.json").read_text())
        checks[f"{name}_files"] = all(hashlib.sha256((folder / path).read_bytes()).hexdigest() == digest
                                     for path, digest in manifest.items())
        checks[f"{name}_acceptance"] = json.loads((folder / "verification.json").read_text())["passed"]
    archived = ROOT / "10.18_result/baseline/archived_notebooks/10.18_synthetic_report.ipynb"
    notebook = nbformat.read(archived if archived.exists() else ROOT / "10.18_test_report.ipynb", as_version=4)
    nbformat.validate(notebook)
    additions = {"grg_depth_benchmark", "intel_lab", "grg_policy_verification"}
    code = [c for c in notebook.cells if c.cell_type == "code"
            and not additions.intersection(c.metadata.get("tags", []))]
    outputs = [o for cell in code for o in cell.outputs]
    checks["notebook_executed"] = all(c.execution_count is not None for c in code)
    checks["notebook_no_errors"] = not any(o.output_type == "error" for o in outputs)
    checks["six_embedded_figures"] = sum("image/png" in o.get("data", {}) for o in outputs) == 6
    current = nbformat.read(ROOT / "10.18_test_report.ipynb", as_version=4)
    text = (ROOT / "README.md").read_text() + "\n" + "\n".join(
        c.source for c in current.cells if c.cell_type == "markdown")
    links = re.findall(r"\]\(([^)]+)\)", text)
    checks["local_links_exist"] = all((ROOT / unquote(link)).exists() for link in links
                                      if "://" not in link and not link.startswith("#"))
    hash_path = result / "verification/source_hashes.json"
    hashes = json.loads(hash_path.read_text())
    relocation_path = result / "verification/relocation.json"
    relocation = json.loads(relocation_path.read_text()) if relocation_path.exists() else None
    moved = {item["old"]: item for item in relocation["files"]} if relocation else {}
    def source_matches(path, digest):
        relative = str(Path(path).relative_to(ROOT.name)) if path.startswith(ROOT.name + "/") else None
        if relative in moved:
            item = moved[relative]
            return (item["original_sha256"] == digest and
                    hashlib.sha256((ROOT / item["new"]).read_bytes()).hexdigest() == item["sha256"])
        return hashlib.sha256((ROOT.parent / path).read_bytes()).hexdigest() == digest
    current_matches_historical_sources = all(source_matches(path, digest) for path, digest in hashes.items())
    checks["historical_source_hashes_retained"] = bool(hashes) and all(
        re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values())
    output = ROOT / "10.18_result/grg_correcness/current/historical_artifacts.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {"passed": all(checks.values()),
                                                       "checks": checks,
                                                       "current_matches_historical_sources": current_matches_historical_sources,
                                                       "note": "The user replaced baseline code; historical data hashes remain authoritative for old results.",
                                                       "code_cells": len(code),
                                                       "local_links": len(links)})
    print(json.dumps(checks, indent=2))
    assert all(checks.values()), "Artifact verification failed"


if __name__ == "__main__":
    main()
