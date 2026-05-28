"""Universal export orchestrator — run any combination of export formats.

Entry point for the Swarm Export tab "Generate Configs" button and the
CLI `python -m distill.export.export_matrix`.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Format key → (module, function)
_EXPORTERS: dict[str, tuple[str, str]] = {
    "gguf":          ("distill.orchestration.export_utils",   "export_gguf"),
    "mlx":           ("distill.orchestration.export_utils",   "export_mlx_quant"),
    "coreml":        ("distill.orchestration.export_utils",   "export_coreml"),
    "safetensors":   ("distill.export.safetensors_export",    "export_safetensors"),
    "awq":           ("distill.export.awq_gptq_export",       "export_awq"),
    "gptq":          ("distill.export.awq_gptq_export",       "export_gptq"),
    "exl2":          ("distill.export.exl2_export",           "export_exl2"),
    "onnx":          ("distill.export.onnx_export",           "export_onnx"),
}

# UI label → format key
LABEL_TO_KEY: dict[str, str] = {
    "GGUF (llama.cpp)":     "gguf",
    "MLX":                  "mlx",
    "CoreML":               "coreml",
    "Safetensors + HF Hub": "safetensors",
    "AWQ (4-bit GPU)":      "awq",
    "GPTQ (4-bit GPU)":     "gptq",
    "EXL2 (high-perf GPU)": "exl2",
    "vLLM config":          "vllm_config",
    "ONNX":                 "onnx",
}


def run_export_matrix(
    model_path: str,
    formats: list[str],
    output_dir: str,
    quant_method: str = "q4_k_m",
    merge_lora: bool = False,
    base_model_id: str | None = None,
    hub_repo_id: str = "",
    hub_token: str | None = None,
    vllm_port: int = 8000,
) -> dict[str, Any]:
    """Run exports for all requested formats.

    Args:
        model_path:    Path to the trained checkpoint (merged or LoRA).
        formats:       List of format keys or UI labels.
        output_dir:    Root output directory; sub-dirs created per format.
        quant_method:  GGUF/MLX quantization method.
        merge_lora:    If True, merge LoRA before export (needed for GGUF/ONNX).
        base_model_id: Required if merge_lora=True.
        hub_repo_id:   HF Hub repo for safetensors push.
        hub_token:     HF Hub token.
        vllm_port:     Port for vLLM config.

    Returns:
        Dict mapping format_key → result_dict.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Normalise format labels → keys
    keys = [LABEL_TO_KEY.get(f, f) for f in formats]

    # Optionally merge LoRA first
    src_path = model_path
    if merge_lora and base_model_id:
        logger.info("Merging LoRA before export...")
        try:
            from distill.export.lora_merger import merge_lora_into_base
            merged = merge_lora_into_base(
                base_model_id, model_path, str(out_root / "merged")
            )
            src_path = str(merged)
        except Exception as exc:
            logger.error("LoRA merge failed: %s — using original path", exc)

    results: dict[str, Any] = {}

    for key in keys:
        fmt_out = str(out_root / key)
        logger.info("Exporting format: %s → %s", key, fmt_out)

        if key == "vllm_config":
            try:
                from distill.export.vllm_config_generator import (
                    generate_vllm_config, save_config,
                )
                cfg = generate_vllm_config(src_path, port=vllm_port)
                results["vllm_config"] = save_config(cfg, fmt_out)
            except Exception as exc:
                results["vllm_config"] = {"error": str(exc)}
            continue

        if key not in _EXPORTERS:
            results[key] = {"error": f"Unknown export format: {key}"}
            continue

        module_path, fn_name = _EXPORTERS[key]
        try:
            import importlib
            mod  = importlib.import_module(module_path)
            fn   = getattr(mod, fn_name)
            # Call with positional args each exporter expects
            if key in ("gguf", "mlx", "coreml"):
                from pathlib import Path as _P
                from distill.infra.paths import project_dir
                result = fn(_P(src_path), project_dir(), quant_method)
            elif key == "safetensors":
                result = fn(src_path, fmt_out,
                            push_to_hub=bool(hub_repo_id),
                            hub_repo_id=hub_repo_id,
                            hub_token=hub_token)
            else:
                result = fn(src_path, fmt_out)
            results[key] = result if isinstance(result, dict) else {"output_path": str(result)}
        except Exception as exc:
            logger.error("Export %s failed: %s", key, exc)
            results[key] = {"error": str(exc)}

    return results


def zip_export_results(results: dict[str, Any], output_dir: str) -> str | None:
    """Bundle all successful exports into a ZIP file. Returns path or None."""
    out = Path(output_dir)
    zip_path = out / "production_pack.zip"
    found_files: list[Path] = []

    for key, result in results.items():
        path_str = result.get("output_path", "") if isinstance(result, dict) else ""
        if not path_str:
            continue
        p = Path(path_str)
        if p.is_file():
            found_files.append(p)
        elif p.is_dir():
            found_files.extend(f for f in p.rglob("*") if f.is_file())

    if not found_files:
        return None

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in found_files:
            zf.write(f, f.relative_to(out))

    logger.info("Production pack: %s (%d files)", zip_path, len(found_files))
    return str(zip_path)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Universal export matrix CLI")
    p.add_argument("--model", required=True)
    p.add_argument("--formats", default="gguf,mlx,safetensors")
    p.add_argument("--output-dir", default="exports")
    p.add_argument("--quant", default="q4_k_m")
    p.add_argument("--merge-lora", action="store_true")
    p.add_argument("--base-model", default="")
    args = p.parse_args()

    results = run_export_matrix(
        model_path=args.model,
        formats=args.formats.split(","),
        output_dir=args.output_dir,
        quant_method=args.quant,
        merge_lora=args.merge_lora,
        base_model_id=args.base_model or None,
    )
    for fmt, result in results.items():
        status = "✅" if not result.get("error") else "❌"
        print(f"  {status} {fmt}: {result.get('output_path', result.get('error', ''))}")


if __name__ == "__main__":
    main()
