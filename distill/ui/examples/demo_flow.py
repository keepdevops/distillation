#!/usr/bin/env python3
"""End-to-end demo flow — verifies the full Wow Sausage Maker pipeline.

Exercises every major subsystem without requiring a live model or GPU:
  1. Config validation (schemas)
  2. Registry (tabs, backends, formats)
  3. Event bus (pub/sub)
  4. LoRA bridge (VRAM estimation, metrics)
  5. Export bridge (format specs, manifest)
  6. Production pack (ZIP generation)
  7. Data pipeline (safety filter, deduplication)
  8. Checkpoint resume detection
  9. Webhook config (dry-run)
  10. App build (all tabs load)

Run: python distill/ui/examples/demo_flow.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

_PASS = "✅"
_FAIL = "❌"
_results: list[tuple[str, bool, str]] = []


def step(name: str, fn) -> None:
    try:
        detail = fn() or ""
        _results.append((name, True, str(detail)))
        print(f"  {_PASS} {name}{': ' + str(detail)[:80] if detail else ''}")
    except Exception as exc:
        _results.append((name, False, str(exc)))
        print(f"  {_FAIL} {name}: {exc}")


def main() -> int:
    print("\n🌭 Wow Sausage Maker — End-to-End Demo Flow")
    print("=" * 60)

    # ── 1. Config schemas ─────────────────────────────────────────────
    print("\n[1/10] Config Schemas")

    def _lora_schema():
        from distill.config.schemas import LoRAConfig
        lc = LoRAConfig(rank=32, alpha=64)
        assert lc.scaling == 2.0
        return f"rank={lc.rank}, scaling={lc.scaling}, vram={lc.estimated_vram_mb():.1f}MB"

    def _training_schema():
        from distill.config.schemas import TrainingConfig
        cfg = TrainingConfig(teacher="Qwen/Qwen2-1.5B-Instruct",
                             student="Qwen/Qwen2-0.5B-Instruct")
        assert "--backend" in cfg.to_cli_args()
        return f"backend={cfg.backend}, eff_batch={cfg.effective_batch_size()}"

    def _preset_schema():
        from distill.config.schemas import from_preset
        p = from_preset("Code Specialist")
        return f"preset loaded: {p.backend} / rank={p.lora.rank}"

    step("LoRAConfig",     _lora_schema)
    step("TrainingConfig", _training_schema)
    step("from_preset",    _preset_schema)

    # ── 2. Registry ───────────────────────────────────────────────────
    print("\n[2/10] Plugin Registry")

    def _registry_summary():
        from distill.ui.core.registry import registry
        s = registry.summary()
        assert s["tabs"] >= 12 and s["backends"] >= 5 and s["export_formats"] >= 9
        return f"tabs={s['tabs']}, backends={s['backends']}, formats={s['export_formats']}"

    def _registry_platform_filter():
        from distill.ui.core.registry import registry
        mps  = registry.backend_choices(platform="mps")
        cuda = registry.backend_choices(platform="cuda")
        assert "mlx" in mps and "mlx" not in cuda
        return f"mps={mps}, cuda-only={[b for b in cuda if b not in mps]}"

    step("Registry summary",         _registry_summary)
    step("Platform-filtered backends", _registry_platform_filter)

    # ── 3. Event bus ──────────────────────────────────────────────────
    print("\n[3/10] Event Bus")

    def _event_bus():
        from distill.ui.core.event_bus import EventBus, Topic
        bus = EventBus()
        events = []
        bus.on(Topic.TRAINING_STEP, lambda e: events.append(e))
        bus.emit(Topic.TRAINING_STEP, {"step": 100, "loss": 0.85})
        bus.emit_async(Topic.EXPORT_COMPLETE, {"format": "gguf"})
        time.sleep(0.05)
        assert len(events) == 1
        return f"fired 2 topics, received {len(events)} sync event"

    step("EventBus pub/sub", _event_bus)

    # ── 4. LoRA bridge ────────────────────────────────────────────────
    print("\n[4/10] LoRA Bridge")

    def _lora_vram():
        from distill.backends.lora_bridge import estimate_vram
        v = estimate_vram(rank=16, hidden_size=2048, num_layers=24,
                          base_model_gb=3.0, batch_size=4)
        assert v["total_gb"] > 3.0
        return f"total={v['total_gb']:.2f}GB"

    def _lora_dead():
        from distill.backends.lora_bridge import push_training_metrics
        m = push_training_metrics(1, adapter_norms=[0.0, 0.15], update_ratios=[], grad_norms=[])
        return f"dead_layers={m['dead_layers']}, health={m['health_ok']}"

    step("VRAM estimation", _lora_vram)
    step("Dead layer detection", _lora_dead)

    # ── 5. Export bridge ──────────────────────────────────────────────
    print("\n[5/10] Export Bridge")

    def _export_manifest():
        from distill.backends.export_bridge import build_result, build_manifest
        results = [
            build_result("gguf", "/tmp/model.gguf", success=True),
            build_result("awq",  "",                success=False, error="no GPU"),
        ]
        m = build_manifest("demo-model", "/tmp/ckpt", results)
        assert m["succeeded"] == 1 and m["failed"] == 1
        return f"succeeded={m['succeeded']}, failed={m['failed']}"

    step("Export manifest", _export_manifest)

    # ── 6. Production pack ────────────────────────────────────────────
    print("\n[6/10] Production Pack")

    def _prod_pack():
        import zipfile
        from distill.ui.components.production_pack import build_pack
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = build_pack(
                model_path="/tmp/demo-model",
                export_results={"gguf": {"output_path": "/tmp/model.gguf", "success": True}},
                output_dir=tmp, vllm_port=8000, llama_port=8080,
            )
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
            return f"{Path(zip_path).name} ({len(names)} files)"

    step("Build ZIP pack", _prod_pack)

    # ── 7. Data pipeline ──────────────────────────────────────────────
    print("\n[7/10] Data Pipeline")

    def _safety_filter():
        from distill.data.safety_filter import check_safety, detect_pii
        clean = check_safety("The weather today is sunny and warm.")
        pii   = detect_pii("Email me at user@example.com")
        assert clean.is_safe
        assert "email" in pii
        return f"clean={clean.is_safe}, pii_types={pii}"

    def _dedup():
        from distill.data.deduplication import exact_dedup
        texts = ["Hello world", "Hello world", "Different text"]
        out, n = exact_dedup(texts)
        assert n == 1 and len(out) == 2
        return f"removed {n} exact duplicate"

    step("Safety filter", _safety_filter)
    step("Deduplication", _dedup)

    # ── 8. Checkpoint resume ──────────────────────────────────────────
    print("\n[8/10] Checkpoint Resume")

    def _checkpoint_scan():
        from distill.orchestration.checkpoint_resume import find_checkpoints
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint-200"
            ckpt.mkdir()
            (ckpt / "trainer_state.json").write_text(json.dumps({
                "global_step": 200, "log_history": [{"step": 200, "loss": 0.72}]
            }))
            found = find_checkpoints(tmp)
            assert found and found[0].step == 200
            return f"found checkpoint at step {found[0].step}, loss={found[0].loss}"

    step("Checkpoint detection", _checkpoint_scan)

    # ── 9. Webhook dry-run ────────────────────────────────────────────
    print("\n[9/10] Webhook (dry-run)")

    def _webhook():
        from distill.notifications.webhook import notify_run_complete
        notify_run_complete("demo-model", "mlx", 0.72, 0.88, "00:45", dry_run=True)
        return "fired dry-run notification"

    step("Webhook notification", _webhook)

    # ── 10. App build ─────────────────────────────────────────────────
    print("\n[10/10] Full App Build")

    def _app_build():
        from distill.ui.app import build_app
        app = build_app()
        assert app is not None
        return "all tabs loaded cleanly"

    step("build_app()", _app_build)

    # ── Summary ───────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total  = len(_results)

    print(f"\n{'=' * 60}")
    print(f"  Demo complete: {passed}/{total} steps passed")
    if failed:
        print(f"\n  Failed steps:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    {_FAIL} {name}: {detail}")
    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
