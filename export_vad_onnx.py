#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────
#  export_vad_onnx.py – ONNX exporter for Mel‑Performer VAD
#  *checkpoint projections are kept exactly as‑is*
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, logging, pathlib, sys
import torch
import torch.nn as nn
from packaging import version
from models import VADLightning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("export-onnx")

# ─────────────────────────────── CLI ────────────────────────────────

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export Mel‑Performer VAD to ONNX",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=pathlib.Path, required=True, help="Lightning .ckpt path")
    p.add_argument("--output", type=pathlib.Path, default="vad.onnx", help="ONNX output file")
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--dummy-frames", type=int, default=1000, help="dummy T for tracing")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--use-dynamo", action="store_true", help="torch.onnx.export(dynamo=True)")
    return p.parse_args()

# ───────────────────── 1  Load checkpoint ───────────────────────────

def load_net(path: pathlib.Path, dev: torch.device) -> nn.Module:
    ckpt = torch.load(str(path), map_location=dev)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
        sys.exit("❌  Not a Lightning checkpoint – abort")
    net = VADLightning.load_from_checkpoint(str(path), hp=ckpt["hyper_parameters"], map_location=dev).net
    net.eval()
    log.info("Loaded model – %.2f M params", sum(p.numel() for p in net.parameters()) / 1e6)
    return net


# ───────────────────── 2  Wrapper & export ──────────────────────────

class Wrapper(nn.Module):
    def __init__(self, m: nn.Module, max_t: int): super().__init__(); self.m, self.t = m, max_t
    def forward(self, x): return self.m(x[:, : self.t])

def do_export(model: nn.Module, dummy: torch.Tensor, out: pathlib.Path, opset: int, dyn: bool):
    if dyn and version.parse(torch.__version__.split("+",1)[0]) >= version.parse("2.6.0"):
        log.info("→ torch.onnx.export(dynamo=True)")
        torch.onnx.export(model, dummy, str(out), dynamo=True, opset_version=opset,
                          input_names=["mel"], output_names=["frame_prob"],
                          dynamic_axes={"mel": {1: "T"}, "frame_prob": {1: "T"}})
    else:
        log.info("→ classic torch.onnx.export()")
        torch.onnx.export(model, dummy, str(out), opset_version=opset,
                          input_names=["mel"], output_names=["frame_prob"],
                          dynamic_axes={"mel": {1: "T"}, "frame_prob": {1: "T"}},
                          do_constant_folding=True)

# ───────────────────── 3  Main ──────────────────────────────────────

def main():
    a = cli(); dev = torch.device(a.device)
    net = load_net(a.ckpt, dev)
    import types
    import performer_pytorch.reversible as _rev

    if not hasattr(_rev, "_Patched"):
        def _plain_forward(*args, **kwargs):
            """Call the raw forward and skip custom backward."""
            return _rev._ReversibleFunction.forward(None, *args, **kwargs)
        # Monkey-patch .apply so tracing sees only ordinary ops
        _rev._ReversibleFunction.apply = _plain_forward          # type: ignore[attr-defined]
        _rev._Patched = True

    wrapped = Wrapper(net, a.dummy_frames).to(dev)
    dummy   = torch.randn(1, a.dummy_frames, a.n_mels, device=dev)

    torch.set_grad_enabled(False)
    do_export(wrapped, dummy, a.output, a.opset, a.use_dynamo)
    log.info("✅  Saved to %s", a.output)

if __name__ == "__main__":
    main()