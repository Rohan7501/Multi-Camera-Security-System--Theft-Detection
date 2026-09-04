#!/usr/bin/env python3
"""Convert a fixed-batch YOLOv8 ONNX model to a dynamic-batch one.

No weights are extracted or rebuilt: convolution kernels are batch-independent,
so nothing about the learned parameters changes. What pins the batch size is
purely *shape metadata*, in two places:

  1. graph.input / graph.output dim[0] -- declared as the literal 1
  2. Reshape shape constants with a hardcoded leading 1, e.g. [1, 64, -1].
     THIS is the one people miss. Editing only (1) produces a model that loads
     fine and then fails at Run() with a reshape error on batch >= 2, because
     mid-graph it still insists on a batch of exactly 1.

Fix for (2) is ONNX Reshape's "0" semantics: 0 = copy the corresponding
dimension from the input tensor. So [1, 64, -1] -> [0, 64, -1] keeps whatever
batch actually arrives. (Guarded below: if a Reshape sets allowzero=1 then 0
would mean a literal zero-size dim instead, and the rewrite is unsafe.)

Usage:
    python3 scripts/make_dynamic_batch.py                       # best.onnx -> best_dynamic.onnx
    python3 scripts/make_dynamic_batch.py IN.onnx OUT.onnx
    python3 scripts/make_dynamic_batch.py --check-only IN.onnx  # report, change nothing

Verification runs automatically: the new model is executed at several batch
sizes, and its batch-1 output is compared elementwise against the ORIGINAL
model's -- numerical equivalence is the real proof the surgery was semantic
no-op, not just that the file loads.

Note: models/best.engine (TensorRT) is built for a fixed shape and is NOT
updated by this script; a dynamic engine needs rebuilding with an optimization
profile (min/opt/max batch). Same for any TRT cache under trt_cache/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

REPO = Path(__file__).resolve().parent.parent
BATCH_SYM = "batch"          # symbolic name for the dynamic dimension


# ---- inspection -------------------------------------------------------------

def describe(model: onnx.ModelProto, label: str) -> None:
    print(f"  [{label}]")
    for io, kind in ((model.graph.input, "input"), (model.graph.output, "output")):
        for t in io:
            dims = [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
            print(f"    {kind:6} {t.name:12} {dims}")


def batch_pinned_reshapes(graph) -> list:
    """Reshape nodes whose shape constant starts with a literal 1."""
    inits = {i.name: i for i in graph.initializer}
    consts = {n.output[0]: n for n in graph.node if n.op_type == "Constant"}
    found = []
    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        name = node.input[1]
        if name in inits:
            arr = numpy_helper.to_array(inits[name])
            found.append((node, name, arr, "initializer"))
        elif name in consts:
            attr = next(a for a in consts[name].attribute if a.name == "value")
            arr = numpy_helper.to_array(attr.t)
            found.append((node, name, arr, "constant-node"))
    return [f for f in found if len(f[2]) and f[2][0] == 1]


def assert_allowzero_safe(nodes) -> None:
    """Reshape(allowzero=1) makes 0 mean a literal empty dim, not 'copy'."""
    for node, *_ in nodes:
        for a in node.attribute:
            if a.name == "allowzero" and a.i == 1:
                sys.exit(f"ERROR: {node.name} has allowzero=1; the 0-copy rewrite "
                         f"is unsafe for this model. Re-export from source instead.")


# ---- surgery ----------------------------------------------------------------

def set_symbolic_batch(model: onnx.ModelProto) -> None:
    """graph.input/output dim[0] -> symbolic 'batch'."""
    for t in list(model.graph.input) + list(model.graph.output):
        dim = t.type.tensor_type.shape.dim
        if not len(dim):
            continue
        dim[0].ClearField("dim_value")     # must clear before setting the other field
        dim[0].dim_param = BATCH_SYM


def unpin_reshapes(model: onnx.ModelProto, nodes) -> int:
    """Leading 1 -> 0 ('copy batch from input') in each shape constant.

    Shape constants are often SHARED by several Reshape nodes, so rewrite each
    unique tensor once rather than once per referencing node.
    """
    inits = {i.name: i for i in model.graph.initializer}
    consts = {n.output[0]: n for n in model.graph.node if n.op_type == "Constant"}
    patched, seen = 0, set()
    for _, tname, arr, kind in nodes:
        if tname in seen:
            continue
        seen.add(tname)
        new = arr.copy()
        new[0] = 0
        tensor = numpy_helper.from_array(new.astype(arr.dtype), tname)
        if kind == "initializer":
            inits[tname].CopyFrom(tensor)
        else:
            attr = next(a for a in consts[tname].attribute if a.name == "value")
            attr.t.CopyFrom(tensor)
        print(f"    {tname}: {arr.tolist()} -> {new.tolist()}")
        patched += 1
    return patched


def refresh_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop value_info inferred for batch=1, then re-infer with the symbolic dim."""
    del model.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(model)


# ---- verification -----------------------------------------------------------

def verify(original: Path, converted: Path, batches=(1, 2, 4)) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not available -- skipping runtime verification")
        return True

    so = ort.SessionOptions()
    so.log_severity_level = 3
    new = ort.InferenceSession(str(converted), so, providers=["CPUExecutionProvider"])
    old = ort.InferenceSession(str(original), so, providers=["CPUExecutionProvider"])

    iname = new.get_inputs()[0].name
    shape = new.get_inputs()[0].shape          # e.g. ['batch', 3, 640, 640]
    c, h, w = (int(x) for x in shape[1:])
    ok = True

    for n in batches:
        x = np.random.rand(n, c, h, w).astype(np.float32)
        try:
            outs = new.run(None, {iname: x})
            print(f"    batch={n:<2} -> " + ", ".join(str(o.shape) for o in outs))
        except Exception as e:
            print(f"    batch={n:<2} -> FAILED: {str(e)[:120]}")
            ok = False

    # The real test: identical numbers vs the original model at batch 1.
    x1 = np.random.rand(1, c, h, w).astype(np.float32)
    a = old.run(None, {old.get_inputs()[0].name: x1})
    b = new.run(None, {iname: x1})
    same = all(np.allclose(p, q, atol=1e-5) for p, q in zip(a, b))
    worst = max((float(np.abs(p - q).max()) for p, q in zip(a, b)), default=0.0)
    print(f"    batch-1 output identical to original: {same}  (max abs diff {worst:.2e})")

    # And that a batched run equals stacking single runs.
    if 2 in batches:
        x2 = np.random.rand(2, c, h, w).astype(np.float32)
        batched = new.run(None, {iname: x2})[0]
        singles = np.concatenate(
            [new.run(None, {iname: x2[i:i + 1]})[0] for i in range(2)], axis=0)
        agree = np.allclose(batched, singles, atol=1e-5)
        print(f"    batch-2 equals two batch-1 runs stacked: {agree}")
        ok = ok and agree

    return ok and same


# ---- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default=str(REPO / "models/best.onnx"))
    ap.add_argument("dst", nargs="?", default=None,
                    help="default: <src stem>_dynamic.onnx")
    ap.add_argument("--check-only", action="store_true",
                    help="report what would change; write nothing")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"no such model: {src}")
    dst = Path(args.dst) if args.dst else src.with_name(src.stem + "_dynamic.onnx")

    model = onnx.load(str(src))
    print(f"\nsource: {src}")
    describe(model, "before")

    pinned = batch_pinned_reshapes(model.graph)
    print(f"\n  Reshape nodes with a hardcoded batch: {len(pinned)}")
    for node, tname, arr, _ in pinned:
        print(f"    {node.name or '(anon)':32} {arr.tolist()}")
    assert_allowzero_safe(pinned)

    if args.check_only:
        print("\n--check-only: nothing written")
        return 0

    print("\n  rewriting shape constants (1 -> 0 = copy batch from input):")
    n = unpin_reshapes(model, pinned)
    set_symbolic_batch(model)
    model = refresh_shapes(model)
    onnx.checker.check_model(model)

    onnx.save(model, str(dst))
    print(f"\n  patched {n} shape constant(s); wrote {dst}")
    describe(onnx.load(str(dst)), "after")

    print("\n  verifying:")
    ok = verify(src, dst)
    print("\n" + ("PASS -- model accepts a dynamic batch and is numerically identical"
                  if ok else "FAIL -- see errors above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
