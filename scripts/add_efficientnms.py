#!/usr/bin/env python3
"""Bake an EfficientNMS_TRT plugin node into models/best.onnx -> models/best_nms.onnx.

YOLOv8 output0 is [1, 8, 8400] channel-major: 4 box coords (cx,cy,w,h) + 4 class
scores, already sigmoid'd, no objectness. EfficientNMS_TRT needs boxes [1,N,4] and
scores [1,N,num_classes], so we transpose to [1,8400,8] then slice.

The resulting ONNX runs ONLY on a TensorRT path (ORT TensorRT EP or a TRT engine) --
the EfficientNMS_TRT op is a TensorRT plugin, not a standard ONNX op.
"""
import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs

p = argparse.ArgumentParser(
    description="Bake an EfficientNMS_TRT node into a YOLOv8 ONNX model.")
p.add_argument("--src",         default="models/best.onnx",
               help="input YOLOv8 ONNX (output0 = [1, 4+num_classes, 8400])  [%(default)s]")
p.add_argument("--dst",         default="models/best_nms.onnx",
               help="output ONNX with EfficientNMS baked in  [%(default)s]")
p.add_argument("--num-classes", type=int,   default=4,
               help="number of classes (0=Product 1=Product-Picked 2=Regular 3=Shoplifting)  [%(default)s]")
p.add_argument("--topk",        type=int,   default=100, help="max detections kept  [%(default)s]")
p.add_argument("--conf-thres",  type=float, default=0.25,
               help="score threshold (match postprocess.cpp CONF_THRESHOLD)  [%(default)s]")
p.add_argument("--iou-thres",   type=float, default=0.45,
               help="NMS IoU threshold (match postprocess.cpp IOU_THRESHOLD)  [%(default)s]")
args = p.parse_args()

SRC, DST    = args.src, args.dst
NUM_CLASSES = args.num_classes
TOPK        = args.topk
CONF_THRESH = args.conf_thres
IOU_THRESH  = args.iou_thres

graph = gs.import_onnx(onnx.load(SRC))
out = next(o for o in graph.outputs if o.name == "output0")   # [1, 8, 8400]

# 1. [1,8,8400] -> [1,8400,8]
t = gs.Variable("bs_t", dtype=np.float32)
graph.nodes.append(gs.Node("Transpose", attrs={"perm": [0, 2, 1]}, inputs=[out], outputs=[t]))

# 2. slice boxes (chans 0:4 = cx,cy,w,h) and scores (chans 4:4+C)
def _slice(name, data, start, end):
    v = gs.Variable(name, dtype=np.float32)
    graph.nodes.append(gs.Node("Slice", name=name + "_n", inputs=[data,
        gs.Constant(name + "_s", np.array([start], np.int64)),
        gs.Constant(name + "_e", np.array([end],   np.int64)),
        gs.Constant(name + "_a", np.array([2],     np.int64))], outputs=[v]))
    return v

boxes  = _slice("boxes",  t, 0, 4)
scores = _slice("scores", t, 4, 4 + NUM_CLASSES)

# 3. EfficientNMS_TRT plugin node
num_dets    = gs.Variable("num_dets",    np.int32,   [1, 1])
det_boxes   = gs.Variable("det_boxes",   np.float32, [1, TOPK, 4])
det_scores  = gs.Variable("det_scores",  np.float32, [1, TOPK])
det_classes = gs.Variable("det_classes", np.int32,   [1, TOPK])

graph.nodes.append(gs.Node(
    "EfficientNMS_TRT", name="EfficientNMS",
    attrs={
        "plugin_version":  "1",
        "background_class": -1,          # no background / objectness
        "max_output_boxes": TOPK,
        "score_threshold":  CONF_THRESH,
        "iou_threshold":    IOU_THRESH,
        "box_coding":       1,           # 1 = BoxCenterSize (cx,cy,w,h) -- YOLOv8
        "score_activation": 0,           # scores already sigmoid'd
        # NOTE: do NOT set class_agnostic -- the EfficientNMS_TRT plugin in TRT 10.16
        # rejects it as an unrecognized attribute. Its default (false = per-class NMS)
        # is what we want anyway; the parser's "not found" message is just a warning.
    },
    inputs=[boxes, scores],
    outputs=[num_dets, det_boxes, det_scores, det_classes]))

graph.outputs = [num_dets, det_boxes, det_scores, det_classes]
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), DST)
print(f"wrote {DST}  (outputs: num_dets, det_boxes, det_scores, det_classes)")
