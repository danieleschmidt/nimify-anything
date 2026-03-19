"""ModelInspector: reads ONNX model metadata (inputs, outputs, shapes, dtypes).

Uses the `onnx` library when available; falls back to raw protobuf parsing,
then to a deterministic mock for testing without a real model.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TensorInfo:
    """Describes a single model input or output tensor."""
    name: str
    dtype: str               # e.g. "float32", "int64", "uint8"
    shape: List[Optional[int]]  # None means dynamic (unknown) dim

    def shape_str(self) -> str:
        dims = ["?" if d is None else str(d) for d in self.shape]
        return f"[{', '.join(dims)}]"

    def __repr__(self) -> str:
        return f"TensorInfo(name={self.name!r}, dtype={self.dtype!r}, shape={self.shape})"


@dataclass
class ModelMetadata:
    """Collected metadata for an ONNX model."""
    model_path: str
    opset_version: int
    domain: str
    inputs: List[TensorInfo]
    outputs: List[TensorInfo]
    doc_string: str = ""
    ir_version: int = 0

    @property
    def name(self) -> str:
        return Path(self.model_path).stem


# ---------------------------------------------------------------------------
# ONNX element-type → Python dtype name
# ---------------------------------------------------------------------------

_ELEM_TYPE_MAP = {
    0:  "undefined",
    1:  "float32",
    2:  "uint8",
    3:  "int8",
    4:  "uint16",
    5:  "int16",
    6:  "int32",
    7:  "int64",
    8:  "string",
    9:  "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}


def _elem_type_to_dtype(elem_type: int) -> str:
    return _ELEM_TYPE_MAP.get(elem_type, f"unknown_{elem_type}")


# ---------------------------------------------------------------------------
# Protobuf field number / wire type decoder (minimal, sufficient for ONNX)
# ---------------------------------------------------------------------------
# ONNX .proto layout (fields we need):
#   ModelProto:
#     field 1 (varint)  ir_version
#     field 2 (length)  opset_import  (OperatorSetIdProto)
#       field 2 (varint)  version
#     field 4 (length)  graph  (GraphProto)
#       field 11 (length) input  (ValueInfoProto)
#       field 12 (length) output (ValueInfoProto)
#       ValueInfoProto:
#         field 1 (length) name
#         field 2 (length) type (TypeProto)
#           TypeProto:
#             field 1 (length) tensor_type (TypeProto.Tensor)
#               TypeProto.Tensor:
#                 field 1 (varint) elem_type
#                 field 2 (length) shape (TensorShapeProto)
#                   TensorShapeProto:
#                     field 1 (length) dim (TensorShapeProto.Dimension)
#                       TensorShapeProto.Dimension:
#                         field 1 (varint) dim_value  (-OR-)
#                         field 2 (length) dim_param (name like "batch_size")
#     field 7 (length)  doc_string
#     field 8 (varint)  model_version
#     field 10 (length) domain


def _read_varint(buf: bytes, pos: int):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
    return result, pos


def _read_field(buf: bytes, pos: int):
    """Read one protobuf field, return (field_number, wire_type, value, new_pos)."""
    tag, pos = _read_varint(buf, pos)
    field_number = tag >> 3
    wire_type = tag & 0x07
    if wire_type == 0:  # varint
        value, pos = _read_varint(buf, pos)
    elif wire_type == 2:  # length-delimited
        length, pos = _read_varint(buf, pos)
        value = buf[pos: pos + length]
        pos += length
    elif wire_type == 1:  # 64-bit
        value = buf[pos: pos + 8]
        pos += 8
    elif wire_type == 5:  # 32-bit
        value = buf[pos: pos + 4]
        pos += 4
    else:
        raise ValueError(f"Unsupported wire type {wire_type} at pos {pos}")
    return field_number, wire_type, value, pos


def _parse_fields(buf: bytes) -> dict:
    """Parse all fields in a protobuf message into a dict keyed by field_number."""
    fields: dict = {}
    pos = 0
    while pos < len(buf):
        fn, wt, val, pos = _read_field(buf, pos)
        fields.setdefault(fn, []).append((wt, val))
    return fields


def _decode_tensor_info(buf: bytes) -> TensorInfo:
    """Decode a ValueInfoProto into TensorInfo."""
    fields = _parse_fields(buf)
    name = fields.get(1, [(None, b"")])[0][1].decode("utf-8", errors="replace")
    dtype = "float32"
    shape: List[Optional[int]] = []

    if 2 in fields:  # TypeProto
        type_buf = fields[2][0][1]
        type_fields = _parse_fields(type_buf)
        if 1 in type_fields:  # tensor_type
            tensor_buf = type_fields[1][0][1]
            tensor_fields = _parse_fields(tensor_buf)
            if 1 in tensor_fields:
                dtype = _elem_type_to_dtype(tensor_fields[1][0][1])
            if 2 in tensor_fields:  # shape
                shape_buf = tensor_fields[2][0][1]
                shape_fields = _parse_fields(shape_buf)
                if 1 in shape_fields:  # dims
                    for _, dim_buf in shape_fields[1]:
                        dim_fields = _parse_fields(dim_buf)
                        if 1 in dim_fields:  # dim_value
                            v = dim_fields[1][0][1]
                            shape.append(v if v != 0 else None)
                        elif 2 in dim_fields:  # dim_param (symbolic)
                            shape.append(None)
                        else:
                            shape.append(None)

    return TensorInfo(name=name, dtype=dtype, shape=shape)


def _parse_onnx_protobuf(model_path: str) -> ModelMetadata:
    """Parse ONNX file via raw protobuf without the onnx library."""
    data = Path(model_path).read_bytes()
    fields = _parse_fields(data)

    ir_version = fields[1][0][1] if 1 in fields else 0
    opset_version = 0
    if 2 in fields:
        for _, opset_buf in fields[2]:
            opset_fields = _parse_fields(opset_buf)
            if 2 in opset_fields:
                opset_version = opset_fields[2][0][1]

    doc_string = b""
    if 7 in fields:
        doc_string = fields[7][0][1]

    inputs: List[TensorInfo] = []
    outputs: List[TensorInfo] = []
    if 4 in fields:  # graph
        graph_buf = fields[4][0][1]
        graph_fields = _parse_fields(graph_buf)
        if 11 in graph_fields:
            for _, vi_buf in graph_fields[11]:
                inputs.append(_decode_tensor_info(vi_buf))
        if 12 in graph_fields:
            for _, vi_buf in graph_fields[12]:
                outputs.append(_decode_tensor_info(vi_buf))

    return ModelMetadata(
        model_path=model_path,
        ir_version=ir_version,
        opset_version=opset_version,
        domain="",
        doc_string=doc_string.decode("utf-8", errors="replace") if isinstance(doc_string, bytes) else doc_string,
        inputs=inputs,
        outputs=outputs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ModelInspector:
    """Inspect an ONNX model file and return its metadata.

    Strategy (in order):
      1. Use ``onnx`` library if installed.
      2. Fall back to raw protobuf parsing.
      3. If *mock=True* or if model_path doesn't exist, return synthetic metadata.
    """

    def __init__(self, model_path: str, mock: bool = False):
        self.model_path = model_path
        self.mock = mock

    def inspect(self) -> ModelMetadata:
        if self.mock or not Path(self.model_path).exists():
            return self._mock_metadata()
        try:
            return self._inspect_onnx_library()
        except ImportError:
            pass
        try:
            return _parse_onnx_protobuf(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse ONNX model at '{self.model_path}': {exc}\n"
                "Install the 'onnx' package for full support: pip install onnx"
            ) from exc

    def _inspect_onnx_library(self) -> ModelMetadata:
        import onnx  # noqa: F401 (guarded by try/except ImportError above)
        from onnx import TensorProto

        model = onnx.load(self.model_path)
        opset = model.opset_import[0].version if model.opset_import else 0

        def _vi_to_tensor(vi) -> TensorInfo:
            t = vi.type.tensor_type
            dtype = _elem_type_to_dtype(t.elem_type)
            shape: List[Optional[int]] = []
            if t.HasField("shape"):
                for d in t.shape.dim:
                    shape.append(d.dim_value if d.dim_value else None)
            return TensorInfo(name=vi.name, dtype=dtype, shape=shape)

        inputs = [_vi_to_tensor(vi) for vi in model.graph.input]
        outputs = [_vi_to_tensor(vi) for vi in model.graph.output]

        return ModelMetadata(
            model_path=self.model_path,
            ir_version=model.ir_version,
            opset_version=opset,
            domain=model.domain,
            doc_string=model.doc_string,
            inputs=inputs,
            outputs=outputs,
        )

    def _mock_metadata(self) -> ModelMetadata:
        """Return a mock ResNet-50-like metadata for demo/testing."""
        return ModelMetadata(
            model_path=self.model_path,
            ir_version=8,
            opset_version=17,
            domain="",
            doc_string="Mock ONNX model (ResNet-50 style)",
            inputs=[
                TensorInfo(name="input", dtype="float32", shape=[None, 3, 224, 224]),
            ],
            outputs=[
                TensorInfo(name="output", dtype="float32", shape=[None, 1000]),
            ],
        )
