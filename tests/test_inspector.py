"""Tests for ModelInspector using mock metadata (no real ONNX file needed)."""

import pytest
from src.nimify.inspector import ModelInspector, ModelMetadata, TensorInfo


class TestModelInspectorMock:
    """Tests using mock=True — no onnx library or real file needed."""

    def test_returns_metadata(self):
        meta = ModelInspector("model.onnx", mock=True).inspect()
        assert isinstance(meta, ModelMetadata)

    def test_has_inputs_and_outputs(self):
        meta = ModelInspector("model.onnx", mock=True).inspect()
        assert len(meta.inputs) > 0
        assert len(meta.outputs) > 0

    def test_input_dtype(self):
        meta = ModelInspector("model.onnx", mock=True).inspect()
        assert meta.inputs[0].dtype == "float32"

    def test_output_dtype(self):
        meta = ModelInspector("model.onnx", mock=True).inspect()
        assert meta.outputs[0].dtype == "float32"

    def test_input_shape(self):
        meta = ModelInspector("model.onnx", mock=True).inspect()
        # Mock is ResNet-50 style: [None, 3, 224, 224]
        assert meta.inputs[0].shape == [None, 3, 224, 224]

    def test_output_shape(self):
        meta = ModelInspector("model.onnx", mock=True).inspect()
        assert meta.outputs[0].shape == [None, 1000]

    def test_model_name_from_path(self):
        meta = ModelInspector("resnet50.onnx", mock=True).inspect()
        assert meta.name == "resnet50"

    def test_nonexistent_path_returns_mock(self):
        """If the file doesn't exist, inspector returns mock metadata (not error)."""
        meta = ModelInspector("/tmp/does_not_exist_xyz.onnx", mock=False).inspect()
        assert isinstance(meta, ModelMetadata)

    def test_shape_str(self):
        t = TensorInfo(name="x", dtype="float32", shape=[None, 3, 224, 224])
        assert t.shape_str() == "[?, 3, 224, 224]"

    def test_opset_and_ir(self):
        meta = ModelInspector("model.onnx", mock=True).inspect()
        assert meta.opset_version >= 0
        assert meta.ir_version >= 0
