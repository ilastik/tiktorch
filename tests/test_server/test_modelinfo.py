import random
from unittest import mock

import pytest
from bioimageio.spec.shared.raw_nodes import ImplicitOutputShape, ParametrizedInputShape
from marshmallow import missing

from tiktorch.converters import NamedExplicitOutputShape, NamedImplicitOutputShape, NamedParametrizedShape
from tiktorch.server.session.process import ModelInfo


@pytest.fixture
def implicit_output_spec():
    """output spec with ImplicitOutputShape"""
    shape = ImplicitOutputShape(
        reference_tensor="blah",
        scale=[1.0] + [float(random.randint(0, 2**32)) for _ in range(4)],
        offset=[0.0] + [float(random.randint(0, 2**32)) for _ in range(4)],
    )
    output_spec = mock.Mock(axes=("x", "y"), shape=shape, halo=[5, 12])
    output_spec.name = "implicit_out"
    return output_spec


@pytest.fixture
def parametrized_input_spec():
    shape = ParametrizedInputShape(
        min=[random.randint(0, 2**32) for _ in range(5)], step=[float(random.randint(0, 2**32)) for _ in range(5)]
    )
    input_spec = mock.Mock(axes=("b", "x", "y", "z", "c"), shape=shape)
    input_spec.name = "param_in"
    return input_spec


@pytest.fixture
def explicit_input_spec():
    input_shape = [random.randint(0, 2**32) for _ in range(3)]
    input_spec = mock.Mock(axes=("b", "x", "y"), shape=input_shape)
    input_spec.name = "explicit_in"
    return input_spec


@pytest.fixture
def explicit_output_spec():
    output_shape = [random.randint(0, 2**32) for _ in range(3)]
    halo = [0] + [random.randint(0, 2**32) for _ in range(2)]
    output_spec = mock.Mock(axes=("b", "x", "y"), shape=output_shape, halo=halo)
    output_spec.name = "explicit_out"
    return output_spec


def test_model_info_explicit_shapes(explicit_input_spec, explicit_output_spec):
    prediction_pipeline = mock.Mock(input_specs=[explicit_input_spec], output_specs=[explicit_output_spec], name="bleh")

    model_info = ModelInfo.from_prediction_pipeline(prediction_pipeline)

    assert model_info.input_axes == ["".join(explicit_input_spec.axes)]
    assert model_info.output_axes == ["".join(explicit_output_spec.axes)]
    assert len(model_info.input_shapes) == 1
    assert len(model_info.output_shapes) == 1
    assert isinstance(model_info.input_shapes[0], list)
    assert model_info.input_shapes[0] == [(ax, s) for ax, s in zip(explicit_input_spec.axes, explicit_input_spec.shape)]
    assert isinstance(model_info.output_shapes[0], NamedExplicitOutputShape)
    assert model_info.output_shapes[0].shape == [
        (ax, s) for ax, s in zip(explicit_output_spec.axes, explicit_output_spec.shape)
    ]
    assert model_info.output_shapes[0].halo == [
        (ax, s) for ax, s in zip(explicit_output_spec.axes, explicit_output_spec.halo)
    ]
    assert model_info.input_names == ["explicit_in"]
    assert model_info.output_names == ["explicit_out"]


def test_model_info_explicit_shapes_missing_halo(explicit_input_spec, explicit_output_spec):
    explicit_output_spec.halo = missing

    prediction_pipeline = mock.Mock(input_specs=[explicit_input_spec], output_specs=[explicit_output_spec], name="bleh")

    model_info = ModelInfo.from_prediction_pipeline(prediction_pipeline)

    assert model_info.input_axes == ["".join(explicit_input_spec.axes)]
    assert model_info.output_axes == ["".join(explicit_output_spec.axes)]
    assert len(model_info.input_shapes) == 1
    assert len(model_info.output_shapes) == 1
    assert isinstance(model_info.input_shapes[0], list)
    assert model_info.input_shapes[0] == [(ax, s) for ax, s in zip(explicit_input_spec.axes, explicit_input_spec.shape)]
    assert isinstance(model_info.output_shapes[0], NamedExplicitOutputShape)
    assert model_info.output_shapes[0].shape == [
        (ax, s) for ax, s in zip(explicit_output_spec.axes, explicit_output_spec.shape)
    ]
    assert model_info.output_shapes[0].halo == [(ax, s) for ax, s in zip(explicit_output_spec.axes, [0, 0, 0])]


def test_model_info_implicit_shapes(parametrized_input_spec, implicit_output_spec):
    prediction_pipeline = mock.Mock(
        input_specs=[parametrized_input_spec], output_specs=[implicit_output_spec], name="bleh"
    )

    model_info = ModelInfo.from_prediction_pipeline(prediction_pipeline)
    assert model_info.input_axes == ["".join(parametrized_input_spec.axes)]
    assert model_info.output_axes == ["".join(implicit_output_spec.axes)]
    assert len(model_info.input_shapes) == 1
    assert len(model_info.output_shapes) == 1
    assert isinstance(model_info.input_shapes[0], NamedParametrizedShape)
    assert model_info.input_shapes[0].min_shape == [
        (ax, s) for ax, s in zip(parametrized_input_spec.axes, parametrized_input_spec.shape.min)
    ]
    assert model_info.input_shapes[0].step_shape == [
        (ax, s) for ax, s in zip(parametrized_input_spec.axes, parametrized_input_spec.shape.step)
    ]
    assert isinstance(model_info.output_shapes[0], NamedImplicitOutputShape)
    assert model_info.output_shapes[0].offset == [
        (ax, s) for ax, s in zip(implicit_output_spec.axes, implicit_output_spec.shape.offset)
    ]
    assert model_info.output_shapes[0].scale == [
        (ax, s) for ax, s in zip(implicit_output_spec.axes, implicit_output_spec.shape.scale)
    ]
    assert model_info.output_shapes[0].halo == [
        (ax, s) for ax, s in zip(implicit_output_spec.axes, implicit_output_spec.halo)
    ]
    assert model_info.output_shapes[0].reference_tensor == implicit_output_spec.shape.reference_tensor

    assert model_info.input_names == ["param_in"]
    assert model_info.output_names == ["implicit_out"]


def test_model_info_implicit_shapes_missing_halo(parametrized_input_spec, implicit_output_spec):
    implicit_output_spec.halo = missing
    prediction_pipeline = mock.Mock(
        input_specs=[parametrized_input_spec], output_specs=[implicit_output_spec], name="bleh"
    )

    model_info = ModelInfo.from_prediction_pipeline(prediction_pipeline)
    assert model_info.input_axes == ["".join(parametrized_input_spec.axes)]
    assert model_info.output_axes == ["".join(implicit_output_spec.axes)]
    assert len(model_info.input_shapes) == 1
    assert len(model_info.output_shapes) == 1
    assert isinstance(model_info.input_shapes[0], NamedParametrizedShape)
    assert model_info.input_shapes[0].min_shape == [
        (ax, s) for ax, s in zip(parametrized_input_spec.axes, parametrized_input_spec.shape.min)
    ]
    assert model_info.input_shapes[0].step_shape == [
        (ax, s) for ax, s in zip(parametrized_input_spec.axes, parametrized_input_spec.shape.step)
    ]
    assert isinstance(model_info.output_shapes[0], NamedImplicitOutputShape)
    assert model_info.output_shapes[0].offset == [
        (ax, s) for ax, s in zip(implicit_output_spec.axes, implicit_output_spec.shape.offset)
    ]
    assert model_info.output_shapes[0].scale == [
        (ax, s) for ax, s in zip(implicit_output_spec.axes, implicit_output_spec.shape.scale)
    ]
    assert model_info.output_shapes[0].halo == [(ax, s) for ax, s in zip(implicit_output_spec.axes, [0, 0, 0, 0, 0])]
    assert model_info.output_shapes[0].reference_tensor == implicit_output_spec.shape.reference_tensor
