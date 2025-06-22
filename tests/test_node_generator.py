# ### <!-- // /*  d a r k s h a p e s */ -->

from pytest import raises
from sdbx.nodes.generate import terminal_gen
from mir.constants import PkgType


def test_terminal_gen_invalid_pkg():
    with raises(ModuleNotFoundError) as exc_info:
        terminal_gen(class_name="CLIPModel", pkg_name=PkgType.DIFFUSERS)
    assert exc_info.value.args == ("No module named 'diffusers.pipelines.'",)


def test_terminal_gen_correct_pkg():
    from transformers import CLIPConfig

    result = terminal_gen(class_name="CLIPModel", pkg_name=PkgType.TRANSFORMERS)
    expected = {"aux": {}, "generation": {}, "pipe": {"config": CLIPConfig}}

    assert result == expected


def test_terminal_alternate_class():
    from transformers import CLIPTextConfig

    result = terminal_gen(class_name="CLIPTextModel", pkg_name=PkgType.TRANSFORMERS)
    expected = {"aux": {}, "generation": {}, "pipe": {"config": CLIPTextConfig}}

    assert result == expected


def test_terminal_diffusers_pkg_invalid_class():
    with raises(AttributeError) as exc_info:
        terminal_gen(class_name="StableDiffusionXL", pkg_name=PkgType.DIFFUSERS)
    assert exc_info.value.args == ("module diffusers.pipelines.stable_diffusion_xl has no attribute __call__",)


def test_terminal_diffusers_correct_class_gen():
    from diffusers import StableDiffusionXLPipeline

    result = terminal_gen(class_name="StableDiffusionXLPipeline", pkg_name=PkgType.DIFFUSERS)
    assert result is not None
    expected_gen_args = list(StableDiffusionXLPipeline.__call__.__annotations__)
    assert bool(isinstance(result["generation"], dict)) is True
    assert list(result["generation"]) == expected_gen_args


def test_terminal_diffusers_alternate_class_gen():
    from diffusers import FluxPipeline

    result = terminal_gen(class_name="FluxPipeline", pkg_name=PkgType.DIFFUSERS)
    expected_gen = list(FluxPipeline.__call__.__annotations__)
    assert bool(isinstance(result["generation"], dict)) is True
    assert list(result["generation"]) == expected_gen


def test_terminal_diffusers_correct_class_pipe():
    from diffusers import StableDiffusionXLPipeline

    result = terminal_gen(class_name="StableDiffusionXLPipeline", pkg_name=PkgType.DIFFUSERS)
    expected_pipe_args = list(StableDiffusionXLPipeline.__init__.__annotations__)
    assert bool(isinstance(result["pipe"], dict)) is True
    assert list(result["pipe"]) == expected_pipe_args


def test_terminal_diffusers_alternate_class_pipe():
    from diffusers import FluxPipeline

    result = terminal_gen(class_name="FluxPipeline", pkg_name=PkgType.DIFFUSERS)
    expected_pipe = list(FluxPipeline.__init__.__annotations__)
    assert bool(isinstance(result["pipe"], dict)) is True
    assert list(result["pipe"]) == expected_pipe


def test_terminal_diffusers_correct_class_aux():
    from typing import Optional, Dict
    from diffusers import StableDiffusionXLPipeline

    result = terminal_gen(class_name="StableDiffusionXLPipeline", pkg_name=PkgType.DIFFUSERS)
    pipe_args = StableDiffusionXLPipeline.__init__.__annotations__
    expected_aux_args = {k: Optional[v] for k, v in pipe_args.items() if k in StableDiffusionXLPipeline._optional_components}
    assert bool(isinstance(result["aux"], dict)) is True
    assert result["aux"] == expected_aux_args


def test_terminal_diffusers_alternate_class_aux():
    from typing import Optional

    from diffusers import FluxPipeline

    result = terminal_gen(class_name="FluxPipeline", pkg_name=PkgType.DIFFUSERS)
    pipe_args = FluxPipeline.__init__.__annotations__
    expected_aux_args = {k: Optional[v] for k, v in pipe_args.items() if k in FluxPipeline._optional_components}
    assert bool(isinstance(result["aux"], dict)) is True
    assert result["aux"] == expected_aux_args
