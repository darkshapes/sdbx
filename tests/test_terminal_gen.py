import pytest
from nnll.metadata.helpers import snake_caseify
from sdbx.nodes.generate import terminal_gen
from zodiac.providers.constants import PkgType


# Define a test case for the terminal_gen function
def test_terminal_gen_diffusers_cls():
    # Mocked values for demonstration purposes
    from diffusers import AllegroPipeline

    class_obj = AllegroPipeline
    class_name = class_obj.__name__
    # Expected output dictionary structure
    expected_output = {
        "node_name": f"{class_name}Node",
        "func_name": snake_caseify(class_name),
        "class_obj": class_obj,
        "generation_args": class_obj.__call__.__annotations__,
        "pipeline_args": class_obj.__init__.__annotations__,
        "aux_classes": {},
    }

    # Call the function with mocked values
    result = terminal_gen(class_obj, PkgType.DIFFUSERS)

    # Assert that the output matches the expected structure
    assert result == expected_output


def test_terminal_gen_transformers_cls():
    # Mocked values for demonstration purposes
    from transformers import AlbertModel

    class_obj = AlbertModel
    class_name = class_obj.__name__
    # Expected output dictionary structure
    expected_output = {
        "node_name": f"{class_name}Node",
        "func_name": snake_caseify(class_name),
        "class_obj": class_obj,
        "generation_args": class_obj.__call__.__annotations__,
        "pipeline_args": class_obj.__init__.__annotations__,
        "aux_classes": {},
    }

    # Call the function with mocked values
    result = terminal_gen(class_name, PkgType.TRANSFORMERS)

    # Assert that the output matches the expected structure
    assert result == expected_output


def test_terminal_gen_diffusers_with_aux_classes():
    # Mocked values for demonstration purposes

    from diffusers import FluxPipeline
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    class_obj = FluxPipeline
    class_name = class_obj.__name__
    # Expected output dictionary structure
    expected_output = {
        "node_name": f"{class_name}Node",
        "func_name": snake_caseify(class_name),
        "class_obj": class_obj,
        "generation_args": class_obj.__call__.__annotations__,
        "pipeline_args": class_obj.__init__.__annotations__,
        "aux_classes": {"feature_extractor": CLIPImageProcessor, "image_encoder": CLIPVisionModelWithProjection},
    }

    # Call the function with mocked values
    result = terminal_gen(class_obj, PkgType.TRANSFORMERS)

    # Assert that the output matches the expected structure
    assert result == expected_output


def test_terminal_gen_invalid_class():
    # Mocked values for demonstration purposes
    from transformers import AyaVisionConfig

    class_obj = AyaVisionConfig
    class_name = class_obj.__name__

    result = terminal_gen(AyaVisionConfig, PkgType.TRANSFORMERS)
    assert not result["generation_args"]
    assert not result["pipeline_args"]
    assert not result["aux_classes"]


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
