from dataclasses import dataclass
from typing import Any, Callable, Union


@dataclass
class Name:
    name: str = ""


@dataclass
class Slider:
    min: Union[int, float]
    max: Union[int, float]
    step: Union[int, float] = 1.0
    round: bool = False


@dataclass
class Numerical(Slider):
    randomizable: bool = False


@dataclass
class Text:
    multiline: bool = False
    dynamic_prompts: bool = False


@dataclass
class Dependent:
    on: str
    when: Any


@dataclass
class Validator:
    condition: Callable[[Any], bool]
    error_message: str
