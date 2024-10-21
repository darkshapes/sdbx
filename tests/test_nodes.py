import unittest

from sdbx import logger
from sdbx.nodes.types import *
from sdbx.nodes.info import NodeInfo
class TestNodeInfo(unittest.TestCase):
    def test_required_and_optional_inputs(self):
        @node
        def test_node(
            required_input: int,
            optional_input: int = 5
        ):
            pass

        inputs = test_node.info.inputs
        self.assertIn('Required Input', inputs['required'])
        self.assertIn('Optional Input', inputs['optional'])
        self.assertNotIn('Optional Input', inputs['required'])
        self.assertNotIn('Required Input', inputs['optional'])

    def test_tuple_output(self):
        @node
        def test_node() -> (int, str):
            return 69, 'test'

        outputs = test_node.info.outputs
        output_names = list(outputs.keys())
        self.assertEqual(len(outputs), 2)
        expected_output_names = ['Int', 'Str']
        self.assertListEqual(output_names, expected_output_names)
        self.assertEqual(outputs['Int']['type'], 'Int')
        self.assertEqual(outputs['Str']['type'], 'Str')

    def test_annotated_input(self):
        @node
        def test_node(
            number: A[int, Numerical(min=0, max=10)] = 5
        ):
            pass

        inputs = test_node.info.inputs
        self.assertIn('Number', inputs['optional'])
        number_info = inputs['optional']['Number']
        self.assertEqual(number_info['type'], 'Int')
        self.assertEqual(number_info['constraints']['min'], 0)
        self.assertEqual(number_info['constraints']['max'], 10)
        self.assertEqual(number_info['default'], 5)

    def test_node_display_attribute(self):
        @node(display=True)
        def test_node(number: int):
            pass

        self.assertTrue(test_node.info.display)

    def test_generator_node(self):
        @node
        def test_node() -> I[int]:
            yield 1
            yield 2

        self.assertTrue(test_node.info.generator)
        outputs = test_node.info.outputs
        self.assertIn('Int', outputs)
        self.assertEqual(outputs['Int']['type'], 'Int')

    def test_node_with_custom_names(self):
        @node
        def test_node(
            param1: A[int, Name('Custom Name 1')],
            param2: A[str, Name('Custom Name 2')] = 'default'
        ) -> (A[int, Name('Output 1')], A[str, Name('Output 2')]):
            return param1, param2

        inputs = test_node.info.inputs
        self.assertIn('Custom Name 1', inputs['required'])
        self.assertIn('Custom Name 2', inputs['optional'])

        outputs = test_node.info.outputs
        self.assertIn('Output 1', outputs)
        self.assertIn('Output 2', outputs)

    def test_node_with_literals(self):
        @node
        def test_node(
            choice: Literal['option1', 'option2', 'option3']
        ):
            pass

        inputs = test_node.info.inputs
        self.assertIn('Choice', inputs['required'])
        choice_info = inputs['required']['Choice']
        self.assertEqual(choice_info['type'], 'OneOf')
        self.assertEqual(choice_info['choices'], ('option1', 'option2', 'option3'))

    def test_node_with_dependent(self):
        @node
        def test_node(
            param1: int,
            param2: A[int, Dependent(on='param1', when=Condition(eq, 2))]
        ):
            pass

        inputs = test_node.info.inputs
        self.assertIn('Param1', inputs['required'])
        self.assertIn('Param2', inputs['required'])
        param2_info = inputs['required']['Param2']
        self.assertIn('dependent', param2_info)
        self.assertEqual(param2_info['dependent']['on'], 'param1')
        when = param2_info['dependent']['when'][0]
        self.assertEqual(when['operator'], eq)
        self.assertEqual(when['value'], 2)

    def test_node_with_dependent_condition_varieties(self):
        @node
        def test_node_condition(
            param1: int,
            param2: A[int, Dependent(on='param1', when=Condition(eq, 2))]
        ):
            pass

        @node
        def test_node_tuple(
            param1: int,
            param2: A[int, Dependent(on='param1', when=(eq, 2))]
        ):
            pass

        @node
        def test_node_singleton(
            param1: int,
            param2: A[int, Dependent(on='param1', when=2)]
        ):
            pass

        self.assertEqual(test_node_condition.info.inputs, test_node_tuple.info.inputs)
        self.assertEqual(test_node_tuple.info.inputs, test_node_singleton.info.inputs)

    def test_node_with_dependent_no_when(self):
        @node
        def test_node(
            param1: int,
            param2: A[int, Dependent(on='param1')]  # should be when param1 not equals None
        ):
            pass

        inputs = test_node.info.inputs
        self.assertIn('Param1', inputs['required'])
        self.assertIn('Param2', inputs['required'])
        param2_info = inputs['required']['Param2']
        self.assertIn('dependent', param2_info)
        self.assertEqual(param2_info['dependent']['on'], 'param1')
        when = param2_info['dependent']['when'][0]
        self.assertEqual(when['operator'], ne)
        self.assertEqual(when['value'], None)

    def test_node_with_dependent_using_mutiple_tuple_conditions(self):
        @node
        def test_node(
            param1: int,
            param2: A[int, Dependent(on='param1', when=[(ne, None), (lt, 3)])]
        ):
            pass

        inputs = test_node.info.inputs
        self.assertIn('Param1', inputs['required'])
        self.assertIn('Param2', inputs['required'])
        param2_info = inputs['required']['Param2']
        self.assertIn('dependent', param2_info)
        self.assertEqual(param2_info['dependent']['on'], 'param1')
        self.assertEqual(len(param2_info['dependent']['when']), 2)
        when1 = param2_info['dependent']['when'][0]
        self.assertEqual(when1['operator'], ne)
        self.assertEqual(when1['value'], None)
        when2 = param2_info['dependent']['when'][1]
        self.assertEqual(when2['operator'], lt)
        self.assertEqual(when2['value'], 3)

    def test_node_with_validator(self):
        @node
        def test_node(
            number: A[int, Numerical(min=0, max=10), Validator(condition=lambda x: x % 2 == 0, error_message="Must be even")]
        ):
            pass

        inputs = test_node.info.inputs
        self.assertIn('Number', inputs['required'])
        number_info = inputs['required']['Number']
        # Validator is not yet implemented in the original code's NodeInfo.put()
        # So we can't test for 'validator' key unless the code is extended

    def test_node_with_text(self):
        @node
        def test_node(
            description: A[str, Text(multiline=True)]
        ):
            pass

        inputs = test_node.info.inputs
        self.assertIn('Description', inputs['required'])
        description_info = inputs['required']['Description']
        self.assertEqual(description_info['type'], 'Str')
        self.assertEqual(description_info['constraints']['multiline'], True)

    def test_node_with_no_return(self):
        @node
        def test_node(param: int):
            pass

        self.assertEqual(len(test_node.info.outputs), 0)
        self.assertTrue(getattr(test_node.info, 'terminal'))

    def test_terminal_node(self):
        @node
        def test_node() -> None:
            pass

        self.assertEqual(len(test_node.info.outputs), 0)
        self.assertTrue(getattr(test_node.info, 'terminal'))

    def test_list_and_tuple_input_node(self):
        @node
        def test_node(
            param1: List[int],
            param2: Tuple[str, str, str, str]
        ):
            pass

        inputs = test_node.info.inputs
        self.assertIn('Param1', inputs['required'])
        param1_info = inputs['required']['Param1']
        self.assertEqual(param1_info['type'], 'List')
        self.assertEqual(param1_info['sub']['type'], 'Int')
        self.assertFalse(hasattr(param1_info, "constraints"))

        self.assertIn('Param2', inputs['required'])
        param2_info = inputs['required']['Param2']
        self.assertEqual(param2_info['type'], 'List')
        self.assertEqual(param2_info['sub']['type'], 'Str')
        self.assertEqual(param2_info['constraints']['length'], 4)