import pytest
from pydantic import ValidationError
from unittest.mock import patch, mock_open
import os
import struct
import json
import tempfile

from sdbx.indexer import ModelType, ModelCodeData, EvalMeta, ReadMeta, ModelIndexer

def test_model_type_enum():
    assert ModelType.DIFFUSION.value == 'DIF'
    assert ModelType.LANGUAGE.value == 'LLM'
    assert ModelType.LORA.value == 'LOR'
    assert ModelType.TRANSFORMER.value == 'TRA'
    assert ModelType.VAE.value == 'VAE'

def test_model_code_data_validation():
    # Test with valid data
    valid_data = [123456, '/path/to/model', 'float32']
    model_code_data = ModelCodeData.parse_obj(valid_data)
    assert model_code_data.size == 123456
    assert model_code_data.path == '/path/to/model'
    assert model_code_data.dtype_or_context_length == 'float32'

    # Test with invalid data (missing one element)
    invalid_data = [123456, '/path/to/model']
    with pytest.raises(ValidationError):
        ModelCodeData.parse_obj(invalid_data)

def test_eval_meta_with_diffusion_model():
    extract = {
        'unet': 100,
        'filename': 'diffusion_model.safetensors',
        'size': 1e9,
        'path': '/path/to/diffusion_model.safetensors',
        'tensor_params': 5000000,
        'shape': [64, 3, 256, 256],
        'dtype': 'float32',
    }
    eval_meta = EvalMeta(extract)
    result = eval_meta.data()
    # Since actual data processing is complex, we check if result is not None
    assert result is not None
    code, data = result
    assert code == ModelType.DIFFUSION
    assert data[0] == 'diffusion_model.safetensors'

def test_eval_meta_with_vae_model():
    extract = {
        'unet': 96,
        'shape': [512],
        'filename': 'vae_model.safetensors',
        'size': 167335343,
        'path': '/path/to/vae_model.safetensors',
    }
    eval_meta = EvalMeta(extract)
    result = eval_meta.data()
    assert result is not None
    code, data = result
    assert code == ModelType.VAE
    assert data[0] == 'vae_model.safetensors'

@patch('os.path.exists')
@patch('os.path.getsize')
def test_read_meta_safetensors(mock_getsize, mock_exists):
    mock_exists.return_value = True
    mock_getsize.return_value = 123456789

    # Mock the open function to simulate reading a safetensors file
    fake_header_size = 100
    fake_header = json.dumps({'metadata': {'some_key': 'some_value'}}).encode('utf-8')
    fake_file_content = struct.pack('<Q', fake_header_size) + fake_header

    with patch('builtins.open', mock_open(read_data=fake_file_content)) as mock_file:
        read_meta = ReadMeta('/path/to/model.safetensors')
        data = read_meta.data()
        assert data['size'] == 123456789
        assert data['filename'] == 'model.safetensors'
        assert data['extension'] == 'safetensors'
        assert 'some_key' in read_meta.meta.get('metadata', {})

@patch('os.path.exists')
@patch('os.path.getsize')
def test_read_meta_gguf(mock_getsize, mock_exists):
    mock_exists.return_value = True
    mock_getsize.return_value = 987654321

    # Mock the Llama class to return fake metadata
    fake_metadata = {
        'general.name': 'Test LLM',
        'general.architecture': 'transformer',
    }
    with patch('llama_cpp.Llama') as mock_llama_class:
        mock_llama = mock_llama_class.return_value
        mock_llama.metadata = fake_metadata

        read_meta = ReadMeta('/path/to/model.gguf')
        data = read_meta.data()
        assert data['size'] == 987654321
        assert data['filename'] == 'model.gguf'
        assert data['extension'] == 'gguf'
        assert 'general.name' in read_meta.meta
        assert read_meta.meta['general.name'] == 'Test LLM'

def test_model_indexer_with_mocked_classes():
    # Mock ReadMeta.data to return predefined extract data
    with patch('indexer.ReadMeta') as mock_read_meta_class:
        mock_read_meta_instance = mock_read_meta_class.return_value
        mock_read_meta_instance.data.return_value = {
            'filename': 'mock_model.safetensors',
            'size': 123456789,
            'path': '/path/to/mock_model.safetensors',
            'unet': 100,
            'tensor_params': 5000000,
            'shape': [64, 3, 256, 256],
            'dtype': 'float32',
        }

        # Mock EvalMeta.data to return predefined code and data
        with patch('indexer.EvalMeta') as mock_eval_meta_class:
            mock_eval_meta_instance = mock_eval_meta_class.return_value
            mock_eval_meta_instance.data.return_value = (
                ModelType.DIFFUSION,
                ('mock_model.safetensors', 'lookup_value', 123456789, '/path/to/mock_model.safetensors', 'float32')
            )

            # Mock config.get_path to return a temporary directory
            with tempfile.TemporaryDirectory() as temp_models_dir:
                with patch('config.config.get_path', return_value=temp_models_dir):
                    # Mock config.get_path_tree to return a list of files
                    with patch('config.config.get_path_tree') as mock_get_path_tree:
                        mock_get_path_tree.return_value = [
                            {'path': '/path/to/mock_model.safetensors'}
                        ]

                        # Initialize the indexer
                        indexer = ModelIndexer()

                        # Check that indexer.index contains our mocked data
                        assert ModelType.DIFFUSION.value in indexer.index
                        diffusion_index = indexer.index[ModelType.DIFFUSION.value]
                        assert 'mock_model.safetensors' in diffusion_index
                        assert 'lookup_value' in diffusion_index['mock_model.safetensors']
                        value = diffusion_index['mock_model.safetensors']['lookup_value']
                        assert value == [123456789, '/path/to/mock_model.safetensors', 'float32']

                        # Test fetch_id
                        tag, category, value = indexer.fetch_id('mock_model.safetensors')
                        assert tag == ModelType.DIFFUSION.value
                        assert category == 'lookup_value'
                        assert value == [123456789, '/path/to/mock_model.safetensors', 'float32']

                        # Check that index.json was written in temp_models_dir
                        index_file = os.path.join(temp_models_dir, 'index.json')
                        assert os.path.exists(index_file)

                        # Read the index.json file
                        with open(index_file, 'r') as f:
                            index_data = json.load(f)
                            assert index_data == indexer.index

def test_fetch_compatible():
    # Mock the indexer's index data
    mock_index = {
        ModelType.VAE.value: {
            'vae_model.safetensors': {
                'lookup_value': [167335343, '/path/to/vae_model.safetensors', 'float32']
            }
        },
        ModelType.TRANSFORMER.value: {
            'transformer_model.safetensors': {
                'lookup_value': [123456789, '/path/to/transformer_model.safetensors', 'float32']
            }
        },
        ModelType.LORA.value: {
            'lora_model.safetensors': {
                'lookup_value': [987654321, '/path/to/lora_model.safetensors', 'float32']
            }
        },
    }

    with patch.object(ModelIndexer, 'index', new=mock_index):
        indexer = ModelIndexer()
        # Mock the config.get_default function
        with patch('indexer.config.get_default') as mock_get_default:
            mock_get_default.side_effect = [
                {'some_key': 'some_value'},  # For clip_data
                {'lora_priority': 1}         # For lora_priority
            ]
            vae_sorted, tra_sorted, lora_sorted = indexer.fetch_compatible('lookup_value')
            assert vae_sorted is not None
            assert 'vae_model.safetensors' in vae_sorted[0][0][0]
            assert tra_sorted is not None
            assert 'transformer_model.safetensors' in tra_sorted
            assert lora_sorted is not None
            assert 'lora_model.safetensors' in lora_sorted

def test_fetch_id():
    mock_index = {
        ModelType.DIFFUSION.value: {
            'mock_model.safetensors': {
                'lookup_value': [123456789, '/path/to/mock_model.safetensors', 'float32']
            }
        }
    }
    indexer = ModelIndexer()
    indexer.index = mock_index
    tag, category, value = indexer.fetch_id('mock_model.safetensors')
    assert tag == ModelType.DIFFUSION.value
    assert category == 'lookup_value'
    assert value == [123456789, '/path/to/mock_model.safetensors', 'float32']

def test_fetch_refiner():
    mock_index = {
        ModelType.DIFFUSION.value: {
            'STA-XR': {
                'lookup_value': [123456789, '/path/to/sta_xr_model.safetensors', 'float32']
            }
        }
    }
    indexer = ModelIndexer()
    indexer.index = mock_index
    refiner = indexer.fetch_refiner()
    assert refiner is not None
    assert 'lookup_value' in refiner
    assert refiner['lookup_value'][1] == '/path/to/sta_xr_model.safetensors'