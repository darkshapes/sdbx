import os
import pytest
import logging
import argparse
from unittest.mock import patch, mock_open, MagicMock

from sdbx.config import Config, get_config_location, LatentPreviewMethod, parse, VRAM, PrecisionConfig, WebConfig
from sdbx.nodes.manager import NodeManager

@pytest.fixture
def mock_config_path(tmp_path):
    """Fixture to create a temporary path for config."""
    return os.path.join(tmp_path, "config.toml")

@pytest.fixture
def mock_config_instance(mock_config_path):
    """Fixture to create a mock Config instance."""
    with patch("sdbx.config.os.path.exists", return_value=True):
        config = Config(str(mock_config_path))
        yield config

def test_get_config_location():
    """Test the get_config_location function returns the correct config path based on the platform."""
    with patch('platform.system', return_value='Windows'):
        assert get_config_location() == os.path.join(os.environ.get('LOCALAPPDATA', os.path.join(os.path.expanduser('~'), 'AppData', 'Local')), 'Shadowbox', 'config.toml')

    with patch('platform.system', return_value='Linux'):
        assert get_config_location() == os.path.join(os.path.expanduser('~'), '.config', 'shadowbox', 'config.toml')

    with patch('platform.system', return_value='Darwin'):
        assert get_config_location() == os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'Shadowbox', 'config.toml')

def test_config_initialization(mock_config_path):
    """Test Config initialization with a given path."""
    with patch('sdbx.config.os.path.exists', return_value=True):
        config = Config(str(mock_config_path))
        assert isinstance(config, Config)
        assert config.path == os.path.dirname(mock_config_path)

def test_config_initialization_creates_new_config(mock_config_path):
    """Test Config initialization when the path doesn't exist should generate a new config."""
    with patch('sdbx.config.os.path.exists', return_value=False):
        with patch('sdbx.config.Config.generate_new_config') as mock_generate_config:
            config = Config(str(mock_config_path))
            mock_generate_config.assert_called_once()

def test_generate_new_config(mock_config_instance):
    """Test generate_new_config creates required directories and copies files."""
    with patch('sdbx.config.os.makedirs') as mock_makedirs, patch('sdbx.config.shutil.copytree') as mock_copytree:
        mock_config_instance.generate_new_config()
        mock_makedirs.assert_called()
        mock_copytree.assert_called()

def test_rewrite_config(mock_config_instance):
    """Test rewriting a config key."""
    with patch("sdbx.config.open", mock_open()) as mocked_file:
        mock_config_instance.rewrite("web.listen", "0.0.0.0")
        # TODO: IMPLEMENT REWRITE
        # mocked_file.assert_called()
    
def test_get_path(tmp_path, mock_config_instance):
    clients_dir = os.path.join(tmp_path, "clients")

    assert mock_config_instance.get_path("clients") == clients_dir

def test_get_path_contents(tmp_path, mock_config_instance):
    """Test get_path_contenst checks .extension files in the directory."""
    
    # Create a mock directory path for "clients" inside a temporary path
    clients_dir = os.path.join(tmp_path, "clients")
    os.mkdir(clients_dir)
    
    # Create some fake .toml files in the mock clients directory
    file1 = os.path.join(clients_dir, "file1.toml")
    file2 = os.path.join(clients_dir, "file2.toml")
    open(file1, 'a').close()
    open(file2, 'a').close()
    
    # Check if the get_path_contents method returns the correct files
    toml_files = mock_config_instance.get_path_contents("clients", extension="toml")
    
    # Verify that the returned files match the created files
    assert toml_files == [str(file1), str(file2)]

def test_memory_config_defaults(mock_config_instance):
    """Test MemoryConfig default values."""
    memory_config = mock_config_instance.memory
    assert memory_config.vram == VRAM.NORMAL
    assert memory_config.smart_memory is True

def test_precision_config_defaults(mock_config_instance):
    """Test PrecisionConfig default values."""
    precision_config = mock_config_instance.precision
    assert precision_config.fp == "mixed"
    assert precision_config.unet == "float32"
    assert precision_config.vae == "mixed"
    assert precision_config.text_encoder == "float16"

def test_web_config_defaults(mock_config_instance):
    """Test WebConfig default values."""
    web_config = mock_config_instance.web
    assert web_config.listen == "127.0.0.1"
    assert web_config.port == 8188
    assert web_config.external_address == "localhost"
    assert web_config.preview_mode == LatentPreviewMethod.AUTO

def test_parse_function():
    """Test the parse function with command line arguments."""
    mock_args = ['-c', 'test_config.toml', '-v']
    with patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(config='test_config.toml', verbose=True, silent=False, daemon=False, help=False)), \
         patch('sdbx.config.Config.__init__', return_value=None) as mock_init:
        config = parse()
        mock_init.assert_called_with('test_config.toml')

def test_node_manager_property(mock_config_instance):
    """Test lazy loading of node_manager."""
    with patch('sdbx.nodes.manager.NodeManager') as mock_node_manager:
        manager = mock_config_instance.node_manager
        mock_node_manager.assert_called_once()

def test_client_manager_property(mock_config_instance):
    """Test lazy loading of client_manager."""
    with patch('sdbx.clients.manager.ClientManager') as mock_client_manager:
        manager = mock_config_instance.client_manager
        mock_client_manager.assert_called_once()