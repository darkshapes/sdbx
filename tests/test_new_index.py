import unittest
from unittest.mock import patch, MagicMock
import os
from collections import defaultdict
import json
import struct

from new_index import BlockIndex

class TestBlockIndex(unittest.TestCase):

    @patch('os.path.getsize')
    @patch('safetensors.torch.load_file')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('new_index.BlockIndex.error_handler')  # Assuming error_handler is part of BlockIndex
    def test_main_safetensors(self, mock_error_handler, mock_open, mock_load_file, mock_getsize):
        # Setup
        block_index = BlockIndex()
        file_name = "model.safetensors"
        path = "/path/to/model"

        mock_getsize.return_value = 1024  # Mock file size
        mock_load_file.return_value = {"key": "value"}  # Mock successful load

        # Act
        block_index.main(file_name, path)

        # Assert
        mock_load_file.assert_called_once_with(file_name)  # Ensure the file loader is called
        self.assertEqual(block_index.identifying_values["file_size"], 1024)  # Ensure file size is set
        self.assertFalse(mock_error_handler.called)  # Ensure error handler was not called on success

    @patch('os.path.getsize')
    @patch('safetensors.torch.load_file', side_effect=Exception("Failed to load file"))  # Simulate load failure
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('new_index.BlockIndex.error_handler')  # Assuming error_handler is part of BlockIndex
    def test_main_safetensors_failure(self, mock_error_handler, mock_open, mock_load_file, mock_getsize):
        # Setup
        block_index = BlockIndex()
        file_name = "model.safetensors"
        path = "/path/to/model"

        mock_getsize.return_value = 1024  # Mock file size

        # Act
        block_index.main(file_name, path)

        # Assert
        mock_load_file.assert_called_once_with(file_name)  # Ensure the file loader is called
        self.assertEqual(block_index.identifying_values["file_size"], 1024)

    @patch('os.path.getsize')
    @patch('safetensors.torch.load_file', side_effect=Exception("Failed to load file"))
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('struct.unpack', return_value=(16,))
    @patch('json.loads', return_value={"header": "value"})
    @patch('new_index.BlockIndex.error_handler')
    def test_main_unsafetensors_retry(self, mock_error_handler, mock_json_loads, mock_struct_unpack, mock_open, mock_load_file, mock_getsize):
        # Setup
        block_index = BlockIndex()
        file_name = "model.safetensors"
        path = "/path/to/model"
        block_index.path = path

        mock_getsize.return_value = 1024  # Mock file size

        # Act
        block_index.main(file_name, path)

        # Assert
        mock_load_file.assert_called_once_with(file_name)  # Ensure the file loader is called
        mock_open.assert_called_once_with(path, 'rb')  # Ensure the file open is attempted in retry
        mock_error_handler.assert_called_with(kind="retry", error_log=mock_load_file.side_effect, identity=".safetensors", reference=file_name)

    # You can write similar tests for .pt/.pth and .gguf by mocking __unpickle and __ungguf similarly.

if __name__ == "__main__":
    unittest.main()
