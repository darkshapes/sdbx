
#### CLASS Config
#### IMPORT from sdbx import config
#### METHODS get_default, get_path_contents, get_path
#### SYNTAX get_default(filename with no extension, header)
####        get_path_contents("string_to_folder.string_to_sub_folder") (see config.json, directories.json)
####        get_path("filename") or get_path("string_to_folder.filename")


#### CLASS ReadMeta
#### IMPORT from sdbx.indexer import ReadMeta
#### METHODS data
#### SYNTAX instance_name = ReadMeta(full_path_to_file).data()
#### OUTPUT dict of int and str, a form filled model_tag[] 

#### CLASS EvalMeta
#### IMPORT from sdbx.indexer import EvalMeta
#### METHODS, process_vae, process_vae_no_12, process_lora, process_tf, process_model 
#### SYNTAX  instance_name = EvalMeta(dict_metadata_from_ReadMeta).data()
#### OUTPUT list of type str: tag code, file size, full path (see tuner.json)

#### Exception handling
#### IMPORT from sdbx.config import logger
#### logger.debug(self.path, error_log) 
#### logger.exception(self.path, error_log) - hard lockup/os freeze only
