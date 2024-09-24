
# API CODE

#### CLASS Config
#### IMPORT from sdbx import config
#### METHODS get_default, get_path_contents, get_path
#### PURPOSE find source directories and data
#### SYNTAX
```
        config.get_default(filename with no extension, key)               (!cannot find sub-keys on its own)
        config.get_path_contents("string_to_folder.string_to_sub_folder") (see config/config.json, config/directories.json)
        config.get_path("filename") or config.get_path("string_to_folder.filename")
```
### OUTPUT contents of a value for a key, file contents of a directory and sub directories, path to a file

#### CLASS ReadMeta
#### IMPORT from sdbx.indexer import ReadMeta
#### METHODS data
#### PURPOSE extract metadata from model files
#### SYNTAX 
```
        instance_name = ReadMeta(full_path_to_file).data()                 (see config/tuning.json)
```
#### OUTPUT dict of int and str, a form filled model_tag[] 

#### CLASS EvalMeta
#### IMPORT from sdbx.indexer import EvalMeta
#### METHODS, process_vae, process_vae_no_12, process_lora, process_tf, process_model 
#### PURPOSE interpret metadata from model files
#### SYNTAX 
```
        instance_name = EvalMeta(dict_metadata_from_ReadMeta).data()        (see config/tuning.json)
                        tag = item[0]                   (TRA, LOR, LLM, DIF, VAE)
                        filename = item[1][0]           (base-name only)
                        compatability = item[1][1:2][0] (short code)
                        data = item[1][2:5]             (meta data dict)
```
#### OUTPUT list of type str: 0: tag code, 1: file size, 2: full path (see tuner.json)

#### CLASS IndexManager
#### IMPORT from sdbx.nodes.tuner import IndexManager
#### METHODS write_index, fetch_compatible
#### PURPOSE manage model type lookups, search for compatibility data
#### SYNTAX 
```


        store_writing = IndexManager().write_index(optional_filename)       (defaults to config/index.json)
        path = IndexManager().fetch_matching(index, query)                  (single type search)
        vae, lora, tra = IndexManager().fetch_compatible(query)             (automated all type search)
                value[0][0] filename
                value[0][1] compatability short code
                value[1][0] size
                value[1][1] path
                value[1][2] dtype

```
#### OUTPUT json file with model metadata, a set of dicts with all compatible models, a dict of model compatible codes



#### Exception handling
#### IMPORT from sdbx.config import logger
#### SYNTAX
```
        logger.debug(self.path, error_log, , exc_info=True)                 (quiet log)
        logger.exception(self.path, error_log)                              (hard lockup/os freeze only)
```
#### OUTPUT detailed error message in log or console