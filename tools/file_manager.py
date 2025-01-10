# Standard libraries
import os
import sys
import shutil
from pathlib import Path

# Third-party libraries
import yaml
from addict import Dict
from yaml_config_override import add_arguments


def directory_exists(path, message):
    '''
       directory_exists() checks if a directory truly exists. If it does not exist, the program will be terminated.

         Args:
         path: The directory path.
         message: The Error Message that will be printed before the program terminates. 

        Returns:
        True: if the directory exists.
    '''
    if os.path.isdir(path):
        return True  
    else:
        print(message)
        sys.exit()
        
def file_exists(file_path, message):
    '''
       file_exists() checks if a file truly exists. If it does not exist, the program will be terminated.

         Args:
         path: The file path.
         message: The Error Message that will be printed before the program terminates. 

        Returns:
        True: if the file exists.
    '''
    if os.path.isfile(file_path):
        return True
    else: 
        print(message)
        sys.exit()

def move_directory(source, destination):
    '''
       move_directory() moves a directory into another directory.

         Args:
         source: The path of the directory that will be moved.
         destination: The destination directory path.
    '''
    path = os.path.join(destination, source)
    if os.path.isdir(path): 
        shutil.rmtree(path)
    shutil.move(source, destination)

    
def configuration(main_path=None):
    '''
       configuration() extract from the configuration file (config\base_config.yaml) the configuration parameters
       and save it in a Dict object. Also it checks if the configuration file exists.

        Args:
        main_path(path string): The path of the parent folder. Default value: None. 

        Returns:
        config (addict Dict): Dict containing configuration parameters.
    '''

    # Set path.
    relative_path = os.path.join('config', 'base_config.yaml')
    if main_path is None: 
        path = relative_path
    else:
        path = os.path.join(main_path, relative_path)

    file_exists(path, " NO CONFIGURATION FILE DETECTED --> ADD A base_config.yaml FILE TO config DIRECTORY ")
    config_source = yaml.safe_load(Path(path).read_text())
    config = Dict(add_arguments(config_source))
    return config