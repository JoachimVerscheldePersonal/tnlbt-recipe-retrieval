from munch import Munch
import yaml

def parse_config(config_path: str) -> Munch:
    """
    Parses a YAML configuration file and returns its contents as a Munch object.

    This function reads a YAML file from the specified path and converts the
    YAML structure into a Munch object. Munch is a dictionary-like object that
    allows accessing dictionary keys as object attributes, making it convenient
    for managing configuration data.

    Args:
        config_path (str): The file path to the YAML configuration file.

    Returns:
        Munch: A Munch object containing the parsed configuration data.

    Example:
        # Example usage of the parse_config function
        config = parse_config("config.yaml")
        print(config.some_key)  # Accessing the configuration using dot notation.
        print(config["some_key"])  # Accessing the configuration using dictionary notation.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If the YAML file contains syntax errors.
    """

    with open(config_path) as config_file:
        config_dict = yaml.safe_load(config_file)
        return Munch(config_dict)
