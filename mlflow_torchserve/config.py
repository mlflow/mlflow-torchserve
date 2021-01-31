import os


class Config(dict):
    def __init__(self):
        """
        Initializes constants from Environment variables
        """
        super().__init__()
        self["export_path"] = os.environ.get("EXPORT_PATH")
        self["config_properties"] = os.environ.get("CONFIG_PROPERTIES")
        self["torchserve_address_names"] = ["inference_address", "management_address", "export_url"]
        self["export_uri"] = os.environ.get("EXPORT_URL")
