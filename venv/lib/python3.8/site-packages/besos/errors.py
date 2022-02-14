"""
This file contains custom errors used by BESOS
"""
import platform


class ModeError(ValueError):
    """An error for when an invalid mode is encountered."""

    def __init__(self, mode=None, message=None):  # pragma: no cover
        if message is None:
            message = f'Invalid mode {mode}. Expected "idf" or "json"'
        super().__init__(message)


class InstallationError(ValueError):
    """
    idf version X does not match with installed EP, please supply ep_path to correct EP version.
    We expect EP to be installed in {}.
    """

    def __init__(self, version=None, ep_path=None, message=None):
        if message is None and ep_path and version:
            message = f"EnergyPlus in IDF does not match ep_path EP version, please supply correct ep_path to v{version}"
            message += f"\n ep_path: {ep_path}\n Version: {version}"
        elif message is None and version:
            message = f"\n Could not locate EnergyPlus V{version}"
            version = version.replace(".", "-")

            if platform.system() == "Windows":
                ep_directory = f"Detected Windows OS: C:/EnergyPlusV{version}"
            elif platform.system() == "Linux":
                ep_directory = f"Detected Linux OS: /usr/local/EnergyPlus-{version}"
            else:
                ep_directory = f"Detected MacOS: /Applications/EnergyPlus-{version}"

            message += f"\n We expected EnergyPlus to be installed in the following path\n {ep_directory}\n Or specify in ep_path argument\n"

        super().__init__(message)
