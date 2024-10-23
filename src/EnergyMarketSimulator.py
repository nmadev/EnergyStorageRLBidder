class EnergyMarketSimulator:
    def __init__(self, params):
        self.simulation_name = self._safe_lookup(params, "simulation_name", None)

        return 0

    def _safe_lookup(dictionary: dict = None, key: str = None, default=None):
        """
        Safely lookup a key in a dictionary and return a default value if not found

        :param dictionary: dictionary containing key-value pairs
        :param key: key to access value from dictionary
        :param default: default value to return
        :return: None if dictionary or key does not exist or if key not in dictionary, otherwise returns value from key in dictionary
        """
        if dictionary == None or key == None or key not in dictionary:
            return default
        return dictionary[key]
