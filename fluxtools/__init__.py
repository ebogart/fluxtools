import os
_TEMP_DIR = os.path.join(os.getcwd(), '.fluxtools')

if not os.path.exists(_TEMP_DIR):
    os.makedirs(_TEMP_DIR)
