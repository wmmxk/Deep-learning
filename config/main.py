from load_config import Config

config = Config.load(file_name="config.yaml")
print(config.paths.DATA_DIR)
print(config.paths.fig_paths)
