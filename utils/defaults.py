from yacs.config import CfgNode as CN

# --------------------------------------------------------------------------------------
# Default experiment configuration (YACS)
# --------------------------------------------------------------------------------------

_BASE_CFG = CN()
_BASE_CFG.CONFIG_PATH = None
_BASE_CFG.EXPERIMENT_NAME = "default-experiment"

# ---- Data settings ----
_BASE_CFG.DATA = CN()
_BASE_CFG.DATA.PATH = None
_BASE_CFG.DATA.DATASET = "pascal"
_BASE_CFG.DATA.IMG_SIZE = 700

# ---- Training settings ----
_BASE_CFG.TRAIN = CN()
_BASE_CFG.TRAIN.SEED = 0
_BASE_CFG.TRAIN.EPOCHS = 80
_BASE_CFG.TRAIN.LEARNING_RATE = 1e-3
_BASE_CFG.TRAIN.DATA_LOADER = CN()
_BASE_CFG.TRAIN.DATA_LOADER.BATCH_SIZE = 12
_BASE_CFG.TRAIN.DATA_LOADER.NUM_WORKERS = 4

# ---- Method-specific settings ----
_BASE_CFG.METHOD = CN()
_BASE_CFG.METHOD.NAME = "emat"
_BASE_CFG.METHOD.BACKBONE_CHECKPOINT = None
_BASE_CFG.METHOD.SUPPORT_DIM = 401


def get_default_cfg() -> CN:
    """Return a clone of the default configuration.

    Returns
    -------
    yacs.config.CfgNode
        An editable copy of the base configuration.
    """

    return _BASE_CFG.clone()
