import logging
from npcm import npcm
from apcm import apcm
from upcm import upcm
from npcm_etaWithoutControl import npcm_eta_zero
from npcm_plot_without_background_noise_handle import npcm_without_handle_backnoise
from npcm_plot import npcm_plot
from npcm_variance import npcm_variance

# log setup
logger = logging.getLogger('algorithm')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logging.log', 'w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)