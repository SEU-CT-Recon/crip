__title__ = 'crip'
__author__ = 'z0gSh1u'
__license__ = 'MIT License'
__doc__ = '''
CT Recon in Python: An all-in-one tool for Data IO, Pre/Post-process, Physics, Dual Energy, Low Dose and everything.
https://github.com/SEU-CT-Recon/crip
'''

__all__ = [
    'de', 'io', 'lowdose', 'physics', 'postprocess', 'preprocess', 'shared', 'utils', 'mangoct', 'plot', 'metric', '_rc'
]

from . import _rc
from . import de
from . import io
from . import lowdose
from . import physics
from . import postprocess
from . import preprocess
from . import shared
from . import utils
from . import mangoct
from . import plot
from . import metric
