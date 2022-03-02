'''
    Utilities of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''
import os


def mgfbp(config_path):
    """
        Use mangoct-1.1 'mgfbp' to reconstruct: https://gitee.com/njjixu/mangoct/releases/Mangoct_ver1.1
    """
    mangoct_mgfbp = r'..\..\mangoct\mgfbp.exe'  # win environment
    cmd = mangoct_mgfbp + ' ' + config_path
    os.system(cmd)
