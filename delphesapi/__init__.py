from os.path import isfile
from os.path import dirname
from os.path import join
from os import getenv

version_file = '{}/version.txt'.format(dirname(__file__))

if isfile(version_file):
    with open(version_file) as version_file:
        __version__ = version_file.read().strip()

from dotenv import load_dotenv
env_path = join(dirname(__file__), '..','env.sh')
load_dotenv(dotenv_path=env_path)
