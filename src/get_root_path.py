from os import environ
from os.path import dirname, abspath, expanduser
from socket import gethostname

def get_root_path():
    return dirname(dirname(abspath(__file__)))

def get_user():
    try:
        home_user = expanduser('~').split('/')[-1]
    except:
        home_user = 'user'
    return home_user

def get_host():
    host = environ.get('HOSTNAME')
    if host is not None:
        return host
    return gethostname()