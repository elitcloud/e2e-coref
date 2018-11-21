from setuptools import setup
from setuptools.command.install import install
from sys import platform
from subprocess import check_call
import atexit


def _post_install():
    if platform == "linux" or platform == "linux2":
        check_call(["./setup_kernel_linux.sh"])
    elif platform == "darwin":
        check_call(["./setup_kernel_mac.sh"])
    else:
        raise Exception('Unknown OS: ' + platform)


class PostInstallCommand(install):
    def __init__(self, *args, **kwargs):
        super(PostInstallCommand, self).__init__(*args, **kwargs)
        atexit.register(_post_install)


setup(name='uw-e2e-coref',
      version='0.1',
      description='End-to-End Coreference Resolution',
      url='https://github.com/elitcloud/e2e-coref',
      install_requires=[
          'tensorflow==1.7.0',
          'tensorflow-hub',
          'h5py',
          'pyhocon',
          'scipy',
          'sklearn',
          'elit==0.1.25.dev1542393574'
      ],
      cmdclass={'install': PostInstallCommand})
