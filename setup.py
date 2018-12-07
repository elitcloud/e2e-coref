from setuptools import setup
from setuptools.command.install import install
import atexit

# Out of date

def _post_install():
    pass


class PostInstallCommand(install):
    def __init__(self, *args, **kwargs):
        super(PostInstallCommand, self).__init__(*args, **kwargs)
        atexit.register(_post_install)


setup(name='uw_e2e_coref',
      version='0.1',
      description='End-to-End Coreference Resolution',
      url='https://github.com/elitcloud/e2e-coref',
      packages=['uw_e2e_coref'],
      package_data={'uw_e2e_coref': ['coref_kernels_*.so', 'experiments.conf']},
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
