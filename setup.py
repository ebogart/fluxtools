from distutils.core import setup

setup(
    name='fluxtools',
    version='0.1',
    author='Eli Bogart',
    author_email='elb87@cornell.edu',
    packages=['fluxtools', 'fluxtools.test', 'fluxtools.expr_manip',
              'fluxtools.reaction_networks', 'fluxtools.utilities'],
    url='https://github.com/ebogart/fluxtools',
    description='Nonlinear constraint-based metabolic modeling tools',
    long_description=open('README.md').read(),
    )
