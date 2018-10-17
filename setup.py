# TO DO:
#   * Upload to GitHub and put down the url 
#   * Check Licenses

from setuptools import setup

setup(
    name='SOcial ANalysis',
    url='',
    author='Maarten Grootendorst',
    author_email='maartengrootendorst@gmail.com',
    packages=['soan'],
    description='Used to analyze whatsapp data',
    long_description=open('README.txt').read(),
    install_requires=['numpy', 'PIL', 'scipy', 'sklearn', 'regex', 're', 'emoji', 'seaborn',
                     'datetime', 'itertools', 'pandas', 'operator', 'requests', 'mpl_toolkits', 
                      'wordcloud', 'pattern', 'palettable'],
    version='0.1',
    license='MIT',
)