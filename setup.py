# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import io
import sys
import setuptools
import subprocess

from aqua import __VERSION__

NAME = 'aqua'
VERSION = __VERSION__

with io.open('README.md', 'r', encoding="utf-8") as fp:
    description = fp.read()

with open('requirements.txt', 'r') as reqfile:
    req = [line.strip() for line in reqfile if line and not line.startswith('#')]

def run(args):
    subprocess.run(args, stdout=sys.stdout, stderr=sys.stdout, check=True, encoding='utf8')
    sys.stdout.flush()

#pkgs = [elem.replace('autonml/', '') for elem in glob('autonml/static/*', recursive=True) if os.path.isfile(elem)]

setuptools.setup(
    name=NAME,
    version=VERSION,
    install_requires=req,
    url="https://github.com/autonlab/aqua",
    description=r"AQuA: A Benchmarking Tool for Label Quality Assessment",
    long_description=description,
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    include_package_data=True,
    author='Mononito Goswami, Vedant Sanil, Arjun Choudhry, Arvind Srinivasan, Chalisa Udompanyawit, Artur Dubrawski',
    maintainer='Mononito Goswami, Vedant Sanil, Arvind Srinivasan',
    maintainer_email='vsanil@andrew.cmu.edu',
    keywords=['data-science', 'machine-learning', 'data-cleaning', 'robust-machine-learning', 'data-centric-ai'],
    license='MIT-License',
    long_description_content_type='text/markdown',
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",]
)
