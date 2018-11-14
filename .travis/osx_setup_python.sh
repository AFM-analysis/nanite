#!/bin/bash

if [ -z $1 ]; then
    echo "Please specify Python version as command line argument!"
    exit 1
fi

# get previous directory
OLD="$(pwd)"

# get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

# setup MacPython version management
source osx_utils.sh

# get MacPython version
MPV="$(fill_pyver $1)"

# create download directory
DLD=${DIR}/dl_cache
mkdir -p $DLD

# download MacPython
PKG="python-${MPV}-macosx10.6.pkg"
curl https://www.python.org/ftp/python/${MPV}/${PKG} > ${DLD}/${PKG}

# install MacPython
sudo installer -pkg ${DLD}/${PKG} -target /

# install latest version of pip
curl https://bootstrap.pypa.io/get-pip.py -o ${DLD}/get-pip.py
python ${DLD}/get-pip.py

# install virtualenv
python -m pip install virtualenv

# create virtualenv
PP="/Library/Frameworks/Python.framework/Versions/${MPV::3}/bin/python${MPV::3}"
python -m virtualenv --no-site-packages -p $PP .env
source .env/bin/activate

# install ca certificates
# (resolves [SSL: CERTIFICATE_VERIFY_FAILED])
pip install certifi
/Applications/Python\ ${MPV::3}/Install\ Certificates.command

# Use TkAgg to avoid
# "ImportError: Python is not installed as a framework"
echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc

# go back
cd $OLD
