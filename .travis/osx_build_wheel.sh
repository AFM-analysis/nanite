pip install delocate
python setup.py bdist_wheel
delocate-wheel -v dist/*.whl
