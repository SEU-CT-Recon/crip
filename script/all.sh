# generate docs
rmdir /s docs
pdoc -o ./docs ./crip

# packup
python setup.py sdist bdist_wheel

# publish from local
twine upload ./dist/crip-$1-py3-none-any.whl

# run all unittests
python -m unittest discover