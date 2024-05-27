# generate docs
rmdir /s docs
pdoc -o ./docs ./crip

# packup
python setup.py sdist bdist_wheel

# publish from local
twine upload ./dist/crip-$1-py3-none-any.whl

# run all unittests
pytest test # basic
pytest --cov-report html --cov=crip test # with coverage report
