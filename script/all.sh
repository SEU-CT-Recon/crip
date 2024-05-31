# generate docs
rmdir /s docs
rmdir /s htmlcov
pdoc -o ./docs ./crip
pytest --cov-report html --cov=crip test
move htmlcov docs

# packup
python setup.py sdist bdist_wheel

# publish from local
twine upload ./dist/crip-$1-py3-none-any.whl

# run all unittests
pytest test # basic
