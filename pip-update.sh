# pip install twine 
# pip install build

rm -rf dist
rm -rf build
rm -rf *.egg-info
python -m build

# use twine
twine upload dist/*