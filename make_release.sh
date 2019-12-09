#!/bin/bash

# v1.0.0, v1.5.2, etc.
versionLabel=v$1

echo "tagging $versionLabel..."
git tag -d $versionLabel
git push -d origin $versionLabel
git commit --allow-empty -m "REL: $versionLabel"
git tag $versionLabel  # Don't forget the leading v

echo "updating origin master..."
git push origin master
git push origin $versionLabel

echo "removing kruft..."
git clean -dfx

echo "making the wheel..."
python setup.py sdist
python setup.py bdist_wheel

echo "uploading to PyPI..."
twine upload dist/* --cert ~/cert.pem