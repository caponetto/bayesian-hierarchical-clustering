#!/bin/sh
flake8 .

coverage erase
coverage run -m pytest
coverage report -m
coverage html -d coverage_report

rm -fr dist
rm -fr build
python -m build --wheel --sdist
