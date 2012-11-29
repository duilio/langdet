#!/bin/sh

# This script downloads wikipedia abstracts

for lang_code in $*; do
    echo Downloading ${lang_code}...
    curl http://dumps.wikimedia.org/${lang_code}wiki/latest/${lang_code}wiki-latest-abstract.xml | gzip > ${lang_code}-abstract.xml.gz
done
