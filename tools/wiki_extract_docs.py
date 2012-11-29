#!/usr/bin/env python
import os
import sys
import gzip
import re
import codecs
from functools import partial

try:
    from lxml.etree import iterparse
except:
    from warnings import warn
    warn('lxml not found, fallback to ElementTree. Might be very slow!')
    from xml.etree.ElementTree import iterparse


def main():
    fout = codecs.getwriter('utf8')(sys.stdout)
    filename = sys.argv[1]
    lang = os.path.basename(filename)[:2]

    replace_spaces = partial(re.compile('\s+', re.U).sub, ' ')

    with gzip.open(sys.argv[1], 'rb') as fin:
        for _, el in iterparse(fin, tag='doc', encoding='utf8'):
            abstract = replace_spaces(el.findtext('abstract'))
            if abstract.strip() in ('__NOTOC__', '.', ''):
                continue
            try:
                print >>fout, '%s\t%s' % (lang, abstract)
            except IOError:
                break
            el.clear()

if __name__ == '__main__':
    sys.exit(main())
