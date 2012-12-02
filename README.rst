==========================
Simple language classifier
==========================

This repository contain a simple python module for doing language
classification with some basic algorithms.

This is not supposed to be used in any production environment, is just an
example to show how language detection works.

Classes
-------

The classes defined in the ``langdet`` modules implement some algorithms
to compute language detection using `cosine similarity`_.

If you want to use one of these classes in your code, is probably better
to serialize a model and use it:

.. code-block:: python

  >>> from langdet import TrigramCosineLanguageDetector as LD, stream_sample
  >>> ld = LD()
  >>> ld.train(stream_sample('datasets/train.txt'))
  >>> import pickle
  >>> with open('model.ldm', 'wb') as fout:
  ...     pickle.dump(ld, fout)
  ... 


And then load in your code:

.. code-block:: python

  import pickle
  ld = pickle.load(open('model.ldm', 'rb'))
  lang, score = ld.detect('some text to test it work')


Datasets
--------

The datasets included in the ``datasets`` directory are extracted from
`Wikipedia abstracts`_, you can find the tools used to generate the datasets in
the ``tools`` dir.

To dump the data from wikipedia use the ``dump_wiki.sh`` script:

.. code-block:: bash

  $ ./tools/dump_wiki.sh LANG1 LANG2 ...

Where ``LANGX`` is the `two letter language code`_. The script requires `curl`_
(by default installed in most *nix and BSD-derivative distributions like OSX or
Ubuntu) and download the abstract dumps and create the files
``LANGX-abstract.xml.gz``.

After you've dumped the abstract, you can extract the text of each abstract with
the ``wiki_extract_docs.py`` script and get two random samples for training and
testing the language classifiers using ``gen_datasets.sh``:

.. code-block:: bash

  $ for lang in de en es fr it pt; do
  >   echo computing ${lang}...;
  >   ./tools/wiki_extract_docs.py ${lang}-abstract.xml.gz | ./tools/gen_datasets.sh 10000;
  > done


.. _`cosine similarity`: http://en.wikipedia.org/wiki/Cosine_similarity
.. _`Wikipedia abstracts`: http://en.wikipedia.org/wiki/Wikipedia:Database_download
.. _`two letter language code`: http://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
.. _`curl`: http://curl.haxx.se/
