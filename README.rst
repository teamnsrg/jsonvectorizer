===============
JSON Vectorizer
===============

.. image:: https://readthedocs.org/projects/jsonvectorizer/badge/?version=latest
    :target: http://jsonvectorizer.readthedocs.io

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: ./LICENSE

.. sphinx-start

Overview
========

This package contains tools for extracting vector representations of JSON
documents, for various machine learning applications. The implementation
follows a scikit-learn-like interface, and uses three stages for extracting
features, summarized as follows:

- **Learning a schema**: First, a sequence of sample documents are processed to
  learn their structure, or *schema*. This package uses a simplified version of
  the `JSON schema`_, supporting *objects* (Python dictionaries), *arrays*,
  *nulls*, *booleans*, *numbers*, and *strings*. During this stage, sample
  values are also collected for feature generation, e.g., for extracting tokens
  from strings. Additionally, optional fields (i.e., those that are not present
  in any of the inspected documents) are recognized for dictionaries.

- **Pruning**: Once a schema has been learned from sample documents, one can
  optionally prune the built schema using various rules. For instance, sparse
  fields (i.e., those that are only observed for a small percentage of sample
  documents) can be removed, or regular expressions can be used to prune fields
  that cannot produce meaningful features, e.g., hashes.

- **Feature generation**: Finally, features are generated using a list of
  pre-defined rules. Each rule specifies the data type (e.g., string) that it
  can be applied to, along with (optionally) a regular expression for matching
  specific fields. Each rule must also specify a vectorizer for generating
  features from sample values. This package already includes vectorizers for
  booleans, numbers, strings, and timestamps. Additionally, the presense of
  optional fields in dictionaries, and the type of polymorphic fields (e.g.,
  a field that can be null or a string) are also used to generate features from
  nested documents.

**Notes**

- The current implementation only supports binary features. This means that
  numerical data types must be transformed using binning followed by one-hot
  encoding to mark the range that a sample value belongs to.
- An array can be regarded as a list or a tuple. When using lists (default), it
  is assumed that all items in a list conform to the same schema, and features
  are aggregated (by taking the logical or) over all items in the list. In
  contrast, when using tuples the first stage maintains a separate schema for
  each item in a tuple, and features are generated for each item accordingly.

Installation
============

Install using:

.. code-block:: sh

    python setup.py install

Usage
=====

The following example shows how one can build vectorizers for JSON documents
using the aforementioned three-stage process. You can further customize and
fine-tune the parameters in each stage for your specfic data set.

First we instantiate a ``JsonVectorizer`` object, and learn the schema of, and
collect samples from, a set of JSON documents stored in a file (with one record
per line):

.. code-block:: python

    import json
    from jsonvectorizer import JsonVectorizer, vectorizers
    from jsonvectorizer.utils import fopen

    # Load data
    docs = []
    with utils.fopen('samples.json.gz') as f:
        for line in f:
            doc = json.loads(line)
            docs.append(doc)
    
    # Learn the schema of sample documents
    vectorizer = JsonVectorizer()
    for doc in docs:
        vectorizer.extend(doc)

We then prune fields that are present for less than 1% of all observed samples,
and also those starting with an underscore:

.. code-block:: python

    vectorizer.prune(patterns=['^_'], min_f=0.01)

Finally, we create a list of vectorizers for individual data types, and use
them to build a vectorizer for JSON documents:

.. code-block:: python

    # Report booleans as is
    bool_vectorizer = {
        'type': 'boolean',
        'vectorizer': vectorizers.BoolVectorizer
    }

    # For numbers, use one-hot encoding with 10 bins
    number_vectorizer = {
        'type': 'number',
        'vectorizer': vectorizers.NumberVectorizer,
        'kwargs': {'n_bins': 10},
    }

    # For strings use tokenization, ignoring sparse (<1%) tokens
    string_vectorizer = {
        'type': 'string',
        'vectorizer': vectorizers.StringVectorizer,
        'kwargs': {'min_df': 0.01}
    }

    # Build JSON vectorizer
    vectorizers = [
        bool_vectorizer,
        number_vectorizer,
        string_vectorizer
    ]
    vectorizer.fit(vectorizers=vectorizers)

The generated features can be inspected by printing the following property:

.. code-block:: python

    for i, feature_name in enumerate(vectorizer.feature_names_):
        print('{}: {}'.format(i, feature_name))

The constructed vectorizer can then compute feature vectors from any set of
JSON documents, generating SciPy List of Lists (LIL) sparse matrices:

.. code-block:: python

    # Convert to CSR format for efficient row slicing
    X = vectorizer.transform(docs).tocsr()

Note that vectorizer objects are picklable, which means they can be stored on
disk, and later be loaded in a separate session:

.. code-block:: python

    import pickle

    # Saving
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Loading
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

To-Do
=====

- Supporting non-binary features.
- The ability to specifiy an aggregation function (e.g., mean) for lists.

.. _JSON schema: https://spacetelescope.github.io/understanding-json-schema
