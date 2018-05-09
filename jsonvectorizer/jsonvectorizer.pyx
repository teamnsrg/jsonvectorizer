cimport cython
cimport numpy as np

import copy
import numpy as np
import re
import scipy.sparse as sp

from .jsontype cimport *
from .lil cimport *
from .schema cimport Schema


cdef tuple get_vectorizer(dict[:] vectorizers, JsonType json_type, str path):
    # Return the first matching vectorizer from the provided list
    for d in vectorizers:
        if 'type' in d:
            if isinstance(d['type'], str):
                if type2str(json_type) != d['type']:
                    continue
            else:
                if type2str(json_type) not in d['type']:
                    continue
        if 'pattern' in d and not re.search(d['pattern'], path):
            continue

        return d['vectorizer'], d.get('args', []), d.get('kwargs', {})

    return None


cdef int hasmatch(str s, str[:] patterns) except -1:
    # Determine whether a string matches any of the given regex patterns
    for pattern in patterns:
        if re.search(pattern, s):
            return 1

    return 0


cdef class JsonVectorizer(Schema):
    """Class for extracting features from JSON documents

    Parameters
    ----------
    schema : dict, optional (default={})
        A valid JSON schema for initializing the object.
    path : tuple of str, optional (default=('root',))
        Path from the top-most node (including the root) to this node.
    tuple_items : bool, optional (default=False)
        If True, JSON arrays are regarded as tuples with different
        schemas for each index, otherwise it is assumed that all items
        conform to the same schema.

    Attributes
    ----------
    path : tuple of str
        Path from the top-most node (including the root) to this node.
    tuple_items : bool
        If True, JSON arrays are regarded as tuples with different
        schemas for each index, otherwise it is assumed that all items
        conform to the same schema.
    type : set
        Valid data types for documents conforming to the current schema.
    required : set of str
        Set of required properties for JSON objects (dictionaries).
    properties : dict
        Mapping between property names, and :class:`JsonVectorizer`
        instances  corresponding to different properties in JSON objects
        (Python dictionaries).
    items : list
        :class:`JsonVectorizer` instances corresponding to different
        items in JSON arrays.
    feature_names_ : list of str
        Array mapping from feature integer indices to feature names.

    """

    cdef:
        dict vectorizers
        int pos
        list feature_names

    cdef readonly:
        dict counts, values
        int n_features

    def __cinit__(self, *args, **kwargs):
        self.counts = {}
        self.values = {}
        self.vectorizers = {}
        self.feature_names = []

    @property
    def feature_names_(self):
        # Names of extracted features
        feature_names = copy.copy(self.feature_names)
        for name in sorted(self.properties):
            feature_names.extend(self.get_property(name).feature_names_)
        for i in range(len(self.items)):
            feature_names.extend(self.get_item(i).feature_names_)

        return feature_names

    cdef JsonVectorizer get_property(self, str name):
        # Retrieve a child property and cast it to the correct type
        return <JsonVectorizer>self.properties[name]

    cdef JsonVectorizer get_item(self, int index):
        # Retrieve a child item and cast it to the correct type
        return <JsonVectorizer>self.items[index]

    cdef int from_dict(self, dict schema) except -1:
        # Restore current object from a dictionary
        Schema.from_dict(self, schema)
        if 'counts' in schema:
            self.counts = {str2type(k): v for k, v in schema['counts'].items()}
        if 'values' in schema:
            self.values = {str2type(k): v for k, v in schema['values'].items()}
        if 'pos' in schema:
            self.pos = schema['pos']
        if 'vectorizers' in schema:
            self.vectorizers = {
                str2type(k): v for k, v in schema['vectorizers'].items()
            }
        if 'feature_names' in schema:
            self.feature_names = schema['feature_names']
        if 'n_features' in schema:
            self.n_features = schema['n_features']

        return 0

    cdef dict to_dict(self):
        # Convert current object to a dictionary
        schema = Schema.to_dict(self)
        if self.counts:
            schema['counts'] = {type2str(k): v for k, v in self.counts.items()}
        if self.values:
            schema['values'] = {type2str(k): v for k, v in self.values.items()}
        if self.pos is not None:
            schema['pos'] = self.pos
        if self.vectorizers:
            schema['vectorizers'] = {
                type2str(k): v for k, v in self.vectorizers.items()
            }
        if self.feature_names:
            schema['feature_names'] = self.feature_names
        if self.n_features is not None:
            schema['n_features'] = self.n_features

        return schema

    cdef int  _extend_self(self, object doc, JsonType json_type) except -1:
        # Extend this node from the provided document
        Schema._extend_self(self, doc, json_type)
        self.counts[json_type] = self.counts.get(json_type, 0) + 1
        if json_type != OBJECT and json_type != ARRAY and json_type != JNULL:
            if json_type in self.values:
                self.values[json_type].append(doc)
            else:
                self.values[json_type] = [doc]

        return 0

    cdef list _prune_self(self, int min_f):
        # Prune this node
        cdef:
            JsonType json_type
            str path = ':'.join(self.path)
            list paths = []

        for json_type in sorted(self.type):
            if self.counts[json_type] < min_f:
                paths.append('{} -> {}'.format(path, type2str(json_type)))
                self.type.remove(json_type)
                del self.counts[json_type]
                if json_type in self.values:
                    del self.values[json_type]
                if json_type == OBJECT:
                    self.properties.clear()
                    self.required.clear()
                elif json_type == ARRAY:
                    del self.items[:]

        if self.type:
            return paths
        else:
            return [path]

    cdef list _prune(self, str[:] patterns, int min_f):
        # Recursively prune this node and its children
        paths = self._prune_self(min_f)
        for name in sorted(self.properties):
            property_ = self.get_property(name)
            path = ':'.join(property_.path)
            drop = hasmatch(path, patterns)
            if drop:
                paths.append(path)
            else:
                paths.extend(property_.prune(patterns=patterns, min_f=min_f))
                drop = not bool(property_.type)
            if drop:
                del self.properties[name]
                if name in self.required:
                    self.required.remove(name)
        for i in reversed(range(len(self.items))):
            item = self.get_item(i)
            path = ':'.join(item.path)
            drop = hasmatch(path, patterns)
            if hasmatch(path, patterns):
                paths.append(path)
            else:
                paths.extend(item.prune(patterns=patterns, min_f=min_f))
                drop = not bool(item.type)
            if drop:
                del self.items[i]

        return paths

    cdef int _fit_self(
        self, int pos, double n_total,
        dict[:] vectorizers, str[:] ignore_patterns
    ) except -1:
        # Extract features from this node
        cdef:
            JsonType json_type
            str path = ':'.join(map(str, self.path))
            dict vectorizers_ = {}

        for json_type in self.type:
            vectorizer = get_vectorizer(vectorizers, json_type, path)
            if vectorizer is not None:
                vectorizers_[json_type] = vectorizer

        self.pos = pos
        for json_type in sorted(self.type):
            if len(self.type) > 1:
                self.feature_names.append(
                    '{} is {}'.format(path, type2str(json_type))
                )
            if json_type == OBJECT:
                for name in sorted(self.properties):
                    if name not in self.required:
                        self.feature_names.append(
                            '{} has property "{}"'.format(path, name)
                        )
            if not hasmatch(path, ignore_patterns):
                if json_type in vectorizers_ and json_type in self.values:
                    Vectorizer, args_, kwargs_ = vectorizers_[json_type]
                    vectorizer = Vectorizer(*args_, **kwargs_)
                    vectorizer = vectorizer.fit(
                        self.values[json_type], n_total=n_total, path=self.path
                    )
                    if vectorizer is not None:
                        self.vectorizers[json_type] = vectorizer
                        self.feature_names.extend([
                            path + ' ' + fn for fn in vectorizer.feature_names_
                        ])
            if json_type in self.values:
                del self.values[json_type]

        return pos + len(self.feature_names)

    cdef int _fit(
        self, int pos, double n_total,
        dict[:] vectorizers, str[:] ignore_patterns
    ) except -1:
        # Recursively extract features from this node and its children
        pos = self._fit_self(pos, n_total, vectorizers, ignore_patterns)
        for name in sorted(self.properties):
            pos = self.get_property(name)._fit(
                pos, n_total, vectorizers, ignore_patterns
            )
        for i in range(len(self.items)):
            item = self.get_item(i)
            pos = item._fit(pos, n_total, vectorizers, ignore_patterns)

        self.n_features = len(self.feature_names_)
        return pos

    cdef int _transform_self(
        self, list docs, list[:] X_rows, list[:] X_data,
        np.ndarray[np.int32_t] indices,
        dict indices_by_type, dict indices_by_property
    ) except -1:
        # Transform documents at this node
        cdef:
            JsonType json_type
            int pos = self.pos

        for json_type in sorted(self.type):
            if len(self.type) > 1:
                if json_type in indices_by_type:
                    indices_ = indices_by_type[json_type]
                    lil_set_col(X_rows, X_data, indices[indices_], pos)

                pos += 1
            if json_type == OBJECT:
                for name in sorted(self.properties):
                    if name not in self.required:
                        if name in indices_by_property:
                            indices_ = indices_by_property[name]
                            lil_set_col(X_rows, X_data, indices[indices_], pos)

                        pos += 1
            if json_type in self.vectorizers:
                vectorizer = self.vectorizers[json_type]
                if json_type in indices_by_type:
                    indices_ = indices_by_type[json_type]
                    docs_ = [docs[i] for i in indices_]
                    rs, cs = vectorizer.transform(docs_).nonzero()
                    if cs.dtype != np.int32:
                        cs = cs.astype(np.int32)

                    lil_set(X_rows, X_data, indices[indices_][rs], pos + cs)

                if hasattr(vectorizer, 'feature_names_'):
                    pos += len(vectorizer.feature_names_)
                else:
                    pos += len(vectorizer.feature_names)

        return 0

    cdef int _transform(
        self, list docs, list[:] X_rows, list[:] X_data,
        np.ndarray[np.int32_t] indices
    ) except -1:
        # Recursively transform documents at this node and its children
        cdef:
            int i, j
            dict indices_by_type = {}
            dict indices_by_property = {}
            list indices_by_item = []

        for i in range(len(docs)):
            json_type = typeof(docs[i])
            if json_type in self.type:
                if json_type in indices_by_type:
                    indices_by_type[json_type].append(i)
                else:
                    indices_by_type[json_type] = [i]

        if OBJECT in indices_by_type:
            for i in indices_by_type[OBJECT]:
                for name in docs[i]:
                    if name in self.properties:
                        if name in indices_by_property:
                            indices_by_property[name].append(i)
                        else:
                            indices_by_property[name] = [i]

        if ARRAY in indices_by_type:
            for i in indices_by_type[ARRAY]:
                for j in range(len(docs[i])):
                    if j * self.tuple_items < len(self.items):
                        if j < len(indices_by_item):
                            indices_by_item[j].append(i)
                        else:
                            indices_by_item.append([i])

        self._transform_self(
            docs, X_rows, X_data,
            indices, indices_by_type, indices_by_property
        )
        if OBJECT in indices_by_type:
            for name, indices_ in sorted(indices_by_property.items()):
                docs_ = [docs[i][name] for i in indices_]
                self.get_property(name)._transform(
                    docs_, X_rows, X_data, indices[indices_]
                )
        if ARRAY in indices_by_type:
            for j, indices_ in enumerate(indices_by_item):
                docs_ = [docs[i][j] for i in indices_]
                if self.tuple_items:
                    self.get_item(i)._transform(
                        docs_, X_rows, X_data, indices[indices_]
                    )
                else:
                    self.get_item(0)._transform(
                        docs_, X_rows, X_data, indices[indices_]
                    )

        return 0

    def prune(self, patterns=[], min_f=1):
        """Prune the learned schema using the provided rules

        Parameters
        ----------
        patterns : list of str (default=[])
            List containing regular expressions. Node paths that match
            any of these patterns will be dropped. Node names in a path
            are separated by colons, e.g., 'foo:bar'.
        min_f : int or float, optional (default=1)
            For all nodes in the learned schema, removes data types with
            less than this many collected samples. An integer is taken
            as an absolute count, and a float indicates the proportion
            of all documents. If all data types in a node are removed,
            the node itself will be dropped from the schema.

        Returns
        -------
        paths : list of str
            List of node paths that were dropped, e.g., 'foo:bar' if a
            node is dropped, or 'foo:bar -> string' if a specific data
            type is removed.

        """
        patterns = np.asarray(patterns, dtype=object)
        if isinstance(min_f, float):
            min_f = int(min_f * sum(self.counts.values()))

        return self._prune(patterns, min_f)

    def fit(self, docs=[], vectorizers=[], ignore_patterns=[]):
        """Fit vectorizer to the provided data

        For each node, the first matching vectorizer is used to extract
        features from the node. Each item of `vectorizers` must be a
        dictionary containing the following fields:

        * **vectorizer** : Class for extracting features. For currently
          supported classes, see :mod:`jsonvectorizer.vectorizers`.
        * **type** (str or list of str, optional) : Data type(s) that
          can be used with `vectorizer`. If not provided, matches all
          supported data types: {'object', 'array', 'null', 'boolean',
          'integer', 'number', 'string'}.
        * **pattern** (str, optional) : When provided, nodes that do not
          match this regular expression will be ignored.
        * **args** (list, optional) : Positional arguments passed to
          `vectorizer` for initialization.
        * **kwargs** (dict, optional) : Keyword arguments passed to
          `vectorizer` for initialization.

        Parameters
        ----------
        docs : iterable object, optional (default=[])
            Iterable containing JSON documents for learning a schema and
            fitting vectorizers. Alternatively, the :meth:`extend`
            method can be used for this step.
        vectorizers : list of dict, optional (default=[])
            List of vectorizer definitions (see above for details) for
            extracting features from individual nodes.
        ignore_patterns : list of str, optional (default=[])
            List containing regular expressions. Node paths that match
            any of these patterns will be ignored. Node names in a path
            are separated by colons, e.g., 'foo:bar'.

        Returns
        -------
        self

        """
        for doc in docs:
            self._extend(doc)

        vectorizers = np.asarray(vectorizers, dtype=object)
        ignore_patterns = np.asarray(ignore_patterns, dtype=object)
        self._fit(0, sum(self.counts.values()), vectorizers, ignore_patterns)

        return self

    def transform(self, docs):
        """Transform JSON documents to feature matrix.

        Parameters
        ----------
        docs: iterable object
            Iterable containing JSON documents.

        Returns
        -------
        X: sparse LIL matrix, [n_samples, n_features]
            Feature matrix.

        """
        cdef np.ndarray[np.int32_t] indices
        if not isinstance(docs, list):
            docs = list(docs)

        X = sp.lil_matrix((len(docs), self.n_features), dtype=bool)
        indices = np.arange(len(docs), dtype=np.int32)
        self._transform(docs, X.rows, X.data, indices)

        return X
