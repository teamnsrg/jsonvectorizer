cimport cython

import re
import sys

from .jsontype cimport *


# Python version (for handling unicode strings)
cdef int VERSION = sys.version_info.major


cdef class Schema:
    """Class for learning a schema from JSON documents

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
        Mapping between property names, and :class:`Schema` instances
        corresponding to different properties in JSON objects (Python
        dictionaries).
    items : list
        :class:`Schema` instances corresponding to different items in
        JSON arrays.

    """

    def __cinit__(self, *args, **kwargs):
        self.type = set()
        self.properties = {}
        self.items = []

    def __init__(
        self, dict schema={}, tuple path=('root',), bint tuple_items=False
    ):
        self.path = path
        schema['tuple_items'] = schema.get('tuple_items', tuple_items)
        self.from_dict(schema)

    def __reduce__(self):
        return (self.__class__, (self.to_dict(), self.path, self.tuple_items))

    cdef int add_property(self, str name, dict schema) except -1:
        # Add a child property
        path = self.path + (name,)
        self.properties[name] = type(self)(
            schema=schema, path=path, tuple_items=self.tuple_items
        )
        return 0

    cdef int add_item(self, dict schema) except -1:
        # Add a child item
        path = self.path
        path += (len(self.items),) if self.tuple_items else ('any',)
        item = type(self)(
            schema=schema, path=path, tuple_items=self.tuple_items
        )
        self.items.append(item)
        return  0

    cdef Schema get_property(self, str name):
        # Retrieve a child property and cast it to the correct type
        return <Schema>self.properties[name]

    cdef Schema get_item(self, int index):
        # Retrieve a child item and cast it to the correct type
        return <Schema>self.items[index]

    cdef int from_dict(self, dict schema) except -1:
        # Restore this object from a dictionary
        self.tuple_items = schema['tuple_items']
        if 'type' in schema:
            if isinstance(schema['type'], str):
                self.type.add(str2type(schema['type']))
            else:
                for json_type in schema['type']:
                    self.type.add(str2type(json_type))
        if 'required' in schema:
            self.required = set(schema['required'])
        if 'properties' in schema:
            for key, value in schema['properties'].items():
                self.add_property(key, value)
        if 'items' in schema:
            if isinstance(schema['items'], dict):
                self.add_item(schema['items'])
            else:
                for item in schema['items']:
                    self.add_item(item)

        return 0

    cdef dict to_dict(self):
        # Convert this object to a dictionary
        schema = dict(tuple_items=self.tuple_items)
        if self.type:
            schema['type'] = [type2str(t) for t in sorted(self.type)]
        if self.required is not None:
            schema['required'] = self.required
        if self.properties:
            schema['properties'] = {
                name: self.get_property(name).to_dict()
                for name in self.properties
            }
        if self.items:
            schema['items'] = [
                self.get_item(i).to_dict() for i in range(len(self.items))
            ]

        return schema

    cdef int  _extend_self(self, object doc, JsonType json_type) except -1:
        # Extend this node from the provided document
        self.type.add(json_type)
        if json_type == OBJECT:
            for key in doc:
                if VERSION == 2 and type(key) is unicode:
                    key = key.encode('utf-8')
                if key not in self.properties:
                    self.add_property(key, {})
            if self.required is None:
                self.required = set(doc.keys())
            else:
                self.required &= set(doc.keys())
        elif json_type == ARRAY:
            if self.tuple_items:
                while len(self.items) < len(doc):
                    self.add_item({})
            elif not self.items:
                self.add_item({})

        return 0

    cdef int _extend(self, object doc) except -1:
        # Recursively extend this node and its children
        json_type = typeof(doc)
        self._extend_self(doc, json_type)
        if json_type == OBJECT:
            for key, value in doc.items():
                if VERSION == 2 and type(key) is unicode:
                    key = key.encode('utf-8')
                self.get_property(key)._extend(value)
        elif json_type == ARRAY:
            if self.tuple_items:
                for i, item in enumerate(doc):
                    self.get_item(i)._extend(item)
            else:
                for item in doc:
                    self.get_item(0)._extend(item)

        return 0

    def find_nodes(self, patterns):
        """Find nodes that match any of the provided regular expressions

        Parameters
        ----------
        patterns : list of str
            List of regular expression patterns for finding nodes.

        Returns
        -------
        paths : list of tuple
            List of paths for matching nodes. Each item is a tuple,
            containing the path from the top-most node (including the
            root) to a matching node.

        """
        paths = []
        for pattern in patterns:
            if re.search(pattern, ':'.join(self.path)):
                paths.append(self.path)
                break

        for name in sorted(self.properties):
            property_ = self.get_property(name)
            paths.extend(property_.find_nodes(patterns))
        for i in range(len(self.items)):
            item = self.get_item(i)
            paths.extend(item.find_nodes(patterns))

        return paths

    def extend(self, docs):
        """Extend the schema to conform to the provided documents

        Parameters
        ----------
        docs : iterable object
            Iterable containing JSON documents.

        """
        for doc in docs:
            self._extend(doc)
