from .jsontype cimport *

cdef class Schema:
    cdef readonly:
        tuple path
        bint tuple_items

        set type
        set required
        dict properties
        list items

    cdef int add_property(self, str name, dict schema) except -1

    cdef int add_item(self, dict schema) except -1

    cdef Schema get_property(self, str name)

    cdef Schema get_item(self, int index)

    cdef int from_dict(self, dict schema) except -1

    cdef dict to_dict(self)

    cdef int  _extend_self(self, object doc, JsonType json_type) except -1

    cdef int _extend(self, object doc) except -1
