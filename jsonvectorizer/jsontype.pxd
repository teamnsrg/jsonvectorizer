# Supported types for JSON documents
cdef enum JsonType: OBJECT, ARRAY, JNULL, BOOLEAN, INTEGER, NUMBER, STRING, TIMESTAMP

cdef JsonType typeof(object doc) except? JNULL

cdef JsonType str2type(str json_type) except? JNULL

cdef str type2str(JsonType json_type)
