# List of lists (LIL) representation of a sparse matrix
# Causes memory leak when passed to cdef functions
ctypedef struct LilMatrix:
    int m, n
    list[:] rows, data

cdef int lil_set_col(list[:] rows, list[:] data, int[:] rs, int col) except -1

cdef int lil_set(list[:] rows, list[:] data, int[:] rs, int[:] cols) except -1
