cimport cython


@cython.cdivision(True)
cdef inline int bisect_left(list a, int x) except -1:
    # Find index at which to insert x to maintain order
    # Simplified version of counterpart in scipy.sparse._csparsetools
    cdef:
        int lo = 0
        int hi = len(a)
        int mid, v

    if hi > 0:
        v = a[hi - 1]
        if x > v:
            return hi

    while lo < hi:
        mid = lo + (hi - lo) // 2
        v = a[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid

    return lo


cdef int lil_insert(list[:] rows, list[:] datas, int i, int j) except -1:
    # Insert a single entry into a binary LIL matrix
    # Simplified version of counterpart in scipy.sparse._csparsetools
    cdef:
        list row = rows[i]
        list data = datas[i]
        int pos = bisect_left(row, j)

    if pos == len(row):
        row.append(j)
        data.append(True)
    elif row[pos] != j:
        row.insert(pos, j)
        data.insert(pos, True)
    else:
        data[pos] = True

    return 0


cdef int lil_set_col(list[:] rows, list[:] data, int[:] rs, int col) except -1:
    # Set rows in a given column of a LIL matrix
    cdef int i
    for i in range(rs.shape[0]):
        lil_insert(rows, data, rs[i], col)

    return 0


cdef int lil_set(list[:] rows, list[:] data, int[:] rs, int[:] cols) except -1:
    # Set arbitrary entries in a LIL matrix
    cdef int i
    for i in range(rs.shape[0]):
        lil_insert(rows, data, rs[i], cols[i])

    return 0
