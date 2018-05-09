import lz4framed


class Lz4File(object):
    """Class for reading lines from lz4 files

    Parameters
    ----------
    filename : str
        Path to a lz4 file.
    decode : bool, optional (default=False)
        If True, decodes each line and returns strings on each
        iteration, otherwise returns bytes.

    """

    def __init__(self, filename, decode=False):
        self.filename = filename
        self.decode = decode
        self._f = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __iter__(self):
        if self._f is None:
            raise RuntimeError('lz4 file has not been opened')

        decoded = b''
        for chunk in lz4framed.Decompressor(self._f):
            decoded += chunk
            decoded = decoded.split(b'\n')
            for data in decoded[:-1]:
                if self.decode:
                    yield data.decode() + '\n'
                else:
                    yield data + b'\n'

            decoded = decoded[-1]

        if decoded:
            yield decoded.decode()

    def open(self):
        """Open the lz4 file"""
        self._f = open(self.filename, 'rb')

    def close(self):
        """Close the lz4 file"""
        self._f.close()
        self._f = None
