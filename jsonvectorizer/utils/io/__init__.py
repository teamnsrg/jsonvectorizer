import gzip
import os

from .lz4file import Lz4File


def fopen(filename, mode='r'):
    """Open a regular or compressed file

    Detects the type of the file, and uses the relevent module to open
    it. Supports regular (uncompressed), gz, and lz4 files.

    Parameters
    ----------
    filename : str
        Path to a file.
    mode : str, optional (default='r')
        Mode for opening the file, similar to Python's open method. lz4
        files can only be opened for reading.

    Returns
    -------
    file-like object

    Raises
    ------
    ValueError
        If a lz4 file is opened for writing or appending.

    """
    if len(mode) == 1:
        mode += 't'

    ext = os.path.splitext(filename)[1]
    if ext == '.gz':
        return gzip.open(filename, mode)
    elif ext == '.lz4':
        if mode in ['r', 'rt']:
            return Lz4File(filename, decode=True)
        elif mode == 'rb':
            return Lz4File(filename, decode=False)
        else:
            raise ValueError("mode must be 'r', 'rb', or 'rt' for lz4 files")
    else:
        return open(filename, mode)
