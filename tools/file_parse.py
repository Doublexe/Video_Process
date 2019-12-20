import os
import glob


class FileNode(object):
    """An utility to construct hierachical file structure.

    Init:
    Parameters
    ----------
    root : str
        The root (can be relative) for this file / directory
    path : str
        The full (can be relative) path for this file / directory.
        Assume the file has a suffix with '.' and the direcotry doesn't have
        any '.' in its name.
    """

    def __init__(self, root, path):
        self.root = os.path.abspath(root)
        self.path = os.path.abspath(path)
        self.hierachy = os.path.dirname(os.path.relpath(path, root))
        self.file_name = os.path.basename(path)  # Name for file or directory
        self.name = '.'.join(self.file_name.split('.')[
                             :-1])  # Name without suffix

    def echo(self, new_root, make_dir=False):
        """Construct the hierachy structure at another root.

        Parameters
        ----------
        new_root : str
            The new root (can be relative) to echo.
        make_dor : bool
            If true, construct a directory named the same as the file node.

        Example:
            root = 'root/'
            path = 'root/dir1/file1'

            then if new_root = 'new_root/'
            echoed_path = 'new_root/dir1/'
            echoed_dir = 'new_root/dir1/file1.name/'
        """
        new_root = os.path.abspath(new_root)
        new_path = os.path.join(new_root, self.hierachy)
        if make_dir:
            new_path = os.path.join(new_path, self.name)

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        return new_path


def parse_hierachy(root, filt='**/*'):
    """Parse the data hierachy and return FileNode for each leaf file (or directory).

    Parameters
    ----------
    root : str
        The root (can be relative) to parse.
    filter : str
        The glob regex to match, default on all files.
    """
    # If recursive is true, the pattern “**” will match any files and zero or more directories,
    # subdirectories and symbolic links to directories. If the pattern is followed by an os.sep or os.altsep then files will not match.
    matches = glob.iglob(os.path.join(
        os.path.abspath(root), filt), recursive=True)
    for match in matches:
        yield FileNode(root, match)
