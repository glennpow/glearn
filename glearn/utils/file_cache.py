import os
import time
import json
import tempfile
import logging
from glearn.utils.printing import colorize


TEMP_DIR = "/tmp/glearn"

logger = logging.getLogger(__name__)


class FileCache(object):
    """
    This is a simple local-temp-file cache
    """

    def __init__(self, dir=None, verbose=False):
        # dir can be None, in which case a tmp dir is created
        if dir is None:
            self.dir = tempfile.TemporaryDirectory(prefix="cache")
        else:
            self.dir = dir
        self.verbose = verbose

        # prepare cache directory
        os.makedirs(self.file_path(), exist_ok=True)

    def log(self, message):
        if self.verbose:
            logger.info(colorize(message, "magenta"))

    def warn(self, message):
        logger.warn(message)

    def err(self, message):
        logger.err(message)

    def file_path(self, path=None):
        # get full file path to cache
        if path is not None and len(path) > 0:
            # TODO - could do some validation on the path format here
            return os.path.join(self.dir, path)
        return self.dir

    def has(self, path, expires=None):
        # check file/dir existence and possibly age
        file_path = self.file_path(path)
        if os.path.exists(file_path):
            if os.path.isfile(file_path) and expires is not None:
                age = time.time() - os.path.getmtime(file_path)
                return age < expires
            return True
        return False

    def get(self, path, default=None, reader=None, load_json=None):
        """
        Fetch the cached file  (TODO - could allow reading entire directories of files too)
        reader - You can also give a custom reader of signature: read(path) -> value
        load_json - Will load file as json  (superceded by reader)
        """
        # check for cached file
        file_path = self.file_path(path)
        if not os.path.isfile(file_path):
            self.log(f"No cached value found for path: {path}.  Using default: {default}.")
            return default

        # read file
        self.log(f"Reading cached file: {file_path}")
        if reader is not None:
            try:
                value = reader(file_path)
            except Exception as e:
                value = None
                self.warn(f"Exception reading cache for path: '{path}':  {e}")
        else:
            with open(file_path, "r") as f:
                value = f.read()

            # process raw value
            if value is not None:
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                if isinstance(value, str):
                    if load_json is None:
                        load_json = value.startswith("{") or value.startswith("[")
                    if load_json:
                        return json.loads(value)
        return value

    def set(self, path, value, writer=None):
        """
        Writes value to cache file defined by path
        writer - You can specify a custom file writer for the value to cache: write(value, path)
        """
        # serialize value if possible
        if isinstance(value, (dict, list)):
            value = self._make_serializable(value)
            serialized = json.dumps(value)
        else:
            serialized = value

        # check if cached data exists
        file_path = self.file_path(path)
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                raise Exception("Attempting to write a cache file over an existing directory")
            os.remove(file_path)
        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # write serialized value to cache file
        if writer is not None:
            trimmed = str(value)[:200]
            self.log(f"Writing cached file: {file_path}:  {trimmed}")
            try:
                writer(value, file_path)
            except Exception as e:
                self.warn(f"Exception writing cache file for path: '{path}': {e}")
        else:
            trimmed = serialized[:200]
            self.log(f"Writing cached file: {file_path}: [size: {len(serialized)}]  {trimmed}")
            with open(file_path, "w") as f:
                f.write(serialized)

    def delete(self, path):
        # remove cached file/dir
        file_path = self.file_path(path)
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                os.removedirs(file_path)
            else:
                os.remove(file_path)

    def clear(self):
        # remove root cache directory
        os.removedirs(self.file_path())

    def block(self, path, call, writer=None, reader=None, expires=None, bust=False, **kwargs):
        """
        This function will check cache for `path`, and if not found, then invoke `call` to retrieve
        desired value.  It will automatically cache the resulting value at path for future.
        writer - You can specify a custom file writer for the value to cache: write(value, path)
        reader - You can also give a custom reader of signature: read(path) -> value
        expires - Seconds until cached file data should be disregarded.
        bust - This will force the cache to be busted and refreshed for this path
        Remaining *kwargs will be passed along to the get() call
        """
        # first try to get cached
        value = None
        if not bust and self.has(path, expires=expires):
            value = self.get(path, reader=reader, **kwargs)

        # if no cached value, then invoke call
        if value is None:
            value = call()

            # cache
            if value is not None:
                empty = (isinstance(value, str) and len(value.strip()) == 0) \
                    or (hasattr(value, '__len__') and len(value) == 0)
                if not empty:
                    self.set(path, value, writer=writer)
        return value

    def _make_serializable(self, value):
        encodables = [json.JSONEncoder, str, int, float, bool]

        serializable_value = value
        if serializable_value is not None:
            if isinstance(value, dict):
                serializable_value = {}
                for k, v in value.items():
                    serializable_value[k] = self._make_serializable(v)
            elif isinstance(value, list) or isinstance(value, tuple):
                serializable_value = []
                for i, v in enumerate(value):
                    serializable_value.append(self._make_serializable(v))
            elif not any(isinstance(value, enc) for enc in encodables):
                serializable_value = str(value)
        return serializable_value
