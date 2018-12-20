import os
import urllib.request
import tarfile
import zipfile
from glearn.utils.printing import print_update


def ensure_download(url, download_dir, extract=False):
    # use the filename from the URL and add it to the download_dir
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # check if the file already exists
    if not os.path.exists(file_path):
        os.makedirs(download_dir, exist_ok=True)

        print(f"Downloading: {url}")

        def report(count, block_size, total_size):
            pct_complete = min(1, float(count * block_size) / total_size)
            print_update(f"Downloading | Progress: {pct_complete:.1%}")

        # download the file from the internet
        file_path, _ = urllib.request.urlretrieve(url=url, filename=file_path, reporthook=report)

        print(f"Download complete: {file_path}")

        if extract:
            print(f"Extracting: {file_path}")

            if file_path.endswith(".zip"):
                zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
            elif file_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
