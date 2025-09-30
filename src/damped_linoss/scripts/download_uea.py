"""
This script downloads and unzips the UEA data from the timeseriesclassification website.
"""

import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent


def download_and_unzip(url, save_dir, zipname):
    """Downloads and unzips a (g)zip file from a url.

    Args:
        url (str): The url to download from.
        save_dir (str): The directory to save the (g)zip file to.
        zipname (str): The name of the (g)zip file.

    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if len(os.listdir(save_dir)) == 0:
        urllib.request.urlretrieve(url, zipname)
        print("Downloaded data to {}".format(zipname))
        if zipname.split(".")[-1] == "gz":
            with tarfile.open(zipname, "r:gz") as tar:
                tar.extractall(save_dir)
        else:
            with zipfile.ZipFile(zipname, "r") as zip:
                zip.extractall(save_dir)
    else:
        print("Data already exists in {}".format(save_dir))


if __name__ == "__main__":
    save_dir = BASE_DIR / "data" / "raw" / "UEA"
    url = (
        "http://www.timeseriesclassification.com/aeon-toolkit/Archives"
        "/Multivariate2018_arff.zip"
    )
    zipname = save_dir / "uea.zip"

    download_and_unzip(
        url,
        str(save_dir),
        str(zipname),
    )
