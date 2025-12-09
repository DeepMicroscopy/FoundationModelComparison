"""

This script downloads the MIDOG 2022 dataset. The total size of the dataset is 65 GB so running the script might take a while.  


"""

from pathlib import Path
from tqdm import tqdm
from urllib.request import urlretrieve
import argparse
import json
import logging 
import os
import pandas as pd 


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

log = logging.getLogger(__name__)

IMAGES = [
    f"{str(x + 1).zfill(3)}.png" for x in range(405)
    ]

DATABASES = [
    "MIDOG2022_training_png.json", 
    "MIDOG2022_training_png.sqlite"
    ] 

FILES = IMAGES + DATABASES

LOCATION = 'MIDOG2022'


MESSAGE = """
####################################################################################################################################

This script downloads the MIDOG 2022 (png) dataset. The total size of the dataset is 30 GB so running the script might take a while.

####################################################################################################################################
"""



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=LOCATION, help='Location to store images.')
    return parser.parse_args()



def main(args):

    # Initialize the download
    log.info(MESSAGE)

    # Create location to store images 
    location = Path(args.location)
    if not location.exists():
        location.mkdir(exist_ok=True, parents=True)
        log.info(f'Created folder to store images at: {location}.')
    else:
        log.info(f'Images will be stored at: {location}.')

    # Start the download 
    log.info('Start downloading.')
    for file in tqdm(FILES):
        try:
            url = f'https://zenodo.org/records/6547151/files/{file}'
            file_location = location.joinpath(file)
            if file_location.exists():
                log.info(f'File already exists: {file}.')
                continue
            else:
                urlretrieve(url, file_location)
        except Exception as e:
            log.error(f'Failed to download {file}: {str(e)}')
            raise

    # Download complete
    log.info('Download complete.')


if __name__ == '__main__':
    args = get_args()
    main(args)




    