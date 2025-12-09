"""

This script downloads the MIDOGpp dataset. The total size of the dataset is 65 GB so running the script might take a while.  

Reference: https://github.com/DeepMicroscopy/MITOS_WSI_CCMCT/blob/master/Setup.ipynb

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


FILENAMES = {'databases/MITOS_WSI_CCMCT_Tumorzone.sqlite': 
                          'https://ndownloader.figshare.com/files/16261586' ,
                     'databases/MITOS_WSI_CCMCT_ODAEL.sqlite':
                          'https://ndownloader.figshare.com/files/16261571' ,
                     'databases/MITOS_WSI_CCMCT_HEAEL.sqlite':
                          'https://ndownloader.figshare.com/files/16261583' ,
                     'databases/MITOS_WSI_CCMCT_MEL.sqlite':
                          'https://ndownloader.figshare.com/files/16261574' ,
                     'WSI/96274538c93980aad8d6.svs': # 3
                          'https://ndownloader.figshare.com/files/16261559', 
                     'WSI/1018715d369dd0df2fc0.svs': # 20
                          'https://ndownloader.figshare.com/files/16261562',
                     'WSI/9374efe6ac06388cc877.svs': # 26
                          'https://ndownloader.figshare.com/files/16261553',
                     'WSI/552c51bfb88fd3e65ffe.svs': # 27
                          'https://ndownloader.figshare.com/files/16261556',
                     'WSI/285f74bb6be025a676b6.svs': # 29
                          'https://ndownloader.figshare.com/files/16261550',
                     'WSI/91a8e57ea1f9cb0aeb63.svs': # 24
                          'https://ndownloader.figshare.com/files/16261544',
                     'WSI/70ed18cd5f806cf396f0.svs': # 35
                          'https://ndownloader.figshare.com/files/16261541',
                     'WSI/066c94c4c161224077a9.svs': # 25 
                          'https://ndownloader.figshare.com/files/16261547',
                     'WSI/39ecf7f94ed96824405d.svs': # 19
                          'https://ndownloader.figshare.com/files/16261529',
                     'WSI/34eb28ce68c1106b2bac.svs': # 14
                          'https://ndownloader.figshare.com/files/16261538',
                     'WSI/20c0753af38303691b27.svs': # 21
                          'https://ndownloader.figshare.com/files/16261532',
                     'WSI/3f2e034c75840cb901e6.svs': # 15
                          'https://ndownloader.figshare.com/files/16261505',
                     'WSI/2efb541724b5c017c503.svs': #22 
                          'https://ndownloader.figshare.com/files/16261520',
                     'WSI/2f2591b840e83a4b4358.svs':#23
                          'https://ndownloader.figshare.com/files/16261514',
                     'WSI/8bebdd1f04140ed89426.svs': # 17
                          'https://ndownloader.figshare.com/files/16261523',
                     'WSI/8c9f9618fcaca747b7c3.svs': # 9
                          'https://ndownloader.figshare.com/files/16261526',
                     'WSI/2f17d43b3f9e7dacf24c.svs': # 8
                          'https://ndownloader.figshare.com/files/16261535',
                     'WSI/f3741e764d39ccc4d114.svs': # 31
                          'https://ndownloader.figshare.com/files/16261493',
                     'WSI/fff27b79894fe0157b08.svs': # 7
                          'https://ndownloader.figshare.com/files/16261490',
                     'WSI/f26e9fcef24609b988be.svs': # 6
                          'https://ndownloader.figshare.com/files/16261496',
                     'WSI/dd4246ab756f6479c841.svs': # 18
                          'https://ndownloader.figshare.com/files/16261487',
                     'WSI/c3eb4b8382b470dd63a9.svs': # 4
                          'https://ndownloader.figshare.com/files/16261466',
                     'WSI/c86cd41f96331adf3856.svs': # 30
                          'https://ndownloader.figshare.com/files/16261475',
                     'WSI/c91a842257ed2add5134.svs': # 1
                          'https://ndownloader.figshare.com/files/16261481',
                     'WSI/dd6dd0d54b81ebc59c77.svs': # 28
                          'https://ndownloader.figshare.com/files/16261478',
                     'WSI/be10fa37ad6e88e1f406.svs': # 11
                          'https://ndownloader.figshare.com/files/16261469',
                     'WSI/ce949341ba99845813ac.svs': # 34
                          'https://ndownloader.figshare.com/files/16261484',
                     'WSI/a0c8b612fe0655eab3ce.svs': # 13
                          'https://ndownloader.figshare.com/files/16261424',
                     'WSI/add0a9bbc53d1d9bac4c.svs': # 2
                          'https://ndownloader.figshare.com/files/16261436',
                     'WSI/2e611073cff18d503cea.svs': # 32
                          'https://ndownloader.figshare.com/files/16261439',
                     'WSI/0e56fd11a762be0983f0.svs': # 31
                          'https://ndownloader.figshare.com/files/16261442',
                     'WSI/ac1168b2c893d2acad38.svs': # 12
                          'https://ndownloader.figshare.com/files/16261445',
                    }


LOCATION = 'MITOS_WSI_CCMCT'

MESSAGE = """
#########################################################################################################################################

This script downloads the MITOS_WSI_CCMCT dataset. The total size of the dataset is above 44GB so running the script might take a while.

#########################################################################################################################################
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=LOCATION, help=f'Location to store images (default: {LOCATION}).')
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
    for file in tqdm(list(FILENAMES.keys())):
        try:
            url = FILENAMES[file]
            file_location = location.joinpath(file)
            if file_location.exists():
                log.info(f'File already exists: {file}.')
                continue
            elif not file_location.parent.exists():
                file_location.parent.mkdir(parents=True, exist_ok=True)
                urlretrieve(url, file_location)
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
