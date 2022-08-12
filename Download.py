import requests, zipfile, io
from tqdm import tqdm
import logging

class downloader:
    datafiles = ["https://figshare.com/ndownloader/files/3381290",
            "https://figshare.com/ndownloader/files/3381293",
            "https://figshare.com/ndownloader/files/3381296",
            "https://figshare.com/ndownloader/files/3381302"
            ]
    
    def __init__(self, datapath= "./Data/brainTumorDataPublic/") -> None:
        self.datapath  =  datapath

    def get(self) -> None:
        for j in  tqdm(range(len(self.datafiles))):
            logging.info(f"Downloading and extracting {self.datafiles[j]}. This may take a while depending on your Internet speed!")
            r = requests.get(self.datafiles[j])
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.datapath)
        logging.info(f"Dataset download and extraction completed successfully to Path={self.datapath}")

if __name__ == "__main__":
    """
      This script downloads all data zip files from https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
      into the ./Data/brainTumorDataPublic directory. 
      The 'cvind.mat' and 'Readme.txt' files are already in the ./Data directory
    """
    pass 
