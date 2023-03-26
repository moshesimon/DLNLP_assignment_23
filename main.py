import pandas as pd
import tarfile
import bson

# tar = tarfile.open("/scratch/zceemsi/Datasets/dump_small.tar.gz", "r:gz")

# tar.extractall('/scratch/zceemsi/Datasets/')

# tar.close()

#load data
# load bson file

data = bson.decode_file_iter(open('/scratch/zceemsi/Datasets/dump/sefaria/dafyomi.bson', 'rb'))

# convert bson to pandas dataframe

df = pd.DataFrame(data)

print(df.head())

print(df.loc[0]['displayValue'])

#df = pd.read_json('/scratch/zceemsi/Datasets/dump/sefaria/dafyomi.metadata.json')

#print(df.head())

