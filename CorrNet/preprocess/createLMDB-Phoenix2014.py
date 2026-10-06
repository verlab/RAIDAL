import lmdb
import zstandard as zstd
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# path to the root folder containing 'features'
DATA_ROOT = '../dataset/phoenix2014/phoenix-2014-multisigner'
OUTPUT_LMDB = os.path.join(DATA_ROOT, "features", 'phoenix2014_videos.lmdb')
MAP_SIZE = 1099511627776
COMMIT_FREQUENCY = 100

MODES = ['train', 'test', 'dev']

def create_dataset():
    total_counter=0

    if not os.path.exists(OUTPUT_LMDB):
        os.makedirs(OUTPUT_LMDB)
    env = lmdb.open(OUTPUT_LMDB, map_size=MAP_SIZE)

    txn = env.begin(write=True)

    compressor = zstd.ZstdCompressor(level=16, threads=16)
    
    for mode in MODES:
        # construct path: .../features/fullFrame-256x256px/train/
        search_path = os.path.join(DATA_ROOT, 'features/fullFrame-256x256px', mode)
        
        # find all video folders (e.g. 10April_2010.../1/)
        video_folders = glob.glob(os.path.join(search_path, '*', '1'))
        
        print(f"Processing {mode}: Found {len(video_folders)} videos.")
        
        for folder_path in tqdm(video_folders):
            img_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))
            
            if not img_paths:
                continue
                
            # read all images into a single Numpy Buffer (Time, H, W, C)
            video_frames = []
            for img_p in img_paths:
                # Read as standard BGR
                img = cv2.imread(img_p)
                video_frames.append(img)
            
            # stack into one array of shape (T, 256, 256, 3)
            video_array = np.array(video_frames, dtype=np.uint8)
            
            # store shape in the first 4 integers so we can reconstruct it
            shape                 = np.array(video_array.shape, dtype=np.int32)
            data_bytes_compressed = compressor.compress(video_array.tobytes())
            shape_bytes           = shape.tobytes()
            

            video_folder_path = os.path.dirname(folder_path)
            key               = os.path.basename(video_folder_path)
            
            # write to LMDB
            txn.put(key.encode('ascii'), shape_bytes + data_bytes_compressed)

            total_counter += 1
            if total_counter % COMMIT_FREQUENCY == 0:
                txn.commit()  # flush to disk
                print(f"Committed {total_counter} videos...") # for debugging
                txn = env.begin(write=True) # start new bucket

    txn.commit()
    env.close()
    print("LMDB Created Successfully!")

if __name__ == "__main__":
    create_dataset()