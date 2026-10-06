import lmdb
import zstandard as zstd
import numpy as np
import cv2
import glob
import os

from tqdm import tqdm
import concurrent.futures


# --- CONFIGURATION ---
# path to the root folder containing 'features'
DATA_ROOT        = '../dataset/Isharah'
OUTPUT_LMDB      = os.path.join(DATA_ROOT, "features", 'Isharah_videos.lmdb')
MAP_SIZE         = 1099511627776  # 1 TB reserved space
COMMIT_FREQUENCY = 20
LEVEL            = 18  # 0 = fastest, lowest compression ratio | 20 = slowest, highest compression ratio


def create_dataset():
    if not os.path.exists(OUTPUT_LMDB):
        os.makedirs(OUTPUT_LMDB)
    env = lmdb.open(OUTPUT_LMDB, map_size=MAP_SIZE)

    txn = env.begin(write=True)
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        counter = 0
        # construct path: .../features/fullFrame-256x256px/train/
        search_path = os.path.join(DATA_ROOT, 'features/fullFrame-256x256px')
        
        # find all video folders
        video_folders = glob.glob(os.path.join(search_path, '*'))
        
        print(f"Processing: Found {len(video_folders)} videos.")

        results_iterator = executor.map(processImg, video_folders)

        for result in tqdm(results_iterator, total=len(video_folders), desc="Writing to LMDB"):
            if result is None:
                continue
        
            key, data = result
            txn.put(key, data)

            counter += 1
            if counter % COMMIT_FREQUENCY == 0:
                txn.commit()
                txn = env.begin(write=True)

    txn.commit()
    env.close()
    print("LMDB Created Successfully!")


def processImg(folder_path):
        compressor = zstd.ZstdCompressor(level=LEVEL, threads=1)

        img_paths = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))

        video_frames = []
        # read all images
        for img_p in img_paths:
            # read as standard BGR
            img = cv2.imread(img_p)
            video_frames.append(img)
        
        # stack into one array of shape (T, 256, 256, 3)
        video_array = np.array(video_frames, dtype=np.uint8)

        # store shape in the first 4 integers so we can reconstruct it
        shape                 = np.array(video_array.shape, dtype=np.int32)
        data_bytes_compressed = compressor.compress(video_array.tobytes())
        shape_bytes           = shape.tobytes()
        
        # write to LMDB
        key               = os.path.basename(folder_path)

        key               = key.encode('ascii')
        payload           = shape_bytes + data_bytes_compressed
        
        return (key, payload)


if __name__ == "__main__":
    create_dataset()