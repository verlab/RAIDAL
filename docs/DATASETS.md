# Dataset preparation

The active learning loop creates labeled and unlabeled dataset copies next to the original dataset. The large frame folders and LMDB files are linked instead of copied. The dataset directory and its parent directory therefore need write permission.

## PHOENIX-2014

Download the original PHOENIX-2014 dataset from [the original PHOENIX-2014 website](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) and place the extracted files inside `CorrNet/dataset/phoenix2014`.

The folders should follow this exact structure:

```text
CorrNet/
└── dataset/
    └── phoenix2014/
        └──phoenix-2014-multisigner
           ├──annotations
           ├──evaluation
           └──features
              └──fullFrame-210x260px
```

Before training the model, you must run the preprocessing code inside `CorrNet/preprocess`:

```bash
cd CorrNet/preprocess
```

```bash
python3 dataset_preprocess.py --process-image --multiprocessing --save-gloss-dict
```

This creates the 256 by 256 frames and the files used by CorrNet under `dataset/phoenix2014/.../fullFrame-256x256px`. It also creates the gloss dictionary and the information dictionaries for the train, test and dev splits under `preprocess/phoenix2014/`.

Note that `dataset_preprocess.py` edits the original `train.corpus.csv` in place. It removes the annotation for `13April_2011_Wednesday_tagesschau_default-14`, which a known broken sample that crashes training. This was inherited from CorrNet's and Swin-MSTP's codebase, so keep a copy of the file if you need the untouched original.

## PHOENIX-T

PHOENIX-T follows a very similar procedure to PHOENIX-2014.

Download it from [the original PHOENIX-T website](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) and place the extracted files inside `CorrNet/dataset`.

Before running the preprocessing script, make sure the dataset folder follows this exact structure:

```text
CorrNet/
└── dataset/
    └── phoenix2014-T/
        ├──annotations
        ├──evaluation
        └──features
           └──fullFrame-210x260px
```

Then run:


```bash
cd CorrNet/preprocess
```

```bash
python3 dataset_preprocess-T.py --process-image --multiprocessing --save-gloss-dict
```


## Isharah

Isharah requires a custom folder structure before it can be used. The first step is to download it from [its public repository](https://snalyami.github.io/Isharah_CSLR/). Look for the Isharah-1000 download link.

The Isharah-1000 dataset is split into multiple files. Download everything, extract the files and then merge all parts into a single folder.

1) First, create this folder structure:

```text
CorrNet/
└── dataset/
    └── Isharah/
        ├──Annotations
        │  └──isharah1000
        │     └──SI
        └──features
           └──fullFrame-256x256px
```

And place all the Signer Independent (SI) annotation files inside `isharah1000/SI`.

2) Move all the video files from the extracted folders into `fullFrame-256x256px/`, so that each folder inside `fullFrame-256x256px/` contains the frames of exactly one video. 

**Heads-up about `14.zip`.** On my machine, `unzip` flagged this file as a zip bomb and refused to write out some of its folders, such as `14_0012/`. The command still runs, but since `14_0012/` is never extracted, it's easy to end up with videos silently missing. Extract this part with the detection turned off like this:

```bash
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip 14.zip
```

Afterwards, check if the .jpg files inside `14_0012/` are actually there before moving on. `14.zip` is the only file that needs this. The rest can be extracted normally.

The final structure of Isharah should be exactly this:

```text
CorrNet/
└── dataset/
    └── Isharah/
        ├──Annotations
        │  └──isharah1000
        │     └──SI
        │        ├──train.txt
        │        ├──test.txt
        │        └──dev.txt
        └──features
           └──fullFrame-256x256px
              ├──00_0001
              │  ├──frame0000.jpg
              │  ├──frame0001.jpg
              │  ├──frame0002.jpg
              │  ├──...
              │  └──frame0053.jpg
              │
              ├──00_0002
              │  ├──frame0000.jpg
              │  ├──frame0001.jpg
              │  ├──...
              │  └──frame0146.jpg
              │
              ├──...
              │
              └──17_1000
                 ├──frame0000.jpg
                 ├──frame0001.jpg
                 ├──...
                 └──frame0238.jpg
```

3) Run the preprocessing script:

```bash
cd CorrNet/preprocess
```

```bash
python3 dataset_preprocess-Isharah.py --save-gloss-dict
```

Notice that dataset_preprocess-Isharah.py does **NOT** use `--process-image` and `--multiprocessing` because the original images are already 256x256px.

## LMDB

If your storage device is slow (HDD, slow SATA SSD, etc.), I provide scripts that convert the dataset into an LMDB database, which makes reading much faster by storing each video as a single ZSTD-compressed record instead of hundreds of standalone image files (.png for PHOENIX, .jpg for Isharah) that have to be read and decoded individually. If you have a good NVMe M.2 SSD this is usually optional, but in my experience using LMDB helps slower storage devices by a lot. Only run these scripts after having already preprocessed the data.

There is one script per dataset in `preprocess/`. Their `DATA_ROOT` is a relative path, so run them from inside `preprocess/`:

```bash
cd CorrNet/preprocess
python createLMDB-Phoenix2014.py   # or createLMDB-Phoenix-T.py / createLMDB-Isharah.py
```

If your dataset is stored somewhere else, change the `DATA_ROOT` constant at the top of the script.

By default, each script writes its LMDB here:

| Script | Output |
|---|---|
| `createLMDB-Phoenix2014.py` | `dataset/phoenix2014/phoenix-2014-multisigner/features/phoenix2014_videos.lmdb` |
| `createLMDB-Phoenix-T.py` | `dataset/phoenix2014-T/features/phoenix2014-T_videos.lmdb` |
| `createLMDB-Isharah.py` | `dataset/Isharah/features/Isharah_videos.lmdb` |

Keep these names and locations, since the dataloader and the active learning loop look for the LMDB at exactly these paths.

**You need to change the config files if you want to use LMDB!** The current configuration uses the standard `video` setting, which will read the videos as PNGs. Go into the config files in `configs/` and change `datatype: 'video'` to `datatype: 'lmdb'` and you'll be good to go.

