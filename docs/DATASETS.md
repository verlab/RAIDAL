# Dataset preparation

The active learning loop creates labeled and unlabeled dataset copies next to the original dataset. The large frame folders and LMDB files are linked instead of copied. The dataset directory and its parent directory therefore need write permission.

## PHOENIX 2014

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
python dataset_preprocess.py --process-image --multiprocessing --save-gloss-dict
```

This creates the 256 by 256 frames and the files used by CorrNet under `preprocess/phoenix2014`, including the gloss dictionary and the information dictionaries for the three splits.

## PHOENIX-T

PHOENIX-T follows a very similar procedure to PROENIX-2014.

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
python dataset_preprocess-T.py --process-image --multiprocessing --save-gloss-dict
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
python dataset_preprocess-Isharah.py --save-gloss-dict
```

Notice that dataset_preprocess-Isharah.py does **NOT** use `--process-image` and `--multiprocessing` because the original images are already 256x256px.

## LMDB

If your storage device is slow (HDD, slow SATA SSD, etc.), I provide a script that converts the dataset into an LMDB file, which makes reading the files much faster by turning each video into a single ZSTD-compressed file rather than storing it as hundreds of standalone .png files that have to be read and decoded individually. If you have a good NVME M.2 SSD this is usually optional, but in my experience using LMDB helps slower storage devices by a lot. Only run this script after having already preprocessed the data.

You can find the appropriate scripts for each dataset in `preprocess/createLMDB-X.py`. They currently have a fixed `DATA_ROOT`. Change that constant if your dataset is stored somewhere else.

The script will write the LMDB file here by default:

```text
CorrNet/dataset/phoenix2014/phoenix-2014-multisigner/features/phoenix2014_videos.lmdb
```

**You need to change the config files if you want to use LMDB!** The current configuration uses the standard `video` setting, which will read the videos as PNGs. Go into the config files in `configs/` and change `datatype: 'video'` to `datatype: 'lmdb'` and you'll be good to go.

