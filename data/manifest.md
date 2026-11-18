# Local dataset manifest

Dataset images and annotations are local inputs and are not redistributed with the project repository. They are excluded from version control because of their size and dataset terms. See the dataset sources in `../README.md` and preserve the folder structure checked below.

## Raw image counts

| Dataset partition | Files |
|---|---:|
| HICO-DET train2015 | 38,118 |
| HICO-DET test2015 | 9,658 |
| COCO 2014 train2014 | 82,783 |
| COCO 2014 val2014 | 40,504 |

## UPT loader preflight

Every UPT annotation-referenced image in the following partitions exists and the first sample decoded successfully:

| UPT dataset partition | Loader images | Missing referenced images |
|---|---:|---:|
| HICO-DET train2015 | 37,633 | 0 |
| HICO-DET test2015 | 9,546 | 0 |
| V-COCO trainval | 4,969 | 0 |
| V-COCO test | 4,532 | 0 |

The UPT dataset classes exclude images without labeled interaction pairs, so loader lengths are lower than raw image file counts.

## UPT dataset annotation JSON checksums (SHA-256)

| File | SHA-256 |
|---|---|
| `instances_train2015.json` | `09A967CA9EA154765F75694C09940483CE1741CD9E45789095AD7A99781A743F` |
| `instances_test2015.json` | `843E8EFED473AF94FEE13E6428AB2F49652210C82D2FD2697E4D38CA02BC6815` |
| `instances_vcoco_train.json` | `712CB26ACD8123089667C310F3B36F85CEB6A022626233D3138C34029A64B0AB` |
| `instances_vcoco_val.json` | `BE8FF0B360F8DBB7023EB4B2A8B42AFA13DBA588EC009B1905A18A9AD2EFD24C` |
| `instances_vcoco_trainval.json` | `A9CD182D9FE4574E6743E43A40367FE524BAB08D9CB56377F9BE98BC7430D272` |
| `instances_vcoco_test.json` | `CCBE12875ABB459692D85D2366381240BC883E5FDC22CF15610EA524DE062326` |
