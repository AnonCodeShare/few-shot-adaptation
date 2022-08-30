# few-shot-adaptation
This repository contains code and resources for the paper _Few-shot Adaptation Works with UnpredicTable Data_, currently under review.

![Tables-to-tasks](/img/tables_to_tasks.png)

This repository contains submodules. To clone the full repository along with submodules (required for reproducing training/results), please use
```
git clone --recurse-submodules git@github.com:Anon/few-shot-adaptation.git
```

## UnpredicTable dataset

### Download
Our datasets are available [on the HuggingFace Hub](https://huggingface.co/datasets/unpredictable/unpredictable_full). We provide the complete dataset `UnpredicTable-full` as well as the various sub-distributions discussed in our paper, for a total of 57 dataset options.

To download a dataset, simply `pip install datasets` and download the dataset using `load_dataset`:
```python
from datasets import load_dataset

distribution_names = [
    # Full dataset
    "unpredictable/unpredictable_full",
    # 5k random tasks from full dataset
    "unpredictable/unpredictable_5k",
    # Filtered to 1 task per website
    "unpredictable/unpredictable_unique",
    #  Single website tasks
    "unpredictable/unpredictable_baseball-fantasysports-yahoo-com",
    "unpredictable/unpredictable_bulbapedia-bulbagarden-net",
    "unpredictable/unpredictable_cappex-com",
    "unpredictable/unpredictable_cram-com",
    "unpredictable/unpredictable_dividend-com",
    "unpredictable/unpredictable_dummies-com",
    "unpredictable/unpredictable_en-wikipedia-org",
    "unpredictable/unpredictable_ensembl-org",
    "unpredictable/unpredictable_gamefaqs-com",
    "unpredictable/unpredictable_mgoblog-com",
    "unpredictable/unpredictable_mmo-champion-com",
    "unpredictable/unpredictable_msdn-microsoft-com",
    "unpredictable/unpredictable_phonearena-com",
    "unpredictable/unpredictable_sittercity-com",
    "unpredictable/unpredictable_sporcle-com",
    "unpredictable/unpredictable_studystack-com",
    "unpredictable/unpredictable_support-google-com",
    "unpredictable/unpredictable_w3-org",
    "unpredictable/unpredictable_wiki-openmoko-org",
    "unpredictable/unpredictable_wkdu-org",
    # Single cluster tasks
    "unpredictable/unpredictable_cluster00", "unpredictable/unpredictable_cluster01", "unpredictable/unpredictable_cluster02", "unpredictable/unpredictable_cluster03", "unpredictable/unpredictable_cluster04", "unpredictable/unpredictable_cluster05", "unpredictable/unpredictable_cluster06", "unpredictable/unpredictable_cluster07", "unpredictable/unpredictable_cluster08", "unpredictable/unpredictable_cluster09", "unpredictable/unpredictable_cluster10", "unpredictable/unpredictable_cluster11", "unpredictable/unpredictable_cluster12", "unpredictable/unpredictable_cluster13", "unpredictable/unpredictable_cluster14", "unpredictable/unpredictable_cluster15", "unpredictable/unpredictable_cluster16", "unpredictable/unpredictable_cluster17", "unpredictable/unpredictable_cluster18", "unpredictable/unpredictable_cluster19", "unpredictable/unpredictable_cluster20", "unpredictable/unpredictable_cluster21", "unpredictable/unpredictable_cluster22", "unpredictable/unpredictable_cluster23", "unpredictable/unpredictable_cluster24", "unpredictable/unpredictable_cluster25", "unpredictable/unpredictable_cluster26", "unpredictable/unpredictable_cluster27", "unpredictable/unpredictable_cluster28", "unpredictable/unpredictable_cluster29", "unpredictable/unpredictable_cluster-noise", 
    # Manual-rated tasks
    "unpredictable/unpredictable_rated-low", "unpredictable/unpredictable_rated-medium", "unpredictable/unpredictable_rated-high",
]

# Get the 5k sample dataset
dataset = load_dataset('unpredictable/unpredictable_5k')
```

We provide a demo of loading and inspecting tasks from the dataset at `dataset_demo.ipynb`. Click the badge below to try it out with Colab!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnonCodeShare/few-shot-adaptation/blob/master/dataset_demo.ipynb)


### Recreate

This section provides instructions for recreating the UnpredicTable dataset.

Install requirements:
```
conda create -n unpredictable python=3.8
conda activate unpredictable
python -m pip install -r requirements.txt
```

Recreating the dataset involves the following steps:
1. Download the `English-Language Relational Web Tables 2015` source tables from [WDC Web Table Corpus 2015](http://webdatacommons.org/webtables/2015/downloadInstructions.html).
2. Extract the files.
3. Convert the tables into tasks (.jsonl format).

Since the source tables are provided as 51 separate slices, we process each of the slices separately:
```bash
SLICE="00" # Repeat for each of 00, 01, 02 ... 50
# Download
wget http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/$SLICE.tar.gz
# Extract
tar -xvf $SLICE.tar.gz
# Convert
python tables_to_tasks.py --tarfile $SLICE.tar --outdir ./unpredictable/ --max_source_files 10000
```

For convenience, we provide sbatch scripts for performing the the above steps in a parallelized manner on a SLURM system. To download and extract all 51 slices via 51 parallel batch jobs, simply run `bash download_and_process_all.sh`. (Caution: Will generate ~150GB and ~500k files)

## MetaICL training and evaluation
This section provides instructions for reproducing our main results with [MetaICL](https://github.com/facebookresearch/MetaICL).

We use a modified copy of the MetaICL repository to simplify working with our dataset, found at `few-shot-adaptation/MetaICL`.
 
To install the required dependencies, please follow the "Installation" section of `MetaICL/README.md`.

### Model weights & Training
The weights for our fine-tuned GPT2-large model can be downloaded below:
- Fine-tuned on `UnpredicTable-5k` - [weights](https://drive.google.com/file/d/1Q1mh9rKxD6MX0lTD_okWEjINWRNfqhXY/view?usp=sharing)
- Fine-tuned on `support.google.com` - [weights](https://drive.google.com/file/d/1AM_3tXJjAixrJ3R5q3chSnFuW3Uk_6YR/view?usp=sharing)

To train your own models, please follow the instructions in the "Training" section of `MetaICL/README.md`.

For training on our task datasets, you can use the HuggingFace dataset path with the prefix "huggingface:" as the `$task`. For example, to train on `unpredictable/unpredictable_5k`, use
```bash
cd MetaICL/

task="huggingface:unpredictable/unpredictable_5k"
python train.py \
  --task $task --k 16384 --test_k 16 --seed 100 --use_demonstrations --method channel \
  --do_tensorize --n_gpu 8 --n_process 40
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 8 \
  --batch_size 1 --lr 1e-05 --fp16 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task
```

### Evaluation
Given the trained model, you can use the `MetaICL/reproduce.sh` script to evaluate the test scores for each of the task settings:

```bash
cd MetaICL/

MODEL_PATH="/PATH/TO/gpt2large-unpredictable5k.pt"
bash reproduce.sh hr_to_lr metaicl 100,13,21,42,87 32 $MODEL_PATH
bash reproduce.sh class_to_class metaicl 100,13,21,42,87 32 $MODEL_PATH
bash reproduce.sh qa_to_qa metaicl 100,13,21,42,87 32 $MODEL_PATH
bash reproduce.sh non_nli_to_nli metaicl 100,13,21,42,87 32 $MODEL_PATH
bash reproduce.sh non_paraphrase_to_paraphrase metaicl 100,13,21,42,87 32 $MODEL_PATH
```
