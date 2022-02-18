# making test datasets

## one motif, plus strand, binary y-vals

```bash
conda activate motifer
python create_synthetic_data.py \
    --outdir ~/test_data_binary_plus \
    --recnum 3000 \
    --outpre test_binary_one_motif_plus_strand \
    --seqlen 60 \
    --motifs ACATGCAGTC \
    --ncats 2 \
    --dtype categorical \
    --motif-count-plus 1 \
    --motif_peak_frac 1.0 \
    --motif_nonpeak_frac 0.01
```

## one motif, both strands, binary y-vals

```bash
conda activate motifer
python create_synthetic_data.py \
    --outdir ~/test_data_binary_both \
    --recnum 3000 \
    --outpre test_binary_one_motif_plus_strand \
    --seqlen 60 \
    --motifs ACATGCAGTC \
    --ncats 2 \
    --inter-motif-distance 10 \
    --dtype categorical \
    --motif-count-plus 1 \
    --motif-count-minus 1 \
    --motif_peak_frac 1.0 \
    --motif_nonpeak_frac 0.01
```


## ten motifs, both strands, continuous y-vals

```bash
conda activate motifer
python create_synthetic_data.py \
    --outdir ~/test_data_continuous \
    --recnum 3000 \
    --outpre test_continuous \
    --seqlen 60 \
    --motifs ACATGCAGTC CGTGCGTAAT GGGTGTCACA ACTGTCACAT CCATGGAGCA CCCAAATTTG GATCACACAT GGGGGTTTTT AATTGTTAAT ATATATATCT \
    --ncats 10 \
    --dtype continuous \
    --motif-count-plus 1 \
    --motif-count-minus 1 \
    --motif_peak_frac 1.0 \
    --motif_nonpeak_frac 0.01 \
    --inter-motif-distance 10
```

## ten motifs, both strands, categorical y-vals

```bash
conda activate motifer
python create_synthetic_data.py \
    --outdir ~/test_data_categorical \
    --recnum 3000 \
    --outpre test_categorical \
    --seqlen 60 \
    --motifs ACATGCAGTC CGTGCGTAAT GGGTGTCACA ACTGTCACAT CCATGGAGCA CCCAAATTTG GATCACACAT GGGGGTTTTT AATTGTTAAT ATATATATCT \
    --ncats 10 \
    --dtype categorical \
    --motif-count-plus 1 \
    --motif-count-minus 1 \
    --motif_peak_frac 1.0 \
    --motif_nonpeak_frac 0.01 \
    --inter-motif-distance 10
```

