# making test datasets

## one motif, plus strand, binary y-vals

```bash
cdMotif
conda activate motifer
python ~/src/DNAshape_motif_finder/python3/create_synthetic_data.py \
    --outdir test_data_binary_plus \
    --recnum 3000 \
    --outpre test_data_bindary_plus \
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
cdMotif
conda activate motifer
python ~/src/DNAshape_motif_finder/python3/create_synthetic_data.py \
    --outdir test_data_binary_both \
    --recnum 3000 \
    --outpre test_data_binary_both \
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

## five motifs, both strands, continuous y-vals

```bash
cdMotif
conda activate motifer
python ~/src/DNAshape_motif_finder/python3/create_synthetic_data.py \
    --outdir test_five_cat_continuous \
    --recnum 3000 \
    --outpre test_five_cat_continuous \
    --seqlen 60 \
    --motifs CGTGCGTAAT ACTGTCACAT CCCAAATTTG GGGGGTTTTT ATATATATCT \
    --ncats 5 \
    --dtype continuous \
    --motif-count-plus 1 \
    --motif-count-minus 1 \
    --motif_peak_frac 1.0 \
    --motif_nonpeak_frac 0.01 \
    --inter-motif-distance 10
```

## five motifs, both strands, categorical y-vals

```bash
cdMotif
conda activate motifer
python ~/src/DNAshape_motif_finder/python3/create_synthetic_data.py \
    --outdir test_five_cat_categorical \
    --recnum 3000 \
    --outpre test_five_cat_categorical \
    --seqlen 60 \
    --motifs CGTGCGTAAT ACTGTCACAT CCCAAATTTG GGGGGTTTTT ATATATATCT \
    --ncats 5 \
    --dtype categorical \
    --motif-count-plus 1 \
    --motif-count-minus 1 \
    --motif_peak_frac 1.0 \
    --motif_nonpeak_frac 0.01 \
    --inter-motif-distance 10
```

## ten motifs, both strands, continuous y-vals

```bash
cdMotif
conda activate motifer
python ~/src/DNAshape_motif_finder/python3/create_synthetic_data.py \
    --outdir test_data_continuous \
    --recnum 3000 \
    --outpre test_data_continuous \
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

## ten motifs, both strands, continuous y-vals, many records

```bash
cdMotif
conda activate motifer
python ~/src/DNAshape_motif_finder/python3/create_synthetic_data.py \
    --outdir test_data_continuous_30000 \
    --recnum 30000 \
    --outpre test_data_continuous_30000 \
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
cdMotif
conda activate motifer
python ~/src/DNAshape_motif_finder/python3/create_synthetic_data.py \
    --outdir test_data_categorical \
    --recnum 3000 \
    --outpre test_data_categorical \
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

## five motifs, both strands, categorical y-vals, with extra pivot category

```bash
cdMotif
conda activate motifer
python ~/src/DNAshape_motif_finder/python3/create_synthetic_data.py \
    --outdir test_data_categorical_pivot \
    --recnum 3000 \
    --outpre test_data_categorical_pivot \
    --seqlen 60 \
    --motifs CGTGCGTAAT ACTGTCACAT CCCAAATTTG GGGGGTTTTT ATATATATCT \
    --ncats 10 \
    --dtype categorical \
    --motif-count-plus 1 \
    --motif-count-minus 1 \
    --motif_peak_frac 1.0 \
    --motif_nonpeak_frac 0.01 \
    --inter-motif-distance 10 \
    --pivot-category
```

## getting JASPAR sequence motifs

```bash
cdMotif
cd JASPAR_db
sh ./get_seq_motifs.sh
```

