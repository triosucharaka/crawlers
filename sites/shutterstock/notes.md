# Stages Info.:

- Stage 1: Site Mapping
- Stage 2: Dataset Download and organizing
- Stage 3: Dataset De-watermarking (im2im/*-swmm)
- Stage 4: Frame Encoding (VAE/AutoencoderKL)
- Stage 5: Caption Encoding (CLIPTextModel/clip-vit-large-patch14)

### SQL DB download for stage 2

SQL DB download path (authenticated):

https://huggingface.co/datasets/shinonomelab/cleanvid-15m_map/resolve/main/sql/database.db

### TPU/JAX Speeds for the VAE

fancy pants (w/cgpt4):

| Per Image | Total Time Taken (10 rounds) | Devices; Batch Size |
|-----------|-----------------------------|---------------------|
| 0.007595  | 19.441504                    | 8; 32               |
| 0.007988  | 20.449864                    | 8; 32               |
| 0.007887  | 20.191192                    | 8; 32               |
| 0.013348  | 17.084991                    | 8; 16               |
| 0.013620  | 17.433118                    | 8; 16               |
| 0.012902  | 16.513953                    | 8; 16               |
| 0.013376  | 8.560501                     | 8; 8                |
| 0.013048  | 8.350401                     | 8; 8                |
| 0.012520  | 8.012182                     | 8; 8                |

og results:

```
Per Image               |  Total Time Taken (10 rounds)| Devices; Batch Size
0.007594532426446676        19.441503763198853          8; 32
0.007988434843719005        20.449864387512207          8; 32
0.007887406926602124        20.19119167327881           8; 32
0.013347899541258812        17.084990739822388          8; 16
0.013619952462613582        17.433118104934692          8; 16
0.012901734933257103        16.51395297050476           8; 16
0.013376232236623764        8.560500860214233           8; 8
0.013047899678349495        8.350401401519775           8; 8
0.012519768625497817        8.012182235717773           8; 8
```