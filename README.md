Prepare data
Reference:Johns N I, Gomes A L, Yim S S, et al.Metagenomic mining of regulatory elements enables programmable species-selective gene expression.Nature methods,2018, 15 (5): 323-329.

ecoli_data\seq.txt and ecoli_data\exp.txt The data used by the generator
ecoli_data\seq.txt and ecoli_data\exp.txt The data used by the predictor
Design Promoter Sequence

We take design promoters in E.coli as an example,to illustrate how to train the ProCLRDM model and design the promoter sequences

1.Training the vae
run \training\train_ecoli_vae.py
Train this data augmentation module to obtain latent space data.A VAE model checkpoint file will be generated：my_vae_model_state_dict.pth

run \training\train_ecoli_diffusion.py
Use above VAE model checkpoint file，Train the conditional latent diffusion model(ProCLRDM).

run \Generation\ecoli_data.py_generation.py
Use the VAE model checkpoint：my_vae_model_state_dict.pth and the diffusion model checkpoint:best_model_overall.pth.Generate promoter sequence data using conditional information.


2.Evaluate the performance of ProSTR model generated sequences
DNAshape:run valid\dnashape.py

GC:run valid\GCviolin.py

Diversity:run valid\editdistance_combine.py

K-mer:run valid\kmer.py

BLAST search:run valid\blast.py