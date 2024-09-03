# GenSelfDiff-HIS: Generative Self-Supervision Using Diffusion for Histopathological Image Segmentation (TMI 2024)
This is the official repository for the implementation of (paper link)

# Abstract
 Histopathological image segmentation is a laborious and time-intensive task, often requiring analysis from experienced pathologists for accurate examinations. To reduce this burden, supervised machine-learning approaches have been adopted using large-scale annotated datasets for histopathological image analysis. However, in several scenarios, the availability of large-scale annotated data is a bottleneck while training such models. Self-supervised learning (SSL) is an alternative paradigm that provides some respite by constructing models utilizing only the unannotated data which is often abundant. The basic idea of SSL is to train a network to perform one or many pseudo or pretext tasks on unannotated data and use it subsequently as the basis for a variety of downstream tasks. It is seen that the success of SSL depends critically on the considered pretext task. While there have been many efforts in designing pretext tasks for classification problems, there have not been many attempts on SSL for histopathological image segmentation. Motivated by this, we propose an SSL approach for segmenting histopathological images via generative diffusion models. Our method is based on the observation that diffusion models effectively solve an image-to-image translation task akin to a segmentation task. Hence, we propose generative diffusion as the pretext task for histopathological image segmentation. We also utilize a multi-loss function-based fine-tuning for the downstream task. We validate our method using several metrics on two publicly available datasets along with a newly proposed head and neck (HN) cancer dataset containing Hematoxylin and Eosin (H&E) stained images along with annotations.

# Methodology
<p align="center">
<img width="578" alt="diff_model" src="https://github.com/PurmaVishnuVardhanReddy/GenSelfDiff-HIS/assets/103281951/c6d68807-bafe-4589-aef6-d337a185b588">
 </p>
 
 <p align="center">
  <em>Fig-1: An overview of the proposed framework. (a) Self-supervised pre-training using diffusion: The U-Net model (encoder-decoder) takes the corrupted version ![x_t](https://latex.codecogs.com/png.latex?\mathbf{x}_t)  of the image $\mathbf{x}_0$ and the corresponding time embedding $t_e$ as the input to predict the noise that takes $\mathbf{x}_0$ to $\mathbf{x}_t$, using the P2 weighted loss. $f(\cdot)$ denotes the function that recovers $\mathbf{x}_{t-1}$ from $\mathbf{x}_t$. (b) Downstream segmentation: The self-supervised pre-trained U-Net is fine-tuned end-to-end in a supervised manner to predict the segmentation masks.</em>
</p>

# Pre-process

# Pretrain
One can run the following script using multi-gpus for generative self-supervision using diffusion. Change the path of the unlabeled images directory on which we need to pretrain on, in the script "./GenSelfDiff/pretrain/utils.py". 
<pre> CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./GenSelfDiff/pretrain/SS_diff.py </pre>

# Downstream train
One can run the following script using multi-gpus for the downstream segmentation task on the pretrained generative self-supervision using diffusion. Change the path of the images, labels, and pre-trained model in the script "./GenSelfDiff/downstream_train/Diff_Seg.py". 
<pre> CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./GenSelfDiff/downstream_train/Diff_Seg.py </pre>

# Evaluation
One can run the following script metric evaluation on the downstream segmentation train model. Change the path of the images, labels, and downstream model in the script "./GenSelfDiff/test/test_script.py". 
<pre> python3 ./GenSelfDiff/test/test_script.py </pre>

# Baselines
The git-hub repositories for some of the baselines are: [CS-CO](https://github.com/easonyang1996/CS-CO/tree/main) | [Wolleb et al.](https://github.com/JuliaWolleb/Diffusion-based-Segmentation/tree/main) | [Baranchuk et al.](https://github.com/yandex-research/ddpm-segmentation/tree/master) | [CIMD](https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models) | [MedSegDiff](https://github.com/MedicineToken/MedSegDiff) | [FCT](https://github.com/Thanos-DB/FullyConvolutionalTransformer)

# Acknowledgment
The code implementation has been adopted from [Diffusion](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=3a159023)
# Citation
