# GenSelfDiff-HIS: Generative Self-Supervision Using Diffusion for Histopathological Image Segmentation (TMI 2024)
This is the official repository for the implementation of (paper link)

# Abstract
 Histopathological image segmentation is a laborious and time-intensive task, often requiring analysis from experienced pathologists for accurate examinations. To reduce this burden, supervised machine-learning approaches have been adopted using large-scale annotated datasets for histopathological image analysis. However, in several scenarios, the availability of large-scale annotated data is a bottleneck while training such models. Self-supervised learning (SSL) is an alternative paradigm that provides some respite by constructing models utilizing only the unannotated data which is often abundant. The basic idea of SSL is to train a network to perform one or many pseudo or pretext tasks on unannotated data and use it subsequently as the basis for a variety of downstream tasks. It is seen that the success of SSL depends critically on the considered pretext task. While there have been many efforts in designing pretext tasks for classification problems, there have not been many attempts on SSL for histopathological image segmentation. Motivated by this, we propose an SSL approach for segmenting histopathological images via generative diffusion models. Our method is based on the observation that diffusion models effectively solve an image-to-image translation task akin to a segmentation task. Hence, we propose generative diffusion as the pretext task for histopathological image segmentation. We also utilize a multi-loss function-based fine-tuning for the downstream task. We validate our method using several metrics on two publicly available datasets along with a newly proposed head and neck (HN) cancer dataset containing Hematoxylin and Eosin (H&E) stained images along with annotations.

# Methodology
<p align="center">
<img width="578" alt="diff_model" src="https://github.com/PurmaVishnuVardhanReddy/GenSelfDiff-HIS/assets/103281951/c6d68807-bafe-4589-aef6-d337a185b588">
 </p>
 
 <p align="center">
  <em>Fig-1: An overview of the proposed framework. (a) Self-supervised pre-training using diffusion: The U-Net model (encoder-decoder) takes the corrupted version $\mathbf{x}_t$ of the image $\mathbf{x}_0$ and the corresponding time embedding $t_e$ as the input to predict the noise that takes $\mathbf{x}_0$ to $\mathbf{x}_t$, using the P2 weighted loss. $f(\cdot)$ denotes the function that recovers $\mathbf{x}_{t-1}$ from $\mathbf{x}_t$. (b) Downstream segmentation: The self-supervised pre-trained U-Net is fine-tuned end-to-end in a supervised manner to predict the segmentation masks.</em>
</p>

# Acknowledgment

# Citation
