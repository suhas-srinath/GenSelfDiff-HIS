# GenSelfDiff-HIS: Generative Self-Supervision Using Diffusion for Histopathological Image Segmentation (TMI 2024)
This is the official repository for the implementation of [GenSelfDiff-HIS](https://ieeexplore.ieee.org/abstract/document/10663482) by Vishnuvardhan Purma, Suhas Srinath, Seshan Srirangarajan, Aanchal Kakkar, and Prathosh AP.

# Abstract
 Histopathological image segmentation is a laborious and time-intensive task, often requiring analysis from experienced pathologists for accurate examinations. To reduce this burden, supervised machine-learning approaches have been adopted using large-scale annotated datasets for histopathological image analysis. However, in several scenarios, the availability of large-scale annotated data is a bottleneck while training such models. Self-supervised learning (SSL) is an alternative paradigm that provides some respite by constructing models utilizing only the unannotated data which is often abundant. The basic idea of SSL is to train a network to perform one or many pseudo or pretext tasks on unannotated data and use it subsequently as the basis for a variety of downstream tasks. It is seen that the success of SSL depends critically on the considered pretext task. While there have been many efforts in designing pretext tasks for classification problems, there have not been many attempts on SSL for histopathological image segmentation. Motivated by this, we propose an SSL approach for segmenting histopathological images via generative diffusion models. Our method is based on the observation that diffusion models effectively solve an image-to-image translation task akin to a segmentation task. Hence, we propose generative diffusion as the pretext task for histopathological image segmentation. We also utilize a multi-loss function-based fine-tuning for the downstream task. We validate our method using several metrics on two publicly available datasets along with a newly proposed head and neck (HN) cancer dataset containing Hematoxylin and Eosin (H&E) stained images along with annotations.

# Methodology
<p align="center">
<img width="578" alt="diff_model" src="https://github.com/PurmaVishnuVardhanReddy/GenSelfDiff-HIS/assets/103281951/c6d68807-bafe-4589-aef6-d337a185b588">
 </p>
 
 <p align="center">
  <em>Fig-1: An overview of the proposed framework.</em>
</p>

# Data Pre-processing
The public datasets can be downloaded from [MoNuSeg](https://monuseg.grand-challenge.org/Data/) and [GlaS](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation). These datasets should be placed into the appropriate folders within the pre_process directory for data pre-processing. Our proposed HN dataset will be made available for research purposes in the future.

# Pre-training
One can run the following script using multi-gpus for generative self-supervision using diffusion. Specify the path to the directory of the unlabeled images on which pre-training needs to be done in the script "GenSelfDiff/pretrain/utils.py". 
<pre> CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./GenSelfDiff/pretrain/SS_diff.py </pre>

# Downstream Training for Segmentation
One can run the following script using multi-gpus for the downstream segmentation task on the pretrained generative self-supervision using diffusion. Change the path of the images, labels, and pre-trained model in the script "./GenSelfDiff/downstream_train/Diff_Seg.py". 
<pre> CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./GenSelfDiff/downstream_train/Diff_Seg.py </pre>

# Evaluation
One can run the following script for evaluation on the trained segmentation model. Specify the path of the images, labels, and the model in the script "./GenSelfDiff/test/test_script.py". 
<pre> python3 ./GenSelfDiff/test/test_script.py </pre>

# Baselines
The git-hub repositories for some of the baselines are: [CS-CO](https://github.com/easonyang1996/CS-CO/tree/main) | [Wolleb et al.](https://github.com/JuliaWolleb/Diffusion-based-Segmentation/tree/main) | [Baranchuk et al.](https://github.com/yandex-research/ddpm-segmentation/tree/master) | [CIMD](https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models) | [MedSegDiff](https://github.com/MedicineToken/MedSegDiff) | [FCT](https://github.com/Thanos-DB/FullyConvolutionalTransformer)

# Acknowledgment
The code implementation for the Attention-based UNet architecture has been adopted from [Diffusion](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=3a159023).
# Citation
<pre>
@ARTICLE{10663482,
  author={Purma, Vishnuvardhan and Srinath, Suhas and Srirangarajan, Seshan and Kakkar, Aanchal and Prathosh, A.P.},
  journal={IEEE Transactions on Medical Imaging}, 
  title={GenSelfDiff-HIS: Generative Self-Supervision Using Diffusion for Histopathological Image Segmentation}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Task analysis;Image segmentation;Diffusion models;Biomedical imaging;Annotations;Image analysis;Histopathology;Diffusion;H&E-stained Histopathological Images;Representation Learning;Self-Supervised Learning},
  doi={10.1109/TMI.2024.3453492}}
</pre>
