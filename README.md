# SuperPoint-Pytorch (A Pure Pytorch Implementation)
SuperPoint: Self-Supervised Interest Point Detection and Description  


# Thanks  
This work is based on:  
- Tensorflow implementation by [Rémi Pautrat and Paul-Edouard Sarlin](https://github.com/rpautrat/SuperPoint)  
- Official [SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork). 
- [Kornia](https://kornia.github.io/)  

# Existing Problems
1. Different performances in training and evaluation modes, which is 
a famous problem have been discussed in 
[Performance highly degraded when eval() is activated in the test phase](https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323)


# New Update (09/04/2021)
* Convert tf pretrained weight to Pytorch   
* Usage:
    - 1 Construct network by [superpoint_bn.py](model/superpoint_bn.py) (Refer to [train.py](./train.py) for more details)
    - 2 Set parameter eps=1e-3 for all the BatchNormalization functions in model/modules/cnn/*.py
    - 3 Set parameter momentum=0.01 (**not tested**)
    - 4 Load pretrained weight [superpoint_bn.pth](./superpoint_bn.pth) and run forward propagation
 
 
# Usage
* 1 Prepare your data. Make directories *data* and *export*. The data directory should look like,
    ```
    data
    |-- coco
    |  |-- train2017
    |  |     |-- a.jpg
    |  |     |-- ...
    |  --- test2017
    |        |-- b.jpg
    |        |-- ...
    |-- hpatches
    |   |-- i_ajuntament
    |   |   |--1.ppm
    |   |   |--...
    |   |   |--H_1_2
    |   |-- ...
    ```
    You can create *soft links* if you already have *coco, hpatches* data sets, the commands are,
    ```
    cd data
    ln -s dir_to_coco ./coco
    ```
* 2 The training steps are much similar to [rpautrat/Superpoint](https://github.com/rpautrat/SuperPoint). 
    **However we strongly suggest you read the scripts first so that you can give correct settings for your envs.**
    - 2.0 Modify save model conditions in train.py, line 61  
          `if (i%118300==0 and i!=0) or (i+1)==len(dataloader['train']):`  
          and set proper epoch in _*.yaml_.
    - 2.1 Train MagicPoint (>1 hours):  
          `python train.py ./config/magic_point_train.yaml`   
          (Note that you have to delete the directory _./data/synthetic_shapes_ 
          whenever you want to regenerate it)
    - 2.2 Export coco labels (>40 hours):   
          `python homo_export_labels.py #using your data dirs`
    - 2.3 Train MagicPoint on coco labels data set (exported by step 2.2)       
          `python train.py ./config/magic_point_coco_train.py #with correct data dirs` 
    - 2.4 Train SuperPoint following the steps in **Training Steps** (>12 hours)    
          `python train.py ./config/superpoint_train.py #with correct data dirs`  
    - others. Validate detection repeatability or description
              (Better in training mode if you also have eval problem stated in **Existing Problems**)  
                   
        ```
        python export_detections_repeatability.py #(very fast)  
        python compute_repeatability.py  #(very fast)
        ## or
        python export_descriptors.py #(> 5.5 hours) 
        python compute_desc_eval.py #(> 1.5 hours)
        ```   
    **AGAIN: You have to edit _.yaml_ files to run corresponding tasks,
     especially for the _path_ or _dir_ items** 
    ```
    model
        name: superpoint # magicpoint
     ...
    data:
        name: coco #synthetic
        image_train_path: ['./data/mp_coco_v2/images/train2017',] #several data sets can be list here
        label_train_path: ['./data/mp_coco_v2/labels/train2017/',]
        image_test_path: './data/mp_coco_v2/images/test2017/'
        label_test_path: './data/mp_coco_v2/labels/test2017/'
    ```

