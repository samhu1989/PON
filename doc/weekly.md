### Part Based Single View Shape Reconstruction
##### Project Goal
1. Verify the superiority of part based representation in shape reconstruction by proposing a novel neural network for the task.

2. The network should:
   
    - do reconstruction in few shot setting 
    



### report 20191009：

##### <font color=#55ff55>Done  </font>

on open set arrangement learning

I implemented a rule based box placement to generate toy data for exp

![CycleAtlasNet](./img/toy.png "")

 the underline rules:

1. there are five different kind of boxes in the world.

2. box with smaller base must place on top of one with larger base

3. otherwise place box on the ground without interaction

4. the cube never appears in the training data

   

   currently only two boxes are chosen

   

On Part generation 

I fixed some bugs and understand more of the data from Kai Chun.

there will be new results on part generation tomorrow on our project meeting.



<font color=#ff0099>To Do </font>

Implementing a conditional GAN to generate scale and arrangement to place the boxes. 

to find out what kind of knowledge are easily generalized and what are not.



### meeting notes 20191004:

**motivation and priority** 

We believe that to accomplish open-set single view reconstruction, the shape regression neural networks, no matter with what kind of shape representation, are all dead ends. 

A more promising approach is to mimic human behavior to construct objects in three steps: 

a) parse the object into parts from visual input. (part segmentation)

b) forge the primitive material into object parts, one by one. (part generation)

c) conditioned on the visual input, assemble the parts into reasonable configuration. (assemble)



<font color=#e26f46>Now, our priority is focused on step c):</font>



For step c),  we employ the framework of generative adversarial network and propose an

Assembler-VAE-GAN.

For Assembler - VAE - GAN, 

We first train a VAE to encode part assembling image into vector z.

Then we train GAN for assembling

The **generator** is conditioned on two self centered parts (as two point sets), it output two 3D transformations, which are used to transform the parts to be attached. 

random input vector z is sampled based on input image or randomly draw from distribution.

one **discriminator** takes two transformed parts as input and output whether they are  in reasonable configuration.

another  **discriminator**  takes the image of part assembling and the merged point cloud as input to tell whether they are matched.

The <font color=#55ff55>**advantage**</font> of such design is that inference can function with or without the visual input, hence have the  potential to complete the invisible part based purely on the distribution of reasonable configuration.



### report 20191004

##### <font color=#55ff55>Done  </font>

some ugly results

http://171.67.77.236:8082/bv/data~res~_pgen~_000_as

current cd 0.011

should reach somewhere about 0.006 ( or even smaller considering we are doing only the parts? ) 

![CycleAtlasNet](./img/net.png "")

##### <font color=#ff0099>To Do </font>

how to make our method iteration  faster ( smaller dataset doesn't work )

data preparation

1. the view - gt alignment issue

   

#### meeting 20190927 ( kaichun, tiange & siyu  ) 

##### <font color=#ff0099>To Do </font>

complete first version of step 2 and step 3.

part construction.

place shape according to mask ray.



#### meeting 20190826 ( hao, kaichun & siyu ) 

##### <font color=#55ff55>Done </font>

preliminary experiment on part-based shape augmentation:

##### <font color=#ff0099>To Do </font>

Hao's proposal

![zeroshot](./img/zeroshot.png "")



merge-net part-seg

open-context

shape - > super-pixel  -> predict merge score  

image -> part shape (limited visible area)

image -> merge-net -> part  seg -> part shape -> part arrangement -> shape completion

arrangement and shape completion 

shape refine-net by re-projection and compare to input image.



##### Report

The experiment is conducted as follows :

I picked  Chair, Storage Furniture, Table from PartNet 

I choose 1024 shapes from each category

I used parts from Chair and  Storage Furniture to randomly assemble 2048 shapes as a new category "Augment":

( more samples can be view at http://171.67.77.236:8082/_pon_aug )![holisticfewshot](./img/augment.png "")

I trained three different AtlasNet to do single view reconstruction:



NoAug:  trained with data from Chair and  Storage Furniture.

OnlyAug: trained with data from "Augment".

All: train with data from "Augment", Chair, Storage Furniture



These models are all validated with data from Table

Chamfer Distance  after 1024 epochs:

| CD    | NoAug        | OnlyAug  | All          |
| ----- | ------------ | -------- | ------------ |
| train | **0.005520** | 0.008278 | 0.005855     |
| val   | 0.035083     | 0.027681 | **0.020118** |

visual result: (go to the link for more visual results)

NoAug:  http://171.67.77.236:8082/_pon_atlas_noaug

![holisticfewshot](./img/atlas_noaug.png "")

OnlyAug:  http://171.67.77.236:8082/_pon_atlas_onlyaug

![holisticfewshot](./img/atlas_onlyaug.png "")

All:  http://171.67.77.236:8082/_pon_atlas_all

![holisticfewshot](./img/atlas_all.png "")



#### preliminary report 20190809

##### <font color=#55ff55>Done </font>
correctly run the test code of  [Learning to Reconstruct Shapes from Unseen Classes][5]

##### Report 

![holisticfewshot](./img/genre_our_res.png "")




#### meeting 20190806 ( siyu & kaichun ) 

an important zero shot holistic baseline:  [Learning to Reconstruct Shapes from Unseen Classes][5]

tips for the few shot settings:
learn assembly from prototype
learn part generation / possible part relation from known category
    
#### preliminary report 20190805

A **proposal** as holistic baseline for few shot shape completion

The main references: (click the links to paper)
The few shot learning loss come from [Prototypical Networks for Few-shot Learning][4]

![holisticfewshot](./img/holistic_few_shot.png "")

#### meeting 20190802 ( all )

What do we want to do ?

- [ ] reconstruct 3D shape from single image   (using part prior to assist it)
- [ ] recover object parts from single image for other applications like robot interaction planning (using reconstruction loss as regularization to assist it) 
- [x] few shot learning

#### preliminary report 20190719
##### <font color=#55ff55>Previous Attempts </font>

Under point cloud representation, Kaichun have tried 4 different approaches for the task ( including structurenet )
- When the generation of different part was assigned and trained by hungarian matching,  branches/tree nodes would compete to predict same part (generate duplicate parts), especailly if the point number for each part are fixed. The network reprensentation power maybe wasted in such case. This may even lead to the missing of other parts.

- In structurenet, the position shift of predicted parts may cause a large error in quantitative evaluation

##### <font color=#0099ff>Problem / Issue </font>  
- By adopting part based representation, the challenge is shifted from shape prediction to structure prediction. It may not necessarily be a easier approach. Holistic approaches are allowed to learn arbitrary structure inside the space of the structures that they can implicitly represent as long as the loss is minimized. We attempt to learn semantic meaningful structure based on the PartNet annotation. Therefore, our part based approach may be more difficult to reach lower prediction error due to the extra constraint.

- To predict semantic meaningful structure, we would rely on the part annotation data from PartNet.  
=> It makes more sense to utilize **multi-level part annotation** as in structurenet instead of only certain level part annotation.   
=> However, how to utilize **multi-level part annotation** to train network for **cross category objects** remains a challenge for us.


##### <font color=#ff0099>To Do </font> 
- to decide a new design of network as our new starting point. It should:

  - have structure prediction mechanism that can be trained with multi-level part annotation and for cross category objects.
  

<font color=#007fff>if</font> adopt occupancy representation:
- prepare data from PartNet for the training of Occupancy Network.
- port code and retrain Occupancy Network on our data as holistic baseline.

##### Report
A **proposal** by siyu is as follows:

The main references: (click the links to paper)
- The backbone network comes from [Occupancy Networks][1]
- The Gumbel Subset Sampling module is learned from [PAT][2]
- The part loss is inspired by [Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding][3]

A hypothesis behind the design regarding the key issue and the role of prior knowledge on object part:

<font color=#4169E1> What is the key issue in previous holistic approaches ? </font>

unreasonable and inconsistent correlation / association / co-dependency is implicitly established inside neural networks.

For example, the shape variation of part is significantly smaller that shape variation of object, however, the networks can't automatically exploit such characteristic, because unnecessarily strong association may be established between different object part.  This will lead to unnecessary averaging effect.

<font color=#4169E1>What are the possible causes of this issue ?</font> (we should avoid)

- network design:

  - the shape generation of each part depends only on single global shape descriptor

  - the assumption of  spherical homeomorphism

- training method & loss function:

  - using Hungarian matching to assign network branches for each part
  - Chamfer distance loss (correspondence based on nearest neighbor)

<font color=#4169E1>Why prior knowledge of object part is the cure for this issue ?</font> (we should utilize)

- it can be used as GT to guide the network to learn an internal hierarchical part-by-part associations with proper and consistent strength

The network structure is as follows:

![PON](./img/pon.png "")

The part loss function:  
$$
L_{part}=L_{pull} + L_{push}
$$

$$
L_{pull} = \frac{1}{C_{part}}\sum\frac{1}{P_c}\sum max(||\mu_c - x_p||-\delta_v,0)
$$

$$
L_{push} = \frac{1}{C_{part}}\frac{1}{C_{part}}\sum\sum max(\delta_d - ||\mu_{cA} - \mu_{cB}||,0)
$$


The part loss aims at making the part feature similar to the mean feature for the points inside same part and making the mean part feature between different parts distinguishable from each other. The points outside the shape will be discarded.

In order to utilize multi-level part annotation, we split part feature into several group (divide channel into several group) and apply part loss on each group. If two points are in the same part across all level of part annotations, the entire part features should be close. If two points are only in the same part in top level part annotation, then only a fraction of their part features are close.

An illustration for expected effect of multi-level part loss is shown as follows:

![partloss](./img/expected_part_loss.png "")



[1]:https://arxiv.org/pdf/1812.03828.pdf "Occupancy Networks: Learning 3D Reconstruction in Function Space"

[2]:https://arxiv.org/pdf/1904.03375.pdf "Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling"

[3]:https://arxiv.org/pdf/1902.09777.pdf "Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding"

[4]:https://arxiv.org/pdf/1703.05175.pdf "Prototypical Networks for Few-shot Learning"

[5]:http://genre.csail.mit.edu/papers/genre_nips.pdf "Learning to Reconstruct Shapes from Unseen Classes"
