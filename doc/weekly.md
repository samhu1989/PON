### Part Based Single View Shape Reconstruction
##### Project Goal
1. Verify the superiority of part based representation in single view shape reconstruction by proposing a novel neural network for the task.

2. The network should:
    - be designed for cross category objects
    - acheive lower error than holistic baseline

#### preliminary report 20190715
##### <font color=#55ff55>Previous Attempts </font>

Under point cloud representation, Kaichun have tried 4 different approaches for the task ( including structurenet )
- s
- s
- Structurenet have been accepted by SIGGRAPH Aisa. Basing our structure prediction mechanisim on an adapted/extended version of structurenet may make it easier for us to justify our new design.  

##### <font color=#0099ff>Problem / Issue </font>
- In previous part, 

- By adopting part based representation, the challenge is shifted from shape prediction to structure prediction. It may not necessarily be a easier approach.

- The success of our structure prediction would rely on the part annotation data from PartNet.  
=> It makes more sense to utilize multi-level part annotation as in structurenet instead of only certain level part annotation. (only Siyu's opinion for the moment)  
=> However, how to utilize **multi-level part annotation** to train network for **cross category objects** remains a challenge for us.


##### <font color=#ff0099>To Do </font> 
- design a new network as our new starting point. It should:

  - have structure prediction mechanisim that can be trained with multi-level part annotation and for cross category objects.
  
  - the sub-network for shape prediction of each part should be the same ( share memory ).
  
  - use occupancy representation for the output. In such representation, a point inside any part is considered to be inside the objects. The max over part-based occanpancy will be the occanpancy output for the whole object. 

<font color=#007fff>if</font> adopt occupancy representation:
- prepare data from PartNet for the training of Occupancy Network.
- re-implement and retrain Occupancy Network as holistic baseline.
