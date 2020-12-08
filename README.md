# CMU Interactive Data Science Final Project

* **Online URL**: https://share.streamlit.io/cmu-ids-2020/fp-null-pointer-1/main
* **Team members**:
  * Contact person: Tianyang Zhan tzhan@andrew.cmu.edu
  * Ruoxin Huang ruoxinh@andrew.cmu.edu
  * Amanda Xu xinyix2@andrew.cmu.edu
  * Ziming He zimingh@andrew.cmu.edu
* **Track**: Model (one of Narrative, Model, or Interactive Visualization/Application)

## Work distribution

Update towards the end of the project.

## Deliverables

### Proposal

https://docs.google.com/document/d/1TqYg-uk4ryZCuW5axk1SdqnfntXukpZItl3BxQD2i6s/edit?usp=sharing

### Design review

**Visualization:**
* Current Progress:
  * Scatter plot and map plot to demonstrate the effectiveness of wearing masks.
* Next Step:
  * Explore other mask related datasets and potentially add more visualizations.
* Questions:
  * Is this level of visualization and interaction sufficient? 
  * What is the expected level of visualization and interaction?
  * Any suggested additions?

**Model:**
* Current Progress:
  * Combined two pre-trained models for face and face mask detection:
  Face: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
  Mask:https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2
  * Completed the pipeline for face mask detection and input collection for the statistical evaluation model
* Next Step:
  * Design and complete the statistical model for safety-level scoring
  * Try to train our own CV models and explore different datasets. In particular, explore non-artificial mask datasets like https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
* Questions:
  * Can we use the downloaded pre-trained models or do we need to reproduce the code and train our own models?
  * Is this level of interaction sufficient?



### Final deliverables

- [ ] All code for the project should be in the repo.
- [ ] A 5 minute video demonstration.
- [ ] Update Readme according to Canvas instructions.
- [ ] A detailed project report. The contact should submit the video and report as a PDF on Canvas.
