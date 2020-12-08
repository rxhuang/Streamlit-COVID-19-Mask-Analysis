## Design review

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
