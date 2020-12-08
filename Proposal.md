# Final Project Proposal

**Project URL**: https://share.streamlit.io/cmu-ids-2020/fp-null-pointer-1/main
* **Team members**:
  * Contact person: Tianyang Zhan tzhan@andrew.cmu.edu
  * Ruoxin Huang ruoxinh@andrew.cmu.edu
  * Amanda Xu xinyix2@andrew.cmu.edu
  * Ziming He zimingh@andrew.cmu.edu
* **Track**: Model

## Proposal

In the face of the COVID-19 pandemic, it has been shown that wearing masks is an effective way to prevent the spread of the disease. CDC calls people to wear masks, and masks are required in public transportation and restaurants. However, relying on manpower to check whether all people are wearing masks is cumbersome and unrealistic. Thus, it becomes an important task to automatically check whether people are wearing masks under different scenarios. Our project provides a solution by analyzing whether people in a scenario are wearing masks, and providing a score to evaluate the safety level of the scene.

The question that we are trying to explore is: Is it safe for me to be in a certain place without a high risk of getting COVID-19? To be able to access the safety level of a real-world scenario, we need to extract information from the given scene and measure the likelihood of getting COVID-19 in this scene. We hypothesize that the most relevant variables to our task are the number of people in a scene, the distance between each person, and whether they are wearing face masks. Based on this assumption, we choose to address this problem with a computer vision based solution. Given the limited time for developing this project, we also plan to narrow down the scope from real-time image analysis to single-image analysis.

First of all, we want to demonstrate the importance of wearing masks. To achieve that, we are going to produce interactive visualizations such as histograms and correlation charts that illustrate the effect of wearing masks versus not wearing one using epidemiology datasets. Then we are going to implement a face mask recognition pipeline using some of the states of the art computer vision techniques such as deep convolutional neural networks. Given the time limit of the project, we separate the pipeline into two individual models. Having separate modules instead of an end-to-end pipeline can boost the performance of the system. The first model uses YOLOv3 architecture trained on WIDER FACE: A Face Detection Benchmark to detect faces in the image. The second model uses MobileNet (a lightweight convolutional neural network) architecture with pre-trained ImageNet weights and fine-tuned for face-mask detection to recognize masks on faces. The pipelines will be implemented through OpenCV and Keras/Tensorflow. Given the number of faces and the percentage of people wearing masks in the picture, we will give a safety score based on a probabilistic model to rate the chance of contracting COVID-19.

Given an input image that represents a real-world scenario, our application can identify the safety level of the situation by identifying whether people are wearing masks in the image. With the circumstances around the globe, this is a very useful feature that can potentially help prevent the constant spread of COVID-19. In future work, our model can be extended to perform real-time video analysis instead of only single-image analysis. In that case, it can be applied to security cameras to constantly assess the safety level at a given location, and can alert if the safety level falls below a certain threshold. 
 
## References:
* Howard, Andrew G., Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).
* Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
* Yang, Shuo, Ping Luo, Chen-Change Loy, and Xiaoou Tang. "Wider face: A face detection benchmark." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5525-5533. 2016.
 
 
 
 
 
 
 
 
 
 
 

