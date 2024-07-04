
# Driver Alertness based on r-PPG from Facial Videos using Deep Learning

The goal of this project is to develop a robust method using Deep Neural Networks (DNN) to accurately estimate remote Photoplethysmography (rPPG) signals from facial video, particularly in the presence of motion artifacts. rPPG is a non-invasive technique that calculates photoplethysmographic (PPG) signals from video recordings of a person's face, allowing for continuous monitoring of physiological parameters such as heart rate. However, motion artifacts significantly distort the accuracy of rPPG signals, presenting a major challenge in real-world applications.

The primary focus of this algorithm is to effectively remove these motion artifacts and enhance the reliability of rPPG measurements. By leveraging the advanced capabilities of DNNs, the algorithm will be trained to distinguish and filter out noise caused by movements, thus providing cleaner and more accurate rPPG signals.

This technology is particularly critical for applications in driver state-of-mind detection to enhance safety. Monitoring the physiological state of drivers can help in assessing their alertness and stress levels, potentially preventing accidents caused by fatigue or inattention. The project aims to create a solution that operates in real-time, offering reliable health monitoring and contributing to safer driving environments.


## Challenges in Building the Model

Challenges in Building the Model:

1. **Motion Variability**: Diverse facial movements and expressions introduce significant noise.
2. **Angle Dependency**: Variations in head pose affect the consistency of rPPG signal capture.
3. **Illumination Changes**: Fluctuations in lighting conditions, from dark to bright and vice-versa, impact signal accuracy.
4. **Limited Annotated Data**: Difficulty in collecting extensive, synchronized annotated datasets of PPG signals with facial videos.

## Proposed Solutions

### 1. PFE and TFA

To address the limitations of motion variability and angle dependency, we propose the use of two innovative techniques: **Physiological Signal Feature Extraction (PFE)** and **Temporal Face Alignment (TFA)**.

- **Physiological Signal Feature Extraction (PFE)**: This block is designed to alleviate the degradation caused by varying distances between the camera and the subject.
- **Temporal Face Alignment (TFA)**: This block aims to mitigate the degradation caused by head motion. TFA aligns the facial features temporally across video frames, compensating for movements and maintaining the accuracy of the rPPG signals.

### 2. Self-Supervised Learning via Contrastive Learning
To tackle the challenge of limited annotated data, we introduce a **Self-Supervised Learning** approach using **Contrastive Learning**. This method leverages the abundant unlabeled data by training the model to distinguish between similar and dissimilar video segments, thereby learning robust representations that can be fine-tuned with minimal annotated data.

### 3. Dual GAN
Another solution to the annotated data limitation is the use of **Dual Generative Adversarial Networks (Dual GAN)**. This architecture not only models the Blood Volume Pulse (BVP) predictor but also explicitly models the noise distribution through adversarial learning. By incorporating two Generative Adversarial Networks (GANs), this method effectively enhances the robustness of BVP representation against unseen noise.

By combining these innovative techniques, our model is designed to effectively handle motion variability, angle dependency, and data scarcity, ensuring robust and accurate rPPG signal estimation for driver state-of-mind detection and other applications.

## Dataset

We used the following Datasets to train our models:

1) **UBFC** : The UBFC-RPPG database was created with a simple low cost webcam (Logitech C920 HD Pro) at 30fps with a resolution of 640x480 in uncompressed 8-bit RGB format. A CMS50E transmissive pulse oximeter was used to obtain the ground truth PPG data comprising the PPG waveform as well as the PPG heart rates.
During the recording, the subject sits in front of the camera (about 1m away from the camera) with his/her face visible. All experiments are conducted indoors with a varying amount of sunlight and indoor illumination. It contain a total of 42 subjects.


2) **PURE** : This data set consists of 10 persons performing different, controlled head motions in front of a camera. During these sentences the image sequences of the head as well as reference pulse measurements were recorded.
The 10 persons (8 male, 2 female) that were recorded in 6 different setups resulting in a total number of 60 sequences of 1 minute each. The videos were captured with a eco274CVGE camera by SVS-Vistek GmbH at a frame rate of 30 Hz with a cropped resolution of 640x480 pixels and a 4.8mm lens. Reference data have been captured in parallel using a finger clip pulse oximeter (pulox CMS50E) that delivers pulse rate wave and SpO2 readings with a sampling rate of 60 Hz.

    The test subjects were placed in front of the camera with an average distance of 1.1 meters.  Lighting condition was daylight trough a large window frontal to the face with clouds changing llumination conditions slightly over time.

## USAGE AND TRAINING 

Each model are in seprate folder and with their respective readme file. Go through those file for exection of each model seperately.

## Future Work

- Improving the accuracy of the models 
- Extracting the physiological features from the rPPG signal
- Minimizing the size of model so it can be deployed on small processing devices

## Acknowledgements

- “Learning Motion-Robust Remote Photoplethysmography through Arbitrary Resolution Videos”- The Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI-23)

-  “Facial Video-Based Remote Physiological Measurement via Self-Supervised Learning” -  IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 45, NO. 11, NOVEMBER 2023  
- “Dual-GAN: Joint BVP and Noise Modeling for Remote Physiological Measurement” - 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

