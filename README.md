# On-Device Training Mobile App

This Android application demonstrates how **deep neural networks** can be trained **on mobile devices** for the task of **image classification**.

The app was developed as part of my **diploma thesis** at the National Technical University of Athens (NTUA), focusing on the feasibility, performance, and limitations of on-device training compared to conventional server-based training methods. The full thesis is available [here](https://dspace.lib.ntua.gr/xmlui/bitstream/handle/123456789/61628/pervolrakis_dimitrios_DT.pdf?sequence=1&isAllowed=y).

---

## Objective

The primary objective of this application is to study the process of continuous learning and fine-tuning deep learning models on mobile hardware. By leveraging mobile-optimized ML frameworks, the app supports:

- Pretrained model integration
- On-device fine-tuning with real-world user data
- Real-time classification
- Evaluation of training time, accuracy, and energy efficiency

---

## Features

-  Load pretrained models trained on external datasets
- Fine-tune models directly on-device for custom categories
- Capture images via the device camera for training and inference
- Compare on-device training results with server-based training
- Store training history and classification accuracy

---

## Technologies Used

- **Android SDK** (Java/Kotlin)
- **TensorFlow Lite** or **PyTorch Mobile**
- **CameraX / Camera2** for real-time image capture
- **TFLite Model Personalization**

---


