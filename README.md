# Face-Recognition-with-Transfer-Learning
## HOW TO USE MY PROJECT

### Module1: PERFORM FACE DETECTION:

**Step 1: Download file "haarcascade_frontalface_default.xml"**

- This 'XML' file contains a pre-trained model that was created through extensive training and uploaded by Rainer Lienhart on behalf of Intel in 2000

**Step 2: Upload your image in your current folder**

**Step 3: Run file "face_detection.py"" to perform Face Detection**
```
python face_detection.py
```

![image](https://user-images.githubusercontent.com/120365693/225253014-171c2e04-da74-4af5-88d8-f3960abbc2ba.png)

### Module2: PERFORM FACE RECOGNITION:

**Step 1: Collect face data for each invidual from module 1**

- You can store a folder for each inviduals in Datasets folder. 

**Step 2: Train new dataset with pre-trained VGG-16 model**
```
python train.py
```
- After this, you should have a file '.h5' contains weight of model as the result.
**Step 3: Run file "face_detection.py"" to perform Face Recognition**
```
python demoResult.py
```

![image](https://user-images.githubusercontent.com/120365693/234243694-74cc285c-8f70-4b56-8a0c-c3e817f6846e.png)


