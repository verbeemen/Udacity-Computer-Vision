# Image Captioning:
In this project we have tried to create a model that is capable of describing "things" that are present in an image.  
For this, we have used the respective items: COCO-Dataset, ResNet-50, and a LSTM.  

First of all, we will get the images and their descriptions from the COCO database.
Then we chase the images through ResNet-50. This will try to say which objects are present in the image (encoder). And Finally, we feed that information to a LSTM (RNN), which then tries to form a sentence starting from the input of the encoder.

<img src="https://github.com/verbeemen/Udacity-Computer-Vision/blob/main/Project_2_Image_Captioning/images/encoder-decoder.png" width="900px"/>
  
  
# A guiding paper:
**Url**: https://arxiv.org/pdf/1411.4555.pdf
**Title**: Show and Tell: A Neural Image Caption Generator:

# Result:
![Result](https://github.com/verbeemen/Udacity-Computer-Vision/blob/main/Project_2_Image_Captioning/images/result.png "Result of my cat.")


### Weights & Biases Logs:
Weights and biases is a developer tool for: 
 - Machine learning
 - Experiment tracking
 - Hyperparameter optimization
 - Model and dataset versioning  
   
If you are really interested in the exact parameters of my model for this project... then you can find them in the link below.  
https://wandb.ai/verbeemen/udacity_computer_vision-project_2?workspace=user-verbeemen  
  
<img src="https://github.com/verbeemen/Udacity-Computer-Vision/blob/main/Project_2_Image_Captioning/images/image_captioning_loss.png"  width="640px"/>

