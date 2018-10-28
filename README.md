# Traffic Sign Classification

I trained a deep learning model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

See writeup in Project2trafficsigndetection.pdf for full detail.

## Model

### 1st Approach: LeNet (91.8%)

I started off by directly applying LeNet. In the LeNet architecture, the 32x 32 images pass through a convolutional layer with 6 5x5 filters. The output is 28x28x6. After max pooling, the output dimension is 14x14x6. In a similar manner, it goes through a second convolutional iteration with 16 filters. The output after max-pooling is 5x5x16. We flatten this output to 400. Then it undergoes several fully connected layer, condensing to 120 neurons, then 84 neurons, and lastly 43 neurons with one corresponding to each output class.
It goes through 10 epochs of training, with a learning rate of 0.001. It uses the AdamOptimizer. This gets an accuracy of ~89%.

From reading the training log, the validation set still kept increasing at epoch 10. This led me to believe that the model is underfitting. Therefore, I extended to 20 epochs. This leads to an accuracy of 91.8%. The training accuracy is saturating at epoch 20.

To gain more insights, I obtained the precision and recall breakdown by each class. Some of the classes did particularly bad:
0 Speed limit (20km/h) - precision: 0.4 recall 0.07
24 Road narrows on the right - precision: 0.86 recall 0.2
27 Pedestrians - precision: 1.0 recall 0.5
32 End of all speed and passing limits - precision: 0.35 recall 0.27
Most of these classes have relatively few training data.


### 2nd approach: dropout (94.7%)
I applied the dropout technique on the two fully connected layers, with a drop out rate of 0.5. This technique is highly effective: it approves the accuracy from 91.8% to 94.7%.
In particular, it drastically improved the precision and recall of the classes that did the worst in the previous approach:

26 Traffic signals
BEFORE - precision: 0.68 recall 0.53 AFTER - precision: 0.79 recall 0.97
32 End of all speed and passing limits BEFORE - precision: 0.35 recall 0.27 AFTER - precision: 0.9 recall 0.9

However, even with dropout, the model is still doing terribly in the following classes:
0 Speed limit (20km/h) - precision: 0.5 recall 0.03
24 Road narrows on the right - precision: 0.83 recall 0.17

Let’s also take a look at what wrong labels the model predicts these classes to be. The “Speed limit (20 km/h)” is most likely to be predicted as 70 km/h. The two numbers look similar in shape, and the latter class has 10x more training data.
 The “road narrow to the right” sign is most likely to be predicted as “General caution”, followed by traffic signals.

It seems like the network is capturing the contour of traffic signs (triangular signs vs. circular signs) and rough shape of what’s on the signs well, but it is missing out some details.

### 3rd approach: bigger neural network (96.9%)
Given that the model is missing out some details, I decided to increase the number of filters. Separately, with drop out, the last hidden layer now is effectively 42 in size, and it connects to 43 classes. It worked for MINST dataset which predicts 10 classes, but does not quite make sense here. Thus, I decided to increase the network size for the fully the connected layers as well.

In the first convolutional layer, I increased the filter from 6 to 24. In the second consolutionary layer, I increased the filters to 72. The flattened output thus is 5x5x72 = 1800. Going through 3 fully connected layers, it condensed down to 900, 300, and eventually 43 neurons.

Now the accuracy improves to 96.9%.

In particular, it improves on the two classes it did worse at in the previous run:
0 Speed limit (20km/h)
BEFORE - precision: 0.5 recall 0.03 AFTER - precision: 1.0 recall 0.8
24 Road narrows on the right BEFORE - precision: 0.83 recall 0.17 AFTER - precision: 0.95 recall 0.63

### 4th approach: augment dataset (96-97%)
The performances on these small classes are still lackluster. Therefore I decided to augment the data set for these small classes. I did this by randomly rotating an angle between -10 and 10 degree, then randomly translating the x and y direction between 0 and 3 pixels, randomly zooming in on the images between 1x and 1.3x, and lastly adding some noise.
Shown below, the leftmost image in each row is the original image from the data set. The subsequent 5 images are the generated samples.

This increases the data set by 6000 data points. The accuracy is 97.0%. It is questionable whether it actually improved from the last run whose accuracy is 96.9%.
As a next step, now that I have a balanced dataset, I augmented the data for each sample in this fashion, and increased the dataset size in total by 3 times. The accuracy still didn’t improve (highest during training is 96.9% at 15th epoch.)
The final test accuracy is 94.9%.

