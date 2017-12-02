
# coding: utf-8

# In[2]:


import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def loadData(Path):
    
    lines = []
    with open(Path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines


def balance_data(samples, visulization_flag ,N=60, K=1,  bins=100):

    angles = []
    for line in samples:
        angles.append(float(line[3]))

    n, bins, patches = plt.hist(angles, bins=bins, color= 'orange', linewidth=0.1)
    angles = np.array(angles)
    n = np.array(n)

    idx = n.argsort()[-K:][::-1]    # find the largest K bins
    del_ind = []                    # collect the index which will be removed from the data
    for i in range(K):
        if n[idx[i]] > N:
            ind = np.where((bins[idx[i]]<=angles) & (angles<bins[idx[i]+1]))
            ind = np.ravel(ind)
            np.random.shuffle(ind)
            del_ind.extend(ind[:len(ind)-N])

    balanced_samples = [v for i, v in enumerate(samples) if i not in del_ind]
    balanced_angles = np.delete(angles,del_ind)

    plt.subplot(1,2,2)
    plt.hist(balanced_angles, bins=bins, color= 'orange', linewidth=0.1)
    plt.title('modified histogram', fontsize=20)
    plt.xlabel('steering angle', fontsize=20)
    plt.ylabel('counts', fontsize=20)

    if visulization_flag:
        plt.figure
        plt.subplot(1,2,1)
        n, bins, patches = plt.hist(angles, bins=bins, color='orange', linewidth=0.1)
        plt.title('origin histogram', fontsize=20)
        plt.xlabel('steering angle', fontsize=20)
        plt.ylabel('counts', fontsize=20)
        plt.show()

        plt.figure
        aa = np.append(balanced_angles, -balanced_angles)
        bb = np.append(aa, aa)
        plt.hist(bb, bins=bins, color='orange', linewidth=0.1)
        plt.title('final histogram', fontsize=20)
        plt.xlabel('steering angle', fontsize=20)
        plt.ylabel('counts', fontsize=20)
        plt.show()

    return balanced_samples

def process_data(samples):
    angles = []
    for line in samples:
        
        angles.append(float(line[3]))
        
        return angles


def brightness_change(image):
    
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(0.2,0.8)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    return image1


def data_augmentation(images, angles):
    
    augmented_images = []
    augmented_angles = []
    for image, angle in zip(images, angles):

        augmented_images.append(image)
        augmented_angles.append(angle)

        augmented_images.append(cv2.flip(image,1))
        augmented_angles.append(-1.0 * angle)

        augmented_images.append(brightness_change(image))
        augmented_angles.append(angle)
        augmented_images.append(brightness_change(flipped_image))
        augmented_angles.append(flipped_angle)

    return augmented_images, augmented_angles


def network_model():
    
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(32,3,3,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Convolution2D(128,3,3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(256,3,3, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(20))
    model.add(Dense(1))
    return model


def generator(samples, train_flag, batch_size=16):
    
    num_samples = len(samples)
    correction = 0.2  

    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for line in batch_samples:
                
                source_path = line[0]
                filename = source_path.split('\\')[-1]
                c_imagePath = './driving_behavioral/IMG/' + filename
                #print(c_imagePath)
                c_image = cv2.imread(c_imagePath)
                images.append(c_image)
                angle = float(line[3])
                angles.append(angle)

                if train_flag:  # only add left and right images for training data (not for validation)
                    source_path = line[1]
                    filename = source_path.split('\\')[-1]
                    l_imagePath = './driving_behavioral/IMG/' + filename
                    #print(l_imagePath)
                    
                    source_path = line[2]
                    filename = source_path.split('\\')[-1]
                    r_imagePath = './driving_behavioral/IMG/' + filename
                    l_image = cv2.imread(l_imagePath)
                    r_image = cv2.imread(r_imagePath)

                    images.append(l_image)
                    angles.append(angle + correction)
                    images.append(r_image)
                    angles.append(angle - correction)

            augmented_images, augmented_angles = data_augmentation(images, angles)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield shuffle(X_train, y_train)



# load the csv file
Path = './driving_behavioral/driving_log.csv'
samples = loadData(basePath)

# balance the data with smooth the histogram of steering angles
samples = balance_data(samples, visulization_flag=True)
#print(samples)

# process data to obtain steering angles with balanced data (lines)
angles = process_data(samples)

# split data into training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

# compile and train the model using the generator function
train_generator = generator(train_samples, train_flag=True, batch_size=16)
validation_generator = generator(validation_samples, train_flag=False, batch_size=16)


# define the network model
model = network_model()
#model.summary()


model.compile(loss='mse', optimizer='adam')

history = model.fit_generator(train_generator, samples_per_epoch=(len(train_samples)//16)*16, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=4)

model.save('model.h5')

