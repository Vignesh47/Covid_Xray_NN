#importing Libraries 
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from tqdm import tqdm
import cv2
import seaborn as sns

files = ['Normal', 'COVID']
source_folder = "/Users/vigneshnandhan/cnn/Data"
data_dir = os.path.join(source_folder)


#Reading the files and converting into data 
data = []
for id, level in enumerate(files):
    for file in os.listdir(os.path.join(data_dir, level+'/'+'images')):
        data.append(['{}/{}'.format(level, file), level])
        data.append([level +'/' +'images'+ '/'+file, level])

imagePaths = []
for dirname, _, filenames in os.walk(source_folder):
    for filename in filenames:
        if filename.endswith('.png'):
            imagePaths.append(os.path.join(dirname, filename))



data_df = pd.DataFrame(data, columns = ['image_file', 'corona_result'])


data_df['path'] = source_folder + '/' +data_df['image_file']
data_df['corona_result'] = data_df['corona_result'].map({'Normal': 'Normal', 'COVID': 'Covid_positive'})

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Exploratory Data Analysis<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 


covid_positive_count = data_df['corona_result'].value_counts().get('Covid_positive', 0)
normal_count = data_df['corona_result'].value_counts().get('Normal', 0)

#Number of 
print('Covid-19:', covid_positive_count)
print('Normal:', normal_count)

print('Number of Duplicated Samples>>> : %d' %(data_df.duplicated().sum()))
print('Number of Total Samples >>>>> :', data_df.isnull().value_counts().iloc[0])




#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Data Visulaiztion<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
'''
Data = []
Target = []
resize = 150


for imagePath in tqdm(imagePaths):
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (resize, resize)) /255

    Data.append(image)




df = pd.DataFrame(Target,columns=['Labels'])
sns.countplot(df['Labels'])
plt.show()

print('Covid-19:',Target.count('Covid-19'))
print('Normal:',Target.count('Normal'))
print('Pneumonia: ',Target.count('Pneumonia'))


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(imagePaths),25))) : 
    plt.subplot(5,5,n+1)
    plt.imshow(Data[i] , cmap='gray')
    plt.title(Target[i])
    plt.axis('off')     
plt.show()


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(imagePaths),25))) : 
    plt.subplot(5,5,n+1)
    plt.imshow(Data[i] , cmap='gray')
    plt.title(Target[i])
    plt.axis('off')     
plt.show()

'''