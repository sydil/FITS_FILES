# coding: utf-8

# Imports

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans

# #Function to generate more images by perfoming geometrical operations
def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                   mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                   flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    
    #ipdb.set_trace()
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    # we create two instances with the same arguments
    # Provide the same seed and keyword arguments to the fit and flow methods
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    #val_datagen = ImageDataGenerator(**aug_dict)
    
    
#     import ipdb; ipdb.set_trace()
  

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)


#     val_image_generator = val_datagen.flow_from_directory(
#         'ATLBS_IMAGES/final/train_image/v_image',
#         batch_size = batch_size)


#     val_mask_generator = val_datagen.flow_from_directory(
#         'ATLBS_IMAGES/final/test_image/v_mask',
#         batch_size = batch_size)



    
#     val_generator = zip(val_image_generator, val_mask_generator)
    
#     #combine generators into one which yields image and masks
    #code hangs when zipping the generators
    #train_generator1 =zip(image_generator, mask_generator)
    
    '''for (img,mask) in val_generator:
        img,mask = adjustData(img,mask)
        print("passed the stage")
        yield val_image_generator.__next__(), val_mask_generator.__next__()'''
   
    
    while True:
        try:
            yield image_generator.__next__(), mask_generator.__next__()
        except:
            break


#Running the image augmentation funtion

#for loop for each translation
data_gen_args = dict(rotation_range=50,#make many degrees
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
# myGenerator = trainGenerator(20,"ATLBS_IMAGES/final/","actual_Trainimages","actual_Testimages",data_gen_args,save_to_dir="ATLBS_IMAGES/final/aug")


# next(myGenerator)

# # visualize the data augmentation result


# num_batch = 4
# for i,batch in enumerate(myGenerator):
#     print (i)
#     if(i >= num_batch):
#         break


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

# # image_arr,mask_arr = geneTrainNpy("ATLBS_IMAGES/final/aug","ATLBS_IMAGES/final/aug") 

# def labelVisualize(num_class,color_dict,img):
#     img = img[:,:,0] if len(img.shape) == 3 else img
#     img_out = np.zeros(img.shape + (3,))
#     for i in range(num_class):
#         img_out[img == i,:] = color_dict[i]
#     return img_out/ 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        #img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)




