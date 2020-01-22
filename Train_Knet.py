# coding: utf-8



import matplotlib.pyplot as plt
from Aug_n_Train import *
import h5py
import sklearn
from skimage.util import img_as_float
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy import misc



##TRAINING THE MODEL
#myGene = trainGenerator(20,"ATLBS_IMAGES/final/","actual_Trainimages","actual_Testimages",data_gen_args,save_to_dir= None)
# model = unet()
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=0, save_best_only=True)
#,callbacks=[model_checkpoint]
#history = model.fit_generator(myGene,steps_per_epoch=1,epochs=5,callbacks=[model_checkpoint])


# Running my own grid to tune hyperparameters
filters =[3,4,5,6]
kernel_size= [2,3,4]
myGene = trainGenerator(20,"ATLBS_IMAGES/final/","actual_Trainimages","actual_Testimages",data_gen_args,save_to_dir= None)
for filters in filters:
    for kernel_size in kernel_size:
        print(filters,kernel_size)
        from model import *
        model = unet(filters,kernel_size)
        history = model.fit_generator(myGene,steps_per_epoch=1,epochs=2,callbacks=[model_checkpoint])
        print(history.history['acc'],history.history['loss'])







# # reading images in a folder as training set

# list_of_images = []

# for filename in glob.glob("ATLBS_IMAGES/final/imageA/*.png"): 
#     image_set = misc.imread(filename)
#     list_of_images.append(image_set)
# list_of_masks = []

# for filename in glob.glob("ATLBS_IMAGES/final/maskA/*.png"): 
#     mask_set = misc.imread(filename)
#     list_of_masks.append(mask_set)

# #Splitting the data into training and Testing
# list_of_images=np.array(list_of_images)
# list_of_masks=np.array(list_of_masks)

# data_train,data_test,label_train,label_test = train_test_split(list_of_images,list_of_masks,test_size=0)
# print(data_train.shape,data_test.shape)

    
# # BUILDING A GRID SEARCH

# # a, b = next(myGene)
# # Wrap Keras model so it can be used by scikit-learn
# my_classifier = KerasClassifier(model, verbose=0,batch_size=32)
# print('wrapped')

# # # Create hyperparameter space
# # epochs = [2]
# # batches = [5]
# # optimizers = ['adam']
# # filters = [8]
# # kernel_size = [2]
# # #strides = [1]
# # #pool_size = [(2, 2)]
# # steps_per_epoch=[1]
# # parameters = {'optimizers':['adam'], 
# #                'epochs' : [2]}
# # #                'batches': [5],
# # #                'filters' :[8],
# # #                'kernel_size' : [2],
# # #                'strides' : [1],
# # #                'pool_size': (2, 2),
# # #                'steps_per_epoch':[1]}
# # 'epochs': [3, 6],
# # 'pool_size': [2]
# param_grid={'filters': [8],
#             'kernel_size': [3]}
# validator = GridSearchCV(my_classifier,
#                          param_grid=param_grid,
#                          scoring='accuracy',
#                          n_jobs=1)
# validator.fit(data_train,label_train)
# print('The parameters of the best model are: ')
# print(validator.best_params_)

# # # Create hyperparameter options
# hyperparameters = dict(optimizers=optimizers, epochs=epochs, batch_size=batches,filters=filters,kernel_size=kernel_size,steps_per_epoch=steps_per_epoch)

# # Create grid search
# grid = GridSearchCV(estimator=neural_network, param_grid=hyperparameters,cv =5,scoring='accuracy')
# print('grid searched')
# # Fit grid search
# grid_result = grid.fit(data_train,label_train)
# grid_result
# print('fitted')
# print(grid.best_params_)










#history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
# print(history.history.keys())

# # # summarize history for accuracy
# plt.plot(history.history['acc'])
# #plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# #plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# #plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# #plt.legend(['train', 'test'], loc='upper left')
# plt.show()



# #TEST GENERATOR

# f1=np.fromfile('unet_membrane1.hdf5',dtype=float)
# imfolder2=glob.glob('ATLBS_IMAGES/final/mask_Trainimages/*.png')

# def testGenerator(test_path,num_image = 36,target_size = (256,256),flag_multi_class = False,as_gray = True):
#     for image in imfolder2:
#         #import ipdb; ipdb.set_trace()
#         img = io.imread(image,as_gray = as_gray)
#         img = img / 255
#         #img = trans.resize(img,target_size)
#         img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
#         img = np.reshape(img,(1,)+img.shape)
#         yield img
        
# ##SAVING THE RESULTS
# Sky = [128,128,128]
# Building = [128,0,0]
# Pole = [192,192,128]
# Road = [128,64,128]
# Pavement = [60,40,222]
# Tree = [128,128,0]
# SignSymbol = [192,128,128]
# Fence = [64,64,128]
# Car = [64,0,128]
# Pedestrian = [64,64,0]
# Bicyclist = [0,128,192]
# Unlabelled = [0,0,0]

# COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
#                           Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


# def labelVisualize(num_class,color_dict,img):
#     img = img[:,:,0] if len(img.shape) == 3 else img
#     img_out = np.zeros(img.shape + (3,))
#     for i in range(num_class):
#         img_out[img == i,:] = color_dict[i]
#     return img_out / 255



# def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
#     for i,item in enumerate(npyfile):
#         img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
#         io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
#         io.imsave(os.path.join(save_path,"%d_predict.tiff"%i),img_as_float(img))
        
# ##TESTING THE MODEL

# testGene = testGenerator('ATLBS_IMAGES/final/mask_Trainimages/')
# model = unet()
# model.load_weights("unet_membrane2.hdf5")
# model_checkpoint = ModelCheckpoint('unet_membrane2.hdf5', monitor='loss',verbose=1, save_best_only=True)
# results = model.predict_generator(testGene,24,verbose=0)
# print("Done Testing!")
# saveResult("ATLBS_IMAGES/masked/aug",results)
# print("Done saving predicted images!")
# print("CHECKING!")
# print("CHECKED!")


