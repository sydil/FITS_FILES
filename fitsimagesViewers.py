
# coding: utf-8

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


import matplotlib.pylab as plb


# In[23]:


from astropy.visualization import astropy_mpl_style


# In[89]:


from astropy.io import fits
import numpy as np


# In[25]:


fits.info('M84.fits')


# In[26]:


image_data = fits.getdata('M84.fits')


# In[27]:


print(image_data.shape)


# # Plotting an image at a specific position

# In[28]:


plt.figure()


# In[29]:


plt.imshow(image_data)
plt.colorbar()


# # Plotting using WCS

# In[30]:


from astropy.wcs import WCS


hdu = fits.open('M84.fits')[0]
wcs = WCS(hdu.header)

plt.subplot(projection=wcs)
plt.imshow(hdu.data)
plt.xlabel('Galactic Longitude')
plt.ylabel('Galactic Latitude')


# # Making subimages/Cutouts

# In[31]:


import aplpy


# In[32]:


import numpy as np
from PIL import Image
plt.style.use(astropy_mpl_style)


# In[33]:


plt.imshow(image_data)


# In[34]:


from astropy.nddata import Cutout2D


# In[35]:


from astropy import units as u


# In[48]:


hdul = fits.open('M84.fits')


# In[49]:


position = (hdul[0].header['CRVAL1'],hdul[0].header['CRVAL1'])


# In[50]:


position


# In[51]:


size = (hdul[0].header['CRPIX1'],hdul[0].header['CRPIX2'])


# In[52]:


cutout = Cutout2D(image_data,position,size)


# In[53]:


plt.subplot(projection=wcs)
plt.imshow(cutout.data)
plt.xlabel('Galactic Longitude')
plt.ylabel('Galactic Latitude')


# In[96]:



newhdu = fits.PrimaryHDU(image_data)
newhdulist = fits.HDUList([newhdu])
newhdulist.writeto('firstcutOut1.fits')


# In[54]:


plt.imshow(image_data)
cutout.plot_on_original(color='white')


# # CHANGING/MANIPULATING PIXEL VALUES

# In[65]:


print(image_data[1, 4])


# In[66]:


image_data[:]


# In[75]:


print(image_data)


# In[70]:


print(image_data[2:7,2:3])


# In[77]:


image_data[1,5]


# In[78]:


image_data[1,5] =0.5


# In[79]:


image_data[1,5]


# In[87]:


image_data[image_data < 0.01] = 0


# In[88]:


plt.imshow(image_data)


# # CHANGE/ MANIPULATE HEADER FILES

# In[ ]:


fits.info('M84.fits')


# In[ ]:


print("Extension 0:")
print(repr(fits.getheader('M84.fits', 0)))
print()


# In[ ]:


fits_file='/media/sydil/569E-1551/fitsfiles/FIRST_FRI/M84.fits'


# In[63]:


print(hdul[0].header)


# In[ ]:


hdr = hdul[0].header
hdr['targname'] = 'NGC121-a'


# In[ ]:


hdul[0].header['targname']


# In[ ]:


print("Extension 0:")
print(repr(fits.getheader('M84.fits', 0)))
print()


# In[ ]:


del hdr[16:40]


# In[ ]:


hdr = hdul[0].header
hdr['targname'] = 'NGC121-a'  


# In[ ]:


hdul[0].header['targname']


# In[ ]:


fits.writeto('M84.fits',image_data,hdr, clobber=True)


# In[ ]:


print("Extension 0:")
print(repr(fits.getheader('M84.fits', 0)))
print()

