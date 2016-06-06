import code
code.interact(local=locals())

import matplotlib.pyplot as plt
fig, (ax0) = plt.subplots(ncols=1, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax0.imshow(X[700], cmap=plt.cm.gray)
ax0.set_title('Roberts Edge Detection')
ax0.axis('off')
plt.show()

# from skimage.filters import roberts, sobel, scharr, prewitt
