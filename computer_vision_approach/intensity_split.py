import cv2
import numpy as np
import sys


img=cv2.imread('/home/fateh/PycharmProjects/keras/macular_hole/main_data/Fundus_Images_for_Testing/324.jpg',0)
img=cv2.resize(img,(img.shape[1]/4,img.shape[0]/4))
#img = cv2.fastNlMeansDenoising(img,None,10,7,21)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
cl1 = clahe.apply(img)
cl1 = cv2.fastNlMeansDenoising(cl1,None,10,7,21)


cv2.imshow("img",cl1)


num_layers=16
if(256%num_layers!=0):
	print("invalid number of layers")
	sys.exit(0)

rng=np.zeros((num_layers,2),dtype='uint8')
layers=np.ones((num_layers,cl1.shape[0],cl1.shape[1]),dtype='uint8')

print(layers.shape)

print(np.max(cl1))
diff=np.max(cl1)/num_layers
for i in np.arange(0,num_layers,1):
	r_l=i*diff
	r_u=(i+1)*diff-1
	rng[i][0]=r_l
	rng[i][1]=r_u

#print(rng)
def key_get_avg_contour_intensity(carr):
#print(np.asarray(carr).shape)
#for i in range(len(carr)):
	cimg = np.zeros_like(img)
	cv2.drawContours(cimg, carr, -1, color=255, thickness=-1)
	pts = np.where(cimg == 255)

	#lst_intensities.append(img[pts[0], pts[1]])
	print(np.median(np.asarray(img[pts[0], pts[1]])))
	return np.median(np.asarray(img[pts[0], pts[1]]))

count=0
coords=[]
carr=[]
for i in np.arange(0,num_layers,1):
	#print(rng[i][0])
	#print(rng[i][1])
	#p, q = np.where(img >= rng[i][0] & img<rng[i][1])
	arr= np.array(cl1)
	arr[arr>=rng[i][1]]=0
	arr[arr<rng[i][0]]=0
	arr[arr>0]=1
	#print(arr)
	layers[i]=arr
	kernel = np.ones((7,7),np.float32)/49
	dst = cv2.filter2D(layers[i],-1,kernel)
	dst[arr>0]=1
	#cv2.imshow("avg"+str(i),dst*255)	
	im=layers[i]+dst
	im[im>0]=1
	#cv2.imshow("avg+layer"+str(i),im*255)
	el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	image = cv2.dilate(im, el, iterations=3)
	cv2.imshow("dilated"+str(i), image*255)
	contours,hierarchy=cv2.findContours(image,cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	c = max(contours, key = key_get_avg_contour_intensity)
	print(c.shape)
	carr.append(c)

	imcopy=img.copy()
	cv2.drawContours(imcopy,[c], -1, (0,255,255), 5)
	#cv2.imshow('detected circles'+str(i),imcopy)
	cv2.imshow('layers'+str(i),layers[i]*255)

print(len(carr))


#layers=np.asarray(layers,dtype='uint8')
#print(np.max(layers[0]))
#print(np.min(layers[0]))
#print(layers.shape)
'''
for i in np.arange(0,num_layers,1):
	cv2.imshow("layer:"+str(i),layers[i][:]*255)
	#laplacian = cv2.Laplacian(layers[i],cv2.CV_64F)
	#cv2.imshow("laplacian:"+str(i),laplacian)


#print(np.where(layers[num_layers-1][:] > 0))
#coord=np.argwhere(layers[num_layers-1][:] > 0)
#print(coord)
mid=int((len(coord)-1)/2)
#print(len(coord))
#print(mid)
xcl=coord[mid][0]
ycl=coord[mid][1]


#contours,hierarchy=cv2.findContours(layers[num_layers-1][:],cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,64,64), 5)
#contours,hierarchy=cv2.findContours(layers[num_layers-2][:],cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,255,255), 5)
#print(hierarchy)
#contours,hierarchy=cv2.findContours(layers[num_layers-3][:],cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,255,255), 5)
contours,hierarchy=cv2.findContours(layers[num_layers-1][:],cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(img, contours, -1, (0,255,255), 5)
#print(contours)
#hull = cv2.convexHull(contours[1])
#print(hull)
#ellipse = cv2.fitEllipse(contours[10])
#cv2.ellipse(img,ellipse,(0,255,0),2)
#cv2.drawContours(img, hull, -1, (0,255,255), 5)
#print hierarchy

cv2.imshow("cont",img)

#cnt = contours[0]
#M = cv2.moments(cnt)
#print M
#print(xc)
#print(yc)

kernel = np.ones((7,7),np.float32)/49
dst = cv2.filter2D(layers[num_layers-1],-1,kernel)
dst[arr>0]=1
cv2.imshow("avg",dst*255)


im=layers[num_layers-1]+dst
im[im>0]=1
#im = cv2.addWeighted(dst,1,layers[3],1,0)
cv2.imshow("avg+layer",im*255)

#contours,hierarchy=cv2.findContours(im,cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,255,255), 5)
el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
image = cv2.dilate(im, el, iterations=3)

cv2.imshow("dilated.png", image*255)

contours,hierarchy=cv2.findContours(image,cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(img[contours[0]])
c = max(contours, key = cv2.contourArea)
print(c)
cv2.drawContours(img,[c], -1, (0,255,255), 5)
cv2.imshow('detected circles',img)
'''


cv2.waitKey(0)