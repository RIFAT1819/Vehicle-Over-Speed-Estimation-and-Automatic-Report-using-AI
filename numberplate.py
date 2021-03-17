


import cv2
import numpy as np
import os
import pytesseract
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')





def plot_images(img1, img2, title1='', title2=''):
    fig = plt.figure(figsize=[15,15])
    
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap='gray')
    ax1.set(xticks=[], yticks=[], title=title1)
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap='gray')
    ax2.set(xticks=[], yticks=[], title=title2)




path ="./images/rifat1.jpg"





image = cv2.imread(path)





plot_images(image, image)





gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)





plot_images(image, gray, title1="Original", title2="Gray")





blur = cv2.bilateralFilter(gray, 11, 90, 90)





plot_images(gray, blur, title1="Gray", title2="Blur")




edges = cv2.Canny(blur, 30,200)





plot_images(blur, edges, title1="Blur", title2="Edges")





cnts, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)





print(cnts[0])





image_copy = image.copy()





_ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)





plot_images(edges, image_copy, title1="Edegs", title2="Contours")





print(len(cnts))





cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]





image_reduced_cnts = image.copy()
_ = cv2.drawContours(image_reduced_cnts, cnts, -1, (255,0,255),2)
plot_images(image_copy, image_reduced_cnts, title1="Contours", title2="Reduced")




print(len(cnts))





plate = None
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(edges_count) == 4:
        x,y,w,h = cv2.boundingRect(c)
        plate = image[y:y+h, x:x+w]
        break
        
cv2.imwrite("plate.png", plate)





plot_images(plate, plate, title1="plate", title2="plate")












































# In[ ]:




