import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

def zoom_at(img, zoom=2, angle=0, coord=None):
    
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result

original = cv2.imread("car2.jpg")
cv2.imshow("original", original)

x = 240
y = 178

b = original.item(y, x, 0)
g = original.item(y, x, 1)
r = original.item(y, x, 2)

print('pixel:', b, g, r)
print("shape:",original.shape)
# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow("gris", gris)



# y0,y1 y luego x0yx1
#zona_patente = original[375:675, 365:750]

#car2

#zoomDeportivo=zoom_at(zona_patente)
#cv2.imshow('zoom',zoomDeportivo)
#posible ganador
th = cv2.threshold(gris, 72, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("th", th)
plt.imshow(th, cmap='gray')
plt.show()

contours = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
canvas = np.zeros_like(original)
cv2.drawContours(canvas , contours, -1, (0, 255, 0), 2)
#plt.axis('off')
cv2.imshow("canvas", canvas)
#plt.imshow(th, cmap='gray')
#plt.show()

zona_patente = canvas[300:675, 365:600]
cv2.imshow('zona_patente', zona_patente)



# Patente chilena ratio = 360 / 130 = 2.769230769
candidates = []
max_w = 100
min_w = 10

min_h = 10
max_h = 100

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    # SI el aspecratio es similar Y el ancho Y el alto es Similar , es candidato
    #np.isclose(aspect_ratio, ratio, atol=1.0) and
    if ( (max_w > w > min_w) and (max_h > h > min_h)):
        candidates.append(cnt)

# redibujamos los contornos que nos quedan, despues de aplicar un filtro

canvas = np.zeros_like(original)
cv2.drawContours(canvas , candidates, -1, (0, 255, 0), 2)
#plt.axis('off')
cv2.imshow('zona_patente2', canvas)

# Pantente arg 99, 397, 109, 42



cv2.waitKey(0)
cv2.destroyAllWindows()