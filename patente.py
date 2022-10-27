import cv2
import numpy as np

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
 
# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (5,5), 0)
 
cv2.imshow("suavizado", gauss)
print("shape gaussiano:",gauss.shape)

# y0,y1 y luego x0yx1
#zona_patente = original[375:675, 365:750]

#car2
zona_patente = original[300:675, 365:600]
cv2.imshow('zona_patente', zona_patente)
zoomDeportivo=zoom_at(zona_patente)
cv2.imshow('zoom',zoomDeportivo)
# Detectamos los bordes con Canny
#canny = cv2.Canny(gauss, 50, 150)
 
#cv2.imshow("canny", canny)

# Buscamos los contornos
#(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
# Mostramos el número de monedas por consola
#print("He encontrado {} objetos".format(len(contornos)))
 
#cv2.drawContours(original,contornos,-1,(0,0,255), 2)
#cv2.imshow("contornos", original)
 
cv2.waitKey(0)
cv2.destroyAllWindows()