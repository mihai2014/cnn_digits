import base64
from PIL import Image
from io import BytesIO

f = open('base64.txt', 'r')
data = f.read()
f.closed

#remove header: data:image/png;base64
data = data.split(",")[1]

im = Image.open(BytesIO(base64.b64decode(data)))
im.save('image.png', 'PNG')




#from PIL import Image
#from io import BytesIO
#from base64 import b64decode
#imagestr = 'data:image/png;base64,...base 64 stuff....'
#im = Image.open(BytesIO(b64decode(imagestr.split(',')[1])))
#im.save("image.png")
