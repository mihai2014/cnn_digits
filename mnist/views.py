from django.shortcuts import render
import json

import base64
from PIL import Image
from io import BytesIO

import numpy as np

#tensor flow net
#from . import network as cnn
#what = cnn.Recognise()
#what.test()

# Create your views here.

from django.http import HttpResponse

def home(request):
    #return HttpResponse("<h1>Hello!</h1>")
    return render(request, 'mnist/edit.html', {})


def convert(pil_img):      

    #pil image 280 x 280
    img = pil_img.resize((28, 28))
    img_28_28 = img
    #numpy array [0,255]
    pixels = np.array(img)
    #uint [0,255]
    pixels = pixels.astype('float32')
    #normalize [0,1]
    pixels /= 255.0
    #gray(alpha) from rgba
    pixels = pixels[:,:,3]
    #flatten: 28x28 = 784
    pixels = pixels.reshape(1,-1)

    return [pixels, img_28_28]

def send_image(request):
    method = request.method
    if(method == "POST"):
        test = request.POST['test']
        #print(test)

        base64_image = None
        try:
            file = request.FILES['image']
            file.name           # Gives name
            file.content_type   # Gives Content type text/html etc
            file.size           # Gives file's size in byte
            base64_image = file.read().decode()  # Reads file            
        except:
            pass    

        try:
            base64_image = request.POST['image']
        except:
            pass    

        if(base64_image == None):
             return HttpResponse("Result is: " + "error - no image")       

        #remove header: data:image/png;base64
        base64_image = base64_image.split(",")[1]

        # 280 x 280 pil image
        pil_image = Image.open(BytesIO(base64.b64decode(base64_image)))
        pil_image.save('image_280_280.png', 'PNG')


        # python tensor flow network
        #result = what.out(pil_image)
        result = "-"
        # x: flattened [1,784], normalized [0,1] numpy array
        #x = what.convert(pil_image)
        # 28 x 28 pil image
        #what.img_28_28.save('image_28_28.png', 'PNG')


        # x: flattened [1,784], normalized [0,1] numpy array
        x, img_28_28 = convert(pil_image)

        # 28 x 28 pil image
        img_28_28.save('image_28_28.png', 'PNG')

        #python list
        x = x.tolist()


        obj = {"result":str(result), "x":x}    
        #encode obj
        json_obj =  json.dumps(obj)    

    return HttpResponse(json_obj)

