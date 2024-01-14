import time
from PIL import Image
import os
def modify_rgb(image_path :str) -> None:
    #generates a random number
    current_time = int(time.time())
    generated_number  = (current_time % 100) + 50
    if generated_number %2 == 0:
        generated_number +=10
        
    # using Pillow to modify the image.
    image = Image.open(image_path)
    pixels = image.load()

    # achieving the size properties from the Image instance
    width, height = image.size
    #iterating through the image matrix to get the individual pixels
    for i in range(width):
        for j in range(height):
            r,g,b = pixels[i,j]
            new_r = int(r + generated_number)
            new_g= int(g+ generated_number)
            new_b = int(b + generated_number)
            #assigning the new values
            pixels[i,j] = (new_r,new_g,new_b)
    #saving the new image file.
    image.save(os.getcwd()+"\\Question2\\chapter1out.png")

modify_rgb("Question2\\chapter1.jpg")

