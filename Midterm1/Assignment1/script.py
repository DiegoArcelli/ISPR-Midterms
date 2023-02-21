from convolution import *

# input_image = "Images/3_12_s.bmp"
# input_image = "Images/6_1_s.bmp"
# input_image = "Images/7_1_s.bmp"
input_image = "Images/1_9_s.bmp"

image = np.asarray(Image.open(input_image).convert("L"))/255
kernel = log_kernel((15,15), 2)

output = convolution(image, kernel, True)
output = (output*255).astype(np.uint8)
plt.imshow(output, cmap="gray")
plt.show()
out_image = Image.fromarray(output)
out_image.save("output.jpg")