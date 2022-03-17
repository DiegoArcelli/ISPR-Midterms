from http.client import ImproperConnectionState
from cv2 import log, threshold
from convolution import *

input_image = "Selected/1_9_s.bmp"
# input_image = "test.png"
image = np.asarray(Image.open(input_image).convert("L"))/255

kernel = log_kernel((15,15), 1)
# plt.imshow(kernel, cmap="gray")
# plt.show()
print(kernel)
output = convolution(image, kernel, True)
plt.imshow(output, cmap="gray")
plt.show()
output = (output*255).astype(np.uint8)
print(output)
plt.imshow(output, cmap="gray")
plt.show()

# _, output = cv.threshold(output,200,255,cv.THRESH_BINARY)
print(output.shape)
out_image = Image.fromarray(output)

out_image.save("output.jpg")


cv_img = cv.imread(input_image)
cv_out = cv.filter2D(cv_img, 3, kernel)
cv.imwrite("output_cv.jpg", cv_out)
