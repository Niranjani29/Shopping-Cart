# Shopping-Cart
Developed an algorithm for automation in shopping cart using camera to avoid the customer wait time in long queues and enhance their shopping experience. COCO database was extracted and object detection was implemented using SIFT algorithm in Pycharm IDE


<b><h3>INTRODUCTION</h3></b>

To detect and describe local features of an image scale-invariant feature transform (SIFT) is used. Key points of objects are first extracted from a set of reference images and stored in a database. By individually comparing each feature from the new image to the image in the database and then finding candidate matching features based on Euclidean distance, an object is recognized in a new image. True matches that pass all the tests can be identified as correct with high confidence.


<h3><b>Proposed System</h3></b>
The system includes a shopping cart with a detection sheet placed on one side of the cart. A camera of 3 megapixels is placed on the opposite side of the detection sheet. Whenever user puts a product into the cart, it crosses the detection sheet. Once the product crosses the sheet, camera will capture the images. These images are compared with the images present in the database using SIFT algorithm. Once the images are matched, the correct product will be recognized and the corresponding amount will be displayed on the screen. If the user removes any product, images are captured again as it crosses the sheet and compared with the database images. After product recognition the respective amount is reduced from the total amount. The camera will not capture images of the products present in the vicinity of the cart. Thus, the conundrum of scope of visibility is resolved. The payment mode depends on the grocery store or shopping mall authorities. 
