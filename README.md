This code helps us understand how spatial pooling pyramid (SPP) works in context of
AlexNet architecture. The input image size is 227 x 227 as required by
the architecture. Using the optional SPP, you can experiment with different
image sizes and verify that irrespective of the image size(although for an architecture 
this deep we should have images atleast of size 163 x 163 ),say varying image
sizes in a batch, we still should get the same fixed length vector at the
output of SPP. You might be able to get results for even images with smaller dimensions
if you make the network shallower.Run `alexnet_feedforward.py` to achieve this
