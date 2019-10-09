This code helps us understand how spatial pooling pyramid (SPP) works in context of
AlexNet architecture. The input image size is 227 x 227 as required by
the architecture. Using the optional SPP, you can experiment with different
image sizes and verify that irrrespective of the image size,say varying image
sizes in a batch, we still should get the same fixed length vector at the
output of SPP.Run `alexnet_feedforward.py` to achieve this
