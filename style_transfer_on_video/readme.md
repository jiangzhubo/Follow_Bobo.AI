# Neural-Style-Transfer-on-video-data
Neural Style Transfer-Real Time Video Augmentation using pretrained VGG-16 model

### **Neural Style Transfer on real-time Video:**

I will explain the steps for an image, as a video is nothing but a collection of a set of images. These images are called frames and can be combined to get the original video.So we can loop through the steps for all individual frames, recombine and generate the stylized video.

**Step 1: Loading pre-trained VGG-16 CNN model**

Building(training) the CNN from scratch for NST application takes a lot of time and powerful computation infrastructure, which is not readily available as an individual.

So, we will load the weights(trained from images of famous '[**ImageNet**](https://en.wikipedia.org/wiki/ImageNet).' challenge) of a pre-trained CNN- [VGG-16](https://neurohive.io/en/popular-networks/vgg16/) to implement neural style transfer. We will use Keras Applications to load VGG-16 with pre-trained weights. VGG-16 may not optimum(desired complexity)CNN architecture for NST. There are more complex(deeper with advanced **architecture**) networks like InceptionV4, VGG-19,Resnet-101 etc for this application, which will take more time in loading and running. However, as an experiment, we chose a VGG-16(having high classification accuracy and good intrinsic understanding of features).

```
from keras.applications.vgg16 import VGG16
shape = (224,224)
vgg = VGG16(input_shape=shape,weights='imagenet',include_top=False)
```

The shape is important here as the VGG-16 network takes the input image with shape 224 x 224 x 3.

```
vgg.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_21 (InputLayer)        (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
average_pooling2d_101 (Avera (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
average_pooling2d_102 (Avera (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
average_pooling2d_103 (Avera (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
average_pooling2d_104 (Avera (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
average_pooling2d_105 (Avera (None, 7, 7, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
```

​                                              VGG-16 Architecture

**Step 2: Define the content model and cost function**

For high-level content features, we want to account for features across the entire image.So we will replace max-pool(which may throw away some information), with average pool.Then we will pick any deeper layer out of total 13 convolutions as the "output" and define the model up to that layer. Then we will feed our pre-processed content image(X) in the n/w to calculate(predicted) feature/activation map at the output layer wrt this model and model output wrt to any random(white noise)matrix of the defined shape(224 x 224 x 3). We calculate MSE loss and gradients for the content image network.This will help to update the input image(random image) to the opposite direction of the gradient and allow the content loss value to decrease, so that generated image will match that of the input image. The detailed implementation code is kept at my [GitHub repository](https://github.com/nitsourish/Neural-Style-Transfer-on-video-data).

```
content_model = vgg_cutoff(shape, 13) #Can be experimented with other deep layers
# make the target
target = K.variable(content_model.predict(x))
# try to match the input image
# define loss in keras
loss = K.mean(K.square(target - content_model.output))
# gradients which are needed by the optimizer
grads = K.gradients(loss, content_model.input)
```

**Step 3: Define the style model and style loss function**

Two images whose feature maps at a given layer produced the same Gram matrix we would expect both images to have the same style(but not necessarily the same content).So activation maps in early layers in the network would capture some of the finer textures(low level features), whereas activation maps deeper layers would capture more higher-level elements of the image’s style.So to get the best results we will take a combination of both shallow and deep layers as output to compare the style representation for an image and we define the multi-output model accordingly.

Here first we calculate Gram matrix at each layer and calculate total style loss of the style network. We take different weights for different layers to calculate the weighted loss. Then based on style loss(difference in style component) and gradients we update the input image(random image) and reducing style loss value, so that generated image(Z) texture looks similar that of the style image(Y).

```
#Define multi-output model
symb_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers\
if layer.name.endswith('conv1')]
multi_output_model = Model(vgg.input, symb_conv_outputs)
#Style feature map(outputs) of style image
symb_layer_out = [K.variable(y) for y in multi_output_model.predect(x)]


#Defining Style loss
def gram_matrix(img):
    X = K.batch_flatten(K.permute_dimensions(img,(2,0,1)))
    gram_mat = K.dot(X,K.transpose(X))/img.get_shape().num_elements()
    return gram_mat 


def style_loss(y,t):
    return K.mean(K.square(gram_matrix(y)-gram_matrix(t)))


#Style loss calculation through out the network
#Defining layer weights for layers 
weights = [0.2,0.4,0.3,0.5,0.2]
loss=0
for symb,actual,w in zip(symb_conv_outputs,symb_layer_out,weights):
    loss += w * style_loss(symb[0],actual[0])
grad = K.gradients(loss,multi_output_model.input)
get_loss_grad = K.Function(inputs=[multi_output_model.input], outputs=[loss] + grad)
```

**Step 4: Define the total cost(overall loss)**:

Now we can combine both content and style loss to obtain an overall loss of the network.We need to minimize this quantity over iteration using a suitable optimization algorithm.

```
#Content Loss
loss=K.mean(K.square(content_model.output-content_target)) * Wc #Wc is content loss weight(hyperparameter)

#Defining layer weights of layers for style loss 
weights = [0.2,0.4,0.3,0.5,0.2]

#Total loss and gradient
for symb,actual,w in zip(symb_conv_outputs,symb_layer_out,weights):
    loss += Ws * w * style_loss(symb[0],actual[0]) #Wc is content loss weight(hyperparameter)
    
grad = K.gradients(loss,vgg.input)
get_loss_grad = K.Function(inputs=[vgg.input], outputs=[loss] + grad)
```

**Step 5: Solve the optimization problem and loss minimization function**

After defining the entire symbolic computation graph optimization algorithm is the principal component, which will enable to iteratively minimize the overall network cost. Here instead of using keras standard optimizer function (such as optimizers.Adam, optimizers.sgd etc.), which may take more time, we will use [Limited-memory BFGS( Broyden–Fletcher–Goldfarb–Shanno),](https://en.wikipedia.org/wiki/Limited-memory_BFGS) which is an approximate Numerical Optimization algorithm using a limited amount of computer memory. Due to its resulting linear memory requirement, this method is well suited for optimization problem involving large no of variables(parameters). Like normal BFGS it's a standard quasi-Newton method to optimize smooth functions by maximizing the regularized log-likelihood.

Scipy's minimizer function(fmin_l_bfgs_b) allows us to pass back function value f(x) and its gradient f'(x), which we calculated in earlier step. However, we need to unroll the input to minimizer function in1-D array format and both loss and gradient must be np.float64.

```
#Wrapper Function to feed loss and gradient with proper format to L-BFGS 
def get_loss_grad_wrapper(x_vec):
        l,g = get_loss_grad([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)
#Function to minimize loss and iteratively generate the image
def min_loss(fn,epochs,batch_shape):
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = scipy.optimize.fmin_l_bfgs_b(func=fn,x0=x,maxfun=20)
    # bounds=[[-127, 127]]*len(x.flatten())
    #x = np.clip(x, -127, 127)
    # print("min:", x.min(), "max:", x.max())
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)
    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()
    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0]   
```

**Step 6: Run the optimizer function on input content and style image:**

Run the optimizers on the input content frame and style image and as per defined symbolic computation graph, the network does its intended job of minimizing overall loss and generate an image which looks as close as to both content and style image.

![No alt text provided for this image](https://media.licdn.com/dms/image/C5112AQGRcKcKSiDcRg/article-inline_image-shrink_1500_2232/0?e=1574294400&v=beta&t=Fc_291MoUa-VNt5-QcfWD9GLMW573uqdRzzWqx0Gz0Q)

The output image is still noisy, as we ran the network only for 30 iterations. The ideal NST the network should be optimized for thousands of iterations to reach the minimum loss threshold to generate clear blended output.

**Step 7: Repeat the above steps for all image frames:**

Perform network inference on each frame after extracting frames from the short video, generate styled image for each frame and recombine/stitch the styled image frame.

```
#Vedio Reading and extracting frames

cap = cv2.VideoCapture(path)
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(224,224))
    X = preprocess_img(frame)

    #Running the above optimization as per defined comutation graph and generate styled image frame#    
    final_img = min_loss(fn=get_loss_grad_wrapper,epochs=30,batch_shape=batch_shape)    
    plt.imshow(scale(final_img))
    plt.show()
    cv2.imwrite(filename, final_img)

#Recombine styled image frames to form the video
video = cv2.VideoWriter(video_name, 0, 1, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
cv2.destroyAllWindows()
video.release()
```

 We can try with videos using device camera roll as well and try the style transfer in online mode(real-time video), just by tweaking VideoCapture mode.

```
cap = cv2.VideoCapture(0)
cap.release()
```
