
import tensorflow as tf
print(tf.__version__)
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import numpy as np
from PIL import Image
from glob import glob
import imageio
 # 构造生成器和判别器
 # 构造生成器类


#数据集
##获取所有图片路径
datas=glob(os.path.join('./image_align_celeba_trainset/','*.jpg'))#训练集
datasss=glob(os.path.join('./image_align_celeba_sestset/','*.jpg'))#测试集
datass=datasss[16:26]



z_dim = 100 #输入噪声维度
learning_rate = 0.0002  ############################################

alpha = 0.2 #leakyRelu的斜率
beta1 = 0.5 #Adm优化器的衰减率 0.5-0.99
smooth=0.1

batch_size = 4 #内存不够，128->64->32->16->8->4
loss_d=[[],[]]#储存点位信息，用于画图
loss_g=[[],[]]
counter = 0 #训练次数
epoch=400 #迭代次数
image_size=108 #裁剪图像的大小
image_shape=[64,64,3]


lam=0.1
nIndex = 1 #图像的迭代次数+++++++++++++++++++++++++++++++++++++++++++++++++++++ 


class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    # 卷积层1  卷积为shape：(batch_size, 32, 32, 64)
    self._conv1 = tf.keras.layers.Conv2D(64, 5, strides=2, padding='SAME', activation=None)
    self._activation1 = tf.keras.layers.LeakyReLU(alpha=alpha)
    
    # 卷积层2  卷积为shape：(batch_size, 16, 16, 128)
    self._conv2 = tf.keras.layers.Conv2D(128, 5, strides=2, padding='SAME', activation=None)
    self._bn1 = tf.keras.layers.BatchNormalization(scale=False)
    self._activation2 = tf.keras.layers.LeakyReLU(alpha=alpha)
    
    # 卷积层3  卷积为shape：(batch_size, 8, 8, 256)
    self._conv3 = tf.keras.layers.Conv2D(256, 5, strides=2, padding='SAME', activation=None)
    self._bn2 = tf.keras.layers.BatchNormalization(scale=False)
    self._activation3 = tf.keras.layers.LeakyReLU(alpha=alpha)
    
    # 卷积层4  卷积为shape：(batch_size, 4, 4, 512)
    self._conv4 = tf.keras.layers.Conv2D(512, 5, strides=2, padding='SAME', activation=None)
    self._bn3 = tf.keras.layers.BatchNormalization(scale=False)
    self._activation4 = tf.keras.layers.LeakyReLU(alpha=alpha)
    #全连接层
    self.fc1 = tf.keras.layers.Dense(units=512, activation=None)
    
    # BN + ReLU
    self.bn1 = tf.keras.layers.BatchNormalization(scale=False)
    self.activation1 = tf.keras.layers.Activation(activation='relu')
    
    # 转置卷积层1  转置卷积为shape：(batch_size, 8, 8, 256)
    self.transp_conv1 = tf.keras.layers.Conv2DTranspose(256, 5, strides=2, padding="SAME", activation=None)
    
    # BN + ReLU
    self.bn2 = tf.keras.layers.BatchNormalization(scale=False)
    self.activation2= tf.keras.layers.Activation(activation='relu')
    
    # 转置卷积层2  转置卷积为shape：(batch_size, 16, 16, 128)
    self.transp_conv2 = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding="SAME", activation=None)
    
    # BN + ReLU
    self.bn3 = tf.keras.layers.BatchNormalization(scale=False)
    self.activation3= tf.keras.layers.Activation(activation='relu')
    
   # 转置卷积层3  转置卷积为shape：(batch_size, 32, 32, 64)
    self.transp_conv3 = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding="SAME", activation=None)
    
    # BN + ReLU
    self.bn4 = tf.keras.layers.BatchNormalization(scale=False)
    self.activation4= tf.keras.layers.Activation(activation='relu')

    
    # 转置卷积层4  转置卷积为shape：(batch_size, 64, 64, 3)
    self.transp_conv4 = tf.keras.layers.Conv2DTranspose(3, 5, strides=2, padding="SAME", activation=None)
    self.out = tf.keras.layers.Activation(activation='tanh')
  #call函数让类可以作为函数被调用  
  def __call__(self,inputs, is_training):
    #print(inputs.shape)    
    _conv1 = self._conv1(inputs)
    _activation1 = self._activation1(_conv1)
    #print(_activation1.shape)
    
    _conv2 = self._conv2(_activation1)
    _bn1 = self._bn1(_conv2, training=is_training)
    _activation2 = self._activation2(_bn1)
    #print(_activation2.shape)    
    _conv3 = self._conv3(_activation2)
    _bn2 = self._bn2(_conv3, training=is_training)
    _activation3 = self._activation3(_bn2)
    #print(_activation3.shape)
    _conv4 = self._conv4(_activation3)
    _bn3 = self._bn3(_conv4, training=is_training)
    _activation4 = self._activation4(_bn3)
    #print(_activation4.shape)
    
    fc1 = self.fc1(_activation4)
    fc1_reshaped = tf.reshape(fc1, (-1,4,4,512))
    #print(fc1.shape)
    #print(fc1_reshaped.shape)
    bn1 = self.bn1(fc1_reshaped, training=is_training)
    activation1 = self.activation1(bn1)

    trans_conv1 = self.transp_conv1(activation1) 
    bn2 = self.bn2(trans_conv1, training=is_training)
    activation2 = self.activation2(bn2)

    transp_conv2 = self.transp_conv2(activation2) 
    bn3 = self.bn3(transp_conv2, training=is_training)
    activation3 = self.activation3(bn3)
    
    transp_conv3 = self.transp_conv3(activation3) 
    bn4 = self.bn4(transp_conv3, training=is_training)
    activation4 = self.activation4(bn4)
    
    transp_conv4 = self.transp_conv4(activation4) 
    output = self.out(transp_conv4)
    
    return output
    

# ## 构造判别器类

class Discriminator(tf.keras.Model):
  def __init__(self, alpha):
    super(Discriminator, self).__init__()
    # 卷积层1  卷积为shape：(batch_size, 32, 32, 64)
    self.conv1 = tf.keras.layers.Conv2D(64, 5, strides=2, padding='SAME', activation=None)
    self.activation1 = tf.keras.layers.LeakyReLU(alpha=alpha)
    
    # 卷积层2  卷积为shape：(batch_size, 16, 16, 128)
    self.conv2 = tf.keras.layers.Conv2D(128, 5, strides=2, padding='SAME', activation=None)
    self.bn1 = tf.keras.layers.BatchNormalization(scale=False)
    self.activation2 = tf.keras.layers.LeakyReLU(alpha=alpha)
    
    # 卷积层3  卷积为shape：(batch_size, 8, 8, 256)
    self.conv3 = tf.keras.layers.Conv2D(256, 5, strides=2, padding='SAME', activation=None)
    self.bn2 = tf.keras.layers.BatchNormalization(scale=False)
    self.activation3 = tf.keras.layers.LeakyReLU(alpha=alpha)
    
    # 卷积层4  卷积为shape：(batch_size, 4, 4, 512)
    self.conv4 = tf.keras.layers.Conv2D(512, 5, strides=2, padding='SAME', activation=None)
    self.bn3 = tf.keras.layers.BatchNormalization(scale=False)
    self.activation4 = tf.keras.layers.LeakyReLU(alpha=alpha)
    
    # 把输入拉成一维向量  卷积为shape：(batch_size*4*4*512)
    self.flatten = tf.keras.layers.Flatten()
    self.fc1 = tf.keras.layers.Dense(units=1, activation=None)
    self.out = tf.keras.layers.Activation(activation='sigmoid')
  
  def call(self, inputs, is_training):

    conv1 = self.conv1(inputs)
    activation1 = self.activation1(conv1)
    
    conv2 = self.conv2(activation1)
    bn1 = self.bn1(conv2, training=is_training)
    activation2 = self.activation2(bn1)
    
    conv3 = self.conv3(activation2)
    bn2 = self.bn2(conv3, training=is_training)
    activation3 = self.activation3(bn2)
    
    conv4 = self.conv4(activation3)
    bn3 = self.bn3(conv4, training=is_training)
    activation4 = self.activation4(bn3)
    
    flat = self.flatten(activation4)
    logits = self.fc1(flat)
    out = self.out(logits)
    return out, logits


 # 设置参数


# # 定义代价函数


def discriminator_loss(d_logits_real, d_logits_fake, smooth=0.1):
    
    #判别器两个代价函数
    #输入真图片，判断逼近1
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real) * (1 - smooth)))

    #输入假图片，判断逼近0
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss



def generator_loss(d_logits_fake, d_model_fake):
  
    #生成器一个代价函数
    #输入假图片，迷惑判别器判断逼近1
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    return g_loss


 # 定义优化器



def display_images(dataset, figsize=(4,4), denomalize=True):
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=figsize,)
    for ii, ax in enumerate(axes.flatten()):
        img = dataset[ii,:,:,:]
        if denomalize:
            img = ((img + 1)*255 / 2).astype(np.uint8) # Scale back to 0-255
        
        ax.imshow(img)
      
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# Helpers for image handling
def get_image(image_path, image_size, is_crop=True):

    return transform(Image.open(image_path), image_size, is_crop)

def save_images(images, image_path):
    for imgindex in range(images.shape[0]):

        imageio.imwrite(image_path+'output/'+str(imgindex)+'.jpg',images[imgindex])

def imread(path):
    
    return imageio.imread(path).astype(np.float)

def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def center_crop(x, crop_h, crop_w=None, resize_w=64): 
    if crop_w is None:
        crop_w = crop_h
    x=np.array(x)    
 
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return np.array(Image.fromarray(x[j:j+crop_h, i:i+crop_w]).resize((resize_w, resize_w)))

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


# 转换为较低的分辨率
def convert_to_lower_resolution():
    images=glob(os.path.join('cars_train\cars_train/','*.jpg'))
    i=0
    size=108,108
    for image in images:

        im=Image.open(image)
        im_resized=im.resize(size,Image.ANTIALIAS)
        im_resized.save("cars_train/"+str(i)+'.jpg')


 # 运行模型


# # 增加图像修复代价函数


def complete_Inpainting_loss(g_loss, mask, G, images,lam):

    #真图片未破损部分与假图片未破损部分的生成损失

    contextual_loss = tf.reduce_sum(
           #  tf.keras.layers.Flatten(
                #tf.abs((tf.multiply(mask,G) - tf.multiply(mask, images))))
                tf.abs(G -  images))
    #感知信息损失（保证全局结构性）
    perceptual_loss=g_loss
    complete_loss = contextual_loss + lam*perceptual_loss ##########################
    
    return complete_loss


# # 生成MASK矩阵


def generate_Mask(batch_size):
    scale=0.25 #遮挡部分占全部图片的百分比
    #遮挡图片的MASK矩阵
    mask=np.ones([batch_size]+image_shape).astype(np.float32)
    l=int(image_shape[0]*scale)
    u=int(image_shape[0]*(1.0-scale))
    mask[:,l:u,l:u,:]=0.0


    #取出破损部分的MASK矩阵
    scale=0.25
    imask=np.zeros([batch_size]+image_shape).astype(np.float32)
    l=int(image_shape[0]*scale)
    u=int(image_shape[0]*(1.0-scale))
    imask[:,l:u,l:u,:]=1.0
    
    return mask,imask



def show_loss(gen_loss,dis_loss):
    x1 = gen_loss[0]
    y1 = gen_loss[1]
    x2 = dis_loss[0]
    y2 = dis_loss[1]
    fig = plt.figure(figsize = (7,5))       #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
    #pl.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')
    # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
    p2 = pl.plot(x1, y1,'r-', label = u'gen_loss')
    pl.legend()
    #显示图例
    p3 = pl.plot(x2,y2, 'b-', label = u'dis_loss')
    pl.legend()
    pl.xlabel(u'iters')
    pl.ylabel(u'loss')
    plt.title('  gen_loss and dis_loss')
    # plot the box
    tx0 = 0
    tx1 = 2000
    #设置想放大区域的横坐标范围
    ty0 = 0.000
    ty1 = 0.12
    #设置想放大区域的纵坐标范围
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]
    pl.plot(sx,sy,"purple")
    axins = inset_axes(ax1, width=1.5, height=1.5, loc='right')
    #loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"
    axins.plot(x1,y1 , color='red', ls='-')
    axins.plot(x2,y2 , color='blue', ls='-')
    axins.axis([0,20000,0.000,0.12])
    plt.savefig('./doc/train_results_loss.png')
    #pl.show()

generator_net = Generator()
discriminator_net = Discriminator(alpha=alpha)

global_counter = tf.compat.v1.train.get_or_create_global_step()
generator_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
 # 处理数据和数据显示


checkpoint_dir = './training_checkpoints' #############################
checkpoint_dird ='./training_checkpoints/checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint_prefixp = os.path.join(checkpoint_dird, "ckptt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator_net,
                                 discriminator=discriminator_net)


sample_mask,sample_imask = generate_Mask(batch_size)
#缺损区域的噪音块
fake_input = tf.random.uniform(shape=([batch_size]+ image_shape),
                                        minval=-1.0, maxval=1.0, dtype=tf.float32)
fake_part = tf.multiply(fake_input,sample_imask)




#设置模式

is_training =False#训练模型还是使用模型
training_continue = True #True/False#是否继续训练
dcgan_start = True#是否开始调用


if is_training  ==True:
    if training_continue == True:
        print(tf.train.latest_checkpoint(checkpoint_dir))
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))#从检查点开始    
    else:
        pass
    
    #直接开始

    num_batch = (int)(len(datas)/batch_size)

    print(num_batch)
    for i in range(epoch):
            #随机打乱数据
            np.random.shuffle(datas)
            for ii in range(num_batch):

                print('第{}周期/{}的第{}batch'.format(i,epoch,ii))
                #因为数据过大，采用以batch为单位处理数据集的做法/patch张图片
                batch_files=datas[ii*batch_size:(ii+1)*batch_size]
                batch=[get_image(batch_file,image_size,is_crop=True) for batch_file in batch_files]
                batch_images=np.reshape(np.array(batch).astype(np.float32),[batch_size]+image_shape)
                #生成以batch为单位的随机噪声///输入为缺损图片
                real_part = tf.multiply(batch_images,sample_mask)
                fake_input = tf.add(fake_part,real_part)
                 

                with tf.GradientTape(persistent=True) as tape:
            
                    # 运行生成器
                    g_model = generator_net(fake_input, is_training=True)

                    # 输入真图片运行判别器
                    d_model_real, d_logits_real = discriminator_net(batch_images, is_training=True)

                    # 输入假图片运行判别器
                    d_model_fake, d_logits_fake = discriminator_net(g_model, is_training=True)


                    # 计算生成器的损失
                    gen_loss = generator_loss(d_logits_fake, d_model_fake)

                    print('生成器的损失gen_loss ={}'.format(gen_loss))
                    # 计算判别器的损失
                    dis_loss = discriminator_loss(d_logits_real, d_logits_fake, smooth)
                    print('判别器的损失dis_loss ={}'.format(dis_loss))

                    complete_loss = complete_Inpainting_loss(gen_loss, sample_mask, g_model, batch_images,lam)
                
                    discriminator_grads = tape.gradient(dis_loss, discriminator_net.variables)
                    generator_grads = tape.gradient(complete_loss, generator_net.variables)

                    loss_d[0].append(i*num_batch+ii)
                    loss_d[1].append(dis_loss)
                    loss_g[0].append(i*num_batch+ii)
                    loss_g[1].append(gen_loss)
                    #进行梯度更新
                    discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator_net.variables), global_step=global_counter)
                    generator_optimizer.apply_gradients(zip(generator_grads, generator_net.variables), global_step=global_counter)
                    generator_optimizer.apply_gradients(zip(generator_grads, generator_net.variables), global_step=global_counter)

            
                    counter += 1

            if (i + 1) % 50 == 0:

                # display_images(generated_samples.numpy())
                save_images(g_model,'./out/')
                checkpoint.save(file_prefix = checkpoint_prefix)
                        
                    #为变量计算梯度
    
    doc = open('doc/d.txt','w')
    print(loss_d,file=doc)
    doc.close()
    doc1 = open('doc/g.txt','w')
    print(loss_g,file=doc1)
    doc1.close()
    show_loss(loss_g,loss_d)
else:
    pass


# # 运行模型

batch_size =1
# 生成测试噪声输入
if dcgan_start == True:
    print(tf.train.latest_checkpoint(checkpoint_dird))
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dird))  #从检查点开始

    num_batch = (int)(len(datass)/batch_size)
    print(num_batch)
    np.random.shuffle(datass)  
    for i in range((int)(num_batch)):
        
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))  #从检查点开始生成人脸

        print('===================================第{}张图片/{}==================================='.format(i,num_batch))    
        #因为数据过大，采用以batch为单位处理数据集的做法
        batch_files=datass[i*batch_size:(i+1)*batch_size]   #  datas== sample_files
        batch=[get_image(batch_file,image_size,is_crop=True) for batch_file in batch_files]
        batch_images=np.reshape(np.array(batch).astype(np.float32),[batch_size]+image_shape)
        real_part = tf.multiply(batch_images,sample_mask)
        fake_input = tf.add(fake_part,real_part)
        m = 0
        v = 0
        for ii in range(nIndex):
            with tf.GradientTape(persistent=True) as tape:

                # 运行生成器
                g_model = generator_net(fake_input, is_training=False)
                # 输入假图片运行判别器
                d_model_fake, d_logits_fake = discriminator_net(g_model, is_training=False)
                # 计算生成器的损失
                fake_p = tf.multiply(g_model,sample_imask)
                real_p = tf.multiply(batch_images,sample_mask)
                complie = tf.add(fake_p,real_p)
                              
                #生成假图片
                plt.subplot(151)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(g_model[0].numpy())
       
                plt.subplot(152)//图片对比矩阵
                plt.imshow(batch_images[0])
                plt.xticks([])
                plt.yticks([])
                plt.subplot(153)
                plt.imshow(real_part[0])
                plt.xticks([])
                plt.yticks([])
                plt.subplot(154)
                plt.imshow(fake_input[0])
                plt.xticks([])
                plt.yticks([]) 
                plt.subplot(155)
                plt.imshow(complie[0])
                plt.xticks([])
                plt.yticks([])
                              
                #if i%100 == 0:
                plt.savefig('./out/'+str(i)+'-'+str(ii)+'.jpg')
                
            
        counter += 1

else:
    pass
