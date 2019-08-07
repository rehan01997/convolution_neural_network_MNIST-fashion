{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint8 = np.dtype([(\"qint8\", np.int8, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint8 = np.dtype([(\"quint8\", np.uint8, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint16 = np.dtype([(\"qint16\", np.int16, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint16 = np.dtype([(\"quint16\", np.uint16, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint32 = np.dtype([(\"qint32\", np.int32, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  np_resource = np.dtype([(\"resource\", np.ubyte, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint8 = np.dtype([(\"qint8\", np.int8, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint8 = np.dtype([(\"quint8\", np.uint8, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint16 = np.dtype([(\"qint16\", np.int16, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint16 = np.dtype([(\"quint16\", np.uint16, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint32 = np.dtype([(\"qint32\", np.int32, 1)])\n",
      "c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  np_resource = np.dtype([(\"resource\", np.ubyte, 1)])\n",
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import tensorflow\n",
    "from keras import models\n",
    "from keras.layers import Dense,Convolution2D,MaxPooling2D,Input,Flatten,Dropout\n",
    "import matplotlib.pyplot as plt\n",
    "from keras.utils.np_utils import to_categorical"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "train=pd.read_csv(r\"C:\\Users\\zeesh\\OneDrive\\Desktop\\fashion\\fashion-mnist_train.csv\")\n",
    "test=pd.read_csv(r\"C:\\Users\\zeesh\\OneDrive\\Desktop\\fashion\\fashion-mnist_test.csv\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(60000, 785) (10000, 785)\n"
     ]
    }
   ],
   "source": [
    "print(train.shape,test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>pixel1</th>\n",
       "      <th>pixel2</th>\n",
       "      <th>pixel3</th>\n",
       "      <th>pixel4</th>\n",
       "      <th>pixel5</th>\n",
       "      <th>pixel6</th>\n",
       "      <th>pixel7</th>\n",
       "      <th>pixel8</th>\n",
       "      <th>pixel9</th>\n",
       "      <th>...</th>\n",
       "      <th>pixel775</th>\n",
       "      <th>pixel776</th>\n",
       "      <th>pixel777</th>\n",
       "      <th>pixel778</th>\n",
       "      <th>pixel779</th>\n",
       "      <th>pixel780</th>\n",
       "      <th>pixel781</th>\n",
       "      <th>pixel782</th>\n",
       "      <th>pixel783</th>\n",
       "      <th>pixel784</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30</td>\n",
       "      <td>43</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 785 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   label  pixel1  pixel2  pixel3  pixel4  pixel5  pixel6  pixel7  pixel8  \\\n",
       "0      2       0       0       0       0       0       0       0       0   \n",
       "1      9       0       0       0       0       0       0       0       0   \n",
       "2      6       0       0       0       0       0       0       0       5   \n",
       "3      0       0       0       0       1       2       0       0       0   \n",
       "4      3       0       0       0       0       0       0       0       0   \n",
       "\n",
       "   pixel9  ...  pixel775  pixel776  pixel777  pixel778  pixel779  pixel780  \\\n",
       "0       0  ...         0         0         0         0         0         0   \n",
       "1       0  ...         0         0         0         0         0         0   \n",
       "2       0  ...         0         0         0        30        43         0   \n",
       "3       0  ...         3         0         0         0         0         1   \n",
       "4       0  ...         0         0         0         0         0         0   \n",
       "\n",
       "   pixel781  pixel782  pixel783  pixel784  \n",
       "0         0         0         0         0  \n",
       "1         0         0         0         0  \n",
       "2         0         0         0         0  \n",
       "3         0         0         0         0  \n",
       "4         0         0         0         0  \n",
       "\n",
       "[5 rows x 785 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "train=train.values\n",
    "test=test.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 785)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(60000, 784) (60000, 1)\n",
      "(10000, 784) (10000, 1)\n"
     ]
    }
   ],
   "source": [
    "xtrain=train[:,1:]\n",
    "ytrain=train[:,:1]\n",
    "xtest=test[:,1:]\n",
    "ytest=test[:,:1]\n",
    "print(xtrain.shape,ytrain.shape)\n",
    "print(xtest.shape,ytest.shape)\n",
    "xtrain=xtrain/255.0\n",
    "xtest=xtest/255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "xtrain=xtrain.reshape((-1,28,28,1))\n",
    "ytrain=to_categorical(ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 1.], dtype=float32)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.unique(ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x185030ae828>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(xtrain[1].reshape(28,28))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Logging before flag parsing goes to stderr.\n",
      "W0807 13:15:20.647849  5720 deprecation_wrapper.py:119] From c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "model=models.Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0807 13:15:20.866760  5720 deprecation_wrapper.py:119] From c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.\n",
      "\n",
      "W0807 13:15:20.870742  5720 deprecation_wrapper.py:119] From c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.\n",
      "\n",
      "W0807 13:15:20.925765  5720 deprecation_wrapper.py:119] From c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:133: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.\n",
      "\n",
      "W0807 13:15:20.941748  5720 deprecation.py:506] From c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.\n",
      "W0807 13:15:20.970769  5720 deprecation_wrapper.py:119] From c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:3976: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))\n",
    "model.add(Convolution2D(64,(5,5),activation='relu'))\n",
    "model.add(Dropout(0.25))\n",
    "model.add(MaxPooling2D(2,2))\n",
    "model.add(Convolution2D(32,(5,5),activation='relu'))\n",
    "model.add(Convolution2D(8,(5,5),activation='relu'))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(10,activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 22, 22, 64)        51264     \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 22, 22, 64)        0         \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 11, 11, 64)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_3 (Conv2D)            (None, 7, 7, 32)          51232     \n",
      "_________________________________________________________________\n",
      "conv2d_4 (Conv2D)            (None, 3, 3, 8)           6408      \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 72)                0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 10)                730       \n",
      "=================================================================\n",
      "Total params: 109,954\n",
      "Trainable params: 109,954\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0807 13:15:21.455450  5720 deprecation_wrapper.py:119] From c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\keras\\optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.\n",
      "\n",
      "W0807 13:15:21.501477  5720 deprecation_wrapper.py:119] From c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:3295: The name tf.log is deprecated. Please use tf.math.log instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0807 13:15:22.994543  5720 deprecation.py:323] From c:\\users\\zeesh\\anaconda3\\envs\\tensorflow_cpu\\lib\\site-packages\\tensorflow\\python\\ops\\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.where in 2.0, which has the same broadcast rule as np.where\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 48000 samples, validate on 12000 samples\n",
      "Epoch 1/10\n",
      "48000/48000 [==============================] - 166s 3ms/step - loss: 0.7865 - acc: 0.7040 - val_loss: 0.5771 - val_acc: 0.8026\n",
      "Epoch 2/10\n",
      "48000/48000 [==============================] - 146s 3ms/step - loss: 0.4660 - acc: 0.8330 - val_loss: 0.4793 - val_acc: 0.8276\n",
      "Epoch 3/10\n",
      "48000/48000 [==============================] - 157s 3ms/step - loss: 0.4016 - acc: 0.8555 - val_loss: 0.3897 - val_acc: 0.8623\n",
      "Epoch 4/10\n",
      "48000/48000 [==============================] - 153s 3ms/step - loss: 0.3549 - acc: 0.8744 - val_loss: 0.3748 - val_acc: 0.8739\n",
      "Epoch 5/10\n",
      "48000/48000 [==============================] - 156s 3ms/step - loss: 0.3255 - acc: 0.8824 - val_loss: 0.3434 - val_acc: 0.8814\n",
      "Epoch 6/10\n",
      "48000/48000 [==============================] - 156s 3ms/step - loss: 0.3051 - acc: 0.8902 - val_loss: 0.3619 - val_acc: 0.8671\n",
      "Epoch 7/10\n",
      "48000/48000 [==============================] - 156s 3ms/step - loss: 0.2877 - acc: 0.8962 - val_loss: 0.3239 - val_acc: 0.8887\n",
      "Epoch 8/10\n",
      "48000/48000 [==============================] - 162s 3ms/step - loss: 0.2728 - acc: 0.9017 - val_loss: 0.3146 - val_acc: 0.8937\n",
      "Epoch 9/10\n",
      "48000/48000 [==============================] - 167s 3ms/step - loss: 0.2593 - acc: 0.9068 - val_loss: 0.2866 - val_acc: 0.8999\n",
      "Epoch 10/10\n",
      "48000/48000 [==============================] - 155s 3ms/step - loss: 0.2496 - acc: 0.9092 - val_loss: 0.2855 - val_acc: 0.9003\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x18503105ba8>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(xtrain,ytrain,batch_size=256,epochs=10,validation_split=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "xtest=xtest.reshape((-1,28,28,1))\n",
    "ytest=to_categorical(ytest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x1850589c780>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(xtest[0].reshape(28,28))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10000/10000 [==============================] - 7s 713us/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.27312083475589755, 0.9044]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.evaluate(xtest,ytest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
