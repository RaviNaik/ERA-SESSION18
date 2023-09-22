# ERA-SESSION18 UNET & VAE Implementation in PyTorch Lightning


### Tasks:
1. :heavy_check_mark: Train your own UNet from scratch, you can use the dataset and strategy provided in this linkLinks to an external site.. However, you need to train it 4 times:
  - :heavy_check_mark: MP+Tr+BCE
  - :heavy_check_mark: MP+Tr+Dice Loss
  - :heavy_check_mark: StrConv+Tr+BCE
  - :heavy_check_mark: StrConv+Ups+Dice Loss
2. :heavy_check_mark: Design a variation of a VAE that:
  - :heavy_check_mark: takes in two inputs: an MNIST image, and its label (one hot encoded vector sent through an embedding layer)
  - :heavy_check_mark: Training as you would train a VAE
  - :heavy_check_mark: Now randomly send an MNIST image, but with a wrong label. Do this 25 times, and share what the VAE makes (25 images stacked in 1 image)!
  - :heavy_check_mark: Now do this for CIFAR10 and share 25 images (1 stacked image)!
  - :heavy_check_mark: Questions asked in the assignment are:
    - :heavy_check_mark: Share the MNIST notebook link ON GITHUB [100]
    - :heavy_check_mark: Share the CIFAR notebook link ON GITHUB [200]
    - :heavy_check_mark: Upload the 25 MNIST outputs PROPERLY labeled [250]
    - :heavy_check_mark: Upload the 25 CIFAR outputs PROPERLY labeled. [450]
   
### Model Summary
#### UNET
```python
  | Name                   | Type             | Params
-------------------------------------------------------------
0  | contract1              | ContractingBlock | 39.0 K
1  | contract1.conv_block   | Sequential       | 39.0 K
2  | contract1.conv_block.0 | Conv2d           | 1.8 K 
3  | contract1.conv_block.1 | BatchNorm2d      | 128   
4  | contract1.conv_block.2 | ReLU             | 0     
5  | contract1.conv_block.3 | Conv2d           | 36.9 K
6  | contract1.conv_block.4 | BatchNorm2d      | 128   
7  | contract1.conv_block.5 | ReLU             | 0     
8  | contract1.downsample   | MaxPool2d        | 0     
9  | contract2              | ContractingBlock | 221 K 
10 | contract2.conv_block   | Sequential       | 221 K 
11 | contract2.conv_block.0 | Conv2d           | 73.9 K
12 | contract2.conv_block.1 | BatchNorm2d      | 256   
13 | contract2.conv_block.2 | ReLU             | 0     
14 | contract2.conv_block.3 | Conv2d           | 147 K 
15 | contract2.conv_block.4 | BatchNorm2d      | 256   
16 | contract2.conv_block.5 | ReLU             | 0     
17 | contract2.downsample   | MaxPool2d        | 0     
18 | contract3              | ContractingBlock | 886 K 
19 | contract3.conv_block   | Sequential       | 886 K 
20 | contract3.conv_block.0 | Conv2d           | 295 K 
21 | contract3.conv_block.1 | BatchNorm2d      | 512   
22 | contract3.conv_block.2 | ReLU             | 0     
23 | contract3.conv_block.3 | Conv2d           | 590 K 
24 | contract3.conv_block.4 | BatchNorm2d      | 512   
25 | contract3.conv_block.5 | ReLU             | 0     
26 | contract3.downsample   | MaxPool2d        | 0     
27 | contract4              | ContractingBlock | 3.5 M 
28 | contract4.conv_block   | Sequential       | 3.5 M 
29 | contract4.conv_block.0 | Conv2d           | 1.2 M 
30 | contract4.conv_block.1 | BatchNorm2d      | 1.0 K 
31 | contract4.conv_block.2 | ReLU             | 0     
32 | contract4.conv_block.3 | Conv2d           | 2.4 M 
33 | contract4.conv_block.4 | BatchNorm2d      | 1.0 K 
34 | contract4.conv_block.5 | ReLU             | 0     
35 | contract4.downsample   | MaxPool2d        | 0     
36 | contract5              | ContractingBlock | 14.2 M
37 | contract5.conv_block   | Sequential       | 14.2 M
38 | contract5.conv_block.0 | Conv2d           | 4.7 M 
39 | contract5.conv_block.1 | BatchNorm2d      | 2.0 K 
40 | contract5.conv_block.2 | ReLU             | 0     
41 | contract5.conv_block.3 | Conv2d           | 9.4 M 
42 | contract5.conv_block.4 | BatchNorm2d      | 2.0 K 
43 | contract5.conv_block.5 | ReLU             | 0     
44 | contract5.downsample   | MaxPool2d        | 0     
45 | expand1                | ExpandingBlock   | 9.2 M 
46 | expand1.conv_block     | Sequential       | 7.1 M 
47 | expand1.conv_block.0   | Conv2d           | 4.7 M 
48 | expand1.conv_block.1   | BatchNorm2d      | 1.0 K 
49 | expand1.conv_block.2   | ReLU             | 0     
50 | expand1.conv_block.3   | Conv2d           | 2.4 M 
51 | expand1.conv_block.4   | BatchNorm2d      | 1.0 K 
52 | expand1.conv_block.5   | ReLU             | 0     
53 | expand1.upsample       | ConvTranspose2d  | 2.1 M 
54 | expand2                | ExpandingBlock   | 2.3 M 
55 | expand2.conv_block     | Sequential       | 1.8 M 
56 | expand2.conv_block.0   | Conv2d           | 1.2 M 
57 | expand2.conv_block.1   | BatchNorm2d      | 512   
58 | expand2.conv_block.2   | ReLU             | 0     
59 | expand2.conv_block.3   | Conv2d           | 590 K 
60 | expand2.conv_block.4   | BatchNorm2d      | 512   
61 | expand2.conv_block.5   | ReLU             | 0     
62 | expand2.upsample       | ConvTranspose2d  | 524 K 
63 | expand3                | ExpandingBlock   | 574 K 
64 | expand3.conv_block     | Sequential       | 443 K 
65 | expand3.conv_block.0   | Conv2d           | 295 K 
66 | expand3.conv_block.1   | BatchNorm2d      | 256   
67 | expand3.conv_block.2   | ReLU             | 0     
68 | expand3.conv_block.3   | Conv2d           | 147 K 
69 | expand3.conv_block.4   | BatchNorm2d      | 256   
70 | expand3.conv_block.5   | ReLU             | 0     
71 | expand3.upsample       | ConvTranspose2d  | 131 K 
72 | expand4                | ExpandingBlock   | 143 K 
73 | expand4.conv_block     | Sequential       | 110 K 
74 | expand4.conv_block.0   | Conv2d           | 73.8 K
75 | expand4.conv_block.1   | BatchNorm2d      | 128   
76 | expand4.conv_block.2   | ReLU             | 0     
77 | expand4.conv_block.3   | Conv2d           | 36.9 K
78 | expand4.conv_block.4   | BatchNorm2d      | 128   
79 | expand4.conv_block.5   | ReLU             | 0     
80 | expand4.upsample       | ConvTranspose2d  | 32.8 K
81 | final_conv             | Conv2d           | 65    
-------------------------------------------------------------
31.0 M    Trainable params
0         Non-trainable params
31.0 M    Total params
124.174   Total estimated model params size (MB)
```
##### UNET - OXford Pet Dataset Samples
![image](https://github.com/RaviNaik/ERA-SESSION18/assets/23289802/f8adccb1-71b6-442d-9b6a-f04c82fe1b69)

##### MaxPool + ConvTranspose + Dice Loss
**Training log**

**RESULTS**
![image](https://github.com/RaviNaik/ERA-SESSION18/assets/23289802/1b023fd7-a49a-4594-b0df-6dad96f43b8a)


##### MaxPool + ConvTranspose + BCE Loss
**Training log**

**RESULTS**

##### StrConv + ConvTranspose + BCE Loss
**Training log**

**RESULTS**

##### StrConv + Upsample + Dice Loss
**Training log**

**RESULTS**

#### VAE

##### Model Summary
```python

   | Name             | Type              | Params
--------------------------------------------------------
0  | prepare          | Conv2d            | 30    
1  | prepare_out      | Sequential        | 256   
2  | prepare_out.0    | ConvTranspose2d   | 228   
3  | prepare_out.1    | Conv2d            | 28    
4  | encoder          | ResNetEncoder     | 11.2 M
5  | encoder.conv1    | Conv2d            | 1.7 K 
6  | encoder.bn1      | BatchNorm2d       | 128   
7  | encoder.relu     | ReLU              | 0     
8  | encoder.maxpool  | MaxPool2d         | 0     
9  | encoder.layer1   | Sequential        | 147 K 
10 | encoder.layer1.0 | EncoderBlock      | 74.0 K
11 | encoder.layer1.1 | EncoderBlock      | 74.0 K
12 | encoder.layer2   | Sequential        | 525 K 
13 | encoder.layer2.0 | EncoderBlock      | 230 K 
14 | encoder.layer2.1 | EncoderBlock      | 295 K 
15 | encoder.layer3   | Sequential        | 2.1 M 
16 | encoder.layer3.0 | EncoderBlock      | 919 K 
17 | encoder.layer3.1 | EncoderBlock      | 1.2 M 
18 | encoder.layer4   | Sequential        | 8.4 M 
19 | encoder.layer4.0 | EncoderBlock      | 3.7 M 
20 | encoder.layer4.1 | EncoderBlock      | 4.7 M 
21 | encoder.avgpool  | AdaptiveAvgPool2d | 0     
22 | combine          | Linear            | 267 K 
23 | decoder          | ResNetDecoder     | 8.6 M 
24 | decoder.linear   | Linear            | 2.1 M 
25 | decoder.layer1   | Sequential        | 4.9 M 
26 | decoder.layer1.0 | DecoderBlock      | 3.7 M 
27 | decoder.layer1.1 | DecoderBlock      | 1.2 M 
28 | decoder.layer2   | Sequential        | 1.2 M 
29 | decoder.layer2.0 | DecoderBlock      | 918 K 
30 | decoder.layer2.1 | DecoderBlock      | 295 K 
31 | decoder.layer3   | Sequential        | 303 K 
32 | decoder.layer3.0 | DecoderBlock      | 229 K 
33 | decoder.layer3.1 | DecoderBlock      | 74.0 K
34 | decoder.layer4   | Sequential        | 147 K 
35 | decoder.layer4.0 | DecoderBlock      | 74.0 K
36 | decoder.layer4.1 | DecoderBlock      | 74.0 K
37 | decoder.upscale  | Interpolate       | 0     
38 | decoder.upscale1 | Interpolate       | 0     
39 | decoder.conv1    | Conv2d            | 1.7 K 
40 | fc_mu            | Linear            | 131 K 
41 | fc_var           | Linear            | 131 K 
--------------------------------------------------------
20.3 M    Trainable params
0         Non-trainable params
20.3 M    Total params
81.301    Total estimated model params size (MB)
```
##### VAE MNIST Data samples
![image](https://github.com/RaviNaik/ERA-SESSION18/assets/23289802/8c568a1f-a4ad-4a50-9838-1abd907c27d2)

##### VAE MNIST Training log
![image](https://github.com/RaviNaik/ERA-SESSION18/assets/23289802/e4427bc3-59a1-42fe-b54e-8768a55c7df0)
**TensorBoard Plots**
![image](https://github.com/RaviNaik/ERA-SESSION18/blob/main/vae_mnist_tb.png)
##### VAE MNIST Results
![image](https://github.com/RaviNaik/ERA-SESSION18/assets/23289802/2b2273e3-4cd3-4b05-b3a8-b7064999a396)

##### VAE CIFAR Data samples
![image](https://github.com/RaviNaik/ERA-SESSION18/assets/23289802/be4af0a4-fa8a-41d1-8097-d63ed475ff2c)

##### VAE CIFAR Training log
![image](https://github.com/RaviNaik/ERA-SESSION18/assets/23289802/ac8fc51c-6ea8-44a9-b9f8-bda4447b9901)
**TensorBoard Plots**
![image](https://github.com/RaviNaik/ERA-SESSION18/blob/main/vae_cifar_tb.png)

##### VAE CIFAR Results
![image](https://github.com/RaviNaik/ERA-SESSION18/assets/23289802/700992ad-4593-4baa-a352-065d68e34324)





