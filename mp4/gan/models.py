import torch
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
    
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv1 = nn.Conv2d(3, 128, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, 4, stride=1, padding=1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.bn2(self.conv2(x)))
        x = self.leakyrelu(self.bn3(self.conv3(x)))
        x = self.leakyrelu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x
        
        ##########       END      ##########
        #return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        #self.fc = nn.Linear(noise_dim, 1024*4*4)

        self.conv_transpose_1 = nn.ConvTranspose2d(self.noise_dim, 1024, kernel_size=4, stride=2, padding=0)
        self.batch_norm_1 = nn.BatchNorm2d(1024)

        self.conv_transpose_2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(512)

        self.conv_transpose_3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(256)

        self.conv_transpose_4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.batch_norm_4 = nn.BatchNorm2d(128)

        self.conv_transpose_5 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        #x = x.view(-1, 1024, 4, 4)
        # x = self.fc(x)
        # x = x.view(-1, 1024, 4, 4)

        x = self.leakyrelu(self.batch_norm_1(self.conv_transpose_1(x)))
        x = self.leakyrelu(self.batch_norm_2(self.conv_transpose_2(x)))
        x = self.leakyrelu(self.batch_norm_3(self.conv_transpose_3(x)))
        x = self.leakyrelu(self.batch_norm_4(self.conv_transpose_4(x)))
        x = self.conv_transpose_5(x)
        x = self.tanh(x)
        
        ##########       END      ##########
        
        return x
    

