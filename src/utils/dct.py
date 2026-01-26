import torch_dct as dct
import torch
import torch.nn.functional as F


class ImageDCT():
    def __init__(self, block_size):
        self.block_size = block_size

    def blockify(self, image, n_blocks):
        '''image: BxCxHxW'''
        return F.unfold(image, kernel_size=self.block_size, stride=self.block_size).permute(0,2,1).reshape(-1, n_blocks, self.block_size, self.block_size)

    def unblockify(self, image_block, img_size, n_blocks):
        return F.fold(image_block.reshape(-1, n_blocks, self.block_size**2).permute(0, 2, 1), output_size=(img_size[0], img_size[1]), kernel_size=self.block_size, stride=self.block_size)

    def dct_2d(self, img):
        h, w = img.shape[2], img.shape[3]
        n_blocks = (h//self.block_size)*(w//self.block_size)
        img_block = torch.cat((self.blockify(img[:,:1], n_blocks), 
                    self.blockify(img[:,1:2], n_blocks)), dim=1)
        dct_block = dct.dct_2d(img_block, norm='ortho')
        dct_image = torch.cat((self.unblockify(dct_block[:, 0:n_blocks], [h, w], n_blocks),
                                self.unblockify(dct_block[:, n_blocks:2*n_blocks], [h, w], n_blocks)), dim=1)
        dc_ac = F.pixel_unshuffle(dct_image, self.block_size)
        return dc_ac
    
    def idct_2d(self, dct_img):
        img_block = F.pixel_shuffle(dct_img, self.block_size)
        h, w = img_block.shape[2], img_block.shape[3]
        n_blocks = (h//self.block_size)*(w//self.block_size)
        img_block = torch.cat((self.blockify(img_block[:,:1], n_blocks), 
                    self.blockify(img_block[:,1:2], n_blocks)), dim=1)
        idct_block = dct.idct_2d(img_block, norm='ortho')
        idct_image = torch.cat((self.unblockify(idct_block[:, 0:n_blocks], [h, w], n_blocks),
                                self.unblockify(idct_block[:, n_blocks:2*n_blocks], [h, w], n_blocks)), dim=1)
        return idct_image


class dctLayer(torch.nn.Module):
    def __init__(self, block_size):
        super(dctLayer, self).__init__()
        self.block_size = block_size
        self.dct_util = ImageDCT(block_size)

    def forward(self, x):
        return self.dct_util.dct_2d(x)
    

class idctLayer(torch.nn.Module):
    def __init__(self, block_size):
        super(idctLayer, self).__init__()
        self.block_size = block_size
        self.dct_util = ImageDCT(block_size)

    def forward(self, x):
        return self.dct_util.idct_2d(x)