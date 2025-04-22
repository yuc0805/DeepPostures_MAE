# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import to_2tuple

from util.pos_embed import get_2d_sincos_pos_embed

from util.covariance import spectral_mask
from maxcut import max_cut_mask

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=[3, 100], patch_size=[1,5], in_chans=1, 
                 embed_dim=768, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 is_eval=False,
                 ): # changed - added alt
        super().__init__()

        self.in_chans = in_chans #changed - added
        self.img_size = img_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patch_size = self.patch_embed.patch_size
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.is_eval=is_eval
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        self.mse_loss = nn.MSELoss()

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [1, int(self.patch_embed.num_patches)], cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], [1, int(self.patch_embed.num_patches)], cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size[0]*patch_size[1] *in_chans) # changed
        """
        p = self.patch_embed.patch_size # we are dividing by W
        
        assert imgs.shape[3] % p[1] == 0    # we're trying to do rectangular input image, therefore W%p[1] should be 0

        h = imgs.shape[2] // p[0]  # changed  H/p0 = 6/1 = 6
        w = imgs.shape[3] // p[1]  # changed  W/p1 = 800/80 = 10
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p[0], w, p[1]))  # changed 3-> self.in_chans
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p[0]*p[1] * self.in_chans))   # changed 3-> self.in_chans
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size[0]*patch_size[1]*in_chans)    patch_size[0]*patch_size[1] # changed
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size
        #img_size=[6, 800], patch_size=[1,80], in_chans=1

        h = self.img_size[0] // p[0] # changed 
        w = int(x.shape[1]/h)  #281/1 = 281
        x = x.reshape(shape=(x.shape[0], h, w, p[0], p[1], self.in_chans))  # (21, 1, 281, 1, 64, 3 ) # changed 3-> self.in_chans
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p[0], w * p[1]))  #(21, 3, 1*1, 281*64) # changed 3-> self.in_chans

        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim  = [21, 281, 192] = [N, h * w, p[0]*p[1] * self.in_chans]
        len_keep = int(L * (1 - mask_ratio))
        
  
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] #x.device
    
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

            
        return x_masked, mask, ids_restore
    
    def chunked_masking(self, x, mask_c_prob=0.8, mask_r_prob=0):
        """
        Perform chunked masking on 2D inputs with consecutive masked regions.

        Args:
            x (Tensor): Input tensor of shape [N, L, D], where L is the sequence length.
            mask_c_prob (float): Probability of masking along columns (time axis).
            mask_r_prob (float): Probability of masking along rows (frequency axis).

        Returns:
            x_masked (Tensor): Masked input tensor.
            mask (Tensor): Binary mask indicating masked positions (1 for masked, 0 for unmasked).
            ids_restore (Tensor): Indices for restoring the original input order.
        """
        N, L, D = x.shape  # batch, length, dim

        ph, pw = self.patch_embed.patch_size  # [1, 20]
        h, w = self.img_size  # [6, 200]
        T, F = h // ph, w // pw  # Number of patches along each axis

        len_mask_t = int(T * mask_c_prob)
        len_mask_f = int(F * mask_r_prob)

        # Generate consecutive masked indices along the time axis
        mask_t = torch.zeros((N, T), device=x.device)
        for i in range(N):
            start_t = torch.randint(0, T - len_mask_t + 1, (1,), device=x.device)
            mask_t[i, start_t : start_t + len_mask_t] = 1  # Masked positions are 1

        # Generate consecutive masked indices along the frequency axis
        mask_f = torch.zeros((N, F), device=x.device)
        for i in range(N):
            start_f = torch.randint(0, F - len_mask_f + 1, (1,), device=x.device)
            mask_f[i, start_f : start_f + len_mask_f] = 1  # Masked positions are 1

        # Combine masks
        mask = mask_t.unsqueeze(2) + mask_f.unsqueeze(1)  # N, T, F
        mask = torch.clamp(mask, 0, 1)  # Ensure mask values are either 0 or 1

        # Flatten mask
        mask_flat = mask.view(N, -1)

        # Compute restore indices
        ids_restore = torch.argsort(-mask_flat, dim=1)  # Masked items come first
        num_masked = mask_flat.sum(dim=1).to(torch.long)  # Number of masked positions
        ids_keep = torch.stack([ids_restore[i, num_masked[i]:] for i in range(N)])

        # Apply mask to the input
        x = x.view(N, T, F, D)
        x_masked = torch.gather(
            x.view(N, -1, D), dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D).to(torch.long)
        )

        return x_masked, mask_flat, ids_restore

    def forecast_masking(self, x, mask_c_prob=0.8, mask_r_prob=0):
        """
        Perform forecast masking on 2D inputs where masking on the time axis starts from the beginning.

        Args:
            x (Tensor): Input tensor of shape [N, L, D], where L is the sequence length.
            mask_c_prob (float): Fraction of time patches to mask (time axis).
            mask_r_prob (float): Fraction of frequency patches to mask (frequency axis).

        Returns:
            x_masked (Tensor): Masked input tensor.
            mask (Tensor): Binary mask indicating masked positions (1 for masked, 0 for unmasked).
            ids_restore (Tensor): Indices for restoring the original input order.
        """
        N, L, D = x.shape  # batch, length, dim

        ph, pw = self.patch_embed.patch_size  # e.g., [1, 20]
        h, w = self.img_size  # e.g., [6, 200]
        T, F = h // ph, w // pw  # Number of patches along time and frequency

        len_mask_t = int(T * mask_c_prob)
        len_mask_f = int(F * mask_r_prob)

        # For forecast masking, mask the time axis starting from index 0
        mask_t = torch.zeros((N, T), device=x.device)
        mask_t[:, :len_mask_t] = 1  # Always mask the first len_mask_t time patches

        # For the frequency axis, we use random masking if mask_r_prob > 0.
        mask_f = torch.zeros((N, F), device=x.device)
        for i in range(N):
            if len_mask_f > 0:
                start_f = torch.randint(0, F - len_mask_f + 1, (1,), device=x.device)
                mask_f[i, start_f : start_f + len_mask_f] = 1

        # Combine the time and frequency masks. A patch is masked if either axis is masked.
        mask = mask_t.unsqueeze(2) + mask_f.unsqueeze(1)  # shape [N, T, F]
        mask = torch.clamp(mask, 0, 1)  # Ensure values are 0 or 1

        # Flatten the mask to [N, T*F]
        mask_flat = mask.view(N, -1)

        # Compute restore indices so that masked patches come first.
        ids_restore = torch.argsort(-mask_flat, dim=1)
        num_masked = mask_flat.sum(dim=1).to(torch.long)  # Number of masked patches per sample
        ids_keep = torch.stack([ids_restore[i, num_masked[i]:] for i in range(N)])

        # Apply the mask to the input.
        x = x.view(N, T, F, D)
        x_masked = torch.gather(
            x.view(N, -1, D),
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D).to(torch.long)
        )

        return x_masked, mask_flat, ids_restore


    def random_masking_2d(self, x, mask_c_prob=0, mask_r_prob=0.75):
        """
        2D: CWT_imgs (msking row and column under mask_r_prob and mask_c_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        
        ph,pw = self.patch_embed.patch_size # [1,20]
        h,w = self.img_size # [6,200]
        T,F = h//ph, w//pw
        

        #x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_c_prob))
        len_keep_f = int(F * (1 - mask_r_prob))

        #print(f'{len_keep_t} patchs along column(time axis), {len_keep_f} patches along row(F axis).')

        # print('len_keep_t',len_keep_t)
        # print('len_keep_f',len_keep_f)
        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1) # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1) # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f) # N, T, F

        # get masked x
        id2res=torch.Tensor(list(range(N*T*F))).reshape(N,T,F).to(x.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore

    # For visulization
    def fixed_masking(self, x, noise):
        """
        Perform masking using a fixed mask. The mask should have the same sequence length as x.
        x: [N, L, D], sequence
        fixed_mask: [N, L], binary mask where 0 means keep and 1 means remove
        """
        noise = noise.reshape(shape=(-1,10))
        x = x.reshape(shape=(-1,10,768))

        N, L, D = x.shape  # batch, length, dim  = [21, 281, 192] = [N, h * w, p[0]*p[1] * self.in_chans]
        len_keep = int(L * (1 - 0.75))
        
        #noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] #x.device
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = x_masked.reshape(shape=(1,-1,768))
        mask = mask.reshape(shape=(1,-1))
        # FIXME: WHat can I do with ids_restore
        row_indices = torch.arange(ids_restore.size(0), device=x.device).unsqueeze(1) * 10  # Multiply by stride (10)
        ids_restore_flattened = (ids_restore + row_indices).flatten().unsqueeze(0)

        return x_masked, mask, ids_restore_flattened

    def forward_encoder(self, x, mask_ratio, 
                        var_mask_ratio=0, time_mask_ratio=0,
                        masking_scheme='None', max_iter = 1000,
                        ):

        # embed patches
        x = self.patch_embed(x) # bs, num_p,  embed_dim

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # masking: length -> length * mask_ratio
        if self.is_eval:
            if masking_scheme == 'random_imputation':
                x, mask, ids_restore = self.random_masking(x, mask_ratio=mask_ratio)
            elif masking_scheme == 'temporal_imputation':
                x, mask, ids_restore = self.chunked_masking(x, 0,mask_ratio)
            elif masking_scheme == 'sensor_imputation':
                x, mask, ids_restore = self.random_masking_2d(x, mask_ratio, 0)
            elif masking_scheme == 'forecasting':
                x, mask, ids_restore = self.forecast_masking(x, 0,mask_ratio)
            elif mask_ratio <= 0: 
                x, mask, ids_restore = self.random_masking_2d(x, var_mask_ratio, time_mask_ratio)
            ##########################################################################################
            elif masking_scheme == 'custom_sync':
                x, mask, ids_restore = self.systematic_masking_2d(x, mask_c_prob=0, mask_r_prob=0.7)
            else: 
                x, mask, ids_restore = self.random_masking(x, mask_ratio=mask_ratio)
            ########################################################################################
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio=mask_ratio)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
       
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    #def forward_loss(self, imgs, pred, mask):
    #    """
    #    imgs: [N, 3, H, W]
    #    pred: [N, L, p*p*3]
    #    mask: [N, L], 0 is keep, 1 is remove, 
    #    """
    #    target = self.patchify(imgs)
    #    if self.norm_pix_loss:
    #        mean = target.mean(dim=-1, keepdim=True)
    #        var = target.var(dim=-1, keepdim=True)
    #        target = (target - mean) / (var + 1.e-6)**.5
    #
    #    loss = (pred - target) ** 2
    #    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    #
    #    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    #    return loss
    
    def forward_loss(self, imgs, pred, mask):
       """
        imgs: bs x nvar x L
        pred: bs x nvar x num_patches x patch_size
        mask: bs x num_patches 

       """
       # calculate loss for all time_step
       target = self.patchify(imgs) # bs x nvar x num_patches x patch_size
       loss = self.mse_loss(pred, target)

        # Only reconstruct mask patches
        #loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

       return loss
    
    def forward(self, imgs,  mask_ratio=0.75,
                var_mask_ratio=0,time_mask_ratio=0,
                masking_scheme=False,
                max_iter = 1000):

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio,
                                                         var_mask_ratio,
                                                         time_mask_ratio,
                                                         masking_scheme,
                                                         max_iter)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

#################
    def random_masking_2d_views(self, x, mask_c_prob=0, 
                          mask_r_prob=0.75):
        """
        """
        N, L, D = x.shape  # batch, length, dim
        
        ph,pw = self.patch_embed.patch_size # [1,20]
        h,w = self.img_size # [6,200]
        T,F = h//ph, w//pw
        

        #x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_c_prob))
        len_keep_f = int(F * (1 - mask_r_prob))

        #print(f'{len_keep_t} patchs along column(time axis), {len_keep_f} patches along row(F axis).')

        # print('len_keep_t',len_keep_t)
        # print('len_keep_f',len_keep_f)
        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1) # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1) # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f) # N, T, F

        # get masked x
        id2res=torch.Tensor(list(range(N*T*F))).reshape(N,T,F).to(x.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        ## get the makse view of x (only contain the tokens that got mask out)
        ids_masked=id2res2.flatten(start_dim=1)[:,len_keep_f*len_keep_t:]
        masked_view = torch.gather(x, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, D))


        return x_masked, mask, ids_restore, masked_view

    def random_masking_views(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim  = [21, 281, 192] = [N, h * w, p[0]*p[1] * self.in_chans]
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] #x.device
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Find the masked set tokens. 
        ids_masked = ids_shuffle[:, len_keep:]
        masked_view = torch.gather(x, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, masked_view
    
    def random_masking_views(self, x, mask_ratio):
        """
        Perform per-sample Importance score masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim  = [21, 281, 192] = [N, h * w, p[0]*p[1] * self.in_chans]
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] #x.device
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Find the masked set tokens. 
        ids_masked = ids_shuffle[:, len_keep:]
        masked_view = torch.gather(x, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, masked_view
    
    def bases_view(self, x,  
                   mask_ratio=0.75,var_mask_ratio=0,
                   time_mask_ratio=0,masking_scheme=False):
        
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio <= 0: 
            x, mask, ids_restore,masked_view = self.random_masking_2d_views(x, var_mask_ratio, time_mask_ratio)
        elif masking_scheme == 'custom_sync':
            x, mask, ids_restore,masked_view = self.systematic_masking_2d_views(x, mask_c_prob=0, mask_r_prob=mask_ratio)

        else:
            x, mask, ids_restore,masked_view = self.random_masking_views(x, mask_ratio=mask_ratio)

        # base 1
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # base 2
        cls_token2 = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens2 = cls_token2.expand(x.shape[0], -1, -1)
        masked_view = torch.cat((cls_tokens2, masked_view), dim=1)

        for blk in self.blocks:
            masked_view = blk(masked_view)
        masked_view = self.norm(masked_view)

        return x, masked_view, mask
    
    def systematic_masking_2d(self, x, mask_c_prob=0, mask_r_prob=0.75):
        """
        2D: Systematic masking of the first half of each variate, following the logic of the original function.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch size, sequence length, feature dimension

        ph, pw = self.patch_embed.patch_size  # [1, 20]
        h, w = self.img_size  # [6, 200]
        T, F = h // ph, w // pw

        # Systematic masking logic: Mask the first half of each variate
        len_keep_t = int(T * (1 - mask_c_prob))
        len_keep_f = int(F * (1 - mask_r_prob))

        ids_restore_t = torch.arange(T, device=x.device).unsqueeze(0).repeat(N, 1)
        ids_restore_f = torch.arange(F, device=x.device).unsqueeze(0).repeat(N, 1)

        # Generate systematic binary masks
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:, :len_keep_f] = 0  # Systematically mask the first half along the frequency axis
        mask_f = mask_f.unsqueeze(1).repeat(1, T, 1)  # [N, T, F]

        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:, :len_keep_t] = 0  # Systematically mask the first half along the time axis
        mask_t = mask_t.unsqueeze(2).repeat(1, 1, F)  # [N, T, F]

        # Combine masks (logical AND)
        mask = 1 - (1 - mask_t) * (1 - mask_f)  # N, T, F

        # Flatten indices for systematic masking
        id2res = torch.arange(N * T * F, device=x.device).reshape(N, T, F)
        id2res = id2res + 999 * mask  # Add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep = id2res2.flatten(start_dim=1)[:, :len_keep_t * len_keep_f]
        
        # Get masked x
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore
    
    def systematic_masking_2d_views(self, x, mask_c_prob=0, mask_r_prob=0.75):
        """
        2D: Systematic masking of the first half of each variate, following the logic of the original function.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch size, sequence length, feature dimension

        ph, pw = self.patch_embed.patch_size  # [1, 20]
        h, w = self.img_size  # [6, 200]
        T, F = h // ph, w // pw

        # Systematic masking logic: Mask the first half of each variate
        len_keep_t = int(T * (1 - mask_c_prob))
        len_keep_f = int(F * (1 - mask_r_prob))

        ids_restore_t = torch.arange(T, device=x.device).unsqueeze(0).repeat(N, 1)
        ids_restore_f = torch.arange(F, device=x.device).unsqueeze(0).repeat(N, 1)

        # Generate systematic binary masks
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:, :len_keep_f] = 0  # Systematically mask the first half along the frequency axis
        mask_f = mask_f.unsqueeze(1).repeat(1, T, 1)  # [N, T, F]

        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:, :len_keep_t] = 0  # Systematically mask the first half along the time axis
        mask_t = mask_t.unsqueeze(2).repeat(1, 1, F)  # [N, T, F]

        # Combine masks (logical AND)
        mask = 1 - (1 - mask_t) * (1 - mask_f)  # N, T, F

        # Flatten indices for systematic masking
        id2res = torch.arange(N * T * F, device=x.device).reshape(N, T, F)
        id2res = id2res + 999 * mask  # Add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep = id2res2.flatten(start_dim=1)[:, :len_keep_t * len_keep_f]
        
        # Get masked x
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        ## get the makse view of x (only contain the tokens that got mask out)
        ids_masked=id2res2.flatten(start_dim=1)[:,len_keep_f*len_keep_t:]
        masked_view = torch.gather(x, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, mask, ids_restore, masked_view

    # def correlation_mask(self,x,mask_ratio=0.75):
    #     '''
    #     Input: x (tensor): bs, num_patch, d 
    #     '''
    #     N,L,D = x.shape
    #     len_keep = int(L * (1-mask_ratio))

    #     cov_matrices = torch.stack([torch.cov(x[i]) for i in range(N)])  # (bs, num_patches, num_patches)
    #     imp_scores = torch.sum(torch.abs(cov_matrices), dim=1)  # (bs, num_patches)
        
    #     # sort noise for each sample
    #     #ids_shuffle = torch.argsort(imp_scores, dim=1)  # ascend: small is keep, large is remove

    #     # FIXME: check if the sorting is correct, below is leaving the important patch unmask
    #     ids_shuffle = torch.argsort(imp_scores, dim=1,descending=True)  # descent: keep imporant patch, ascend: remove the large ones
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)

    #     # keep the first subset
    #     ids_keep = ids_shuffle[:, :len_keep]
    #     #x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)

    #     return ids_keep, mask, ids_restore
    def correlation_mask(self, x, mask_ratio=0.75):
        '''
        Input: x (tensor): bs, num_patch, d 
        '''
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        # Use top 2*len_keep patches as important, if possible.
        M = min(2 * len_keep, L)
        
        # Compute covariance matrices and importance scores.
        cov_matrices = torch.stack([torch.cov(x[i]) for i in range(N)])  # (N, L, L)
        imp_scores = torch.sum(torch.abs(cov_matrices), dim=1)  # (N, L)
        
        # Sort indices by importance descending.
        ids_sorted = torch.argsort(imp_scores, dim=1, descending=True)  # (N, L)
        
        # Select top M indices as important.
        important = ids_sorted[:, :M]  # (N, M)
        
        # From the important patches, assign even-indexed ones as unmasked.
        ids_keep = important[:, 0::2]  # (N, len_keep) if M equals 2*len_keep
        
        # The remaining indices include the odd-indexed important patches and all patches not in the top M.
        rest = torch.cat([important[:, 1::2], ids_sorted[:, M:]], dim=1)  # (N, L - len_keep)
        
        # Build the new permutation: unmasked indices first, then masked ones.
        new_ids_shuffle = torch.cat([ids_keep, rest], dim=1)  # (N, L)
        
        # Create the binary mask: 0 for unmasked, 1 for masked.
        mask = torch.ones((N, L), device=x.device)
        mask[:, :len_keep] = 0
        
        # Compute ids_restore so that we can revert the shuffling.
        ids_restore = torch.argsort(new_ids_shuffle, dim=1)
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return ids_keep, mask, ids_restore


if __name__ == "__main__":
    model = MaskedAutoencoderViT().to('cuda')
    x = torch.randn(32,1,6,200).to('cuda')
    loss, pred, mask = model(x,
                             mask_ratio=0.75,
                             masking_scheme='spectral')
    print(mask)


# def mae_vit_base_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_large_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_huge_patch14_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_tiny_patch16_dec256d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         embed_dim=192, depth=12, num_heads=3, 
#         decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# # set recommended archs
# mae_vit_tiny_patch16 = mae_vit_tiny_patch16_dec256d8b
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks