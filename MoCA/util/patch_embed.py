import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
#from util.pos_embed import tAPE


class PatchEmbed_ts(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, ts_len=200, 
                 patch_size=20, 
                 embed_dim=512,
                 nvar=6, 
                 stride=20, ): # non-overlapping patches
        super().__init__()

        '''

        Input: raw_series (bs x nvar x L) 'Differ from Howon's construction of (bs x nvar x 1 x L)'
        
        bs x nvar x L -> bs x nvar x num_patches x patch_size -> bs x num_patches x (nvar*patch_size) -> bs x num_patches x E
        '''
        
        self.ts_len = ts_len
        self.patch_size = patch_size
        self.nvar = nvar
        self.num_patches = int(ts_len//patch_size)

        self.proj = nn.Linear(patch_size*nvar,embed_dim)


    def forward(self, x):
        # x: bs x nvar x L
        # Check dimensions consistency

        bs, nvar, L = x.shape
        assert L == self.num_patches * self.patch_size, "L must be equal to num_patches * patch_size"
        
        x = x.view(bs, nvar, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous() # bs x num_patches, nvar, patch_size
        x = x.view(bs, self.num_patches, nvar*self.patch_size) # bs x num_patch x nvar* num_patch_size

        x = self.proj(x) # bs x num_patch x E

        return x







if __name__ == '__main__':
    # patch_emb = PatchEmbed_new(img_size=(387,65), patch_size=(9,5), in_chans=3, embed_dim=64, stride=(9,5))
    # input = torch.rand(8,3,387,65)
    # output = patch_emb(input)
    # print(output.shape) # (8,559,64)

    # patch_emb = PatchEmbed3D_new(video_size=(6,224,224), patch_size=(2,16,16), in_chans=3, embed_dim=768, stride=(2,16,16))
    # input = torch.rand(8,3,6,224,224)
    # output = patch_emb(input)
    #print(output.shape) # (8,64)

    patch_emb = PatchEmbed_ts(ts_len=387,patch_size=9,stride=9)
    input = torch.randn(6,387)
    output = patch_emb(input)
    print(output.shape)
    print(patch_emb.patch_size)