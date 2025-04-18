import torch
import torch.nn as nn
from timm import create_model

# ✅ ViT Model
# def get_vit(pretrained=True):
#     """Loads a pretrained ViT model and modifies it for binary classification."""
#     model = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=1)
#     if pretrained:
#         num_features = model.head.in_features
#         model.head = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(512, 1)
#         )
#     return model


def get_vit(pretrained=True):
    """Loads a pretrained ViT model and modifies it for binary classification."""
    model = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=1)
    if pretrained:
        num_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
    return model

#model.py claude

#model.py claude

import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model

# ✅ Frequency Attention for DFDetectV2
class FrequencyAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# ✅ DFDetectV2 Model with ViT and Frequency Analysis
class DFDetectV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = create_model('vit_base_patch16_224', pretrained=True)
        hidden_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()  # Remove classification head

        self.freq_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, hidden_dim, 1)
        )

        self.freq_attention = nn.ModuleList([FrequencyAttention(hidden_dim) for _ in range(3)])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Binary classification
        )

    def forward(self, x):
        vit_features = self.backbone(x)
        freq_features = torch.fft.rfft2(x).abs()  # Frequency domain
        freq_features = self.freq_conv(x)
        B, C, H, W = freq_features.shape
        freq_features = freq_features.flatten(2).transpose(1, 2)

        features = vit_features + freq_features
        for attn in self.freq_attention:
            features = attn(features)

        cls_token = features[:, 0]
        output = self.head(cls_token)
        return output

# ✅ ResNet-based DeepFake detector (alternative implementation)
def get_dfdetect(pretrained=False):
    """
    Creates a DeepFake detection model based on ResNet50.
    
    Args:
        pretrained (bool): Whether to use pretrained weights for ResNet50
    
    Returns:
        model (nn.Module): The DeepFake detection model
    """
    # Load the base ResNet model
    model = models.resnet50(pretrained=pretrained)
    
    # Get the number of features from the final layer
    num_features = model.fc.in_features  # 2048 for ResNet50
    
    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1)  # Binary classification output
    )
    
    return model

class DeepFakeDetector(nn.Module):
    """
    Alternative implementation as a custom class if needed.
    Wraps the modified ResNet model with additional functionality.
    """
    def __init__(self, pretrained=True):
        super(DeepFakeDetector, self).__init__()
        self.base_model = get_dfdetect(pretrained)
    
    def forward(self, x):
        return self.base_model(x)
    
    def predict(self, x):
        """
        Helper method for getting binary predictions.
        """
        with torch.no_grad():
            outputs = self(x)
            predictions = torch.sigmoid(outputs) > 0.5
        return predictions
# ✅ Xception Model
def get_xception(pretrained=True):
    """Loads a pretrained Xception model and modifies it for binary classification."""
    return create_model("xception", pretrained=pretrained, num_classes=1)
def get_efficientnet(pretrained=True):
    model = create_model(
        "efficientnet_b3",
        pretrained=pretrained,
        num_classes=1,
        drop_path_rate=0.4,
        act_layer=nn.GELU
    )

    # Add classifier dropout
    if hasattr(model, 'classifier'):
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            model.classifier
        )
    elif hasattr(model, 'head'):
        model.head = nn.Sequential(
            nn.Dropout(0.5),
            model.head
        )
    return model

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            # Convert image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        # Add learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Add learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        b = x.shape[0]  # batch size
        x = self.projection(x)
        
        # Add classification token to each sequence
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(0.1)
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = F.softmax(att, dim=-1)
        att = self.att_drop(att)
        
        x = (att @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        x = self.projection(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    """Vision Transformer for binary classification."""
    def __init__(self, 
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer Encoder
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)  # Binary classification
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer blocks
        x = self.transformer(x)
        
        # Classification token
        x = self.norm(x)
        x = x[:, 0]  # Use [CLS] token
        
        # Classification head
        x = self.head(x)
        return x




import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            # Convert image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        # Add learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Add learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        b = x.shape[0]  # batch size
        x = self.projection(x)
        
        # Add classification token to each sequence
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(0.1)
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = F.softmax(att, dim=-1)
        att = self.att_drop(att)
        
        x = (att @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        x = self.projection(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x




import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            # Convert image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        # Add learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Add learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        b = x.shape[0]  # batch size
        x = self.projection(x)
        
        # Add classification token to each sequence
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(0.1)
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = F.softmax(att, dim=-1)
        att = self.att_drop(att)
        
        x = (att @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        x = self.projection(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x




import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            # Convert image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        # Add learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Add learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        b = x.shape[0]  # batch size
        x = self.projection(x)
        
        # Add classification token to each sequence
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(0.1)
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = F.softmax(att, dim=-1)
        att = self.att_drop(att)
        
        x = (att @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        x = self.projection(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    """Vision Transformer for binary classification."""
    def __init__(self, 
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer Encoder
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)  # Binary classification
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer blocks
        x = self.transformer(x)
        
        # Classification token
        x = self.norm(x)
        x = x[:, 0]  # Use [CLS] token
        
        # Classification head
        x = self.head(x)
        return x

def get_vit(pretrained=False):
    """Returns an initialized ViT model for binary classification."""
    model = ViT(
        image_size=224,        # Required input size
        patch_size=16,         # Size of patches
        in_channels=3,         # RGB images
        embed_dim=768,         # Embedding dimension
        num_layers=12,         # Number of transformer blocks
        num_heads=12,          # Number of attention heads
        mlp_ratio=4.0,         # MLP hidden dimension ratio
        dropout=0.1            # Dropout rate
    )
    
    if pretrained:
        # You would typically load pretrained weights here
        # This is left as a placeholder
        pass
        
    return model