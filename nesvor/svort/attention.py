import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# modified from https://github.com/jadore801120/attention-is-all-you-need-pytorch
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=True)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=True)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):

        x = x.unsqueeze(0)

        q = x
        k = x
        v = x

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        #if mask is not None:
        #    mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q = self.dropout(q)
        q += residual

        q = self.layer_norm(q)

        q = q.squeeze(0)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # b x n x lq x dv
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) #(b x n x lq x dv) (b x n x dv x lq)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, attn = self.slf_attn(enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, d_in):
        super().__init__()
        num_w = d_model // 2 // d_in
        self.num_pad = d_model - num_w * 2 * d_in
        w = 1e-3 ** torch.linspace(0, 1, num_w)
        self.w = nn.Parameter(w.view(1, -1, 1).repeat(1, 1, d_in))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.cat((torch.sin(self.w * x), torch.cos(self.w * x)), 1)
        x = x.flatten(1)
        if self.num_pad:
            x = F.pad(x, (0, self.num_pad))
        return x


class TransformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, pos_enc):

        enc_output = self.dropout(x + pos_enc)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, attn = enc_layer(enc_output)

        return enc_output, attn


class ResNet(nn.Module):

    def __init__(self, n_res, d_model, d_in=1, pretrained=False):
        super().__init__()
        resnet_fn = getattr(tvm, 'resnet%d' % n_res)
        model = resnet_fn(pretrained=pretrained, norm_layer=lambda x: nn.BatchNorm2d(x, track_running_stats=False))
        model.fc = nn.Linear(model.fc.in_features, d_model)
        if not pretrained:
            model.conv1 = nn.Conv2d(d_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model = model
        self.pretrained = pretrained
        
    def forward(self, x):
        if self.pretrained:
            x = x.expand(-1, 3, -1, -1)
        return self.model(x)