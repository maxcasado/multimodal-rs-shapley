"""
Sensor encoders and prediction heads used by the paper's models: TempCNN for the time
series (S2_S2VI, S1, weather), an MLP for DEM and for the heads.

Adapted from fmenat/DSensDp (code/models/nn_models.py, single/encoders.py,
single/encoders_sota.py, single/sota_aux/tcnn.py, single/base_{en,de}coders.py), pruned to
the two model types the configs use. Module and attribute names are unchanged, so the
state_dict keys are the original ones. TempCNN (Pelletier et al., Remote Sens. 2019) always
has 3 convolutional blocks: as in the original, ``n_layers`` of the config is not passed on.
"""
import copy

from torch import nn


class Conv1D_BatchNorm_Relu_Dropout(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability))

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability))

    def forward(self, X):
        return self.block(X)


class TempCNN(nn.Module):
    def __init__(self, input_dim, sequence_length, kernel_size=5, hidden_dims=64, dropout=0.5,
                 n_layers=3, **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.sequence_length = sequence_length
        self.conv_layers = nn.Sequential(*[
            Conv1D_BatchNorm_Relu_Dropout(input_dim if i == 0 else hidden_dims, hidden_dims,
                                          kernel_size=kernel_size, drop_probability=dropout)
            for i in range(n_layers)])
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims * sequence_length, 4 * hidden_dims,
                                               drop_probability=dropout)

    def forward(self, x):                    # (B, T, C)
        x = self.conv_layers(x.permute(0, 2, 1))
        return self.dense(x.view(x.size(0), -1))

    def get_output_size(self):
        return 4 * self.hidden_dims


class MLP(nn.Module):
    def __init__(self, feature_size, layer_sizes=None, activation=nn.ReLU, dropout=0,
                 batchnorm=False, **kwargs):
        super().__init__()
        layer_sizes = (feature_size,) + tuple(layer_sizes or (128,))
        self.encoder_output = layer_sizes[-1]
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                activation(),
                nn.BatchNorm1d(layer_sizes[i + 1], affine=True) if batchnorm else nn.Identity(),
                nn.Dropout(p=dropout) if dropout != 0 else nn.Identity())
            for i in range(len(layer_sizes) - 1)])

    def forward(self, x, **kwargs):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        return {"rep": self.layers(x)}

    def get_output_size(self):
        return self.encoder_output


class Generic_Encoder(nn.Module):
    """Encoder body + linear projection to ``latent_dims`` (+ optional LayerNorm)."""

    def __init__(self, encoder, latent_dims, use_norm=False, **kwargs):
        super().__init__()
        self.pre_encoder = encoder
        self.latent_dims = latent_dims
        self.linear_layer = nn.Linear(self.pre_encoder.get_output_size(), latent_dims)
        self.norm_layer = nn.LayerNorm(latent_dims) if use_norm else nn.Identity()

    def forward(self, x):
        out = self.pre_encoder(x)
        out = out["rep"] if isinstance(out, dict) else out
        return self.norm_layer(self.linear_layer(out))

    def get_output_size(self):
        return self.latent_dims


class Generic_Decoder(nn.Module):
    """MLP body + linear layer to ``out_dims`` (+ optional LayerNorm on the output)."""

    def __init__(self, decoder, out_dims, use_norm_last=False, **kwargs):
        super().__init__()
        self.pre_decoder = decoder
        self.out_dims = out_dims
        self.linear_layer = nn.Linear(self.pre_decoder.get_output_size(), out_dims)
        self.norm_layer = nn.LayerNorm(out_dims) if use_norm_last else None

    def forward(self, x):
        out = self.pre_decoder(x)
        out = self.linear_layer(out["rep"] if isinstance(out, dict) else out)
        return self.norm_layer(out) if self.norm_layer is not None else out

    def get_output_size(self):
        return self.out_dims


def create_model(input_dims, emb_dims, model_type="mlp", n_layers=2, batchnorm=False, dropout=0,
                 encoder=True, **args):
    """``encoder=True``: Generic_Encoder(body) -> emb_dims; False: Generic_Decoder(body) -> emb_dims."""
    args = copy.deepcopy(args)
    model_type = model_type.lower()
    if model_type == "mlp":
        size = args.pop("layer_size", 128)
        body = MLP(input_dims, layer_sizes=tuple(size for _ in range(n_layers)),
                   dropout=dropout, batchnorm=batchnorm)
    elif model_type == "tempcnn":
        if "layer_size" in args:
            args["hidden_dims"] = args.pop("layer_size")
        body = TempCNN(input_dim=input_dims, sequence_length=args.pop("seq_len", 12),
                       dropout=dropout, **args)
    else:
        raise ValueError(f"model_type {model_type!r} is not used by the paper (tempcnn | mlp)")
    if encoder:
        return Generic_Encoder(body, emb_dims, **args)
    return Generic_Decoder(body, emb_dims, **args)
