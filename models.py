import jax.numpy as jnp
import flax.linen as nn

class BasicNetwork(nn.Module):
    
    num_layers: int = 5
    d_hidden: int = 128
    act: int = nn.swish
    @nn.compact
    def __call__(self,x):
        for i in range(self.num_layers):
            x = nn.Dense(self.d_hidden)(x)
            x = self.act(x)
        x = nn.Dense(1)(x)
        return x

class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_hidden: int

    @nn.compact
    def __call__(self, x, mask):
        output = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.d_model)(x, mask=mask)
        x = x + output
        x = nn.LayerNorm()(x)
        output = nn.Dense(self.d_hidden)(x)
        output = nn.swish(output)
        output = nn.Dense(self.d_model)(output)
        x = x + output
        x = nn.LayerNorm()(x)
        return x
    
class TransformerModel(nn.Module):
    num_layers: int = 5
    d_model: int = 128
    num_heads: int = 4
    d_hidden: int = 256
    temporal_attention: bool = False
    temporal_num_layers: int = 1

    @nn.compact
    def __call__(self, x, mask):
        if self.temporal_attention:
            batch_size, lookback, num_tokens, _ = x.shape

            x = nn.Dense(self.d_model)(x)
            time_ids = jnp.arange(lookback)[None, :]
            time_embedding = nn.Embed(lookback, self.d_model)(time_ids)
            x = x + time_embedding[:, :, None, :]

            temporal_mask = mask
            stock_mask = mask[:, -1, :]

            x = jnp.transpose(x, (0, 2, 1, 3)).reshape(batch_size * num_tokens, lookback, self.d_model)
            temporal_mask = jnp.transpose(temporal_mask, (0, 2, 1)).reshape(batch_size * num_tokens, lookback)
            temporal_mask = temporal_mask[:, None, None, :]

            for _ in range(self.temporal_num_layers):
                x = TransformerBlock(self.d_model, self.num_heads, self.d_hidden)(x, temporal_mask)

            x = x[:, -1, :].reshape(batch_size, num_tokens, self.d_model)
        else:
            x = nn.Dense(self.d_model)(x)
            stock_mask = mask

        stock_ids = jnp.arange(x.shape[1])[None, :]
        stock_embedding = nn.Embed(x.shape[1], self.d_model)(stock_ids)
        x = x + stock_embedding

        stock_mask = stock_mask[:, None, None, :]

        for _ in range(self.num_layers):
            x = TransformerBlock(self.d_model, self.num_heads, self.d_hidden)(x, stock_mask)
        x = nn.Dense(1)(x)
        return x