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