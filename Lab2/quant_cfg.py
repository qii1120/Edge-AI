from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for DeiT-S
# def get_quant_config_deit(model):
#     quant_config = {}
    
#     # Define quantization settings
#     q4_attn = BaseQuantizeConfig(nbits=4, group_size=32)   # Safer for attention
#     q3_proj = BaseQuantizeConfig(nbits=3, group_size=32)
#     q4_proj = BaseQuantizeConfig(nbits=4, group_size=32)   # Projection tolerates lower bits
#     q2_mlp = BaseQuantizeConfig(nbits=2, group_size=16)    # MLP layers can be heavily quantized
#     q3_mlp = BaseQuantizeConfig(nbits=3, group_size=16)    # MLP layers can be heavily quantized
    

#     n_blocks = len(model.blocks)
#     # print(f"Number of blocks: {n_blocks}")

#     for i in range(n_blocks):
#         # Quantize attention layers (QKV and projection)
#         quant_config[f'blocks.{i}.attn.qkv'] = q4_attn
#         quant_config[f'blocks.{i}.attn.proj'] = q4_proj

#         # Quantize MLP (feedforward) layers
#         quant_config[f'blocks.{i}.mlp.fc1'] = q3_mlp
#         quant_config[f'blocks.{i}.mlp.fc2'] = q3_mlp

#         # Quantize normalization layers (LayerNorm), there is no improvement
#         # quant_config[f'blocks.{i}.norm1'] = BaseQuantizeConfig(nbits=8, group_size=64)
#         # quant_config[f'blocks.{i}.norm2'] = BaseQuantizeConfig(nbits=8, group_size=64)


#     # Optionally, quantize patch embedding layer
#     quant_config['patch_embed.proj'] = BaseQuantizeConfig(nbits=4, group_size=32)
        
#     return quant_config


def get_quant_config_deit(model):
    quant_config = {}

    q4_64 = BaseQuantizeConfig(nbits=4, group_size=64)
    q4_32 = BaseQuantizeConfig(nbits=4, group_size=32)   # Attention safer

    n_blocks = len(model.blocks)

    for i in range(n_blocks):
        if i < (n_blocks/3+1):
            quant_config[f'blocks.{i}.attn.qkv'] = q4_32
            quant_config[f'blocks.{i}.attn.proj'] = q4_32
            quant_config[f'blocks.{i}.mlp.fc1'] = q4_32
            quant_config[f'blocks.{i}.mlp.fc2'] = q4_32
        else:
            quant_config[f'blocks.{i}.attn.qkv'] = q4_64
            quant_config[f'blocks.{i}.attn.proj'] = q4_64
            quant_config[f'blocks.{i}.mlp.fc1'] = q4_64
            quant_config[f'blocks.{i}.mlp.fc2'] = q4_64

    return quant_config

# def get_quant_config_deit(model):
#     quant_config = {}

#     q5 = BaseQuantizeConfig(nbits=5, group_size=32)
#     q4 = BaseQuantizeConfig(nbits=4, group_size=16)
#     q3 = BaseQuantizeConfig(nbits=3, group_size=16)
#     q2 = BaseQuantizeConfig(nbits=2, group_size=16)
#     q4_patch = BaseQuantizeConfig(nbits=4, group_size=32)

#     n_blocks = len(model.blocks)

#     for i in range(n_blocks):
#         if i < 4:  # 前段：高精度
#             quant_config[f'blocks.{i}.attn.qkv'] = q5
#             quant_config[f'blocks.{i}.attn.proj'] = q5
#             quant_config[f'blocks.{i}.mlp.fc1'] = q5
#             quant_config[f'blocks.{i}.mlp.fc2'] = q5
#         elif i < 8:  # 中段：中精度
#             quant_config[f'blocks.{i}.attn.qkv'] = q4
#             quant_config[f'blocks.{i}.attn.proj'] = q4
#             quant_config[f'blocks.{i}.mlp.fc1'] = q4
#             quant_config[f'blocks.{i}.mlp.fc2'] = q4
#         else:  # 後段：壓縮精度
#             quant_config[f'blocks.{i}.attn.qkv'] = q3
#             quant_config[f'blocks.{i}.attn.proj'] = q3
#             quant_config[f'blocks.{i}.mlp.fc1'] = q3
#             quant_config[f'blocks.{i}.mlp.fc2'] = q2  # 精簡 fc2 精度空間

#     # 加上 patch embedding 量化（若之前沒使用）
#     quant_config['patch_embed.proj'] = q4_patch

#     return quant_config



# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    
    # n_layers = model.config.num_hidden_layers
    # q2_config = BaseQuantizeConfig(nbits=2, group_size=64) 
    
    # for i in range(n_layers):
    #     quant_config[f'model.layers.{i}.self_attn.q_proj'] = q2_config
    #     quant_config[f'model.layers.{i}.self_attn.k_proj'] = q2_config
    #     quant_config[f'model.layers.{i}.self_attn.v_proj'] = q2_config
    #     quant_config[f'model.layers.{i}.self_attn.o_proj'] = q2_config
        
    #     quant_config[f'model.layers.{i}.mlp.gate_proj'] = q2_config
    #     quant_config[f'model.layers.{i}.mlp.up_proj'] = q2_config
    #     quant_config[f'model.layers.{i}.mlp.down_proj'] = q2_config
        
    return quant_config