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

    q4_attn = BaseQuantizeConfig(nbits=4, group_size=32)   # Attention safer
    q4_proj = BaseQuantizeConfig(nbits=4, group_size=32)
    q3_mlp = BaseQuantizeConfig(nbits=3, group_size=32)
    q4_patch = BaseQuantizeConfig(nbits=4, group_size=32)  # 小心不能再用 64，精度會低

    # 改回 3-bit patch，提升精度
    # q3_patch = BaseQuantizeConfig(nbits=3, group_size=64)

    n_blocks = len(model.blocks)

    for i in range(n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = q4_attn
        quant_config[f'blocks.{i}.attn.proj'] = q4_proj
        quant_config[f'blocks.{i}.mlp.fc1'] = q4_proj   # ⚠️ 改為 4-bit 回補 MLP 精度
        quant_config[f'blocks.{i}.mlp.fc2'] = q4_proj

    quant_config['patch_embed.proj'] = q4_patch  # ⚠️ 由 4/64 改成 4/32，提升精度但保壓縮

    return quant_config



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