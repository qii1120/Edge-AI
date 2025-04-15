# from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
# import itertools
# import torch
# from hqq_utils import AutoHQQTimmModel, get_size_of_model
# from utils import prepare_data, evaluate_model

# def search_best_quant_config(model, test_loader):
#     # 候選設定：bit 數與 group size 組合
#     bit_options = [2, 3, 4]
#     group_options = [16, 32, 64]

#     best_score = -float('inf')
#     results = []

#     # 搜索組合空間：投影、MLP、QKV、Patch
#     for attn_bit, attn_gs in itertools.product([4], [32]):
#         for proj_bit, proj_gs in itertools.product([3, 4], [32]):
#             for mlp_bit, mlp_gs in itertools.product([2, 3], [16, 32]):
#                 for patch_bit, patch_gs in itertools.product([3, 4], [32]):

#                     # 重建原始模型權重（視情況加上 deepcopy）
#                     # model.cpu()
#                     # model.load_state_dict(torch.load('./0.9099_deit3_small_patch16_224.pth', map_location='cpu', weights_only=False))  # 換成你的原始模型檔
#                     # model = model.cuda()

#                     # 定義 quant config
#                     def get_quant_config_deit(model):
#                         quant_config = {}
#                         q_attn = BaseQuantizeConfig(nbits=attn_bit, group_size=attn_gs)
#                         q_proj = BaseQuantizeConfig(nbits=proj_bit, group_size=proj_gs)
#                         q_mlp  = BaseQuantizeConfig(nbits=mlp_bit, group_size=mlp_gs)
#                         q_patch = BaseQuantizeConfig(nbits=patch_bit, group_size=patch_gs)

#                         for i in range(len(model.blocks)):
#                             quant_config[f'blocks.{i}.attn.qkv'] = q_attn
#                             quant_config[f'blocks.{i}.attn.proj'] = q_proj
#                             quant_config[f'blocks.{i}.mlp.fc1'] = q_mlp
#                             quant_config[f'blocks.{i}.mlp.fc2'] = q_mlp

#                         quant_config['patch_embed.proj'] = q_patch
#                         return quant_config

#                     # 進行量化
#                     quant_config = get_quant_config_deit(model)
#                     AutoHQQTimmModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float32, device=device)


#                     # Evaluate
#                     acc_after_quant = evaluate_model(model, test_loader, 'cuda:0')
#                     size_mib = get_size_of_model(model) / (1024 ** 2)

#                     # 計算 score
#                     score = 20 - max(0, 90 - acc_after_quant) * 10 + (17 - size_mib)

#                     result = {
#                         'score': score,
#                         'acc': acc_after_quant,
#                         'size': size_mib,
#                         'config': {
#                             'attn': (attn_bit, attn_gs),
#                             'proj': (proj_bit, proj_gs),
#                             'mlp': (mlp_bit, mlp_gs),
#                             'patch': (patch_bit, patch_gs),
#                         }
#                     }

#                     results.append(result)
#                     print(f"Tested config: {result['config']} -> Score: {score:.2f}, Acc: {acc_after_quant:.2f}%, Size: {size_mib:.2f} MiB")

#                     if score > best_score:
#                         best_score = score

#     # 排序並顯示最佳組合
#     results.sort(key=lambda x: x['score'], reverse=True)
#     print("\n🏆 Top 5 Quantization Configs:")
#     for r in results[:5]:
#         print(f"[Score {r['score']:.2f}] Acc: {r['acc']:.2f}% | Size: {r['size']:.2f} MiB | Config: {r['config']}")

#     return results

# # Set up
# device = 'cuda:0'
# batch_size = 16
# model = torch.load('./0.9099_deit3_small_patch16_224.pth', map_location='cpu', weights_only=False)
# model = model.to(device)
# _, test_loader, _ = prepare_data(batch_size)
# search_best_quant_config(model, test_loader)
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
import itertools
import torch
from hqq_utils import AutoHQQTimmModel, get_size_of_model
from utils import prepare_data, evaluate_model

def search_best_quant_config(model, test_loader):
    # 候選設定：bit 數與 group size 組合
    bit_options = [2, 3, 4]
    group_options = [16, 32, 64]

    best_score = -float('inf')
    results = []

    # 搜索組合空間：QKV, Proj, fc1, fc2, Patch
    for attn_bit, attn_gs in itertools.product([3, 4], [32]):
        for proj_bit, proj_gs in itertools.product([3, 4], [32]):
            for fc1_bit, fc1_gs in itertools.product([3], [16, 32]):
                for fc2_bit, fc2_gs in itertools.product([2, 3], [16, 32]):
                    for patch_bit, patch_gs in itertools.product([3, 4], [32]):

                        # 定義 quant config
                        def get_quant_config_deit(model):
                            quant_config = {}
                            q_attn = BaseQuantizeConfig(nbits=attn_bit, group_size=attn_gs)
                            q_proj = BaseQuantizeConfig(nbits=proj_bit, group_size=proj_gs)
                            q_fc1 = BaseQuantizeConfig(nbits=fc1_bit, group_size=fc1_gs)
                            q_fc2 = BaseQuantizeConfig(nbits=fc2_bit, group_size=fc2_gs)
                            q_patch = BaseQuantizeConfig(nbits=patch_bit, group_size=patch_gs)

                            for i in range(len(model.blocks)):
                                quant_config[f'blocks.{i}.attn.qkv'] = q_attn
                                quant_config[f'blocks.{i}.attn.proj'] = q_proj
                                quant_config[f'blocks.{i}.mlp.fc1'] = q_fc1
                                quant_config[f'blocks.{i}.mlp.fc2'] = q_fc2

                            quant_config['patch_embed.proj'] = q_patch
                            return quant_config

                        # 重建模型（建議從原始 checkpoint 重新載入）
                        model = torch.load('./0.9099_deit3_small_patch16_224.pth', map_location='cpu', weights_only=False)
                        model = model.to('cuda:0')

                        # 進行量化
                        quant_config = get_quant_config_deit(model)
                        AutoHQQTimmModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float32, device='cuda:0')

                        # Evaluate
                        acc_after_quant = evaluate_model(model, test_loader, 'cuda:0')
                        size_mib = get_size_of_model(model) / (1024 ** 2)

                        # 計算 score
                        score = 20 - max(0, 90 - acc_after_quant) * 10 + (17 - size_mib)

                        result = {
                            'score': score,
                            'acc': acc_after_quant,
                            'size': size_mib,
                            'config': {
                                'attn': (attn_bit, attn_gs),
                                'proj': (proj_bit, proj_gs),
                                'fc1': (fc1_bit, fc1_gs),
                                'fc2': (fc2_bit, fc2_gs),
                                'patch': (patch_bit, patch_gs),
                            }
                        }

                        results.append(result)
                        print(f"Tested config: {result['config']} -> Score: {score:.2f}, Acc: {acc_after_quant:.2f}%, Size: {size_mib:.2f} MiB")

                        if score > best_score:
                            best_score = score

    # 排序並顯示最佳組合
    results.sort(key=lambda x: x['score'], reverse=True)
    print("\n\U0001F3C6 Top 5 Quantization Configs:")
    for r in results[:5]:
        print(f"[Score {r['score']:.2f}] Acc: {r['acc']:.2f}% | Size: {r['size']:.2f} MiB | Config: {r['config']}")

    return results

# Set up
device = 'cuda:0'
batch_size = 16
model = torch.load('./0.9099_deit3_small_patch16_224.pth', map_location='cpu', weights_only=False)
model = model.to(device)
_, test_loader, _ = prepare_data(batch_size)
search_best_quant_config(model, test_loader)
