import torch
import torch.nn as nn
from collections import OrderedDict

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
def unfix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def has_trainable_params(module: torch.nn.Module) -> bool:
    any_require_grad = any(p.requires_grad for p in module.parameters())
    any_bn_in_train_mode = any(m.training for m in module.modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)))
    return any_require_grad or any_bn_in_train_mode

def has_untrainable_params(module: torch.nn.Module) -> bool:
    any_not_require_grad = any((not p.requires_grad) for p in module.parameters())
    any_bn_in_eval_mode = any((not m.training) for m in module.modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)))
    return any_not_require_grad or any_bn_in_eval_mode

def load_model_dict(model, pretrained_dict):
    """ load pretrained state dict, keys may not match with model

    Args:
        model: nn.Module

        pretrained_dict: collections.OrderedDict
    
    """
    # 1. filter out unnecessary keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=0.1)
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=0.1)
        # if hasattr(m, 'bias'):
        #     nn.init.constant_(m.bias, 0)

    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.xavier_normal_(m.weight, gain=0.05)
    #     nn.init.constant_(m.bias, 0)

def rename_model_dict_keys(pretrained_dict_path, rename_dict):
    """ load pretrained state dict, keys may not match with model

    Args:
        model: nn.Module

        pretrained_dict: collections.OrderedDict
    
    """
    pretrained_dict = torch.load(pretrained_dict_path)
    # 1. filter out unnecessary keys
    for oldname, newname in rename_dict.items():
        if oldname.endswith("*"):
            _oldnames = list(pretrained_dict.keys())
            _oldnames = [x for x in _oldnames if x.startswith(oldname[:-1])]
            for _oldname in _oldnames:
                if newname != "":
                    _newname = _oldname.replace(oldname[:-1], newname[:-1])
                    pretrained_dict[_newname] = pretrained_dict[_oldname]
                pretrained_dict.pop(_oldname)
        else:
            if newname != "":
                pretrained_dict[newname] = pretrained_dict[oldname]
            pretrained_dict.pop(oldname)
    torch.save(pretrained_dict, pretrained_dict_path)


def compose_model(model1, keyname1, model2, keyname2):
    pretrained_dict1 = torch.load(model1)
    pretrained_dict2 = torch.load(model2)

    new_dict = OrderedDict()
    for keyname in keyname1:
        if keyname.endswith("*"):
            _oldnames = list(pretrained_dict1.keys())
            _oldnames = [x for x in _oldnames if x.startswith(keyname[:-1])]
            for _oldname in _oldnames:
                new_dict[_oldname] = pretrained_dict1[_oldname]

    for keyname in keyname2:
        if keyname.endswith("*"):
            _oldnames = list(pretrained_dict2.keys())
            _oldnames = [x for x in _oldnames if x.startswith(keyname[:-1])]
            for _oldname in _oldnames:
                new_dict[_oldname] = pretrained_dict2[_oldname]

    torch.save(new_dict, "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs/A_v2xset_heter_sdta_layer0_aligner6/net_epoch1.pth")


def switch_model_dict_keys(pretrained_dict_path, switch_dict):
    """ load pretrained state dict, keys may not match with model

    Args:
        model: nn.Module

        pretrained_dict: collections.OrderedDict
    
        switch_dict: {"cls_head_lidar": "cls_head_camera"}
    """
    pretrained_dict = torch.load(pretrained_dict_path)
    # 1. filter out unnecessary keys
    for key1, key2 in switch_dict.items():
        all_model_keys = list(pretrained_dict.keys())
        all_key1_weight = [x for x in all_model_keys if x.startswith(key1)]
        for key1_weight_name in all_key1_weight:
            key2_weight_name = key1_weight_name.replace(key1, key2)
        
            pretrained_dict[key1_weight_name], pretrained_dict[key2_weight_name] = \
                pretrained_dict[key2_weight_name], pretrained_dict[key1_weight_name]

    torch.save(pretrained_dict, pretrained_dict_path)


def check_trainable_module(model):
    appeared_module_list = []
    has_trainable_list = []
    has_untrainable_list = []
    for name, module in model.named_modules():
        if any([name.startswith(appeared_module_name) for appeared_module_name in appeared_module_list]) or name=='': # the whole model has name ''
            continue
        appeared_module_list.append(name)

        if has_trainable_params(module):
            has_trainable_list.append(name)
        if has_untrainable_params(module):
            has_untrainable_list.append(name)

    print("=========Those modules have trainable component=========")
    print(*has_trainable_list,sep='\n',end='\n\n')
    print("=========Those modules have untrainable component=========")
    print(*has_untrainable_list,sep='\n',end='\n\n')


if __name__ == "__main__":

    # dict_path = "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs/A_v2xset_heter_lidar_and_camera_pretrain_switch_layer2_layer3_shrink_head/net_epoch1.pth"
    # switch_dict = {"lidar_backbone.resnet.layer1": "camera_backbone.resnet.layer1",
    #                "lidar_backbone.renset.layer2": "camera_backbone.resnet.layer2",
    #                "cls_head_lidar": "cls_head_camera",
    #                "reg_head_lidar": "reg_head_camera",
    #                "dir_head_lidar": "dir_head_camera",
    #                "shrink_lidar":"shrink_camera"}
    # switch_model_dict_keys(dict_path, switch_dict)

    # dict_path = "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs/A_v2xset_heter_camera_pretrain_8x_64/net_epoch1.pth"
    # rename_dict = {"camera_encoder.*": "",
    #                 "camera_backbone.*": "",
    #                 "shrink_camera.*": "",
    #                 "cls_head_camera.*": "",
    #                 "reg_head_camera.*": "",
    #                 "dir_head_camera.*": "",}
    # rename_model_dict_keys(dict_path, rename_dict)

    # dict_path = "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs/v2xset_heter_late_fusion/net_epoch_bestval_at28.pth"
    # rename_dict = {"camencode.*": "camera_encoder.camencode.*",
    #                "bevencode.*": "camera_encoder.bevencode.*",
    #                "head.cls_head.*": "cls_head_camera.*",
    #                "head.reg_head.*": "reg_head_camera.*",
    #                "head.dir_head.*": "dir_head_camera.*",
    #                "shrink_conv.*": "shrink_camera.*"}
    # rename_model_dict_keys(dict_path, rename_dict)

    model1 = "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs/A_v2xset_heter_lidar_and_camera_pretrain/net_epoch1.pth" # lidar
    model2 = "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs/A_v2xset_heter_sdta_layer0_aligner6/camera_pretrain_at25.pth" # cam
    
    keyname1 = ['lidar_encoder.*',]
    keyname2 = ['camera_encoder.*',]
    compose_model(model1, keyname1, model2, keyname2)

    # dict_path = "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs/v2xset_heter_late_fusion/net_epoch1.pth"
    # rename_dict = {"camera_encoder.*": "",
    #                "head.cls_head_camera.*": "",
    #                "head.reg_head_camera.*": "",
    #                "head.dir_head_camera.*": "",
    #                "shrink_camera.*": ""}
    # rename_model_dict_keys(dict_path, rename_dict)