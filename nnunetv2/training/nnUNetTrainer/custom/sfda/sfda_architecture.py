from typing import Union, Type, List, Tuple

import numpy as np
import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from nnunetv2.utilities.network_initialization import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from nnunetv2.training.nnUNetTrainer.variants.sfda.utils_sfda import *
from nnunetv2.utilities.helpers import softmax_helper_dim1
from scipy import ndimage

class PlainConvUNetSFDA(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder1 = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        self.decoder2 = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        self.decoder3 = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        self.decoder4 = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        # 회전된 입력 텐서를 생성
        rotated_imgs = [x, rotate_single_with_label(x, 1), rotate_single_with_label(x, 2), rotate_single_with_label(x, 3)]
        
        # 각 회전된 텐서를 인코딩
        encoded_skips = [self.encoder(img) for img in rotated_imgs]
        
        # 디코딩 및 softmax 처리
        outputs = [softmax_helper_dim1(getattr(self, f'decoder{i+1}')(encoded_skips[i])[0]) for i in range(4)]
        
        # 출력 텐서를 원래 방향으로 회전
        outputs[1] = rotate_single_with_label(outputs[1], 3)
        outputs[2] = rotate_single_with_label(outputs[2], 2)
        outputs[3] = rotate_single_with_label(outputs[3], 1)
        
        # 평균 결과 반환
        return outputs[0], outputs[1], outputs[2], outputs[3], sum(outputs) / 4.0

    def forward_for_predict(self, x):
        # 회전된 입력 텐서를 생성
        rotated_imgs = [x, rotate_single_with_label(x, 1), rotate_single_with_label(x, 2), rotate_single_with_label(x, 3)]
        
        # 각 회전된 텐서를 인코딩
        encoded_skips = [self.encoder(img) for img in rotated_imgs]

        # 인코딩 결과의 차원 확장
        for i in range(len(encoded_skips)):
            for j in range(len(encoded_skips[i])):
                if len(encoded_skips[i][j].shape) == 4:  # 만약 4차원이라면
                    encoded_skips[i][j] = encoded_skips[i][j].unsqueeze(0)        
        # 디코딩 및 softmax 처리
        #print(getattr(self, f'decoder{i+1}')(encoded_skips[i])[0].unsqueeze(0).shape)
        outputs = [softmax_helper_dim1(getattr(self, f'decoder{i+1}')(encoded_skips[i])[0].unsqueeze(0)) for i in range(4)]
        
        # 출력 텐서를 원래 방향으로 회전
        outputs[1] = rotate_single_with_label(outputs[1], 3)
        outputs[2] = rotate_single_with_label(outputs[2], 2)
        outputs[3] = rotate_single_with_label(outputs[3], 1)
        
        # 평균 결과 반환
        return sum(outputs) / 4.0
    
    def save_img(self, x, device):
        pl_threshold = 0.99
        rotated_imgs = [x, rotate_single_with_label(x, 1), rotate_single_with_label(x, 2), rotate_single_with_label(x, 3)]
        
        # 각 회전된 텐서를 인코딩
        encoded_skips = [self.encoder(img) for img in rotated_imgs]
        
        # 디코딩 및 softmax 처리
        outputs = [softmax_helper_dim1(getattr(self, f'decoder{i+1}')(encoded_skips[i])[0]) for i in range(4)]
        
        # 출력 텐서를 원래 방향으로 회전
        outputs[1] = rotate_single_with_label(outputs[1], 3)
        outputs[2] = rotate_single_with_label(outputs[2], 2)
        outputs[3] = rotate_single_with_label(outputs[3], 1)
        
        # Convert outputs to numpy and compute the average prediction map
        soft_outputs = [output.detach().cpu().numpy() for output in outputs]
        four_predict_map = np.mean(soft_outputs, axis=0)
        
        # Binarize the prediction map
        four_predict_map = np.where(four_predict_map > pl_threshold, 1, 0)
        #print(x.shape, outputs[0].shape, outputs[1].shape, outputs[2].shape, outputs[3].shape, four_predict_map.shape)
        B, C, D, W, H = four_predict_map.shape
        for j in range(B):
            for i in range(C):
                four_predict_map[j, i, :, :, :] = self.get_largest_component(four_predict_map[j, i, :, :, :])
        
        return torch.from_numpy(four_predict_map).float().to(device)        
    
    def get_largest_component(self, image):
        """
        get the largest component from 2D or 3D binary image
        image: nd array
        """
        dim = len(image.shape)
        if(image.sum() == 0):
            # print('the largest component is null')
            return image
        if(dim == 2):
            s = ndimage.generate_binary_structure(2,1)
        elif(dim == 3):
            s = ndimage.generate_binary_structure(3,1)
        else:
            raise ValueError("the dimension number should be 2 or 3")
        labeled_array, numpatches = ndimage.label(image, s)
        sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
        max_label = np.where(sizes == sizes.max())[0] + 1
        output = np.asarray(labeled_array == max_label[0], np.uint8)
        return  output

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder1.compute_conv_feature_map_size(input_size) + self.decoder2.compute_conv_feature_map_size(input_size) + self.decoder3.compute_conv_feature_map_size(input_size) + self.decoder4.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)