from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue, enum
from cumm.tensorview import Tensor
from cumm.tensorview import CUDAKernelTimer
class ThrustCustomAllocatorV2:
    alloc_func: Callable[int, int]
class SpconvOps:
    @staticmethod
    def cumm_version() -> str: 
        """
        get cumm version when build spconv.
                
        """
        ...
    @staticmethod
    def is_cpu_only_build() -> bool: ...
    @staticmethod
    def pccm_version() -> str: 
        """
        get pccm version when build spconv.
                
        """
        ...
    @staticmethod
    def generate_conv_inds_cpu(indices: Tensor, indice_pairs: Tensor, out_inds: Tensor, indice_num_per_loc: Tensor, batch_size: int, output_dims: List[int], input_dims: List[int], ksize: List[int], stride: List[int], padding: List[int], dilation: List[int], transposed: bool = False) -> int: 
        """
        Args:
            indices: 
            indice_pairs: 
            out_inds: 
            indice_num_per_loc: 
            batch_size: 
            output_dims: 
            input_dims: 
            ksize: 
            stride: 
            padding: 
            dilation: 
            transposed: 
        """
        ...
    @staticmethod
    def generate_subm_conv_inds_cpu(indices: Tensor, indice_pairs: Tensor, out_inds: Tensor, indice_num_per_loc: Tensor, batch_size: int, input_dims: List[int], ksize: List[int], dilation: List[int]) -> int: 
        """
        Args:
            indices: 
            indice_pairs: 
            out_inds: 
            indice_num_per_loc: 
            batch_size: 
            input_dims: 
            ksize: 
            dilation: 
        """
        ...
    @staticmethod
    def indice_maxpool(out_features: Tensor, features: Tensor, indice_pairs: Tensor, indice_pair_num: Tensor, num_activate_out: int, stream: int = 0) -> None: 
        """
        Args:
            out_features: 
            features: 
            indice_pairs: 
            indice_pair_num: 
            num_activate_out: 
            stream: 
        """
        ...
    @staticmethod
    def indice_maxpool_backward(din: Tensor, features: Tensor, out_features: Tensor, out_bp: Tensor, indice_pairs: Tensor, indice_pair_num: Tensor, stream: int = 0) -> None: 
        """
        Args:
            din: 
            features: 
            out_features: 
            out_bp: 
            indice_pairs: 
            indice_pair_num: 
            stream: 
        """
        ...
    @staticmethod
    def global_pool_rearrange(out_indices: Tensor, coords: Tensor, counts: Tensor, stream: int = 0) -> None: 
        """
        Args:
            out_indices: 
            coords: 
            counts: 
            stream: 
        """
        ...
    @staticmethod
    def maxpool_forward_cpu(out: Tensor, inp: Tensor, out_inds: Tensor, in_inds: Tensor) -> None: 
        """
        Args:
            out: 
            inp: 
            out_inds: 
            in_inds: 
        """
        ...
    @staticmethod
    def maxpool_backward_cpu(out: Tensor, inp: Tensor, dout: Tensor, dinp: Tensor, out_inds: Tensor, in_inds: Tensor) -> None: 
        """
        Args:
            out: 
            inp: 
            dout: 
            dinp: 
            out_inds: 
            in_inds: 
        """
        ...
    @staticmethod
    def gather_cpu(out: Tensor, inp: Tensor, inds: Tensor) -> None: 
        """
        Args:
            out: 
            inp: 
            inds: 
        """
        ...
    @staticmethod
    def scatter_add_cpu(out: Tensor, inp: Tensor, inds: Tensor) -> None: 
        """
        Args:
            out: 
            inp: 
            inds: 
        """
        ...
    @staticmethod
    def maximum_value_int(data: Tensor, value: int, stream_int: int) -> None: 
        """
        Args:
            data: 
            value: 
            stream_int: 
        """
        ...
    @staticmethod
    def calc_point2voxel_meta_data(vsize_xyz: List[float], coors_range_xyz: List[float]) -> Tuple[List[float], List[int], List[int], List[float]]: 
        """
        Args:
            vsize_xyz: 
            coors_range_xyz: 
        """
        ...
    @staticmethod
    def point2voxel_cpu(points: Tensor, voxels: Tensor, indices: Tensor, num_per_voxel: Tensor, densehashdata: Tensor, pc_voxel_id: Tensor, vsize: List[float], grid_size: List[int], grid_stride: List[int], coors_range: List[float], empty_mean: bool = False, clear_voxels: bool = True) -> Tuple[Tensor, Tensor, Tensor]: 
        """
        Args:
            points: 
            voxels: 
            indices: 
            num_per_voxel: 
            densehashdata: 
            pc_voxel_id: 
            vsize: 
            grid_size: 
            grid_stride: 
            coors_range: 
            empty_mean: 
            clear_voxels: 
        """
        ...
    @staticmethod
    def get_int32_max() -> int: ...
    @staticmethod
    def get_handcrafted_max_act_out(num_act_in: int, ksize: List[int], stride: List[int], padding: List[int], dilation: List[int]) -> int: 
        """
        Args:
            num_act_in: 
            ksize: 
            stride: 
            padding: 
            dilation: 
        """
        ...
    @staticmethod
    def get_indice_gen_workspace_size(kv: int, num_act_in: int, num_act_out_bound: int, max_act_out_in_theory: int, subm: bool, use_int64_hash_k: bool, direct_table: bool) -> int: 
        """
        Args:
            kv: 
            num_act_in: 
            num_act_out_bound: 
            max_act_out_in_theory: 
            subm: 
            use_int64_hash_k: 
            direct_table: 
        """
        ...
    @staticmethod
    def get_indice_gen_tensors_from_workspace(workspace, kv: int, num_act_in: int, num_act_out_bound: int, max_act_out_in_theory: int, subm: bool, use_int64_hash_k: bool, direct_table: bool) -> Dict[str, Tensor]: 
        """
        Args:
            workspace: 
            kv: 
            num_act_in: 
            num_act_out_bound: 
            max_act_out_in_theory: 
            subm: 
            use_int64_hash_k: 
            direct_table: 
        """
        ...
    @staticmethod
    def get_indice_pairs_implicit_gemm(allocator, indices: Tensor, batch_size: int, input_dims: List[int], algo: int, ksize: List[int], stride: List[int], padding: List[int], dilation: List[int], out_padding: List[int], subm: bool, transposed: bool, is_train: bool, stream_int: int = 0, num_out_act_bound: int = -1, timer: CUDAKernelTimer =  CUDAKernelTimer(False), direct_table: bool = False, do_sort: bool = True, preallocated: Dict[str, Tensor] =  {}) -> Tuple[Tensor, int]: 
        """
        Args:
            allocator: 
            indices: 
            batch_size: 
            input_dims: 
            algo: 
            ksize: 
            stride: 
            padding: 
            dilation: 
            out_padding: 
            subm: 
            transposed: 
            is_train: 
            stream_int: 
            num_out_act_bound: 
            timer: 
            direct_table: 
            do_sort: 
            preallocated: 
        """
        ...
    @staticmethod
    def get_indice_pairs(allocator, indices: Tensor, batch_size: int, input_dims: List[int], algo: int, ksize: List[int], stride: List[int], padding: List[int], dilation: List[int], out_padding: List[int], subm: bool, transposed: bool, stream_int: int = 0, num_out_act_bound: int = -1, num_input_act_bound: int = -1) -> int: 
        """
        Args:
            allocator: 
            indices: 
            batch_size: 
            input_dims: 
            algo: 
            ksize: 
            stride: 
            padding: 
            dilation: 
            out_padding: 
            subm: 
            transposed: 
            stream_int: 
            num_out_act_bound: 
            num_input_act_bound: 
        """
        ...
