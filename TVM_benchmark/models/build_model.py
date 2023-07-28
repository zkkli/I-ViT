from .quantized_vit import Q_VisionTransformer
from .utils import create_workload, QuantizeInitializer


def get_deit(name,
            batch_size,
            image_shape=(3, 224, 224),
            dtype="int8",
            data_layout="NCHW",
            kernel_layout="OIHW",
            debug_unit=None):


    if data_layout == 'NCHW':
        data_shape = (batch_size,) + image_shape
    elif data_layout == 'NHWC':
        data_shape = (batch_size, image_shape[1], image_shape[2], image_shape[0])
    elif data_layout == 'HWCN':
        data_shape = (image_shape[1], image_shape[2], image_shape[0], batch_size)
    elif data_layout == 'HWNC':
        data_shape = (image_shape[1], image_shape[2], batch_size, image_shape[0])
    else:
        raise RuntimeError("Unsupported data layout {}".format(data_layout))


    if name == 'deit_tiny_patch16_224':
        embed_dim = 192
        num_heads = 3
    elif name == 'deit_small_patch16_224':
        embed_dim = 384
        num_heads = 6
    elif name == 'deit_base_patch16_224':
        embed_dim = 768
        num_heads = 12
    else:
        raise RuntimeError("Unsupported model {}".format(name))
    

    return Q_VisionTransformer(data_shape=data_shape,
                            dtype=dtype,
                            patch_size=16,
                            num_patches=196,
                            in_chans=3,
                            num_classes=1000,
                            embed_dim=embed_dim,
                            depth=12,
                            num_heads=num_heads,
                            mlp_ratio=4)


def get_workload(name,
                 batch_size=1,
                 image_shape=(3, 224, 224),
                 dtype="int8",
                 data_layout="NCHW",
                 kernel_layout="OIHW",
                 debug_unit=None):
    
    if batch_size != 1:
        raise RuntimeError("The released project only supports batch_size = 1.")

    net = get_deit(name,
                   batch_size,
                   image_shape=image_shape,
                   dtype=dtype,
                   data_layout=data_layout,
                   kernel_layout=kernel_layout,
                   debug_unit=debug_unit)

    return create_workload(net, QuantizeInitializer())