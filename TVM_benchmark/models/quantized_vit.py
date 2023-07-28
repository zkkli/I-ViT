import tvm
from tvm import relay

from . import layers


def Q_Block(data,
            name,
            dim,
            num_heads,
            mlp_ratio,
            qk_scale,
            batch_size,
            rounding='TRUNCATE'):

    'Attention mudule'
    shortcut = data

    ## layer_norm
    qconfig0 = layers.get_qconfig(name + '_qconfig_norm1')
    norm1_bias = relay.var(name + '_norm1_bias', shape=[dim], dtype='int32')
    norm1 = layers.quantized_layernorm(data, norm1_bias)

    ## attention
    qconfig1 = layers.get_qconfig(name + '_qconfig_qkv')
    req1 = layers.requantize(norm1,
                             input_scale=qconfig0.output_scale,
                             output_scale=qconfig1.input_scale,
                             out_dtype=qconfig1.input_dtype)

    req1 = relay.reshape(req1, [-3,0])
    qkv = layers.quantized_dense(data=req1,
                                  name=name + '_attn_qkv',
                                  input_scale=qconfig1.input_scale,
                                  kernel_scale=qconfig1.kernel_scale,
                                  units=dim*3,
                                  kernel_shape=(dim*3, dim),
                                  kernel_dtype='int8',
                                  add_bias=True)
    qkv = relay.reshape(qkv, [-4,batch_size,-1,-2])

    qconfig2 = layers.get_qconfig(name + '_qconfig_matmul_1') 
    req2 = layers.requantize(qkv,
                             input_scale=qconfig1.output_scale,
                             output_scale=qconfig2.input_scale,
                             out_dtype=qconfig2.input_dtype)

    qkv_reshape = relay.reshape(req2, [0, 0, 3, num_heads, -1])
    qkv = relay.transpose(qkv_reshape, [2, 0, 3, 1, 4])
    qkv = relay.split(qkv, 3, axis=0)
    q = relay.reshape(relay.squeeze(qkv[0], axis=[0]), [-3,-2])
    k = relay.reshape(relay.squeeze(qkv[1], axis=[0]), [-3,-2])
    v = relay.reshape(relay.squeeze(qkv[2], axis=[0]), [-3,-2])

    attn = layers.quantized_matmul(q, k,
                                   input_scale1=qconfig2.input_scale,
                                   input_scale2=qconfig2.input_scale)

    attn = relay.reshape(attn, [-4,-1,num_heads,-2])

    qconfig3 = layers.get_qconfig(name + '_qconfig_softmax') 
    req3 = layers.requantize(attn,
                             input_scale=qconfig2.output_scale * qk_scale,
                             output_scale=qconfig3.input_scale,
                             out_dtype=qconfig3.input_dtype)

    attn = layers.quantized_softmax(req3, qconfig3.input_scale)

    qconfig4 = layers.get_qconfig(name + '_qconfig_matmul_2') 
    attn = relay.reshape(attn,[-3,-2])
    v = relay.transpose(v, [0, 2, 1])
    attn = layers.quantized_matmul(attn, v,
                                   input_scale1=qconfig4.input_scale,
                                   input_scale2=qconfig2.input_scale)

    attn = relay.reshape(attn, [-4,-1,num_heads,-2])

    attn = relay.transpose(attn, [0, 2, 1, 3])
    attn = relay.reshape(attn, [0, 0, -1])

    qconfig5 = layers.get_qconfig(name + '_qconfig_proj')
    req5 = layers.requantize(attn,
                             input_scale=qconfig4.output_scale,
                             output_scale=qconfig5.input_scale,
                             out_dtype=qconfig5.input_dtype)

    req5 = relay.reshape(req5, [-3,0])
    proj = layers.quantized_dense(data=req5, 
                                  name=name + '_attn_proj',
                                  input_scale=qconfig5.input_scale,
                                  kernel_scale=qconfig5.kernel_scale,
                                  units=dim,
                                  kernel_shape=(dim, dim),
                                  kernel_dtype='int8',
                                  add_bias=True)
    proj = relay.reshape(proj, [-4,batch_size,-1,-2])
    
    ## shortcut
    qconfig6 = layers.get_qconfig(name + '_qconfig_add1')
    req6 = layers.requantize(proj,
                             input_scale=qconfig5.output_scale,
                             output_scale=qconfig6.input_scale,
                             out_dtype=qconfig6.input_dtype)

    add1 = layers.add(lhs=req6,
                      rhs=shortcut,
                      lhs_scale=qconfig6.input_scale,
                      rhs_scale=qconfig0.input_scale,
                      output_scale=qconfig6.output_scale)

    'MLP module'
    shortcut = add1
    ## layer_norm
    qconfig7 = layers.get_qconfig(name + '_qconfig_norm2')
    norm2_bias = relay.var(name + '_norm2_bias', shape=[dim], dtype='int32')
    norm2 = layers.quantized_layernorm(add1, norm2_bias)

    ## dense
    qconfig8 = layers.get_qconfig(name + '_qconfig_fc1')
    req8 = layers.requantize(norm2,
                             input_scale=qconfig7.output_scale,
                             output_scale=qconfig8.input_scale,
                             out_dtype=qconfig8.input_dtype)
    
    req8 = relay.reshape(req8, [-3,0])
    fc1 = layers.quantized_dense(data=req8, 
                                 name=name + '_mlp_fc1',
                                 input_scale=qconfig8.input_scale,
                                 kernel_scale=qconfig8.kernel_scale,
                                 units=mlp_ratio*dim,
                                 kernel_shape=(mlp_ratio*dim, dim),
                                 kernel_dtype='int8',
                                 add_bias=True)
    fc1 = relay.reshape(fc1, [-4,batch_size,-1,-2])
    
    qconfig9 = layers.get_qconfig(name + '_qconfig_gelu')
    req9 = layers.requantize(fc1,
                             input_scale=qconfig8.output_scale,
                             output_scale=qconfig9.input_scale,
                             out_dtype=qconfig9.input_dtype)

    act = layers.quantized_gelu(req9, qconfig9.input_scale)
    

    qconfig10 = layers.get_qconfig(name + '_qconfig_fc2')
    req10 = layers.requantize(act,
                              input_scale=qconfig9.output_scale,
                              output_scale=qconfig10.input_scale,
                              out_dtype=qconfig10.input_dtype)

    req10 = relay.reshape(req10, [-3,0])
    fc2 = layers.quantized_dense(data=req10, 
                                 name=name + '_mlp_fc2',
                                 input_scale=qconfig10.input_scale,
                                 kernel_scale=qconfig10.kernel_scale,
                                 units=dim,
                                 kernel_shape=(dim, mlp_ratio*dim),
                                 kernel_dtype='int8',
                                 add_bias=True)
    fc2 = relay.reshape(fc2, [-4,batch_size,-1,-2])
    
    ## shortcut
    qconfig11 = layers.get_qconfig(name + '_qconfig_add2')
    req11 = layers.requantize(fc2,
                              input_scale=qconfig10.output_scale,
                              output_scale=qconfig11.input_scale,
                              out_dtype=qconfig11.input_dtype)

    add2 = layers.add(lhs=req11,
                      rhs=shortcut,
                      lhs_scale=qconfig11.input_scale,
                      rhs_scale=qconfig7.input_scale,
                      output_scale=qconfig11.output_scale)

    add2 = relay.annotation.stop_fusion(add2)

    return add2


def Q_VisionTransformer(data_shape,
                        dtype='int8',
                        patch_size=16,
                        num_patches=196,
                        in_chans=3,
                        num_classes=1000,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4):
    data = relay.var('data', shape=data_shape, dtype=dtype)

    qconfig_embed_conv = layers.get_qconfig('qconfig_embed_conv')
    proj = layers.quantized_conv2d(data=data,
                                    name='embed_conv',
                                    add_bias=True,
                                    input_channels=in_chans,
                                    output_channels=embed_dim,
                                    kernel_dtype=qconfig_embed_conv.kernel_dtype,
                                    input_scale=qconfig_embed_conv.input_scale,
                                    kernel_scale=qconfig_embed_conv.kernel_scale,
                                    kernel_size=(patch_size, patch_size),
                                    strides=(patch_size, patch_size), 
                                    padding=(0, 0),
                                    data_layout='NCHW',
                                    kernel_layout='OIHW')
    proj = relay.reshape(proj, [0, 0, -1])
    body = relay.transpose(proj, [0, 2, 1])

    qconfig_add = layers.get_qconfig('qconfig_addpos')
    body = layers.requantize(body, 
                            input_scale=qconfig_embed_conv.output_scale,
                            output_scale=qconfig_add.input_scale,
                            out_dtype=qconfig_add.input_dtype)
    

    cls_token = relay.var('cls_token_weight', shape=(1, 1, embed_dim))
    cls_token = layers.quantize(cls_token, output_scale=qconfig_add.input_scale, out_dtype=qconfig_add.input_dtype)
    cls_tokens = relay.repeat(cls_token, data_shape[0], axis=0)

    body = relay.concatenate([cls_tokens, body], axis=1)

    pos_embed = relay.var('pos_embed_weight', shape=(1, num_patches+1, embed_dim))
    qconfig_pos = layers.get_qconfig('qconfig_pos')
    pos_embed = layers.quantize(pos_embed, output_scale=qconfig_pos.output_scale, out_dtype=qconfig_add.input_dtype)

    body = layers.add(lhs=body,
                      rhs=pos_embed,
                      lhs_scale=qconfig_add.input_scale,
                      rhs_scale=qconfig_pos.output_scale,
                      output_scale=qconfig_add.output_scale)

    body = relay.annotation.stop_fusion(body)


    qk_scale = (embed_dim//num_heads) ** -0.5

    for i in range(depth):
        body = Q_Block(body, 
                       name='block_%d' % (i), 
                       dim=embed_dim, 
                       num_heads=num_heads, 
                       mlp_ratio=mlp_ratio, 
                       qk_scale=qk_scale,
                       batch_size=data_shape[0],
                       rounding='TONEAREST')


    qconfig_norm = layers.get_qconfig('qconfig_norm')
    norm_bias = relay.var('norm_bias', shape=[embed_dim], dtype='int32')
    norm = layers.quantized_layernorm(body, norm_bias)

    body = relay.split(norm, 197, axis=1)
    body = relay.squeeze(body[0], axis=[1])


    qconfig_head = layers.get_qconfig('qconfig_head')
    req = layers.requantize(body,
                            input_scale=qconfig_norm.output_scale,
                            output_scale=qconfig_head.input_scale,
                            out_dtype=qconfig_head.input_dtype)

    head = layers.quantized_dense(data=req, 
                                  name='head',
                                  input_scale=qconfig_head.input_scale,
                                  kernel_scale=qconfig_head.kernel_scale,
                                  units=num_classes,
                                  kernel_shape=(num_classes, embed_dim),
                                  kernel_dtype='int8',
                                  add_bias=True)


    net = layers.dequantize(head, input_scale=qconfig_head.output_scale)
    net = relay.nn.softmax(data=net)
    return relay.Function(relay.analysis.free_vars(net), net)
