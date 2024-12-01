"""
MIT License

Copyright (c) 2021 leondgarse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is a keras implementation of EfficientViT.
The original pytorch implementation can be found here:
https://github.com/mit-han-lab/efficientvit

This was modified October 2024 by Paul Thompson,
mostly to make it Keras3 compatible.

"""

import keras 


def mb_conv(
    inputs,
    output_channel,
    shortcut=True,
    strides=1,
    expansion=4,
    is_fused=False,
    use_bias=False,
    use_norm=False,
    use_output_norm=False,
    initializer=None,
    drop_rate=0,
    activation="keras.activations.hard_silu",
    name=""
):

    activation_func = eval(activation)
    if initializer is None:
        initializer = keras.initializers.GlorotUniform()

    input_channel = inputs.shape[-1]
    if is_fused:
        nn = keras.layers.Conv2D(
            int(input_channel * expansion),
            3, strides,
            padding="same",
            kernel_initializer=initializer,
            name=name and name + "expand_conv")(inputs)
        if use_norm:
            nn = keras.layers.BatchNormalization(momentum=0.9, name=name + "expand_bn")(nn) 
        nn = keras.layers.Activation(activation_func, name='{}_activation'.format(name))(nn)
    elif expansion > 1:
        nn = keras.layers.Conv2D(
            int(input_channel * expansion),
            1,
            strides=1,
            use_bias=True,
            kernel_initializer=initializer,
            name=name + "expand_conv")(inputs)
        if use_norm:
            nn = keras.layers.BatchNormalization(momentum=0.9, name=name + "expand_bn")(nn)
        nn = keras.layers.Activation(activation_func, name='{}_activation'.format(name))(nn)
    else:
        nn = inputs

    if not is_fused:
        nn = keras.layers.DepthwiseConv2D(
            3,
            strides=strides,
            use_bias=True,
            padding="same",
            depthwise_initializer=initializer,
            name=name + "dw_conv")(nn)
        if use_norm:
            nn = keras.layers.BatchNormalization(momentum=0.9, name=name + "dw_bn")(nn)
        nn = keras.layers.Activation(activation_func, name='{}_dw_activation'.format(name))(nn)

    pw_kernel_size = 3 if is_fused and expansion == 1 else 1

    nn = keras.layers.Conv2D(
        output_channel,
        pw_kernel_size,
        strides=1,
        padding="same",
        use_bias=True,
        kernel_initializer=initializer,
        name=name + "pw_conv")(nn)
    if use_output_norm:
        nn = keras.layers.BatchNormalization(momentum=0.9, gamma_initializer="zeros", name=name + "pw_bn")(nn)
    nn = keras.layers.Dropout(rate=drop_rate, name=name + "dropout")(nn)

    return keras.layers.Add(name=name + "output")([inputs, nn]) if shortcut else keras.layers.Activation("linear", name=name + "output")(nn)


def lite_mhsa(inputs,
              num_heads=8,
              key_dim=16,
              sr_ratio=5,
              qkv_bias=True,  # was False
              out_shape=None,
              out_bias=False,
              use_norm=True,
              dropout=0,
              initializer=None,
              activation="keras.activations.relu",
              name=None
              ):
    
    if initializer is None:
        initializer = keras.initializers.GlorotUniform()

    input_channel = inputs.shape[-1]
    height, width = inputs.shape[1:-1]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    out_shape = input_channel if out_shape is None else out_shape
    emb_dim = num_heads * key_dim

    # query = layers.Dense(emb_dim, use_bias=qkv_bias, name=name and name + "query")(inputs)
    qkv = keras.layers.Conv2D(
        emb_dim * 3,
        1,
        use_bias=qkv_bias,
        kernel_initializer=initializer,
        name=name and name + "qkv_conv")(inputs)
    sr_qkv = keras.layers.DepthwiseConv2D(
        kernel_size=sr_ratio,
        use_bias=qkv_bias,
        padding="same",
        depthwise_initializer=initializer,
        name=name and name + "qkv_dw_conv")(qkv)
    sr_qkv = keras.layers.Conv2D(
        emb_dim * 3,
        1,
        use_bias=qkv_bias,
        groups=3 * num_heads,
        kernel_initializer=initializer,
        name=name and name + "qkv_pw_conv")(sr_qkv)
    qkv = keras.ops.concatenate([qkv, sr_qkv], axis=-1)


    qkv = keras.ops.reshape(qkv, [-1, height * width, qkv.shape[-1] // (3 * key_dim), 3 * key_dim])
    query, key, value = keras.ops.split(qkv, 3, axis=-1)
    query = keras.ops.transpose(query, [0, 2, 1, 3])
    key = keras.ops.transpose(key, [0, 2, 3, 1])
    value = keras.ops.transpose(value, [0, 2, 1, 3])

    activation_func = eval(activation)
    query = keras.layers.Activation(activation_func, name='{}_query_activation'.format(name))(query)
    key = keras.layers.Activation(activation_func, name='{}_key_activation'.format(name))(key)

    query_key = query @ key
    scale = keras.ops.sum(query_key, axis=-1, keepdims=True)
    attention_output = query_key @ value / (scale + 1e-7)  # 1e-7 for also working on float16
    # print(f">>>> {inputs.shape = }, {emb_dim = }, {num_heads = }, {key_dim = }, {attention_output.shape = }")

    output = keras.ops.transpose(attention_output, [0, 2, 1, 3])  # [batch, q_blocks, num_heads * 2, key_dim]
    output = keras.ops.reshape(output, [-1, height, width, output.shape[2] * output.shape[3]])

    # print(f">>>> {output.shape = }")
    output = keras.layers.Conv2D(
        out_shape,
        1,
        use_bias=True,
        kernel_initializer=initializer,
        name=name and name + "out_conv")(output)
    if use_norm:
        output = keras.layers.BatchNormalization(momentum=0.9, name=name and name + "out_bn")(output)
    return output


def EfficientViT_B(
    num_blocks=[2, 2, 3, 3],
    out_channels=[16, 32, 64, 128],
    stem_width=8,
    block_types=["conv", "conv", "transform", "transform"],
    expansions=4,  # int or list, each element in list can also be an int or list of int
    is_fused=False,  # True for L models, False for B models
    head_dimension=16,  # `num_heads = channels // head_dimension`
    output_filters=[1024, 1280],
    input_shape=(224, 224, 3),
    activation="keras.activations.relu",  # "keras.activations.hard_silu" is in the paper, but i find poor performance
    drop_connect_rate=0,
    dropout=0,
    use_norm=True,
    initializer=None,
    model_name="efficientvit",
    kwargs=None,
    unet_output=False,
):
    
    if initializer is None:
        initializer = keras.initializers.GlorotUniform()

    inputs = keras.layers.Input(input_shape)
    is_fused = is_fused if isinstance(is_fused, (list, tuple)) else ([is_fused] * len(num_blocks))

    activation_func = eval(activation)

    unet_outputs = []

    """ stage 0, Stem_stage """
    nn = keras.layers.Conv2D(
        stem_width,
        3,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        name="stem_conv")(inputs)
    if use_norm:
        nn = keras.layers.BatchNormalization(momentum=0.9, name="stem_bn")(nn)
    nn = keras.layers.Activation(activation_func, name="stem_activation_")(nn)

    if unet_output:
        unet_outputs.append(nn) # 2x downsample

    nn = mb_conv(
        nn,
        stem_width,
        shortcut=True,
        expansion=1,
        is_fused=is_fused[0],
        use_norm=use_norm,
        use_output_norm=use_norm,
        activation=activation,
        initializer=initializer,
        name="stem_MB_")

    """ stage [1, 2, 3, 4] """ # 1/4, 1/8, 1/16, 1/32
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        is_conv_block = True if block_type[0].lower() == "c" else False
        cur_expansions = expansions[stack_id] if isinstance(expansions, (list, tuple)) else expansions

        block_use_bias, block_use_norm = (True, False) if stack_id >= 2 else (False, True)  # fewer_norm

        if not use_norm:
            block_use_norm = False
            block_use_bias = True

        cur_is_fused = is_fused[stack_id]
        for block_id in range(num_block):

            name = "stack_{}_block_{}_".format(stack_id + 1, block_id + 1)
            stride = 2 if block_id == 0 else 1
            shortcut = False if block_id == 0 else True
            cur_expansion = cur_expansions[block_id] if isinstance(cur_expansions, (list, tuple)) else cur_expansions

            block_drop_rate = drop_connect_rate * global_block_id / total_blocks

            if is_conv_block or block_id == 0:
                cur_name = (name + "downsample_") if stride > 1 else name
                nn = mb_conv(
                    nn,
                    out_channel,
                    shortcut=shortcut,
                    strides=stride,
                    expansion=cur_expansion,
                    is_fused=cur_is_fused,
                    use_bias=block_use_bias,
                    use_norm=block_use_norm,
                    use_output_norm=use_norm,
                    drop_rate=block_drop_rate,
                    activation=activation,
                    initializer=initializer,
                    name=cur_name)
            else:
                num_heads = out_channel // head_dimension
                attn = lite_mhsa(
                    nn,
                    num_heads=num_heads,
                    key_dim=head_dimension,
                    sr_ratio=5,
                    use_norm=use_norm,
                    initializer=initializer,
                    name=name + "attn_")

                nn = nn + attn

                nn = mb_conv(
                    nn,
                    out_channel,
                    shortcut=shortcut,
                    strides=stride,
                    expansion=cur_expansion,
                    is_fused=cur_is_fused,
                    use_bias=block_use_bias,
                    use_norm=block_use_norm,
                    use_output_norm=use_norm,
                    drop_rate=block_drop_rate,
                    activation=activation,
                    initializer=initializer,
                    name=name)
            global_block_id += 1

        if unet_output:
            unet_outputs.append(nn)

    output_filters = output_filters if isinstance(output_filters, (list, tuple)) else (output_filters, 0)
    if output_filters[0] > 0:
        nn = keras.layers.Conv2D(
            output_filters[0],
            1,
            kernel_initializer=initializer,
            name="features_conv")(nn)
        if use_norm:
            nn = keras.layers.BatchNormalization(momentum=0.9, name="features_bn")(nn)
        nn = keras.layers.Activation(activation_func, name="features_activation")(nn)

    if unet_output:
        #remove last
        unet_outputs.pop()
        unet_outputs.append(nn)
        nn = unet_outputs

    model = keras.models.Model(inputs, nn, name=model_name)

    return model
