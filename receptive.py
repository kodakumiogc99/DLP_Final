vgg16 = {
    'net': [
        [3, 1, 1],
        [3, 1, 1],
        [2, 2, 0],
        [3, 1, 1],
        [3, 1, 1],
        [2, 2, 0],
        [3, 1, 1],
        [3, 1, 1],
        [3, 1, 1],
        [2, 2, 0],
        [3, 1, 1],
        [3, 1, 1],
        [3, 1, 1],
        [2, 2, 0],
        [3, 1, 1],
        [3, 1, 1],
        [3, 1, 1],
        [2, 2, 0]
    ],
    'name': [
        'conv1_1',
        'conv1_2',
        'pool1',
        'conv2_1',
        'conv2_2',
        'pool2',
        'conv3_1',
        'conv3_2',
        'conv3_3',
        'pool3',
        'conv4_1',
        'conv4_2',
        'conv4_3',
        'pool4',
        'conv5_1',
        'conv5_2',
        'conv5_3',
        'pool5'
    ]
}


def output_shape(image_size: int, net: list, num: int) -> tuple:
    input_size = image_size
    output_size = None

    for layer in range(num):
        filter_size, stride, padding = net[layer]

        output_size = (input_size - filter_size + 2 * padding) // stride + 1
        input_size = output_size

    total_stride = 1
    total_padding = 0

    for layer in reversed(range(num)):
        _, stride, padding = net[layer]

        total_stride *= stride
        total_padding = stride * total_padding + padding

    return (output_size, total_stride, total_padding)


def receptive_field(net: list, num: int) -> int:
    field = 1

    for layer in reversed(range(num)):
        filter_size, stride, _ = net[layer]

        field = (field - 1) * stride + filter_size

    return field


if __name__ == '__main__':
    image_size = 224

    for i in range(len(vgg16['net'])):
        output = output_shape(image_size, vgg16['net'], i + 1)
        field = receptive_field(vgg16['net'], i + 1)

        print(f'layer {vgg16["name"][i]}, output size: {output[0]}, stride: {output[1]}, padding: {output[2]}, receptive field: {field}')
