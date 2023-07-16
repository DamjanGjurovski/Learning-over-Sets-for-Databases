import argparse, math
def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', dest='training', action='store_true')
    parser.set_defaults(training=False)
    parser.add_argument('--eval', dest='training', action='store_false')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=100, help='# epochs')
    parser.add_argument('--scale', type=int, default=3, help='Scaling in the presence of cardinality')
    parser.add_argument('--dataset-name', type=str, default='RW_200000_modified', help='The dataset name')
    parser.add_argument('--datasets-path', type=str, default='Not valid for now', help='Path to the train and eval queries')
    parser.add_argument('--encode-layers', type=int, default=1, help='The number of layers in the NN.')
    parser.add_argument('--encode-neurons', type=int, default=8, help='The number of neurons in each of the layers.')
    parser.add_argument('--middle-layers', type=int, default=1, help='The number of layers in the NN.')
    parser.add_argument('--middle-neurons', type=int, default=8, help='The number of neurons in each of the layers.')
    parser.add_argument('--decode-layers', type=int, default=1, help='The number of layers in the NN.')
    parser.add_argument('--decode-neurons', type=int, default=8, help='The number of neurons in each of the layers.')
    parser.add_argument('--error', type=int, default=0, help='0 means we have q error, 1 means we have mae')
    parser.add_argument('--custom-svd', type=int, default=0, help='if custom svd is 0 we continue with the normal, otherwise it denotes the svd')
    parser.add_argument('--onehot', dest='onehot', action='store_true')
    parser.set_defaults(onehot=False)
    parser.add_argument('--concat', dest='concat', action='store_true')
    parser.set_defaults(concat=True)
    parser.add_argument('--embed-size', type=int, default=2, help='The embedding size in the neural network.')
    parser.add_argument('--decay', action="store_true", help='Decay of the learning rate.')
    parser.set_defaults(decay=True)
    parser.add_argument('--max-elem', type=int, default=40000, help='The largest element id.')
    parser.add_argument('--max-length', type=int, default=10, help='The max size of the set.')
    parser.add_argument('--compression', action='store_true', help='Whether to perform compression.')
    parser.set_defaults(compression=False)
    parser.add_argument('--indexing', action='store_true', help='True is indexing, false is cardinality. Not important for Bloom filter.')
    parser.set_defaults(indexing=False)
    # specific for ms_mmq
    parser.add_argument('--dynamic-embedding', dest='training', action='store_true')
    parser.set_defaults(dynamic_embedding=False)
    # parser.add_argument('--compression-threshold', type=int, default=0, help='The compression threshold.')
    # outlier removal params
    parser.add_argument('--outlierremoval', dest='outlierremoval', action='store_true')
    parser.set_defaults(outlierremoval=False)
    parser.add_argument('--startremoval', type=int, default=90, help='The start of the outlier removal will be at which epoch')
    parser.add_argument('--stepremoval', type=int, default=50, help='The step of the outlier removal will be after which epoch')
    parser.add_argument('--boundaryremoval', type=int, default=90, help='The percentage of the outlier removal')

    args = parser.parse_args()

    # if we do not have a compression we for sure have a one hot embedding
    if not args.compression:
        args.onehot = False

    if "sd" in args.dataset_name:
        args.max_elem = 5661
        args.max_length = 10
    elif args.dataset_name == "RW_200000_modified":
        args.max_elem = 30324
        args.max_length = 8
    elif args.dataset_name == "RW_1500000_modified":
        args.max_elem = 231954
        args.max_length = 8
    elif args.dataset_name == "RW_3000000_modified":
        args.max_elem = 346893
        args.max_length = 8
    elif args.dataset_name == "hashtags3":
        args.max_elem = 73618
        args.max_length = 6
    else:
        print("The dataset name is not correct")
        exit(1)
    print("The max elem is")
    print(args.max_elem)
    args.max_elem += 1
    args.max_length += 1

    '''dimension parameters for the network'''
    old_variant = False
    file_dataset_name = args.dataset_name
    if "generated" in args.dataset_name:
        file_dataset_name = "generated"

    if args.error == 0:
        model_name = 'SE-q_loss-'
    else:
        model_name = "SE-mae-"

    if old_variant:

        layers = [args.neurons for i in range(args.decode_layers)]
        nn_size_name = '_'.join([str(neuron) for neuron in layers])
        model_name += nn_size_name + '-' + file_dataset_name + 'scale-' + str(
            args.scale) + '_decay' + str(args.decay) + '_' + str(args.embed_size)
        encode_l = []
        middle_l = []
        decode_l = layers
    else:
        encode_l = [args.encode_neurons for i in range(args.encode_layers)]
        nn_size_name_el = '_'.join([str(neuron) for neuron in encode_l])

        middle_l = [args.middle_neurons for i in range(args.middle_layers)]
        nn_size_name_ml = '_'.join([str(neuron) for neuron in middle_l])

        decode_l = [args.decode_neurons for i in range(args.decode_layers)]
        nn_size_name_dl = '_'.join([str(neuron) for neuron in decode_l])

        if not args.onehot:
            model_name += 'l-e' + nn_size_name_el + '-m' + nn_size_name_ml + '-d' +nn_size_name_dl+ '-'+ file_dataset_name + 'scale-' + str(
                args.scale) + '_decay' + str(args.decay) + '_emb' + str(args.embed_size)
        else:
            model_name += 'l-e' + nn_size_name_el + '-m' + nn_size_name_ml + '-d' + nn_size_name_dl + '-' + file_dataset_name + 'scale-' + str(
                args.scale) + '_decay' + str(args.decay) + '_onehot'

    if args.compression:
        model_name += "_concat" + str(args.concat)


    if args.outlierremoval:
        model_name += "st_" + str(args.startremoval) + "step" + str(args.stepremoval) + "b" + str(args.boundaryremoval)
    else:
        model_name += "_remove_F"

    sv_d = 0
    ns = 2
    if args.compression:
        model_name += "_compression_" + str(ns)

        if args.custom_svd > 0:
            sv_d = args.custom_svd
            model_name += "svd_" + str(args.custom_svd)
        else:
            sv_d = math.ceil(args.max_elem ** (1 / ns))
        print("SV d " + str(sv_d))

    print("The name of the model in creation is " + str(model_name))

    return args, model_name, encode_l, middle_l, decode_l, sv_d, ns, args.indexing