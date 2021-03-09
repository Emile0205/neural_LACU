from dataset import dataset
from create_sets import *
from classifier import *
from generators import *
from critics import *
from utils import Timer
import shutil
import gc


count_path = 0
count_labels = 0

count_labels = 0
while count_labels < 10:
    # TODO: Delete or rename previous model directory
    main_path = './model' + str(count_labels + 1)
    print(main_path)

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    for i in range(0, 10):

        num_lab_classes = 5
        gc.collect()
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        lab_classes, unlab_classes = pre_process_set(10, num_lab_classes, 10 - num_lab_classes, set_unlab_classes = unlab_classes_list[count_labels])
        create_set('MNIST', './data', lab_classes, 100, unlab_classes, 50)

        data = dataset(num_lab_classes, './data', 128)
            
        secondary_path = main_path + '/loop_' + str(i) 

        if not os.path.exists(secondary_path):
            os.makedirs(secondary_path)

        network = DCCritic("classifier", data.img_size_x, data.img_size_y,
                               data.img_size_z, num_lab_classes + 1)

        classify = classifier(network=network,
                                dataset=data,
                                steps = 15000)

        print("TRANING STARTED")
        temp = classify.call(secondary_path, main_path)
        del network
        del classify
        del data
        tf.compat.v1.reset_default_graph()

        print("TRAINING FINISHED")

    count_labels += 1
   



