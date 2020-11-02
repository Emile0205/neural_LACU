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

#unlab_classes_list = [[4, 5, 8, 9], [2, 4, 5, 9], [2, 4, 7, 9]]
#unlab_classes_list = [[4, 5, 7, 8, 9], [1, 4, 5, 6, 8], [1, 3, 4, 8, 9], [0, 2, 4, 5, 9], [0, 2, 7, 8, 9], [1, 2, 4, 7, 9], [1, 2, 4, 5, 8], [0, 3, 7, 8, 9], [0, 1, 2, 4, 8], [0, 1, 3, 6, 8]]
unlab_classes_list = [[4, 5, 7, 8, 9], [1, 4, 5, 6, 8], [1, 3, 4, 8, 9], [0, 2, 4, 5, 9], [0, 2, 7, 8, 9], [1, 2, 4, 7, 9], [1, 2, 4, 5, 8], [0, 3, 7, 8, 9], [0, 1, 2, 4, 8], [0, 1, 3, 6, 8]]

count_labels = 7
while count_labels < 10:
    # TODO: Delete or rename previous model directory
    main_path = './model' + str(count_labels + 1)
    print(main_path)

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    temp_for = 0
    if count_labels == 7:
        temp_for = 8
    for i in range(temp_for, 10):

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

        #generator = FCGenerator(data.img_size_x, data.img_size_y, 
        #                        data.img_size_z)

        #critic = FCCritic("Critic", data.img_size_x, data.img_size_y,
        #                  data.img_size_z, num_lab_classes + 1)

        network = FCCritic("classifier", data.img_size_x, data.img_size_y,
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
   



