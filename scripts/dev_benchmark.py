import os, sys

from sklearn.metrics import f1_score

sys.path.append('/zfsauton2/home/vsanil/projects/misc/aqua')

from aqua.data import load_cifar10_test, load_cifar10_train, load_cifar10H_softlabels, load_cifar10N_softlabels, Aqdata
from aqua.models import TrainAqModel, TestAqModel

data_pardir = '/home/extra_scratch/vsanil/aqua/datasets/'

def load_cifar():
    # Load train data
    data_cifar, label_cifar = load_cifar10_train(os.path.join(data_pardir, 'cifar-10-batches-py/'))
    labels_annot = load_cifar10N_softlabels(os.path.join(data_pardir, 'CIFAR-N/CIFAR-10_human.pt'))

    # Load test data
    data_cifar_test, label_cifar_test = load_cifar10_test(os.path.join(data_pardir, 'cifar-10-batches-py/test_batch'))
    labels_annot_test = load_cifar10H_softlabels(os.path.join(data_pardir, 'cifar-10h/data/cifar10h-raw.csv'), agreement_threshold=0.9)

    return Aqdata(data_cifar, label_cifar, labels_annot), Aqdata(data_cifar_test, label_cifar_test, labels_annot_test)


data_dict = {
    'cifar10' : load_cifar()
}


for dataset in ['cifar10']:
    modality = None
    if dataset in ['cifar10', 'noisycxr']:
        modality = 'image'

    data_aq, data_aq_test = data_dict[dataset]

    noisy_model = TrainAqModel(modality,
                                'noisy',
                                dataset,
                                'cuda:5')

    noisy_train_labels = noisy_model.fit_predict(data_aq.data, data_aq.labels)
    noisy_test_labels = noisy_model.predict(data_aq_test.data)

    for method in ['cleanlab']:
        train_model = TrainAqModel(modality,
                                    method,
                                    dataset,
                                    'cuda:5')

        train_labels = train_model.fit_predict(data_aq.data, data_aq.labels)
        test_labels = train_model.predict(data_aq_test.data)

        print(f"{method} F1 Score: ", f1_score(noisy_test_labels, test_labels, average='weighted'))


raise KeyboardInterrupt

test_model = TestAqModel('image', 'cleanlab', train_model.wrapper_model)