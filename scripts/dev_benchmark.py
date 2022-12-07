import sys

sys.path.append('/zfsauton2/home/vsanil/projects/misc/aqua')

from aqua.data import load_cifar10_test, load_cifar10_train, load_cifar10H_softlabels, load_cifar10N_softlabels, Aqdata
from aqua.models import TrainAqModel, TestAqModel

#data_cifar = load_cifar10_test('/home/scratch/vsanil/aqua/datasets/cifar-10-batches-py/test_batch')
#labels = load_cifar10_softlabels('/home/scratch/vsanil/aqua/datasets/cifar-10h/data/cifar10h-raw.csv', agreement_threshold=0.9)

#aqd = Aqdata(data_cifar, labels)

# Load train data
data_cifar, label_cifar = load_cifar10_train('/home/scratch/vsanil/aqua/datasets/cifar-10-batches-py/')
labels_annot = load_cifar10N_softlabels('/home/scratch/vsanil/aqua/datasets/CIFAR-N/CIFAR-10_human.pt')

# Load test data
data_cifar_test, label_cifar_test = load_cifar10_test('/home/scratch/vsanil/aqua/datasets/cifar-10-batches-py/test_batch')
labels_annot_test = load_cifar10H_softlabels('/home/scratch/vsanil/aqua/datasets/cifar-10h/data/cifar10h-raw.csv', agreement_threshold=0.9)

print(data_cifar.shape, label_cifar.shape)
print(labels_annot.shape)

data_aq = Aqdata(data_cifar, label_cifar, labels_annot)
data_aq_test = Aqdata(data_cifar_test, label_cifar)

train_model = TrainAqModel('image', 'cleanlab', 'cifar10')
train_labels = train_model.fit_predict(data_aq.data, data_aq.labels)
print(train_labels.shape)
print("F1 Score: ")

raise KeyboardInterrupt

test_model = TestAqModel('image', 'cleanlab', train_model.wrapper_model)