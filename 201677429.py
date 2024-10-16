import math
import copy
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time
import copy







id_ = 201677429

# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args(args=[])

# judge cuda is available or not
use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")

torch.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

############################################################################
################    don't change the below code    #####################
############################################################################
train_set = torchvision.datasets.FashionMNIST(root='data', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)


# define fully connected network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output


##############################################################################
#############    end of "don't change the below code"   ######################
##############################################################################

# generate adversarial data, you can define your adversarial method
def adv_attack(model, X, y, device):
    X_adv = Variable(X.data)

    ################################################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ################################################################################################

    # random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-0.1, 0.1).to(device)
    # X_adv = Variable(X_adv.data + random_noise)
    X = X.to(device)
    y = y.to(device)

    class Attack(object):
        r"""
        Base class for all attacks.
        .. note::
            It automatically set device to the device where given model is.
            It basically changes training mode to eval during attack process.
            To change this, please see `set_training_mode`.
        """

        def __init__(self, name, model):
            r"""
            Initializes internal attack state.
            Arguments:
                name (str): name of attack.
                model (torch.nn.Module): model to attack.
            """

            self.attack = name
            self.model = model
            self.model_name = str(model).split("(")[0]
            self.device = next(model.parameters()).device

            self._attack_mode = 'default'
            self._targeted = False
            self._return_type = 'float'
            self._supported_mode = ['default']

            self._model_training = False
            self._batchnorm_training = False
            self._dropout_training = False

        def forward(self, *input):
            r"""
            It defines the computation performed at every call.
            Should be overridden by all subclasses.
            """
            raise NotImplementedError

        def get_mode(self):
            r"""
            Get attack mode.
            """
            return self._attack_mode

        def set_mode_default(self):
            r"""
            Set attack mode as default mode.
            """
            self._attack_mode = 'default'
            self._targeted = False
            print("Attack mode is changed to 'default.'")

        def set_mode_targeted_by_function(self, target_map_function=None):
            r"""
            Set attack mode as targeted.
            Arguments:
                target_map_function (function): Label mapping function.
                    e.g. lambda images, labels:(labels+1)%10.
                    None for using input labels as targeted labels. (Default)
            """
            if "targeted" not in self._supported_mode:
                raise ValueError("Targeted mode is not supported.")

            self._attack_mode = 'targeted'
            self._targeted = True
            self._target_map_function = target_map_function
            print("Attack mode is changed to 'targeted.'")

        def set_mode_targeted_least_likely(self, kth_min=1):
            r"""
            Set attack mode as targeted with least likely labels.
            Arguments:
                kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)
            """
            if "targeted" not in self._supported_mode:
                raise ValueError("Targeted mode is not supported.")

            self._attack_mode = "targeted(least-likely)"
            self._targeted = True
            self._kth_min = kth_min
            self._target_map_function = self._get_least_likely_label
            print("Attack mode is changed to 'targeted(least-likely).'")

        def set_mode_targeted_random(self, n_classses=None):
            r"""
            Set attack mode as targeted with random labels.
            Arguments:
                num_classses (str): number of classes.
            """
            if "targeted" not in self._supported_mode:
                raise ValueError("Targeted mode is not supported.")

            self._attack_mode = "targeted(random)"
            self._targeted = True
            self._n_classses = n_classses
            self._target_map_function = self._get_random_target_label
            print("Attack mode is changed to 'targeted(random).'")

        def set_return_type(self, type):
            r"""
            Set the return type of adversarial images: `int` or `float`.
            Arguments:
                type (str): 'float' or 'int'. (Default: 'float')
            .. note::
                If 'int' is used for the return type, the file size of
                adversarial images can be reduced (about 1/4 for CIFAR10).
                However, if the attack originally outputs float adversarial images
                (e.g. using small step-size than 1/255), it might reduce the attack
                success rate of the attack.
            """
            if type == 'float':
                self._return_type = 'float'
            elif type == 'int':
                self._return_type = 'int'
            else:
                raise ValueError(type + " is not a valid type. [Options: float, int]")

        def set_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
            r"""
            Set training mode during attack process.
            Arguments:
                model_training (bool): True for using training mode for the entire model during attack process.
                batchnorm_training (bool): True for using training mode for batchnorms during attack process.
                dropout_training (bool): True for using training mode for dropouts during attack process.
            .. note::
                For RNN-based models, we cannot calculate gradients with eval mode.
                Thus, it should be changed to the training mode during the attack.
            """
            self._model_training = model_training
            self._batchnorm_training = batchnorm_training
            self._dropout_training = dropout_training

        def save(self, data_loader, save_path=None, verbose=True, return_verbose=False):
            r"""
            Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
            Arguments:
                save_path (str): save_path.
                data_loader (torch.utils.data.DataLoader): data loader.
                verbose (bool): True for displaying detailed information. (Default: True)
                return_verbose (bool): True for returning detailed information. (Default: False)
            """
            if (verbose == False) and (return_verbose == True):
                raise ValueError("Verobse should be True if return_verbose==True.")

            if save_path is not None:
                image_list = []
                label_list = []

            correct = 0
            total = 0
            l2_distance = []

            total_batch = len(data_loader)

            given_training = self.model.training

            for step, (images, labels) in enumerate(data_loader):
                start = time.time()
                adv_images = self.__call__(images, labels)

                batch_size = len(images)

                if save_path is not None:
                    image_list.append(adv_images.cpu())
                    label_list.append(labels.cpu())

                if self._return_type == 'int':
                    adv_images = adv_images.float() / 255

                if verbose:
                    with torch.no_grad():
                        if given_training:
                            self.model.eval()
                        outputs = self.model(adv_images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        right_idx = (predicted == labels.to(self.device))
                        correct += right_idx.sum()
                        end = time.time()
                        delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                        l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))

                        rob_acc = 100 * float(correct) / total
                        l2 = torch.cat(l2_distance).mean().item()
                        progress = (step + 1) / total_batch * 100
                        elapsed_time = end - start
                        self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')

            # To avoid erasing the printed information.
            if verbose:
                self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

            if save_path is not None:
                x = torch.cat(image_list, 0)
                y = torch.cat(label_list, 0)
                torch.save((x, y), save_path)
                print('- Save complete!')

            if given_training:
                self.model.train()

            if return_verbose:
                return rob_acc, l2, elapsed_time

        def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
            print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t' % (
                progress, rob_acc, l2, elapsed_time), end=end)

        def _get_target_label(self, images, labels=None):
            r"""
            Function for changing the attack mode.
            Return input labels.
            """
            if self._target_map_function:
                return self._target_map_function(images, labels)
            raise ValueError('Please define target_map_function.')

        def _get_least_likely_label(self, images, labels=None):
            r"""
            Function for changing the attack mode.
            Return least likely labels.
            """
            outputs = self.model(images)
            if self._kth_min < 0:
                pos = outputs.shape[1] + self._kth_min + 1
            else:
                pos = self._kth_min
            _, target_labels = torch.kthvalue(outputs.data, pos)
            target_labels = target_labels.detach()
            return target_labels.long().to(self.device)

        def _get_random_target_label(self, images, labels=None):
            if self._n_classses is None:
                outputs = self.model(images)
                if labels is None:
                    _, labels = torch.max(outputs, dim=1)
                n_classses = outputs.shape[-1]
            else:
                n_classses = self._n_classses

            target_labels = torch.zeros_like(labels)
            for counter in range(labels.shape[0]):
                l = list(range(n_classses))
                l.remove(labels[counter])
                t = self.random_int(0, len(l))
                target_labels[counter] = l[t]

            return target_labels.long().to(self.device)

        def random_int(self, low=0, high=1, shape=[1]):
            t = low + (high - low) * torch.rand(shape).to(self.device)
            return t.long()

        def _to_uint(self, images):
            r"""
            Function for changing the return type.
            Return images as int.
            """
            return (images * 255).type(torch.uint8)

        def __str__(self):
            info = self.__dict__.copy()

            del_keys = ['model', 'attack']

            for key in info.keys():
                if key[0] == "_":
                    del_keys.append(key)

            for key in del_keys:
                del info[key]

            info['attack_mode'] = self._attack_mode
            info['return_type'] = self._return_type

            return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

        def __call__(self, *input, **kwargs):
            given_training = self.model.training

            if self._model_training:
                self.model.train()
                for _, m in self.model.named_modules():
                    if not self._batchnorm_training:
                        if 'BatchNorm' in m.__class__.__name__:
                            m = m.eval()
                    if not self._dropout_training:
                        if 'Dropout' in m.__class__.__name__:
                            m = m.eval()

            else:
                self.model.eval()
            with torch.enable_grad():
                images = self.forward(*input, **kwargs)

            if given_training:
                self.model.train()

            if self._return_type == 'int':
                images = self._to_uint(images)

            return images

    # # PGD

    # In[16]:

    import torch
    import torch.nn as nn

    class PGD(Attack):
        r"""
        PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
        [https://arxiv.org/abs/1706.06083]
        Distance Measure : Linf
        Arguments:
            model (nn.Module): model to attack.
            eps (float): maximum perturbation. (Default: 0.3)
            alpha (float): step size. (Default: 2/255)
            steps (int): number of steps. (Default: 40)
            random_start (bool): using random initialization of delta. (Default: True)
        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
                       `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        Examples::
            >>>attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
            >>>adv_images = attack(images, labels)
        """

        def __init__(self, model, eps=0.3,
                     alpha=2 / 255, steps=40, random_start=True):
            super().__init__("PGD", model)
            self.eps = eps
            self.alpha = alpha
            self.steps = steps
            self.random_start = random_start
            self._supported_mode = ['default', 'targeted']

        def forward(self, images, labels):
            r"""
            Overridden.
            """
            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)

            if self._targeted:
                target_labels = self._get_target_label(images, labels)

            loss = nn.CrossEntropyLoss()

            adv_images = images.clone().detach()

            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)

                # Calculate loss
                if self._targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            return adv_images

    # # EOT + PGD

    # In[60]:

    class EOTPGD(Attack):
        r"""
        Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"
        [https://arxiv.org/abs/1907.00895]
        Distance Measure : Linf
        Arguments:
            model (nn.Module): model to attack.
            eps (float): maximum perturbation. (Default: 0.3)
            alpha (float): step size. (Default: 2/255)
            steps (int): number of steps. (Default: 40)
            eot_iter (int) : number of models to estimate the mean gradient. (Default: 10)
        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        Examples::
            >>> attack = torchattacks.EOTPGD(model, eps=4/255, alpha=8/255, steps=40, eot_iter=10)
            >>> adv_images = attack(images, labels)
        """

        def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40,
                     eot_iter=10, random_start=True):
            super().__init__("EOTPGD", model)
            self.eps = eps
            self.alpha = alpha
            self.steps = steps
            self.eot_iter = eot_iter
            self.random_start = random_start
            self._supported_mode = ['default', 'targeted']

        def forward(self, images, labels):
            r"""
            Overridden.
            """
            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)

            if self._targeted:
                target_labels = self._get_target_label(images, labels)

            loss = nn.CrossEntropyLoss()

            adv_images = images.clone().detach()

            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for _ in range(self.steps):
                grad = torch.zeros_like(adv_images)
                adv_images.requires_grad = True

                for j in range(self.eot_iter):
                    outputs = self.model(adv_images)

                    # Calculate loss
                    if self._targeted:
                        cost = -loss(outputs, target_labels)
                    else:
                        cost = loss(outputs, labels)

                    # Update adversarial images
                    grad += torch.autograd.grad(cost, adv_images,
                                                retain_graph=False,
                                                create_graph=False)[0]

                # (grad/self.eot_iter).sign() == grad.sign()
                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            return adv_images

    pgdAttack = PGD(model, eps=24/255, alpha=36/255, steps=4)
    X_adv = pgdAttack(X, y)

    # eotpgdAttack = EOTPGD(model, eps=8 / 255, alpha=2 / 255, steps=4)
    # X_adv = eotpgdAttack(X, y)

    ################################################################################################
    ## end of attack method
    ################################################################################################

    return X_adv



# train function, you can use adversarial training
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)

        # use adverserial data to train the defense model
        adv_data = adv_attack(model, data, target, device=device)

        # clear gradients
        optimizer.zero_grad()

        # compute loss
        loss = F.nll_loss(model(adv_data), target)
        # loss = F.nll_loss(model(data), target)

        # get gradients and update
        loss.backward()
        optimizer.step()


# predict function
def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_adv_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            adv_data = adv_attack(model, data, target, device=device)
            output = model(adv_data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# main function, train the dataset and print train loss, test loss for each epoch
def train_model():
    model = Net().to(device)

    ################################################################################################
    ## Note: below is the place you need to edit to implement your own training algorithm
    ##       You can also edit the functions such as train(...).
    ################################################################################################

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # training
        train(args, model, device, train_loader, optimizer, epoch)

        # get trnloss and testloss
        trnloss, trnacc = eval_test(model, device, train_loader)
        advloss, advacc = eval_adv_test(model, device, train_loader)

        # print trnloss and testloss
        print('Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))

    adv_tstloss, adv_tstacc = eval_adv_test(model, device, test_loader)
    print('Your estimated attack ability, by applying your attack method on your own trained model, is: {:.4f}'.format(
        1 / adv_tstacc))
    print('Your estimated defence ability, by evaluating your own defence model over your attack, is: {:.4f}'.format(
        adv_tstacc))
    ################################################################################################
    ## end of training method
    ################################################################################################

    # save the model
    torch.save(model.state_dict(), str(id_) + '.pt')
    return model


# compute perturbation distance
def p_distance(model, train_loader, device):
    p = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)
        data_ = copy.deepcopy(data.data)
        adv_data = adv_attack(model, data, target, device=device)
        p.append(torch.norm(data_ - adv_data, float('inf')))
    print('epsilon p: ', max(p))


################################################################################################
## Note: below is for testing/debugging purpose, please comment them out in the submission file
################################################################################################

# Comment out the following command when you do not want to re-train the model
# In that case, it will load a pre-trained model you saved in train_model()
model = train_model()

# Call adv_attack() method on a pre-trained model'
# the robustness of the model is evaluated against the infinite-norm distance measure
# important: MAKE SURE the infinite-norm distance (epsilon p) less than 0.11 !!!
p_distance(model, train_loader, device)
