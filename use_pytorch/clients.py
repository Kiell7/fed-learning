import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import ImageClassifierOutput

from get_dataset import GetDataset,CustomDataset
import utils.transform as transform

class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters,mu=0.01,if_prox=False):
        # 保存全局参数用于近端项计算
        global_params_copy = {}
        for key, param in global_parameters.items():
            global_params_copy[key] = param.clone().to(self.dev)

        # 加载全局参数到本地模型
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)

                # 前向传播
                preds = Net(data)
                # 这里面如果是从transformers加载的模型，输出为一个特定的类，而非直接给出预测结果
                if not isinstance(preds, torch.Tensor):
                    preds = preds.logits
                loss = lossFun(preds, label)

                if if_prox:
                    # FedProx近端项：计算当前参数与全局参数的差异
                    prox_term = 0.0
                    for name, param in Net.named_parameters():
                        if name in global_params_copy:
                            prox_term += torch.norm(param - global_params_copy[name]) ** 2
                    # 总损失 = 原始损失 + 近端正则项
                    total_loss = loss + (mu / 2) * prox_term
                else:
                    total_loss = loss

                # 反向传播
                total_loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        if self.data_set_name == "mnist":
            train_dataset, test_dataset = GetDataset("mnist","./data").get_dataset()
            client_transform = transform.mnist_transform_client
            train_data = train_dataset.train_data
            train_label = train_dataset.train_labels
        elif self.data_set_name == "cifar10":
            train_dataset, test_dataset = GetDataset("cifar10", "./data").get_dataset()
            client_transform = transform.cifar10_transform_client
            train_data = torch.tensor(train_dataset.data, dtype=torch.float32).permute(0, 3, 1, 2)
            train_label = torch.tensor(train_dataset.targets, dtype=torch.int64)
        else:
            raise ValueError("data_set_name must be MNIST or CIFAR10")
        self.test_data_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


        train_data_size = len(train_data)

        shard_size = train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(train_data_size // shard_size)
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data = torch.cat((data_shards1, data_shards2), dim=0)
            local_label = torch.cat((label_shards1, label_shards2), dim=0)
            someone = client(CustomDataset(local_data, local_label, client_transform), self.dev)
            self.clients_set['client{}'.format(i)] = someone


if __name__ == "__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])
