import torch
import numpy as np
from torch.utils.data import DataLoader
from get_dataset import GetDataset, CustomDataset, BERTSQuADDataset, BERTIMDBDataset

class client(object):
    def __init__(self, client_idx, local_data, local_label, dev, nlp_transform=None, use_bert=False, dataset_name='squad'):
        self.client_idx = client_idx
        self.local_data = local_data
        self.local_label = local_label
        self.dev = dev
        self.use_bert = use_bert
        self.dataset_name = dataset_name
        
        # 根据是否使用 BERT 创建不同的数据集
        if use_bert:
            if isinstance(local_data[0], tuple) and len(local_data[0]) == 3:
                # BERT + SQuAD: 解包 (context, question, answer) 元组
                contexts = [item[0] for item in local_data]
                questions = [item[1] for item in local_data]
                answers = [item[2] for item in local_data]
                self.train_ds = BERTSQuADDataset(
                    contexts, 
                    questions, 
                    answers,
                    tokenizer_path='./bert_cache/bert-base-uncased-local',
                    max_length=384
                )
            else:
                # BERT + IMDB: 文本分类
                self.train_ds = BERTIMDBDataset(
                    local_data,
                    local_label,
                    tokenizer_path='./bert_cache/bert-base-uncased-local',
                    max_length=512
                )
        else:
            # LSTM 或其他：使用 CustomDataset
            self.train_ds = CustomDataset(local_data, local_label, nlp_transform=nlp_transform)

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, mu=0.01, if_prox=False):
        Net.train()
        Net.load_state_dict(global_parameters, strict=True)
        
        # 自定义 collate_fn 用于 SQuAD 和 BERT
        def squad_collate_fn(batch):
            if isinstance(batch[0], dict):
                # 检查是 SQuAD 还是分类任务
                if 'start_pos' in batch[0]:
                    # SQuAD 问答任务
                    if 'input_ids' in batch[0]:
                        # BERT + SQuAD 格式
                        input_ids = torch.stack([item['input_ids'] for item in batch])
                        attention_mask = torch.stack([item['attention_mask'] for item in batch])
                        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
                        start_pos = torch.stack([item['start_pos'] for item in batch])
                        end_pos = torch.stack([item['end_pos'] for item in batch])
                        return input_ids, attention_mask, token_type_ids, start_pos, end_pos
                    else:
                        # LSTM + SQuAD 格式
                        contexts = torch.stack([item['context'] for item in batch])
                        questions = torch.stack([item['question'] for item in batch])
                        start_pos = torch.stack([item['start_pos'] for item in batch])
                        end_pos = torch.stack([item['end_pos'] for item in batch])
                        return contexts, questions, start_pos, end_pos
                elif 'label' in batch[0]:
                    # BERT + IMDB 分类任务
                    return {
                        'input_ids': torch.stack([item['input_ids'] for item in batch]),
                        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                        'token_type_ids': torch.stack([item['token_type_ids'] for item in batch]),
                        'label': torch.stack([item['label'] for item in batch])
                    }
            else:
                return torch.utils.data.dataloader.default_collate(batch)
        
        # 检查数据集类型
        sample = self.train_ds[0]
        if isinstance(sample, dict):
            self.train_dl = DataLoader(
                self.train_ds, 
                batch_size=localBatchSize, 
                shuffle=True, 
                collate_fn=squad_collate_fn,
                num_workers=4,        # 添加多进程加载
                pin_memory=True,      # 加速数据传输到GPU
                persistent_workers=True  # 保持worker进程
            )
        else:
            self.train_dl = DataLoader(
                self.train_ds, 
                batch_size=localBatchSize, 
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
        
        for epoch in range(localEpoch):
            for batch_idx, batch_data in enumerate(self.train_dl):
                if isinstance(sample, dict):
                    # 字典格式数据
                    if 'start_pos' in sample:
                        # SQuAD 问答任务
                        if 'input_ids' in sample:
                            # BERT + SQuAD
                            input_ids, attention_mask, token_type_ids, start_pos, end_pos = batch_data
                            input_ids = input_ids.to(self.dev)
                            attention_mask = attention_mask.to(self.dev)
                            token_type_ids = token_type_ids.to(self.dev)
                            start_pos, end_pos = start_pos.to(self.dev), end_pos.to(self.dev)
                            
                            if epoch == 0 and batch_idx == 0:
                                print(f"Client {self.client_idx}: [BERT] input_ids形状={input_ids.shape}")
                                print(f"start_pos范围: {start_pos.min()}-{start_pos.max()}, end_pos范围: {end_pos.min()}-{end_pos.max()}")
                            
                            start_logits, end_logits = Net(input_ids, attention_mask, token_type_ids)
                            total_loss_batch = lossFun(start_logits, end_logits, start_pos, end_pos)
                        else:
                            # LSTM + SQuAD
                            context, question, start_pos, end_pos = batch_data
                            context, question = context.to(self.dev), question.to(self.dev)
                            start_pos, end_pos = start_pos.to(self.dev), end_pos.to(self.dev)
                            
                            if epoch == 0 and batch_idx == 0:
                                print(f"Client {self.client_idx}: [LSTM] context形状={context.shape}, question形状={question.shape}")
                                print(f"start_pos范围: {start_pos.min()}-{start_pos.max()}, end_pos范围: {end_pos.min()}-{end_pos.max()}")
                            
                            start_logits, end_logits = Net(context, question)
                            total_loss_batch = lossFun(start_logits, end_logits, start_pos, end_pos)
                    elif 'label' in sample:
                        # BERT + IMDB (分类任务)
                        # batch_data 是 collate_fn 处理后的字典
                        input_ids = batch_data['input_ids'].to(self.dev)
                        attention_mask = batch_data['attention_mask'].to(self.dev)
                        token_type_ids = batch_data['token_type_ids'].to(self.dev)
                        labels = batch_data['label'].to(self.dev)
                        
                        if epoch == 0 and batch_idx == 0:
                            print(f"Client {self.client_idx}: [BERT Classification] input_ids形状={input_ids.shape}")
                            print(f"标签唯一值: {torch.unique(labels)}")
                        
                        logits = Net(input_ids, attention_mask, token_type_ids)
                        total_loss_batch = lossFun(logits, labels)
                    else:
                        raise ValueError(f"Unknown dict format in batch_data: {sample.keys()}")
                else:
                    # IMDB + LSTM 或其他
                    data, label = batch_data
                    data, label = data.to(self.dev), label.to(self.dev)
                    
                    if epoch == 0 and batch_idx == 0:
                        print(f"Client {self.client_idx}: 数据形状={data.shape}, 标签形状={label.shape}")
                        print(f"标签唯一值: {torch.unique(label)}")
                    
                    preds = Net(data)
                    total_loss_batch = lossFun(preds, label)
                
                opti.zero_grad()
                total_loss_batch.backward()
                opti.step()
        
        return total_loss_batch.item(), Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev, use_bert=False):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.use_bert = use_bert
        self.clients_set = {}

        # 加载数据集
        self.dataset = GetDataset(dataSetName, '../data', use_bert=use_bert)
        self.train_dataset, self.test_dataset = self.dataset.get_dataset()

        if self.data_set_name in ['mnist', 'cifar10']:
            self.train_data = self.train_dataset.data
            self.train_label = self.train_dataset.targets
            self.nlp_transform = None
            # 创建测试数据加载器
            self.test_data_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)
            
        elif self.data_set_name == 'imdb':
            # 从 GetDataset 获取原始文本数据
            raw_train_data, raw_train_labels = self.dataset.get_raw_data('train')
            raw_test_data, raw_test_labels = self.dataset.get_raw_data('test')
            
            if not self.use_bert:
                # LSTM: 需要构建词汇表
                print("正在为整个数据集构建全局词汇表...")
                from utils.nlp_transform import NLPTransform
                self.nlp_transform = NLPTransform(max_length=256)
                self.nlp_transform.build_vocab(raw_train_data, max_vocab_size=10000)
                print(f"全局词汇表大小: {self.nlp_transform.vocab_size}")
                
                # 创建测试数据集（使用共享的词汇表）
                from get_dataset import CustomDataset
                test_dataset_with_vocab = CustomDataset(raw_test_data, raw_test_labels, nlp_transform=self.nlp_transform)
                self.test_data_loader = DataLoader(test_dataset_with_vocab, batch_size=100, shuffle=False)
            else:
                # BERT: 不需要词汇表
                print("使用 BERT tokenizer 处理 IMDB 数据集...")
                self.nlp_transform = None
                
                # 创建 BERT 测试数据集
                test_dataset_bert = BERTIMDBDataset(
                    raw_test_data,
                    raw_test_labels,
                    tokenizer_path='./bert_cache/bert-base-uncased-local',
                    max_length=512
                )
                self.test_data_loader = DataLoader(test_dataset_bert, batch_size=100, shuffle=False)
            
            # 保存原始数据用于分配给客户端
            self.train_data = raw_train_data
            self.train_label = raw_train_labels
            
        elif self.data_set_name == 'squad':
            # SQuAD 数据集已经在 GetDataset 中构建了词汇表
            self.nlp_transform = self.dataset.nlp_transform
            
            # 获取原始数据用于分配给客户端
            raw_train_contexts, raw_train_questions, raw_train_answers = self.dataset.get_raw_data('train')
            
            # 将 contexts, questions, answers 组合成一个列表
            self.train_data = list(zip(raw_train_contexts, raw_train_questions, raw_train_answers))
            self.train_label = None  # SQuAD 不需要单独的 label
            
            # 创建测试数据加载器
            def squad_collate_fn(batch):
                # 检查是 BERT 还是 LSTM 格式
                if 'input_ids' in batch[0]:
                    # BERT 格式
                    input_ids = torch.stack([item['input_ids'] for item in batch])
                    attention_mask = torch.stack([item['attention_mask'] for item in batch])
                    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
                    start_pos = torch.stack([item['start_pos'] for item in batch])
                    end_pos = torch.stack([item['end_pos'] for item in batch])
                    return input_ids, attention_mask, token_type_ids, start_pos, end_pos
                else:
                    # LSTM 格式
                    contexts = torch.stack([item['context'] for item in batch])
                    questions = torch.stack([item['question'] for item in batch])
                    start_pos = torch.stack([item['start_pos'] for item in batch])
                    end_pos = torch.stack([item['end_pos'] for item in batch])
                    return contexts, questions, start_pos, end_pos
            
            self.test_data_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=False, collate_fn=squad_collate_fn)
        
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        if self.data_set_name in ['mnist', 'cifar10']:
            train_data_size = len(self.train_data)
            shard_size = train_data_size // self.num_of_clients // 2
            shards_id = np.random.permutation(train_data_size // shard_size)
            
            for i in range(self.num_of_clients):
                shards_id1 = shards_id[i * 2]
                shards_id2 = shards_id[i * 2 + 1]
                data_shards1 = self.train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                data_shards2 = self.train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                label_shards1 = self.train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                label_shards2 = self.train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                local_data = torch.cat((data_shards1, data_shards2), dim=0)
                local_label = torch.cat((label_shards1, label_shards2), dim=0)
                
                someone = client(
                    client_idx=i,
                    local_data=local_data,
                    local_label=local_label,
                    dev=self.dev,
                    dataset_name=self.data_set_name
                )
                self.clients_set['client{}'.format(i)] = someone

        elif self.data_set_name == 'imdb':
            # 将数据按标签分组
            data_array = np.array(self.train_data)
            label_array = np.array(self.train_label)
            
            # 找到正类和负类的索引
            pos_indices = np.where(label_array == 1)[0]
            neg_indices = np.where(label_array == 0)[0]
            
            print(f"总样本数: {len(data_array)}, 正类: {len(pos_indices)}, 负类: {len(neg_indices)}")
            
            # 打乱索引
            np.random.seed(42)
            np.random.shuffle(pos_indices)
            np.random.shuffle(neg_indices)
            
            # 为每个客户端分配均衡的正负样本
            pos_per_client = len(pos_indices) // self.num_of_clients
            neg_per_client = len(neg_indices) // self.num_of_clients
            
            for client_idx in range(self.num_of_clients):
                # 获取该客户端的正负样本索引
                pos_start = client_idx * pos_per_client
                pos_end = (client_idx + 1) * pos_per_client if client_idx != self.num_of_clients - 1 else len(pos_indices)
                
                neg_start = client_idx * neg_per_client
                neg_end = (client_idx + 1) * neg_per_client if client_idx != self.num_of_clients - 1 else len(neg_indices)
                
                client_pos_indices = pos_indices[pos_start:pos_end]
                client_neg_indices = neg_indices[neg_start:neg_end]
                
                # 合并正负样本索引
                client_indices = np.concatenate([client_pos_indices, client_neg_indices])
                np.random.shuffle(client_indices)  # 再次打乱以混合正负样本
                
                # 获取该客户端的数据
                local_data = [data_array[i] for i in client_indices]
                local_label = [label_array[i] for i in client_indices]
                
                # 传入共享的 nlp_transform (LSTM) 或 use_bert (BERT)
                self.clients_set['client{}'.format(client_idx)] = client(
                    client_idx=client_idx,
                    local_data=local_data,
                    local_label=local_label,
                    dev=self.dev,
                    nlp_transform=self.nlp_transform,
                    use_bert=self.use_bert,
                    dataset_name=self.data_set_name
                )
        
        elif self.data_set_name == 'squad':
            # 为 SQuAD 分配数据
            data_len = len(self.train_data)
            samples_per_client = data_len // self.num_of_clients
            
            # 打乱数据
            indices = np.random.permutation(data_len)
            
            for client_idx in range(self.num_of_clients):
                start_idx = client_idx * samples_per_client
                end_idx = start_idx + samples_per_client if client_idx != self.num_of_clients - 1 else data_len
                
                client_indices = indices[start_idx:end_idx]
                
                # 获取该客户端的数据
                local_data = [self.train_data[i] for i in client_indices]
                
                print(f"Client {client_idx}: 样本数={len(local_data)}")
                
                # 根据是否使用 BERT 传递不同参数
                if self.use_bert:
                    # BERT 模式：不需要 nlp_transform，直接传递数据
                    self.clients_set['client{}'.format(client_idx)] = client(
                        client_idx=client_idx,
                        local_data=local_data,
                        local_label=None,
                        dev=self.dev,
                        nlp_transform=None,  # BERT 不使用 nlp_transform
                        use_bert=True,
                        dataset_name=self.data_set_name
                    )
                else:
                    # LSTM 模式：传入共享的 nlp_transform
                    self.clients_set['client{}'.format(client_idx)] = client(
                        client_idx=client_idx,
                        local_data=local_data,
                        local_label=None,
                        dev=self.dev,
                        nlp_transform=self.nlp_transform,
                        dataset_name=self.data_set_name
                    )
        else:
            raise ValueError("data_set_name must be MNIST, CIFAR10, IMDB or SQuAD")

    def accuracy_test(self, net):
        """评估函数 - 根据数据集类型选择不同的评估方式"""
        was_training = net.training
        net.eval()
        
        accuracy = 0.0
        
        # ✅ SQuAD 评估
        if self.data_set_name == 'squad':
            exact_match = 0
            f1_total = 0
            total = 0
            
            with torch.no_grad():
                for batch_data in self.test_data_loader:
                    # 检查是 BERT 还是 LSTM 格式
                    if len(batch_data) == 5:
                        # BERT 格式
                        input_ids, attention_mask, token_type_ids, start_pos, end_pos = batch_data
                        input_ids = input_ids.to(self.dev)
                        attention_mask = attention_mask.to(self.dev)
                        token_type_ids = token_type_ids.to(self.dev)
                        start_pos = start_pos.to(self.dev)
                        end_pos = end_pos.to(self.dev)
                        
                        start_logits, end_logits = net(input_ids, attention_mask, token_type_ids)
                    else:
                        # LSTM 格式
                        context, question, start_pos, end_pos = batch_data
                        context = context.to(self.dev)
                        question = question.to(self.dev)
                        start_pos = start_pos.to(self.dev)
                        end_pos = end_pos.to(self.dev)
                        
                        start_logits, end_logits = net(context, question)
                    
                    pred_start = torch.argmax(start_logits, dim=1)
                    pred_end = torch.argmax(end_logits, dim=1)
                    
                    # 确保 end >= start
                    for i in range(len(pred_start)):
                        if pred_end[i] < pred_start[i]:
                            pred_end[i] = pred_start[i]
                    
                    # 计算 EM (精确匹配)
                    exact_match += ((pred_start == start_pos) & (pred_end == end_pos)).sum().item()
                    
                    # 计算 F1
                    for i in range(len(pred_start)):
                        pred_set = set(range(pred_start[i].item(), pred_end[i].item() + 1))
                        gold_set = set(range(start_pos[i].item(), end_pos[i].item() + 1))
                        
                        if len(pred_set & gold_set) > 0:
                            precision = len(pred_set & gold_set) / len(pred_set)
                            recall = len(pred_set & gold_set) / len(gold_set)
                            f1 = 2 * precision * recall / (precision + recall)
                            f1_total += f1
                    
                    total += start_pos.size(0)
            
            em_score = 100 * exact_match / total if total > 0 else 0.0
            f1_score = 100 * f1_total / total if total > 0 else 0.0
            
            print(f"  EM: {em_score:.2f}%, F1: {f1_score:.2f}%")
            accuracy = em_score  # 使用 EM 作为准确率
        
        # ✅ MNIST/CIFAR10/IMDB 评估
        else:
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in self.test_data_loader:
                    # 检查数据格式：BERT 返回字典，LSTM/CNN 返回元组
                    if isinstance(batch, dict):
                        # BERT 格式
                        input_ids = batch['input_ids'].to(self.dev)
                        attention_mask = batch['attention_mask'].to(self.dev)
                        token_type_ids = batch['token_type_ids'].to(self.dev)
                        labels = batch['label'].to(self.dev)
                        
                        outputs = net(input_ids, attention_mask, token_type_ids)
                    else:
                        # LSTM/CNN 格式
                        data, labels = batch
                        data = data.to(self.dev)
                        labels = labels.to(self.dev)
                        
                        outputs = net(data)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total if total > 0 else 0.0
        
        if was_training:
            net.train()
        
        return accuracy
    
    def _evaluate_squad(self, net):
        """SQuAD 专属评估"""
        exact_match = 0
        f1_total = 0
        total = 0
        
        with torch.no_grad():
            for context, question, start_pos, end_pos in self.test_data_loader:
                # SQuAD 评估逻辑
                pass
        
        return 100 * exact_match / total
    
    def _evaluate_classification(self, net):
        """分类任务评估（MNIST/CIFAR10/IMDB）"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in self.test_data_loader:
                # 分类评估逻辑
                pass
        
        return 100 * correct / total


if __name__ == "__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])
