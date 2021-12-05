import random

import torch
from torch import nn
import numpy as np
from transformers import BertModel, RobertaModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss


def l2norm(x: torch.Tensor):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x


class SCLModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SCLModel, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.T = 0.3

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        labels = labels
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        P_mask = labels[:, None]==labels[None]
        N_mask = P_mask.eq(0)

        S = torch.matmul(logits, logits.transpose(0, 1))/self.T
        S = S.exp()

        pos_scores = S*P_mask
        neg_scores = (S*N_mask).sum(dim=-1, keepdim=True)
        no_neg_mask = N_mask.sum(dim=-1, keepdim=True).eq(0)
        loss_con = (pos_scores.masked_fill(no_neg_mask, 0) / (neg_scores + 1e-12)).masked_fill(N_mask, 0).sum()/P_mask.sum()

        loss_cls = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss_cls = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss_cls = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        loss = loss_con * self.config.contrastive_rate_in_training + \
               loss_cls * (1 - self.config.contrastive_rate_in_training)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class SCLRoBertaModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super(SCLRoBertaModel, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.T = 0.3
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        labels = labels
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        P_mask = labels[:, None]==labels[None]
        N_mask = P_mask.eq(0)

        S = torch.matmul(logits, logits.transpose(0, 1)) / self.T
        S = S.exp()

        pos_scores = S*P_mask
        neg_scores = (S*N_mask).sum(dim=-1, keepdim=True)
        no_neg_mask = N_mask.sum(dim=-1, keepdim=True).eq(0)
        loss_con = (pos_scores.masked_fill(no_neg_mask, 0) / (neg_scores + 1e-12)).masked_fill(N_mask, 0).sum()/P_mask.sum()

        loss_cls = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss_cls = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss_cls = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        loss = loss_con * self.config.contrastive_rate_in_training + \
               loss_cls * (1 - self.config.contrastive_rate_in_training)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class ContrastiveMoCo(nn.Module):

    def __init__(self, config):
        super(ContrastiveMoCo, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        if config.load_trained_model:
            self.encoder_q = BertModel(config)
            self.encoder_k = BertModel(config)
        else:
            self.encoder_q = BertModel.from_pretrained(config.model_name, config=config)
            self.encoder_k = BertModel.from_pretrained(config.model_name, config=config)

        self.classifier_liner = ClassificationHead(config, self.num_labels)

        self.contrastive_liner_q = ContrastiveHead(config)
        self.contrastive_liner_k = ContrastiveHead(config)

        self.m = 0.999
        self.T = 0.07
        self.train_multi_head = config.train_multi_head
        self.multi_head_num = config.multi_head_num

        if not config.load_trained_model:
            self.init_weights()

        # create the label_queue and feature_queue
        self.K = config.queue_size

        self.register_buffer("label_queue", torch.randint(0, self.num_labels, [self.K]))
        self.register_buffer("feature_queue", torch.randn(self.K, config.hidden_size))
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.positive_num = config.positive_num
        self.memory_bank = config.memory_bank

    def _dequeue_and_enqueue(self, keys, label):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            head_size = self.K - ptr
            head_keys = keys[: head_size]
            head_label = label[: head_size]
            end_size = ptr + batch_size - self.K
            end_key = keys[head_size:]
            end_label = label[head_size:]
            self.feature_queue[ptr:, :] = head_keys
            self.label_queue[ptr:] = head_label
            self.feature_queue[:end_size, :] = end_key
            self.label_queue[:end_size] = end_label
        else:
            # replace the keys at ptr (dequeue ans enqueue)
            self.feature_queue[ptr: ptr + batch_size, :] = keys
            self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def select_negative_sample(self, liner_q: torch.Tensor,  label_q: torch.Tensor): # batch * 1
        label_queue = self.label_queue.clone().detach() # train_size * 1
        feature_queue = self.feature_queue.clone().detach()
        # 1、将label_queue 扩展到 batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1])

        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 2、将label扩展到batch_size * K
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat((1, self.K))

        # 3、根据label取mask_index
        neg_mask_index = torch.ne(tmp_label_queue, tmp_label)

        # 4、根据mask_index取features

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None

        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)

        return neg_sample

    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self,
                query,
                positive_sample=None,        # batch_size * max_length
                negative_sample=None,
                ):

        labels = query["labels"]
        labels = labels.view(-1)

        if not self.memory_bank:
            with torch.no_grad():
                self.update_encoder_k()
                bert_output_p = self.encoder_k(**positive_sample)
                update_keys = bert_output_p[1]
                update_keys = self.contrastive_liner_k(update_keys)
                update_keys = l2norm(update_keys)
                self._dequeue_and_enqueue(update_keys, labels)
        else:
            with torch.no_grad():
                bert_output_p = self.encoder_q(**positive_sample)
                update_keys = bert_output_p[1]
                update_keys = self.contrastive_liner_q(update_keys)
                update_keys = l2norm(update_keys)

        query.pop("labels")
        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]  # batch * 768
        liner_q = self.contrastive_liner_q(q)
        liner_q = l2norm(liner_q)
        logits_cls = self.classifier_liner(q)

        loss_cls = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss_cls = loss_fct(logits_cls.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss_cls = loss_fct(logits_cls.view(-1, self.num_labels), labels.view(-1))

        l_pos = torch.einsum('nc,nc->n', [liner_q, update_keys]).unsqueeze(-1)
        l_neg = self.select_negative_sample(liner_q, labels)
        if l_neg is not None:
            logits_con = torch.cat([l_pos, l_neg], dim=1)
            logits_con /= self.T
            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
            loss_fct = CrossEntropyLoss()
            loss_con = loss_fct(logits_con, labels_con)
            loss = loss_con * self.config.contrastive_rate_in_training + \
                   loss_cls * (1 - self.config.contrastive_rate_in_training)
        else:
            loss = loss_cls
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits_cls
        )

    def predict(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                ):
        bert_output_q = self.encoder_q(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        q = bert_output_q[1]
        logits_cls = self.classifier_liner(q)
        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits_cls.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss_cls,
            logits=logits_cls,
        )

    def get_features(self, query):
        with torch.no_grad():
            bert_output_k = self.encoder_k(**query)
            contrastive_output = self.contrastive_liner_k(bert_output_k[1])
        return contrastive_output

    def update_queue_by_bert(self,
                             inputs=None,
                             labels=None
                             ):
        with torch.no_grad():
            # update_sample = self.reshape_dict(inputs)
            roberta_output = self.encoder_k(**inputs)
            update_keys = roberta_output[1]
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(update_keys, tmp_labels)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ContrastiveHead(nn.Module):
    def __init__(self, config):
        super(ContrastiveHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ContrastiveRoBertaMoCo(nn.Module):

    def __init__(self, config):
        super(ContrastiveRoBertaMoCo, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        if config.load_trained_model:
            self.encoder_q = RobertaModel(config)
            self.encoder_k = RobertaModel(config)
        else:
            self.encoder_q = RobertaModel(config).from_pretrained(config.model_name, config=config)
            self.encoder_k = RobertaModel(config).from_pretrained(config.model_name, config=config)

        self.classifier = ClassificationHead(config)

        self.contrastive_liner_q = ContrastiveHead(config)
        self.contrastive_liner_k = ContrastiveHead(config)

        self.m = 0.999
        self.T = 0.07
        self.train_multi_head = config.train_multi_head
        self.multi_head_num = config.multi_head_num

        if not config.load_trained_model:
            self.init_weights()

        # create the label_queue and feature_queue
        self.K = config.queue_size

        self.register_buffer("label_queue", torch.randint(0, self.num_labels, [self.K]))
        self.register_buffer("feature_queue", torch.randn(config.hidden_size, self.K))
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.positive_num = config.positive_num

    def _dequeue_and_enqueue(self, keys, label):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[: batch_size]
            label = label[: batch_size]

        # replace the keys at ptr (dequeue ans enqueue)
        self.feature_queue[:, ptr: ptr + batch_size] = keys.T
        self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def select_negative_sample(self, label): # batch * 1
        label_queue = self.label_queue.clone().detach() # train_size * 1
        feature_queue = self.feature_queue.clone().detach().T
        # 1、将label_queue 扩展到 batch_size * K
        batch_size = label.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])

        # 2、将label扩展到batch_size * K
        tmp_label = label.unsqueeze(1)
        tmp_label = tmp_label.repeat((1, self.K))

        # 3、根据label取mask_index
        mask_index = torch.ne(tmp_label_queue, tmp_label)

        # 4、根据mask_index取features
        tmp_feature = feature_queue.unsqueeze(0)
        tmp_feature = tmp_feature.repeat([batch_size, 1, 1]) # batch_size * K * hidden_zise
        tmp_index = mask_index.unsqueeze(-1)
        tmp_index = tmp_index.repeat([1, 1, self.config.hidden_size])
        feature_value = tmp_feature.masked_select(tmp_index)

        # 5、根据mask_index信息将feature填入一个负样本的tensor中 tensor的shape为[batch_size * K * hidden_size]
        negative_sample = torch.zeros([batch_size, self.K, self.config.hidden_size]).to("cuda")
        negative_sample = negative_sample.masked_scatter(tmp_index, feature_value)

        return negative_sample

    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def forward(self,
                inputs,
                positive_sample=None,  # batch_size * max_length
                negative_sample=None,
                ):
        labels = inputs["labels"]
        query = inputs.copy()
        labels = labels.view(-1)
        query.pop("labels")
        roberta_output_q = self.encoder_q(**query)[1]
        liner_q = self.contrastive_liner_q(roberta_output_q)
        logits_cls = self.classifier(roberta_output_q)

        loss_cls = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.num_labels), labels.view(-1))

        loss_con = None
        if positive_sample is not None:
            with torch.no_grad():
                self.update_encoder_k()
                roberta_output_p = self.encoder_k(**positive_sample)
                p = roberta_output_p[1]
                liner_p = self.contrastive_liner_k(p)

                liner_n = self.select_negative_sample(labels)

            liner_p = l2norm(liner_p)
            liner_q = l2norm(liner_q)

            liner_p = l2norm(liner_p)
            liner_q = l2norm(liner_q)

            l_pos = torch.einsum('nc,nc->n', [liner_q, liner_p]).unsqueeze(-1)

            # negative logits: NxK
            l_neg = torch.einsum('nc,nkc->nk', [liner_q, liner_n])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            con_labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # loss_fct = CrossEntropyLoss(ignore_index=0)
            loss_fct = CrossEntropyLoss()
            loss_con = loss_fct(logits, con_labels, )

            self._dequeue_and_enqueue(liner_p, labels)

        if loss_con is None:
            return SequenceClassifierOutput(
                loss=loss_cls,
                logits=logits_cls,
            )
        if loss_cls is None:
            return SequenceClassifierOutput(
                loss=loss_con,
                logits=logits_cls,
            )

        loss = loss_con * self.config.contrastive_rate_in_training + \
               loss_cls * (1 - self.config.contrastive_rate_in_training)

        return SequenceClassifierOutput(
            loss=loss
        )

    def predict(self, query):
        with torch.no_grad():
            bert_output_q = self.encoder_q(**query)
            con_q = self.contrastive_liner_q(bert_output_q[0])
            cls_q = self.classifier(bert_output_q[0])
        return con_q, cls_q

    def get_features(self, query):
        with torch.no_grad():
            bert_output_k = self.encoder_k(**query)
            con_k = self.contrastive_liner_k(bert_output_k[0])
        return con_k


class ContrastiveMoCoKnnBert(nn.Module):

    def __init__(self, config):
        super(ContrastiveMoCoKnnBert, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        if config.load_trained_model:
            self.encoder_q = BertModel(config)
            self.encoder_k = BertModel(config)
        else:
            self.encoder_q = BertModel.from_pretrained(config.model_name, config=config)
            self.encoder_k = BertModel.from_pretrained(config.model_name, config=config)

        self.classifier_liner = ClassificationHead(config, self.num_labels)

        self.contrastive_liner_q = ContrastiveHead(config)
        self.contrastive_liner_k = ContrastiveHead(config)

        self.m = 0.999
        self.T = 0.07
        self.train_multi_head = config.train_multi_head
        self.multi_head_num = config.multi_head_num

        if not config.load_trained_model:
            self.init_weights()

        # create the label_queue and feature_queue
        self.K = config.queue_size

        self.register_buffer("label_queue", torch.randint(0, self.num_labels, [self.K]))
        self.register_buffer("feature_queue", torch.randn(self.K, config.hidden_size))
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.top_k = config.knn_num
        self.end_k = config.end_k
        self.update_num = config.positive_num

        self.memory_bank = config.memory_bank
        self.random_positive = config.random_positive

    def _dequeue_and_enqueue(self, keys, label):
        # TODO 我们训练过程batch_size是一个变动的，每个epoch的最后一个batch数目后比较少，这里需要进一步修改
        # keys = concat_all_gather(keys)
        # label = concat_all_gather(label)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            head_size = self.K - ptr
            head_keys = keys[: head_size]
            head_label = label[: head_size]
            end_size = ptr + batch_size - self.K
            end_key = keys[head_size:]
            end_label = label[head_size:]
            self.feature_queue[ptr:, :] = head_keys
            self.label_queue[ptr:] = head_label
            self.feature_queue[:end_size, :] = end_key
            self.label_queue[:end_size] = end_label
        else:
            # replace the keys at ptr (dequeue ans enqueue)
            self.feature_queue[ptr: ptr + batch_size, :] = keys
            self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def select_pos_neg_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()        # K
        feature_queue = self.feature_queue.clone().detach()    # K * hidden_size

        # 1、将label_queue和feature_queue扩展到batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * K * hidden_size

        # 2、计算相似度
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3、根据label取正样本和负样本的mask_index
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        # 4、根据mask_index取正样本和负样本的值
        feature_value = cos_sim.masked_select(pos_mask_index)
        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5、取所有的负样本和前top_k 个正样本， -M个正样本（离中心点最远的样本）
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.top_k]
        pos_sample_last = pos_sample[:, -self.end_k:]
        # pos_sample_last = pos_sample_last.view([-1, 1])

        pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_sample = neg_sample.repeat([1, self.top_k + self.end_k])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def select_pos_neg_random(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()        # K
        feature_queue = self.feature_queue.clone().detach()    # K * hidden_size

        # 1、将label_queue和feature_queue扩展到batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * K * hidden_size

        # 2、计算相似度
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3、根据label取正样本和负样本的mask_index
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        # 4、根据mask_index取正样本和负样本的值
        feature_value = cos_sim.masked_select(pos_mask_index)
        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5、取所有的负样本和随机取N个正样本
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None

        pos_range = [index for index in range(pos_min)]
        pos_index = random.sample(pos_range, self.top_k)
        pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample = pos_sample[:, pos_index]
        # pos_sample_last = pos_sample[:, -self.end_k:]
        # pos_sample_last = pos_sample_last.view([-1, 1])

        # pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_sample = neg_sample.repeat([1, self.top_k])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def forward(self,
                query,
                positive_sample=None,
                negative_sample=None,
                ):
        labels = query["labels"]
        labels = labels.view(-1)
        if not self.memory_bank:
            with torch.no_grad():
                self.update_encoder_k()
                update_sample = self.reshape_dict(positive_sample)
                bert_output_p = self.encoder_k(**update_sample)
                update_keys = bert_output_p[1]
                update_keys = self.contrastive_liner_k(update_keys)
                update_keys = l2norm(update_keys)
                tmp_labels = labels.unsqueeze(-1)
                tmp_labels = tmp_labels.repeat([1, self.update_num])
                tmp_labels = tmp_labels.view(-1)
                self._dequeue_and_enqueue(update_keys, tmp_labels)

        query.pop("labels")
        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]
        liner_q = self.contrastive_liner_q(q)
        liner_q = l2norm(liner_q)
        logits_cls = self.classifier_liner(q)

        if self.num_labels == 1:
            loss_fct = MSELoss()
            loss_cls = loss_fct(logits_cls.view(-1), labels)
        else:
            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.num_labels), labels)

        if self.random_positive:
            logits_con = self.select_pos_neg_random(liner_q, labels)
        else:
            logits_con = self.select_pos_neg_sample(liner_q, labels)

        if logits_con is not None:
            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
            loss_fct = CrossEntropyLoss()
            loss_con = loss_fct(logits_con, labels_con)

            loss = loss_con * self.config.contrastive_rate_in_training + \
                   loss_cls * (1 - self.config.contrastive_rate_in_training)
        else:
            loss = loss_cls

        return SequenceClassifierOutput(
            loss=loss,
        )


    # 考虑eval过程写在model内部？
    def predict(self, query):
        with torch.no_grad():
            bert_output_q = self.encoder_q(**query)
            q = bert_output_q[1]
            logits_cls = self.classifier_liner(q)
            contrastive_output = self.contrastive_liner_q(q)
        return contrastive_output, logits_cls

    def get_features(self, query):
        with torch.no_grad():
            bert_output_k = self.encoder_k(**query)
            contrastive_output = self.contrastive_liner_k(bert_output_k[1])
        return contrastive_output

    def update_queue_by_bert(self,
                             inputs=None,
                             labels=None
                             ):
        with torch.no_grad():
            update_sample = self.reshape_dict(inputs)
            roberta_output = self.encoder_k(**update_sample)
            update_keys = roberta_output[1]
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(update_keys, tmp_labels)


class ContrastiveRobertaMoCoKnnBert(nn.Module):
    def __init__(self, config):
        super(ContrastiveRobertaMoCoKnnBert, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        if config.load_trained_model:
            self.encoder_q = RobertaModel(config)
            self.encoder_k = RobertaModel(config)
        else:
            self.encoder_q = RobertaModel(config).from_pretrained(config.model_name, config=config)
            self.encoder_k = RobertaModel(config).from_pretrained(config.model_name, config=config)

        self.classifier = ClassificationHead(config, self.num_labels)
        self.contrastive_liner_q = ContrastiveHead(config)
        self.contrastive_liner_k = ContrastiveHead(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.m = 0.999
        self.T = 0.07
        self.train_multi_head = config.train_multi_head
        self.multi_head_num = config.multi_head_num

        if not config.load_trained_model:
            self.init_weights()

        self.K = config.queue_size

        self.register_buffer("label_queue", torch.randint(0, self.num_labels, [self.K]))
        self.register_buffer("feature_queue", torch.randn(self.K, config.hidden_size))
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.top_k = config.knn_num
        self.end_k = config.end_k
        self.update_num = config.positive_num

        self.memory_bank = config.memory_bank

    def _dequeue_and_enqueue(self, keys, label):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            head_size = self.K - ptr
            head_keys = keys[: head_size]
            head_label = label[: head_size]
            end_size = ptr + batch_size - self.K
            end_key = keys[head_size:]
            end_label = label[head_size:]
            self.feature_queue[ptr:, :] = head_keys
            self.label_queue[ptr:] = head_label
            self.feature_queue[:end_size, :] = end_key
            self.label_queue[:end_size] = end_label
        else:
            # replace the keys at ptr (dequeue ans enqueue)
            self.feature_queue[ptr: ptr + batch_size, :] = keys
            self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def select_pos_neg_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()        # K
        feature_queue = self.feature_queue.clone().detach()    # K * hidden_size

        # 1、将label_queue和feature_queue扩展到batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * K * hidden_size

        # 2、计算相似度
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3、根据label取正样本和负样本的mask_index
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        # 4、根据mask_index取正样本和负样本的值
        feature_value = cos_sim.masked_select(pos_mask_index)
        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5、取所有的负样本和前top_k 个正样本， -M个正样本（离中心点最远的样本）
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.top_k]
        pos_sample_last = pos_sample[:, -self.end_k:]
        # pos_sample_last = pos_sample_last.view([-1, 1])

        pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_sample = neg_sample.repeat([1, self.top_k + self.end_k])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def forward(self,
                query,
                positive_sample=None,
                negative_sample=None,
                ):
        labels = query["labels"]
        labels = labels.view(-1)

        if not self.memory_bank:
            with torch.no_grad():
                self.update_encoder_k()
                update_sample = self.reshape_dict(positive_sample)
                bert_output_p = self.encoder_k(**update_sample)
                # bert_output_p = self.encoder_q(**update_sample)
                update_keys = bert_output_p[1]
                update_keys = self.contrastive_liner_k(update_keys)
                # update_keys = self.contrastive_liner_q(update_keys)
                update_keys = l2norm(update_keys)
                tmp_labels = labels.unsqueeze(-1)
                tmp_labels = tmp_labels.repeat([1, self.update_num])
                tmp_labels = tmp_labels.view(-1)
                self._dequeue_and_enqueue(update_keys, tmp_labels)

        query.pop("labels")
        roberta_output_q = self.encoder_q(**query)[1]
        logits_cls = self.classifier(roberta_output_q)
        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits_cls.view(-1, self.num_labels), labels)

        liner_q = self.contrastive_liner_q(roberta_output_q)
        liner_q = l2norm(liner_q)
        logits_con = self.select_pos_neg_sample(liner_q, labels)
        labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
        loss_fct = CrossEntropyLoss()
        loss_con = loss_fct(logits_con, labels_con)

        loss = loss_con * self.config.contrastive_rate_in_training + \
               loss_cls * (1 - self.config.contrastive_rate_in_training)

        return SequenceClassifierOutput(
            loss=loss,
        )

    def predict(self, query):
        with torch.no_grad():
            roberta_output_q = self.encoder_q(**query)
            q = roberta_output_q[1]
            logits_cls = self.classifier(q)
            contrastive_output = self.contrastive_liner_q(q)
        return contrastive_output, logits_cls

    def update_queue_by_bert(self,
                             inputs=None,
                             labels=None
                             ):
        with torch.no_grad():
            update_sample = self.reshape_dict(inputs)
            roberta_output = self.encoder_k(**update_sample)
            update_keys = roberta_output[1]
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(update_keys, tmp_labels)


class ClusterMoCoKnnBert(nn.Module):

    def __init__(self, config):
        super(ClusterMoCoKnnBert, self).__init__()
        self.config = config
        self.num_labels = config.num_cls_labels
        self.cluster_labels = config.cluster_labels
        if config.load_trained_model:
            self.encoder_q = BertModel(config)
            self.encoder_k = BertModel(config)
        else:
            self.encoder_q = BertModel.from_pretrained(config.model_name, config=config)
            self.encoder_k = BertModel.from_pretrained(config.model_name, config=config)

        self.classifier_liner = ClassificationHead(config, self.num_labels)
        self.classifier_cluster_liner = ClassificationHead(config, self.cluster_labels)

        self.contrastive_liner_q = ContrastiveHead(config)
        self.contrastive_liner_k = ContrastiveHead(config)

        self.m = 0.999
        self.T = 0.07
        self.train_multi_head = config.train_multi_head
        self.multi_head_num = config.multi_head_num

        if not config.load_trained_model:
            self.init_weights()

        # create the label_queue and feature_queue
        self.K = config.queue_size

        self.register_buffer("label_queue", torch.randint(0, self.num_labels, [self.K]))
        self.register_buffer("cluster_queue", torch.randint(0, self.cluster_labels, [self.K]))
        self.register_buffer("feature_queue", torch.randn(self.K, config.hidden_size))
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.top_k = config.top_k
        self.update_num = config.positive_num

    def _dequeue_and_enqueue(self, keys, label, cluster_label):
        # TODO 我们训练过程batch_size是一个变动的，每个epoch的最后一个batch数目后比较少，这里需要进一步修改
        # keys = concat_all_gather(keys)
        # label = concat_all_gather(label)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            head_size = self.K - ptr
            head_keys = keys[: head_size]
            head_label = label[: head_size]
            head_cluster_label = cluster_label[: head_size]
            end_size = ptr + batch_size - self.K
            end_keys = keys[head_size:]
            end_label = label[head_size:]
            end_cluster_label = cluster_label[head_size:]
            self.feature_queue[ptr:, :] = head_keys
            self.label_queue[ptr:] = head_label
            self.cluster_queue[ptr:] = head_cluster_label

            self.feature_queue[: end_size, :] = end_keys
            self.label_queue[: end_size] = end_label
            self.cluster_queue[: end_size] = end_cluster_label

        # replace the keys at ptr (dequeue ans enqueue)
        else:
            self.feature_queue[ptr: ptr + batch_size, :] = keys
            self.label_queue[ptr: ptr + batch_size] = label
            self.cluster_queue[ptr: ptr + batch_size] = cluster_label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def select_pos_neg_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor, cluster_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()    # K
        cluster_queue = self.cluster_queue.clone().detach()
        feature_queue = self.feature_queue.clone().detach()    # K * hidden_size

        # 1、将label_queue和feature_queue扩展到batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_cluster_queue = cluster_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * K * hidden_size

        # 2、计算相似度
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3、根据label取正样本和负样本的mask_index
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        tmp_cluster_label = cluster_q.unsqueeze(1)
        tmp_cluster_label = tmp_cluster_label.repeat([1, self.K])

        cluster_mask_index = torch.eq(tmp_cluster_queue, tmp_cluster_label)
        label_mask_index = torch.eq(tmp_label_queue, tmp_label)

        pos_mask_index = torch.eq(cluster_mask_index, label_mask_index)

        neg_mask_index = torch.eq(cluster_mask_index, ~label_mask_index)

        # 4、根据mask_index取正样本和负样本的值
        feature_value = cos_sim.masked_select(pos_mask_index)
        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5、取所有的负样本和前top_k 个正样本， -M个正样本（离中心点最远的样本）
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = min(pos_number.min(), self.top_k)
        if pos_min == 0:
            return None
        pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        # pos_sample_top_k = pos_sample[:, 0:self.top_k]
        # pos_sample_last = pos_sample[:, -1]
        # pos_sample_last = pos_sample_last.view([-1, 1])
        #
        # pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.view([-1, 1])
        #
        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_sample = neg_sample.view([-1, 1])
        neg_sample = neg_sample.repeat([1, pos_min])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def forward(self,
                query,
                positive_sample=None,
                negative_sample=None,
                ):

        if not self.training:
            labels = query["labels"]
            cls_label = labels % self.num_labels
            cls_label = cls_label.view(-1)
            query.pop("labels")
            bert_output_q = self.encoder_q(**query)
            q = bert_output_q[1]
            logits_cls = self.classifier_liner(q)
            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.num_labels), cls_label)
            return SequenceClassifierOutput(
                loss=loss_cls,
                logits=logits_cls
            )

        labels = query["labels"]
        cluster_label = labels // self.num_labels
        cluster_label = cluster_label.view(-1)
        cls_label = labels % self.num_labels
        cls_label = cls_label.view(-1)

        with torch.no_grad():
            self.update_encoder_k()
            update_sample = self.reshape_dict(positive_sample)
            bert_output_p = self.encoder_k(**update_sample)
            update_keys = bert_output_p[1]
            update_keys = self.contrastive_liner_k(update_keys)
            update_keys = l2norm(update_keys)
            tmp_cls_labels = cls_label.unsqueeze(-1)
            tmp_cls_labels = tmp_cls_labels.repeat([1, self.update_num])
            tmp_cls_labels = tmp_cls_labels.view(-1)
            tmp_cluster_label = cluster_label.unsqueeze(-1)
            tmp_cluster_label = tmp_cluster_label.repeat([1, self.update_num])
            tmp_cluster_label = tmp_cluster_label.view(-1)

            self._dequeue_and_enqueue(update_keys, tmp_cls_labels, tmp_cluster_label)

        query.pop("labels")
        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]
        liner_q = self.contrastive_liner_q(q)
        liner_q = l2norm(liner_q)
        logits_cls = self.classifier_liner(q)
        logits_cluster = self.classifier_cluster_liner(liner_q)

        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits_cls.view(-1, self.num_labels), cls_label)
        # loss_cluster = loss_fct(logits_cluster.view(-1, self.cluster_labels), cluster_label)

        logits_con = self.select_pos_neg_sample(liner_q, cls_label, cluster_label)

        if logits_con is not None:
            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
            loss_fct = CrossEntropyLoss()
            loss_con = loss_fct(logits_con, labels_con)
            #
            loss = loss_con * self.config.contrastive_rate_in_training + \
                   loss_cls * (1 - self.config.contrastive_rate_in_training)
            # loss = loss_cls
                   # loss_cluster * (1 - self.config.contrastive_rate_in_training)
            # loss = loss_cls * 0.6 + loss_cluster * 0.2 + loss_con * 0.2
        else:
            loss = loss_cls

        return SequenceClassifierOutput(
            loss=loss,
        )

    def update_queue_by_bert(self,
                             inputs=None,
                             labels=None
                             ):
        with torch.no_grad():
            cluster_label = labels // self.num_labels
            cluster_label = cluster_label.view(-1)
            cls_label = labels % self.num_labels
            cls_label = cls_label.view(-1)
            update_sample = self.reshape_dict(inputs)
            roberta_output = self.encoder_k(**update_sample)
            update_keys = roberta_output[1]
            self._dequeue_and_enqueue(update_keys, cls_label, cluster_label)


    # 考虑eval过程写在model内部？
    def predict(self, query):
        with torch.no_grad():
            bert_output_q = self.encoder_q(**query)
            q = bert_output_q[1]
            logits_cls = self.classifier_liner(q)
            contrastive_output = self.contrastive_liner_q(q)
        return contrastive_output, logits_cls

    def get_features(self, query):
        with torch.no_grad():
            bert_output_k = self.encoder_k(**query)
            contrastive_output = self.contrastive_liner_k(bert_output_k[1])
        return contrastive_output


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


