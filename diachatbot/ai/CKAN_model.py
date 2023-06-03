import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class CKAN(nn.Module):

    def __init__(self, args, n_entity, n_relation):
        super(CKAN, self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.dim = args.dim

        self.n_layer = args.n_layer
        self.agg = args.agg

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.transfer_matrix = nn.Parameter(torch.Tensor(self.n_relation, self.dim, self.dim))

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.transfer_matrix)

        self.attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )

                
        # init attention
        for layer in self.attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


    def build_train(
        self,
        items: torch.LongTensor,
        user_triple_set: list,
        item_triple_set: list,
    ):

        user_embeddings = []
        user_emb_0 = self.entity_emb(user_triple_set[0][0])
        user_embeddings.append(user_emb_0.mean(dim=1))
        
        for i in range(self.n_layer):
            h_emb = self.entity_emb(user_triple_set[0][i])
            r_emb = self.relation_emb(user_triple_set[1][i])
            t_emb = self.entity_emb(user_triple_set[2][i])
            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            user_embeddings.append(user_emb_i)

            
        item_embeddings = []
        item_emb_origin = self.entity_emb(items)
        item_embeddings.append(item_emb_origin)
        
        for i in range(self.n_layer):
            h_emb = self.entity_emb(item_triple_set[0][i])
            r_emb = self.relation_emb(item_triple_set[1][i])
            t_emb = self.entity_emb(item_triple_set[2][i])

            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            item_embeddings.append(item_emb_i)
        
        if self.n_layer > 0 and (self.agg == 'sum' or self.agg == 'pool'):
            item_emb_0 = self.entity_emb(item_triple_set[0][0])
            item_embeddings.append(item_emb_0.mean(dim=1))
        scores, e_v, e_u = self.predict(user_embeddings, item_embeddings)
        

        return scores, e_v, e_u
    
    
    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]

    
        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v),dim=-1)

        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]

        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)

        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores, e_v, e_u
    

    
    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        att_weights = self.attention(torch.cat((h_emb,r_emb),dim=-1)).squeeze(-1)
        att_weights_norm = F.softmax(att_weights,dim=-1)
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        emb_i = emb_i.sum(dim=1)
        return emb_i


    def focal_loss(self, predict, target):
        alpha = 0.75
        gamma = 2
        reduction = 'mean'
        loss = - alpha * (1 - predict) ** gamma * target * torch.log(predict) - (1 - alpha) * predict ** gamma * (1 - target) * torch.log(1 - predict)
        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
        return loss

    def calc_cf_loss(self,labels,items: torch.LongTensor,user_triple_set: list,item_triple_set: list):
        scores, e_v, e_u = self.build_train(items,user_triple_set,item_triple_set)
        # cf_loss = self.focal_loss(scores, labels)
        loss_func = nn.BCELoss()
        cf_loss = loss_func(scores, labels)
        l2_loss = _L2_loss_mean(e_v) + _L2_loss_mean(e_u)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        r_embed = self.relation_emb(r)
        W_r = self.transfer_matrix[r]
        # print(self.entity_emb.num_embeddings)
        # print(len(h))
        h_embed = self.entity_emb(h)
        pos_t_embed = self.entity_emb(pos_t)
        neg_t_embed = self.entity_emb(neg_t)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)

        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def calc_score(self,items: torch.LongTensor,user_triple_set: list,item_triple_set: list):
        scores, e_v, e_u = self.build_train(items,user_triple_set,item_triple_set)
        return scores

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'predict':
            return self.calc_score(*input)