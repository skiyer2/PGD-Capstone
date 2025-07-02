import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from .layers.base_liner import Linear
from .layers.base_context import ContextualEmbedding, ContextNetBlock

class ContextNet(BaseModel):
    def __init__(self, fix_SparseFeat, fix_DenseFeat, args):
        super(ContextNet, self).__init__(fix_SparseFeat, fix_DenseFeat, args)

        _ = self.get_input_dim()
        _ = self.get_field_index()

        if args.model_params['loss'] == 'binary_crossentropy':
            self.pediction_layer = nn.Sequential(
                nn.Sigmoid()
            )
        self.add_regularization_net_weight(self.pediction_layer.parameters())
        
        self.block_num = args.model_params['block_num']
        self.use_ce_every_layer = args.model_params['use_ce_every_layer']

        if self.use_ce_every_layer:
            self.ce = nn.ModuleList()
            for _ in range(args.model_params['block_num']):
                self.ce.append(
                    ContextualEmbedding(
                        num_fields=len(self.field_index),
                        feature_dim=args.model_params['embedding_dim'],
                        increase=args.model_params['increase_ratio'],
                    ))
            self.cn = nn.ModuleList()
            for _ in range(args.model_params['block_num']):
                self.cn.append(ContextNetBlock(
                    num_fields=len(self.field_index),
                    feature_dim=args.model_params['embedding_dim'],
                    net_model=args.model_params['net_model']
                ))
        else:
            self.ce = ContextualEmbedding(
                num_fields=len(self.field_index),
                feature_dim=args.model_params['embedding_dim'],
                increase=args.model_params['increase_ratio'],
            )
            self.cn = nn.ModuleList()
            for _ in range(args.model_params['block_num']):
                self.cn.append(ContextNetBlock(
                    num_fields=len(self.field_index),
                    feature_dim=args.model_params['embedding_dim'],
                    net_model=args.model_params['net_model']
                ))
        self.add_regularization_net_weight(self.ce.parameters())
        self.add_regularization_net_weight(self.cn.parameters())
        # DNN for classic mode, not used in fused GRU mode
        self.dnn_linear = nn.Linear(len(self.field_index) * args.model_params['embedding_dim'], 1, bias=False)
        self.add_regularization_net_weight(self.dnn_linear.parameters())
        
        if self.args.model_params['use_linear']:
            self.linear = Linear(
                self.input_dim,
                dropout=args.model_params['linear_dropout'],
                use_bias=True
            )
            self.add_regularization_net_weight(self.linear.parameters())

        # ===== FUSED SESSION GRU INTEGRATION =====
        self.use_gru = args.model_params.get('use_gru', False)
        self.context_out_dim = len(self.field_index) * args.model_params['embedding_dim']
        if self.use_gru:
            self.gru_hidden_dim = args.model_params.get('gru_hidden_dim', 64)
            self.gru = nn.GRU(
                input_size=self.context_out_dim,
                hidden_size=self.gru_hidden_dim,
                batch_first=True
            )
            self.session_mlp = nn.Linear(self.gru_hidden_dim, 1)

    def forward(self, x):
        """
        Fused mode: x is (batch, seq_len, num_features)
        Classic mode: x is (batch, num_features)
        """
        if self.use_gru:
            batch_size, seq_len, num_features = x.size()
            context_out_seq = []
            # Process each time step in the sequence through context blocks
            for t in range(seq_len):
                x_t = x[:, t, :]  # (batch, num_features)
                sparse_dict, dense_dict = self.get_embeddings(x_t)
                all_emb = [values for res, values in sparse_dict.items()]
                all_emb += [values for res, values in dense_dict.items()]
                all_emb = torch.cat(all_emb, dim=-1)  # (batch, context_out_dim)
                fm_emb = [values.unsqueeze(1) for res, values in sparse_dict.items()]
                fm_emb = torch.cat(fm_emb, dim=1)
                if self.use_ce_every_layer:
                    out = fm_emb
                    for index in range(self.block_num):
                        out = self.ce[index](out)
                        out = self.cn[index](out)
                else:
                    out = self.ce(fm_emb)
                    for index in range(self.block_num):
                        out = self.cn[index](out)
                context_flat = out.reshape(batch_size, -1)  # (batch, context_out_dim)
                context_out_seq.append(context_flat)
            # Stack context features for each time step: (batch, seq_len, context_out_dim)
            context_seq = torch.stack(context_out_seq, dim=1)
            # Pass sequence through GRU
            gru_out, h_n = self.gru(context_seq)  # h_n: (1, batch, hidden_dim)
            session_emb = h_n[-1]  # (batch, hidden_dim)
            logit = self.session_mlp(session_emb)  # (batch, 1)
            y_pred = self.pediction_layer(logit)
            reg_loss = self.get_regularization_loss()
            return y_pred, reg_loss

        else:
            # ======== Classic (row-wise) mode ========
            sparse_dict, dense_dict = self.get_embeddings(x)
            all_emb = [values for res, values in sparse_dict.items()]
            all_emb += [values for res, values in dense_dict.items()]
            all_emb = torch.cat(all_emb, dim=-1)

            fm_emb = [values.unsqueeze(1) for res, values in sparse_dict.items()]
            fm_emb = torch.cat(fm_emb, dim=1)
            
            if self.use_ce_every_layer:
                b = fm_emb.shape[0]
                out = fm_emb
                for index in range(self.block_num):
                    out = self.ce[index](out)
                    out = self.cn[index](out)
            else:
                b = fm_emb.shape[0]
                out = self.ce(fm_emb)
                for index in range(self.block_num):
                    out = self.cn[index](out)
                
            logit = self.dnn_linear(out.reshape(b, -1))
            if self.args.model_params['use_linear']:
                linear_logit = self.linear(all_emb)
                logit += linear_logit
            y_pred = self.pediction_layer(logit)
            reg_loss = self.get_regularization_loss()
            return y_pred, reg_loss