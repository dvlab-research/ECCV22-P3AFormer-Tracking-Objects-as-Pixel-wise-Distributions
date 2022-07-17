import copy
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import normal_
from models.ops.modules import MSDeformAttn
import pdb


class DeformableTransformer(nn.Module):
    """
        encoder:
        reference_points = images shape
        output=src, pos, reference_points, src, spatial_shapes, level_start_index, padding_mask

    decoder:
        reference_points = fc(query) 256-> 2
        output=tgt, query_pos, reference_points_input = reference_points*valid_ratio,
         src= memory = img feat maps +attention, src_spatial_shapes, src_level_start_index, src_padding_mask
    """

    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4):
        super().__init__()
        self.dense_fuse = True
        double_d_model = d_model * 2 if self.dense_fuse else d_model
        self.d_model = d_model
        self.nhead = nhead
        self.real_dim_feedforward = dim_feedforward
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(double_d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.level_embed_decoder = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.level_embed_decoder_pre = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # my_ffn
        self.linear1 = nn.Linear(double_d_model, 512)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, double_d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(double_d_model)

        self.curr_pre = nn.Linear(double_d_model, double_d_model)
        self.curr_pre_dropout = nn.Dropout(dropout)
        self.curr_pre_norm = nn.LayerNorm(double_d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        # if not self.two_stage:

        print('warning YOU DONT INIT REFERENCE_POINTS WEIGHT, BECAUSE IT IS CONSTANT.')
        # if True:
        #     xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        #     constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)
        normal_(self.level_embed_decoder)

    # transform memory to a query embed
    def my_forward_ffn(self, memory):
        memory2 = self.linear2(self.dropout2(self.activation(self.linear1(memory))))
        return self.norm2(memory + self.dropout3(memory2))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_sum_h = torch.sum(~mask, 1, keepdim=True)
        valid_H, _ = torch.max(valid_sum_h, dim=2)
        valid_H.squeeze_(1)
        valid_sum_w = torch.sum(~mask, 2, keepdim=True)
        valid_W, _ = torch.max(valid_sum_w, dim=1)
        valid_W.squeeze_(1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio_h = torch.clamp(valid_ratio_h, min=1e-3, max=1.1)
        valid_ratio_w = torch.clamp(valid_ratio_w, min=1e-3, max=1.1)
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, pre_srcs=None, pre_masks=None, pre_hms=None, pre_pos_embeds=None):
        if pre_srcs is not None:
            # prepare input for encoder #
            src_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            lvl_pos_embed_flatten_decoder = []

            # pre
            pre_src_flatten = []
            pre_mask_flatten = []
            pre_lvl_pos_embed_flatten = []
            pre_lvl_pos_embed_flatten_decoder = []

            spatial_shapes = []  # save spatial shape of each level/scale
            # lvl => feat scale lvl
            for lvl, (src, mask, pos_embed, pre_src, pre_mask, pre_pos_embed) in enumerate(zip(srcs, masks, pos_embeds, pre_srcs, pre_masks, pre_pos_embeds)):
                assert pre_src.shape == src.shape
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                src_flatten.append(src.flatten(2).transpose(1, 2))
                mask_flatten.append(mask.flatten(1))
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                # + a learnable level position offset embed, !!not query embed!!, == deformable
                lvl_pos_embed_flatten.append(pos_embed + self.level_embed[lvl].view(1, 1, -1))
                # xyh #
                lvl_pos_embed_flatten_decoder.append(pos_embed + self.level_embed_decoder[lvl].view(1, 1, -1))
                # pre
                pre_pos_embed = pre_pos_embed.flatten(2).transpose(1, 2)
                pre_lvl_pos_embed_flatten.append(pre_pos_embed + self.level_embed[lvl].view(1, 1, -1))
                pre_lvl_pos_embed_flatten_decoder.append(pre_pos_embed + self.level_embed_decoder_pre[lvl].view(1, 1, -1))

                pre_src_flatten.append(pre_src.flatten(2).transpose(1, 2))
                pre_mask_flatten.append(pre_mask.flatten(1))
                # b,hxw, 1

            src_flatten = torch.cat(src_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

            # pre
            pre_src_flatten = torch.cat(pre_src_flatten, 1)
            pre_mask_flatten = torch.cat(pre_mask_flatten, 1)
            pre_lvl_pos_embed_flatten = torch.cat(pre_lvl_pos_embed_flatten, 1)
            pre_valid_ratios = torch.stack([self.get_valid_ratio(pre_m) for pre_m in pre_masks], 1)
            pre_lvl_pos_embed_flatten_decoder = torch.cat(pre_lvl_pos_embed_flatten_decoder, 1)

            # xyh #
            lvl_pos_embed_flatten_decoder = torch.cat(lvl_pos_embed_flatten_decoder, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
            
            if self.dense_fuse:
                joint_src_flatten = torch.cat([src_flatten, pre_src_flatten], dim=-1)
                joint_pos_embed = torch.cat([lvl_pos_embed_flatten, pre_lvl_pos_embed_flatten], dim=-1)
                # joint_mask_flatten = torch.cat([mask_flatten, pre_mask_flatten], dim=-1)
                joint_mask_flatten = mask_flatten
                joint_memory = self.encoder(joint_src_flatten, spatial_shapes, level_start_index, valid_ratios, joint_pos_embed, joint_mask_flatten)
                joint_pos_embed = torch.cat([lvl_pos_embed_flatten_decoder, pre_lvl_pos_embed_flatten_decoder], dim=-1)
                query_embed = self.my_forward_ffn(joint_memory.detach())
                reference_points = self.encoder.get_reference_points(spatial_shapes, valid_ratios, src.device)
                hs, _ = self.decoder(tgt=query_embed, pre_tgt=None, reference_points=reference_points, src=joint_memory,
                                    src_spatial_shapes=spatial_shapes, src_level_start_index=level_start_index,
                                    query_pos=query_embed, pre_query_pos=None,
                                    src_padding_mask=joint_mask_flatten,
                                    lvl_pos_embed_flatten=joint_pos_embed,
                                    pre_lvl_pos_embed_flatten=None,
                                    pre_src_padding_mask=None, pre_src=None,
                                    pre_ref_pts=None)
            else:
                # encoder
                memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                                    mask_flatten)
                with torch.no_grad():
                    pre_memory = self.encoder(pre_src_flatten, spatial_shapes, level_start_index, pre_valid_ratios,
                                            pre_lvl_pos_embed_flatten, pre_mask_flatten)

                # prepare input for decoder #
                query_embed = self.my_forward_ffn(memory.detach())
                pre_query_embed = self.curr_pre_dropout(self.activation(self.curr_pre(query_embed.detach())))
                tgt = query_embed
                pre_tgt = pre_query_embed
                # todo： we use fixed ref points as encoder
                reference_points = self.encoder.get_reference_points(spatial_shapes, valid_ratios, src.device)
                pre_reference_points = self.encoder.get_reference_points(spatial_shapes, pre_valid_ratios, pre_src.device)

                # decoder
                hs, _ = self.decoder(tgt=tgt, pre_tgt=pre_tgt, reference_points=reference_points, src=memory,
                                    src_spatial_shapes=spatial_shapes, src_level_start_index=level_start_index,
                                    query_pos=query_embed, pre_query_pos=pre_query_embed,
                                    src_padding_mask=mask_flatten,
                                    lvl_pos_embed_flatten=lvl_pos_embed_flatten_decoder,
                                    pre_lvl_pos_embed_flatten=pre_lvl_pos_embed_flatten_decoder,
                                    pre_src_padding_mask=pre_mask_flatten, pre_src=pre_memory,
                                    pre_ref_pts=pre_reference_points)
            return hs
        else:
            # prepare input for encoder #
            src_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            lvl_pos_embed_flatten_decoder = []
            spatial_shapes = []  # save spatial shape of each level/scale
            # lvl => feat scale lvl
            for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                src_flatten.append(src.flatten(2).transpose(1, 2))
                mask_flatten.append(mask.flatten(1))
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                # + a learnable level position offset embed, !!not query embed!!, == deformable
                lvl_pos_embed_flatten.append(pos_embed + self.level_embed[lvl].view(1, 1, -1))
                # xyh #
                lvl_pos_embed_flatten_decoder.append(pos_embed + self.level_embed_decoder[lvl].view(1, 1, -1))

            src_flatten = torch.cat(src_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

            # xyh #
            lvl_pos_embed_flatten_decoder = torch.cat(lvl_pos_embed_flatten_decoder, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
            
            # encoder
            memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                                mask_flatten)

            # prepare input for decoder #
            query_embed = self.my_forward_ffn(memory.detach())
            tgt = query_embed
            # todo： we use fixed ref points as encoder
            reference_points = self.encoder.get_reference_points(spatial_shapes, valid_ratios, src.device)

            # decoder
            hs, _ = self.decoder(tgt=tgt, pre_tgt=None, reference_points=reference_points, src=memory,
                                src_spatial_shapes=spatial_shapes, src_level_start_index=level_start_index,
                                query_pos=query_embed, pre_query_pos=None,
                                src_padding_mask=mask_flatten,
                                lvl_pos_embed_flatten=lvl_pos_embed_flatten_decoder)
            return hs


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.densefusion = True
        self.d_model = d_model
        self.double_d_model = d_model * 2 if self.densefusion else d_model
        self.densefusion = True
        self.self_attn = MSDeformAttn(self.double_d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.double_d_model)

        # ffn
        self.linear1 = nn.Linear(self.double_d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, self.double_d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(self.double_d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))

            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            """
            new ref_y torch.Size([20, 15698]) = [btch, H_*W_]
            new ref_x torch.Size([20, 15698])

            """
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        output_d_model = d_model
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, output_d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(output_d_model)

        # ffn pre
        self.pre_linear1 = nn.Linear(d_model, d_ffn)
        self.pre_dropout3 = nn.Dropout(dropout)
        self.pre_linear2 = nn.Linear(d_ffn, output_d_model)
        self.pre_dropout4 = nn.Dropout(dropout)
        self.pre_norm3 = nn.LayerNorm(output_d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_ffn_pre(self, pre_tgt):
        pre_tgt2 = self.pre_linear2(self.pre_dropout3(self.activation(self.pre_linear1(pre_tgt))))
        pre_tgt = pre_tgt + self.pre_dropout4(pre_tgt2)
        pre_tgt = self.pre_norm3(pre_tgt)
        return pre_tgt

    def forward(self, tgt, pre_tgt, query_pos, pre_query_pos, reference_points, src, src_spatial_shapes,
                level_start_index, src_padding_mask=None, reference_points_static=None, lvl_pos_embed_flatten=None,
                pre_lvl_pos_embed_flatten=None, pre_src_padding_mask=None, pre_src=None, pre_ref_pts=None):
        if pre_tgt is None:
            tgt = tgt + self.dropout1(self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points,
                                            src, src_spatial_shapes, level_start_index, src_padding_mask))
            pre_tgt = None
            return self.forward_ffn(self.norm1(tgt)), None
        else:
            # self attention
            # find objects at t -1
            pre_tgt = pre_tgt + self.dropout2(self.self_attn(self.with_pos_embed(pre_tgt, pre_query_pos),
                                                            pre_ref_pts, pre_src, src_spatial_shapes, level_start_index,
                                                            pre_src_padding_mask))

            # find objects at t
            tgt = tgt + self.dropout1(self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points,
                                                    src, src_spatial_shapes, level_start_index, src_padding_mask))

            # ffn: 2 fc layers with dropout, 256 -> 1024-> 256
            return self.forward_ffn(self.norm1(tgt)), self.forward_ffn_pre(self.norm2(pre_tgt))


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, merge_mode=0):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.reg = None
        self.ida_up = None
        self.wh = None
        self.merge_mode = merge_mode

    def split_featsV2(self, spatial_shapes, level_start_index, hidden, pre_hidden):
        """
        level start index:  tensor([ 0, 26219, 32855, 34535], device='cuda:0'), indicating level start index
        spatial_shapes: list of (H, W) of the PADDED feature maps
        hidden: [b, 30k, 256]
        """
        if pre_hidden is None:
            split_hidden = []
            b, s, c = hidden.shape

            for lvl in range(len(spatial_shapes)):
                curr_shape = spatial_shapes[lvl]
                start_indx = level_start_index[lvl]
                if lvl + 1 == len(spatial_shapes):
                    end_indx = s
                else:
                    end_indx = level_start_index[lvl + 1]

                split_hidden.append(
                    hidden[:, start_indx:end_indx, :].view(b, curr_shape[0], curr_shape[1], c).permute(0, 3, 1, 2).contiguous())
            return split_hidden, None
        else:
            split_hidden = []
            pre_split_hidden = []
            b, s, c = hidden.shape
            assert hidden.shape == pre_hidden.shape

            for lvl in range(len(spatial_shapes)):
                curr_shape = spatial_shapes[lvl]
                start_indx = level_start_index[lvl]
                if lvl + 1 == len(spatial_shapes):
                    end_indx = s
                else:
                    end_indx = level_start_index[lvl + 1]

                split_hidden.append(
                    hidden[:, start_indx:end_indx, :].view(b, curr_shape[0], curr_shape[1], c).permute(0, 3, 1, 2).contiguous())

                pre_split_hidden.append(
                    pre_hidden[:, start_indx:end_indx, :].view(b, curr_shape[0], curr_shape[1], c).permute(0, 3, 1, 2).contiguous())

            return split_hidden, pre_split_hidden

    # xyh #
    def forward(self, tgt, pre_tgt, reference_points, src, src_spatial_shapes, src_level_start_index,
                query_pos=None, pre_query_pos=None, src_padding_mask=None, lvl_pos_embed_flatten=None,
                pre_lvl_pos_embed_flatten=None, pre_src_padding_mask=None, pre_src=None, pre_ref_pts=None):
        output = tgt
        pre_output = pre_tgt

        intermediate = []

        for lid, layer in enumerate(self.layers):
            output, pre_output = layer(tgt=output, pre_tgt=pre_output, query_pos=query_pos, pre_query_pos=pre_query_pos,
                                       reference_points=reference_points, src=src,
                                       src_spatial_shapes=src_spatial_shapes,
                                       level_start_index=src_level_start_index, src_padding_mask=src_padding_mask,
                                       reference_points_static=reference_points,
                                       lvl_pos_embed_flatten=lvl_pos_embed_flatten,
                                       pre_lvl_pos_embed_flatten=pre_lvl_pos_embed_flatten,
                                       pre_src_padding_mask=pre_src_padding_mask,
                                       pre_src=pre_src, pre_ref_pts=pre_ref_pts)

            split, pre_split = self.split_featsV2(src_spatial_shapes, src_level_start_index, output, pre_output)

            if self.return_intermediate:
                intermediate.append([split, pre_split])

        if self.return_intermediate:
            return intermediate, None

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points
    )
