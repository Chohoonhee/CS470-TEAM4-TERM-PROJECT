""" Code for the main model variants. """

import torch
import torch.nn as nn
from Proposed.model_utils import *
import math
import torch.nn.functional as F
import torchvision.models as models
import torch_scatter as ts


class AgentEncoderDecoder(nn.Module):
    def __init__(self, device, agent_embed_dim, nfuture, lstm_layers, lstm_dropout, noise_dim=16):

        super(AgentEncoderDecoder, self).__init__()

        self.device = device
        self.num_layers = lstm_layers
        self.agent_embed_dim = agent_embed_dim
        self.noise_dim = noise_dim

        self.agent_encoder = AgentEncoderLSTM(device=device, embedding_dim=agent_embed_dim,
                                              h_dim=agent_embed_dim, num_layers=lstm_layers, dropout=lstm_dropout)

        self.agent_decoder = AgentDecoderLSTM(
            # decoder has noise_dim more dimension than encoder due to GAN pretraining
            device=device, seq_len=nfuture, embedding_dim=agent_embed_dim + noise_dim,
            h_dim=agent_embed_dim + noise_dim, num_layers=lstm_layers, dropout=lstm_dropout
        )

    def encoder(self, past_agents_traj, past_agents_traj_len, future_agent_masks):
        # Encode Scene and Past Agent Paths
        past_agents_traj = past_agents_traj.permute(1, 0, 2)  # [B X T X D] -> [T X B X D]

        agent_lstm_encodings = self.agent_encoder(past_agents_traj, past_agents_traj_len).squeeze(0)  # [B X H]

        filtered_agent_lstm_encodings = agent_lstm_encodings[future_agent_masks, :]

        return filtered_agent_lstm_encodings

    def decoder(self, agent_encodings, decode_start_vel, decode_start_pos):

        total_agent = agent_encodings.shape[0]
        noise = torch.zeros((total_agent, self.noise_dim), device=self.device)

        fused_noise_encodings = torch.cat((agent_encodings, noise), dim=1)
        decoder_h = fused_noise_encodings.unsqueeze(0)

        predicted_trajs, final_decoder_h = self.agent_decoder(last_pos_rel=decode_start_vel,
                                                              hidden_state=decoder_h,
                                                              start_pos=decode_start_pos)
        predicted_trajs = predicted_trajs.permute(1, 0, 2)  # [B X L X 2]

        return predicted_trajs

    def forward(self, past_agents_traj, past_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos):

        agent_encodings = self.encoder(past_agents_traj, past_agents_traj_len, future_agent_masks)
        decode = self.decoder(agent_encodings, decode_start_vel[future_agent_masks],
                              decode_start_pos[future_agent_masks])

        return decode


class MultiAgentTrajectory(AgentEncoderDecoder):

    def __init__(self, device, embedding_dim, nfuture, att_dropout, lstm_layers=1, lstm_dropout=0.1):

        super(MultiAgentTrajectory, self).__init__(device, embedding_dim, nfuture, lstm_layers, lstm_dropout)
        self.CNN = ResNetBackbone('resnet50')
        self.self_attention = SelfAttention(d_model=embedding_dim, d_k=embedding_dim, d_v=embedding_dim, n_head=1, dropout=att_dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.visual_attention = Visual_Attention(128, 2048, 64, 128)
    def crossagent_attention_block(self, agent_lstm_encodings, num_past_agents):

        ############# Cross-agent Interaction Module ############

        # Mask agents in different scenes
        trj_num = agent_lstm_encodings.size(1) # number of traj
        batch_mask = torch.zeros((trj_num, trj_num), device=self.device) # trj_num x trj_num

        blocks = [torch.ones((i, i), device=self.device) for i in num_past_agents]

        start_i = torch.zeros_like(num_past_agents[0])
        end_i = torch.zeros_like(num_past_agents[0])
        for end_i, block in zip(torch.cumsum(num_past_agents, 0), blocks):
            batch_mask[start_i:end_i, start_i:end_i] = block
            start_i = end_i
        batch_mask = batch_mask.unsqueeze(0) # 1 x trj_num x trj_num

        residual = agent_lstm_encodings # trj_num x embed
        agent_embed = self.layer_norm(agent_lstm_encodings) # T x trj_num x embed
        agent_attended_agents = self.self_attention(agent_embed, agent_embed, agent_embed, batch_mask) # trj_num x embed

        agent_attended_agents += residual # trj_num x embed

        return agent_attended_agents

    def encoder(self, past_agents_traj, past_agents_traj_len, future_agent_masks, num_past_agents):
        # Encode Scene and Past Agent Paths
        past_agents_traj = past_agents_traj.permute(1, 0, 2)  # [B X T X D] -> [T X B X D]

        agent_lstm_encodings = self.agent_encoder(past_agents_traj, past_agents_traj_len).squeeze(0) # [B X H]


        agent_lstm_encodings = agent_lstm_encodings.unsqueeze(0) # 1 x trj_num x embed

        agent_attended_agents = self.crossagent_attention_block(agent_lstm_encodings, num_past_agents)

        agent_attended_agents = agent_attended_agents.squeeze(0) # trj_num x embed)



        filtered_agent_attended_agents = agent_attended_agents[future_agent_masks, :]

        return filtered_agent_attended_agents

    def forward(self, past_agents_traj, past_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, num_past_agents, scene_images):

        agent_encodings = self.encoder(past_agents_traj, past_agents_traj_len, future_agent_masks, num_past_agents)

        scene_images = scene_images.to(torch.device('cuda'))
        context = self.CNN(scene_images)
        context = context[future_agent_masks, :]
        attention = self.visual_attention(agent_encodings, context, agent_encodings)

        agent_encodings = torch.mul(agent_encodings, attention)

        decode = self.decoder(agent_encodings, decode_start_vel, decode_start_pos)
        #print(decode.shape)
        return decode


class AgentEncoderLSTM(nn.Module):
    def __init__(self, device, embedding_dim=32, h_dim=32, num_layers=1, dropout=0.3):
        super(AgentEncoderLSTM, self).__init__()

        self.device = device
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        if self.num_layers > 1:
            self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        else:
            self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers)

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, total_agents):
        # h_0, c_0 of shape (num_layers, batch, hidden_size)
        return (
            torch.zeros(self.num_layers, total_agents, self.h_dim, device=self.device),
            torch.zeros(self.num_layers, total_agents, self.h_dim, device=self.device)
        )

    def forward(self, obs_traj, src_lens):
        total_agents = obs_traj.size(1)
        hidden = self.init_hidden(total_agents)

        # Convert to relative, as Social GAN do
        rel_curr_ped_seq = torch.zeros_like(obs_traj)
        rel_curr_ped_seq[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]

        # Trajectory Encoding
        obs_traj_embedding = self.spatial_embedding(rel_curr_ped_seq.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.reshape(-1, total_agents, self.embedding_dim)

        obs_traj_embedding = nn.utils.rnn.pack_padded_sequence(obs_traj_embedding, src_lens.cpu(), enforce_sorted=False)
        output, (hidden_final, cell_final) = self.encoder(obs_traj_embedding, hidden)

        if self.num_layers > 1:
            hidden_final = hidden_final[0]

        return hidden_final


class AgentDecoderLSTM(nn.Module):

    def __init__(self, device, seq_len, embedding_dim=32, h_dim=32, num_layers=1, dropout=0.0):
        super(AgentDecoderLSTM, self).__init__()

        self.seq_len = seq_len
        self.device = device
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        if self.num_layers > 1:
            self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        else:
            self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers)

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def relative_to_abs(self, rel_traj, start_pos=None):
        """
        Inputs:
        - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
        - start_pos: pytorch tensor of shape (batch, 2)
        Outputs:
        - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
        """
        if start_pos is None:
            start_pos = torch.zeros_like(rel_traj[0])

        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)

        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos

        return abs_traj.permute(1, 0, 2)

    def forward(self, last_pos_rel, hidden_state, start_pos=None):
        """
        Inputs:
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)

        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        total_agents = last_pos_rel.size(0)
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.reshape(1, total_agents, self.embedding_dim)

        if self.num_layers > 1:
            zero_hidden_states = torch.zeros((self.num_layers - 1), hidden_state.shape[1], hidden_state.shape[2],
                                             device=self.device)
            decoder_h = torch.cat((hidden_state, zero_hidden_states), dim=0)
            decoder_c = torch.zeros_like(decoder_h)
            state_tuple = (decoder_h, decoder_c)
        else:
            decoder_c = torch.zeros_like(hidden_state)
            state_tuple = (hidden_state, decoder_c)

        predicted_rel_pos_list = []
        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            predicted_rel_pos = self.hidden2pos(output.reshape(total_agents, self.h_dim))
            predicted_rel_pos_list.append(predicted_rel_pos)  # [B X 2]

            decoder_input = self.spatial_embedding(predicted_rel_pos)
            decoder_input = decoder_input.reshape(1, total_agents, self.embedding_dim)

        predicted_rel_pos_result = torch.stack(predicted_rel_pos_list, dim=0)  # [L X B X 2]

        return self.relative_to_abs(predicted_rel_pos_result, start_pos), state_tuple[0]


