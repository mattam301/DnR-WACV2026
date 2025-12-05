import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityRepresentationAutoencoder(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim):
        super().__init__()
        # Encoder: n -> bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.ReLU()
        )
        # Decoder: bottleneck -> n
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim // 2, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, feature_dim)
        )

    def forward(self, x):
        x_aug = []
        for x_i in x:
            z = self.encoder(x_i)
            recon = self.decoder(z)
            x_aug.append(F.relu(recon))
        return x_aug


def autoencoder_augmentation(self, x):
    feature_dim = x[0].size(-1)
    bottleneck_dim = max(feature_dim // 2, 1)  # ensure >0
    autoencoder = ModalityRepresentationAutoencoder(feature_dim, bottleneck_dim).to(x[0].device)
    
    x_aug = autoencoder(x)

    assert all(x_aug_i.size() == x_i.size() for x_aug_i, x_i in zip(x_aug, x)), \
        f"Augmented representation size {x_aug} does not match original size {x}"
    
    return x_aug
class CoMM(nn.Module):
    def __init__(self, comm_enc, hidden_dim, n_classes, augmentation_style="linear"):
        """
        CoMM module that can be plugged into any multimodal backbone.

        Args:
            comm_enc: the communication encoder (e.g., MMFusion or custom transformer)
            hidden_dim: dimension of modality representations
            n_classes: number of output classes
            augmentation_style: "linear", "gaussian", or "autoencoder"
            late_comm: if True, apply CoMM after cross-modal fusion; 
                       else apply CoMM on raw modality embeddings (early)
        """
        super().__init__()
        
        #hardcoded
        D_text, D_audio, D_visual = 1024, 1024, 1024
        # Temporal convolutional layers
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)
        
        self.comm_enc = comm_enc
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        # self.late_comm = late_comm

        if augmentation_style == "autoencoder":
            self.augment_1 = self.autoencoder_augmentation
            self.augment_2 = self.autoencoder_augmentation
        elif augmentation_style == "linear":
            self.augment_1 = self.modality_representation_linear_augmentation
            self.augment_2 = self.modality_representation_linear_augmentation
        elif augmentation_style == "gaussian":
            self.augment_1 = self.modality_representation_gaussian_augmentation
            self.augment_2 = self.modality_representation_gaussian_augmentation

        # Projection head for z1/z2
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Fusion head for final output
        self.comm_fuse = nn.Linear(2 * hidden_dim, n_classes)

    @staticmethod
    def gen_all_possible_masks(n_mod: int):
        masks = []
        for L in range(n_mod):
            mask = [s == L for s in range(n_mod)]
            masks.append(mask)
        masks.append([True for _ in range(n_mod)])
        return masks

    def modality_representation_linear_augmentation(self, x):
        # Using a simple Linear layer to augment the representation and return a new representation of the same shape
        # making sure all tensors are on the same device
        # print(x[0].device)
        augmentation_layer = nn.Linear(x[0].size(-1), x[0].size(-1)).to(x[0].device)
        x_aug = [augmentation_layer(x_i) for x_i in x]
        assert all(x_aug_i.size() == x_i.size() for x_aug_i, x_i in zip(x_aug, x)), \
            f"Augmented representation size {x_aug} does not match original size {x}"
        # # Using a ReLU activation function to introduce non-linearity
        x_aug = [F.relu(x_aug_i) for x_aug_i in x_aug]
        # Returning the augmented representation
        # This is a simple augmentation, more complex methods can be used
        # depending on the task and the data.
        return x_aug
    def modality_representation_gaussian_augmentation(self, x):
        # Instead of using Linear layer, apply a random Gaussian noise to the representation
        noise_std = 0.8  # Standard deviation of the Gaussian noise
        x_aug = [x_i + torch.randn_like(x_i) * noise_std for x_i in x]
        assert all(x_aug_i.size() == x_i.size() for x_aug_i, x_i in zip(x_aug, x)), \
            f"Augmented representation size {x_aug} does not match original size {x}"
        return x_aug
    def autoencoder_augmentation(self, x):
        feature_dim = x[0].size(-1)
        bottleneck_dim = max(feature_dim // 2, 1)  # ensure >0
        autoencoder = ModalityRepresentationAutoencoder(feature_dim, bottleneck_dim).to(x[0].device)
        x_aug = autoencoder(x)
        assert all(x_aug_i.size() == x_i.size() for x_aug_i, x_i in zip(x_aug, x)), \
            f"Augmented representation size {x_aug} does not match original size {x}"
        return x_aug
    # ---------- forward ----------
    def forward(self, textf, acouf, visuf, all_transformer_out=None):
        """
        Args:
            textf, acouf, visuf: [B, T, D] representations of modalities
            all_transformer_out: [B, T, D] fused multimodal representation 
                                 (required if late_comm=True)

        Returns:
            z1, z2, all_final_out
        """
        # if self.late_comm:
        #     # Late: use post-fusion features
        #     if all_transformer_out is None:
        #         raise ValueError("all_transformer_out is required for late_comm=True")
        #     x = [textf, acouf, visuf]
        # else:
            # Early: use raw projected features
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)
        x = [textf, acouf, visuf]

        # Augment twice
        x1 = self.augment_1(x)
        x2 = self.augment_2(x)

        # All masks
        all_masks = self.gen_all_possible_masks(len(x))

        # Encode with CoMM encoder
        z1 = [self.head(z) for z in self.comm_enc(x1, mask_modalities=all_masks)] # => z1 is a list of 4 cases: [text only, audio only, visual only, all modalities]
        z2 = [self.head(z) for z in self.comm_enc(x2, mask_modalities=all_masks)]

        # If early_comm: add comm_true_out fusion
        if all_transformer_out is not None:
            comm_true_out = self.comm_enc(x, mask_modalities=None)  # [B, D]
            comm_expanded = comm_true_out.unsqueeze(1).expand(-1, all_transformer_out.size(1), -1)
            fused_out = torch.cat([all_transformer_out, comm_expanded], dim=-1)
        else:
            fused_out = None
        # all_final_out = self.comm_fuse(fused_out)

        return z1, z2, fused_out